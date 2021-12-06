import dgl
from pandas.core import base
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from histocartography.ml import CellGraphModel
import random
import copy
import pickle
import os
import uuid
import json
import sys

from torch.utils.data import DataLoader
from histocartography.interpretability import (
    GraphGradCAMPPExplainer
)

LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"

def safe_create(dir):
    if dir == '' or dir == None:
        return
    if os.path.exists(dir):
        return
    os.makedirs(dir)

def data_to_dgl_graph(pt_data, cell_data, k=5, thresh=50, mode="distance", normalize=False, label_col="Group", negative_val=1):
    labs, ids = np.unique(cell_data.loc[:,'ClusterName'].to_numpy(), return_inverse=True)
    cell_data['ClusterID'] = ids
    graphs = []
    patients = []
    targets = []
    for spot_id,spot in enumerate(cell_data["spots"].unique()):
        subset = cell_data.loc[cell_data.loc[:,'spots'] == spot,:]
        features = subset.loc[:,'size':'Treg-PD-1+'].to_numpy() # TODO: This step depends on column order
        if normalize:
            features = np.nan_to_num((features-features.mean(axis=0))/features.std(axis=0))
        centroids = subset.loc[:,'X':'Y'].to_numpy()
        annotation = subset.loc[:,'ClusterID'].to_numpy()
        cell_ids = subset.loc[:,'CellID'].to_numpy()
        num_nodes = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)
        graph.ndata[FEATURES] = torch.FloatTensor(features)
        if annotation is not None:
            graph.ndata[LABEL] = torch.FloatTensor(annotation.astype(float))
        graph.ndata['cell_ids'] = torch.Tensor(cell_ids)
        graph.ndata['spot_id'] = torch.Tensor(np.array([spot_id for _ in range(num_nodes)]))
        adj = kneighbors_graph(
            centroids,
            k,
            mode=mode,
            include_self=False,
            metric="euclidean").toarray()
        if thresh is not None:
            adj[adj > thresh] = 0
        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))
        graphs.append(graph)
        assert(len(subset['patients'].unique()) == 1)
        patients.append(subset['patients'].unique()[0])
    for pt in patients:
        cp = pt_data.loc[pt_data.loc[:,'Patient'] == pt,label_col].values[0]
        #assert(cp == 1 or cp == 2)
        t = 0 if cp == negative_val else 1
        targets.append(t)
    return list(zip(graphs,targets)), labs, ids


def collate(batch):
    g = dgl.batch([example[0] for example in batch])
    l = torch.LongTensor([example[1] for example in batch])
    return g, l

def collate_graph(batch):
    return dgl.batch(batch)

def dataset_split(data, val_prop):
    if val_prop == 0:
        return data, data
    random.shuffle(data)
    train_data = data[:int(len(data)*val_prop)]
    val_data = data[int(len(data)*val_prop):]
    return train_data, val_data

#TODO: More sophisticated oversample
def oversample_positive(data, oversample_factor=2):
    negative = []
    positive = []
    for item in data:
        if item[1] == 0:
            negative.append(item)
        else:
            positive.append(item)
    positive = oversample_factor*positive
    return positive+negative

class CGModel():
    def __init__(self, gnn_params, classification_params, node_dim=74, num_classes=2):
        self.gnn_params = gnn_params
        self.classification_params = classification_params
        self.node_dim = node_dim
        self.num_classes = num_classes
        self.cgm = CellGraphModel(gnn_params, classification_params, node_dim, num_classes=2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        self.cgm.to(self.device)

    def train(self, data, val_prop=0, oversample_factor=1, lr=10e-4, weight_decay=5e-4, num_epochs=50, batch_size=8, output_dir=None):
        optimizer = torch.optim.Adam(
            self.cgm.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        # define loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        # training
        loss = 10e5
        val_accuracy = 0.
        train_dataloader = None
        val_dataloader = None
        loss_list = []
        val_accuracy_list = []
        train_data, val_data = dataset_split(data, val_prop)
        print("Train positive prop: {}".format(sum([i[1] for i in train_data])/len(train_data)))
        print("Val positive prop: {}".format(sum([i[1] for i in val_data])/len(val_data)))
        train_data = oversample_positive(train_data, oversample_factor=oversample_factor)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
        with trange(num_epochs) as t:
            for i in t:
                t.set_description('Loss={} | Val Accuracy={}'.format(loss, val_accuracy))
                self.cgm.train()
                for graphs, labels in train_dataloader:
                    graphs = graphs.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.cgm(graphs)
                    loss = loss_fn(logits, labels)
                    loss_list.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.cgm.eval()
                all_val_logits = []
                all_val_labels = []
                for graphs, labels in val_dataloader:
                    graphs = graphs.to(self.device)
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        logits = self.cgm(graphs)
                    all_val_logits.append(logits)
                    all_val_labels.append(labels)
                all_val_logits = torch.cat(all_val_logits).cpu()
                all_val_labels = torch.cat(all_val_labels).cpu()
                with torch.no_grad():
                    _, predictions = torch.max(all_val_logits, dim=1)
                    correct = torch.sum(predictions.to(int) == all_val_labels.to(int))
                    val_accuracy = round(correct.item() * 1.0 / len(all_val_labels), 2)
                    val_accuracy_list.append(val_accuracy)
                if output_dir is not None:
                    self.save("{}/model-{}.pkl".format(output_dir, i))
        return loss_list, val_accuracy_list, train_data, val_data
    
    def infer(self, graphs, batch_size=8):
        val_dataloader = DataLoader(graphs, batch_size=batch_size, collate_fn=collate_graph)
        with torch.no_grad():
            all_preds = []
            for graph_batch in val_dataloader:
                graph_batch = graph_batch.to(self.device)
                logits = self.cgm(graph_batch)
                _, predictions = torch.max(logits, dim=1)
                all_preds.append(predictions)
            all_preds = torch.cat(all_preds).cpu()
            return all_preds
    
    def test(self, data, batch_size=8):
        val_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate)
        self.cgm.eval()
        all_val_logits = []
        all_val_labels = []
        for graphs, labels in val_dataloader:
            graphs = graphs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                logits = self.cgm(graphs)
            all_val_logits.append(logits)
            all_val_labels.append(labels)
        all_val_logits = torch.cat(all_val_logits).cpu()
        all_val_labels = torch.cat(all_val_labels).cpu()
        with torch.no_grad():
            _, predictions = torch.max(all_val_logits, dim=1)
            correct = torch.sum(predictions.to(int) == all_val_labels.to(int))
            val_accuracy = round(correct.item() * 1.0 / len(all_val_labels), 2)
        return val_accuracy
    
    def save(self, model_path):
        safe_create(os.path.dirname(model_path))
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# cgm_reps is a list of CellGraphModels (not CGModels)
def get_model_score_dict(cgm_reps, data, labs):
    graph_frames = []
    for e in data:
        graph = e[0]
        target = e[1]
        l = graph.ndata['label'].numpy()
        pos = graph.ndata['centroid'].numpy()
        l = [labs[int(i)] for i in l]
        cell_id = graph.ndata['cell_ids']
        spot_id = graph.ndata['spot_id']
        d = pd.DataFrame({'cell_type': l, 'X':pos[:,0],'Y':pos[:,1],'cell_id':cell_id,'spot_id':spot_id})
        d.attrs['group'] = target+1
        num_models = len(cgm_reps)
        for i,cgm in enumerate(cgm_reps):
            grad_campp_explainer = GraphGradCAMPPExplainer(model=cgm)
            score = grad_campp_explainer.process(graph)[0]
            d[f'campp_{i}'] = score
            with torch.no_grad():
                c = copy.deepcopy(graph)
                logits = cgm(c)
                pred = torch.argmax(logits)
            d.attrs[f'pred_{i}'] = pred+1
        d['campp_median'] = d.loc[:,'campp_0':f'campp_{num_models-1}'].median(axis=1)
        d['campp_mad'] = d.loc[:,'campp_0':f'campp_{num_models-1}'].mad(axis=1)
        graph_frames.append(d)
    return graph_frames

DATA_PREPROC_SEARCH_SPACE = {
    "k": [10],
    "thresh": [10],
    "mode": ["connectivity", "distance"],
    "normalize": [True],
    "label_col": ["Group"],
    "negative_val": [1]
}


GNN_PARAMS_SEARCH_SPACE = {
    'readout_op': ['lstm'],
    'layer_type': ['gin_layer'],
    'hidden_dim': [64,128],
    'agg_type': ['mean'],
    'output_dim': [16, 32, 64],
    'num_layers': [1,2,3],
    'readout_type': ['mean'],
    'batch_norm': [True],
    'act': ['relu']
}

CLASSIFICATION_PARAM_SEARCH_SPACE = {
    'hidden_dim': [16,32,64,128],
    'num_layers': [1,2]
}

TRAIN_PARAM_SEARCH_SPACE = {
    "val_prop": [0.7],
    "oversample_factor": [1],
    "lr": [10e-3, 10e-4],
    "weight_decay": [5e-3],
    "num_epochs": [50],
    "batch_size": [4, 8]
}

class Experiment:

    def __init__(self, pt_data_path, cell_data_path, output_dir):
        self.pt_data_path = pt_data_path
        self.cell_data_path = cell_data_path
        self.output_dir = output_dir
        self.pt_data = pd.read_excel(pt_data_path)
        self.cell_data = pd.read_csv(cell_data_path)
        self.id = str(uuid.uuid4())[:8]
        self.base_path = "{}/{}".format(self.output_dir, self.id)
    def run_experiment(self, data_preproc_params, gnn_params, classification_params, train_params):
        data, labs, ids = data_to_dgl_graph(self.pt_data, self.cell_data, k=data_preproc_params['k'], thresh=data_preproc_params['thresh'], mode=data_preproc_params['mode'], normalize=data_preproc_params['normalize'], label_col=data_preproc_params['label_col'], negative_val=data_preproc_params['negative_val'])
        self.data_preproc_params = data_preproc_params
        self.gnn_params = gnn_params
        self.classification_params = classification_params
        self.train_params = train_params
        self.model = CGModel(gnn_params=gnn_params, classification_params=classification_params)
        self.loss_list, self.val_accuracy_list, self.train_data, self.val_data = self.model.train(data, val_prop=train_params['val_prop'], oversample_factor=train_params['oversample_factor'], lr=train_params['lr'], weight_decay=train_params['weight_decay'], num_epochs=train_params['num_epochs'], batch_size=train_params['batch_size'], output_dir=self.base_path)

    def save_results(self):
        safe_create(self.base_path)
        with open("{}/data_preproc_params.json".format(self.base_path), 'w') as f:
            json.dump(self.data_preproc_params, f, indent=4)

        with open("{}/gnn_params.json".format(self.base_path), 'w') as f:
            json.dump(self.gnn_params, f, indent=4)
        
        with open("{}/classification_params.json".format(self.base_path), 'w') as f:
            json.dump(self.classification_params, f, indent=4)
        
        with open("{}/train_params.json".format(self.base_path), 'w') as f:
            json.dump(self.train_params, f, indent=4)
        
        with open("{}/loss_list.json".format(self.base_path), 'w') as f:
            json.dump(self.loss_list, f, indent=4)
        
        with open("{}/val_accuracy_list.json".format(self.base_path), 'w') as f:
            json.dump(self.val_accuracy_list, f, indent=4)
        
        with open("{}/train_data.pkl".format(self.base_path), 'wb') as f:
            pickle.dump(self.train_data, f)
        
        with open("{}/val_data.pkl".format(self.base_path), 'wb') as f:
            pickle.dump(self.val_data, f)
        
        self.model.save("{}/model.pkl".format(self.base_path))

    def run(self):
        data_preproc_params = {k: random.choice(v) for k, v in DATA_PREPROC_SEARCH_SPACE.items()}
        gnn_params = {k: random.choice(v) for k, v in GNN_PARAMS_SEARCH_SPACE.items()}
        classification_params = {k: random.choice(v) for k, v in CLASSIFICATION_PARAM_SEARCH_SPACE.items()}
        train_params = {k: random.choice(v) for k, v in TRAIN_PARAM_SEARCH_SPACE.items()}
        print(gnn_params)
        print()
        print()
        print(classification_params)
        self.run_experiment(data_preproc_params, gnn_params, classification_params, train_params)
        self.save_results()

def main():
    pt_data_path = sys.argv[1]
    cell_data_path = sys.argv[2]
    output_dir = sys.argv[3]
    e = Experiment(pt_data_path, cell_data_path, output_dir)
    e.run()

if __name__ == "__main__":
    main()

