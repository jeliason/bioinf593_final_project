import json
import os
max_val = 0
max_f = ""
for fi in os.listdir():
    if '.py' in fi:
        continue
    with open("{}/val_accuracy_list.json".format(fi)) as f:
        d=json.load(f)
    if max(d) > 0.8:
        print(max(d))
        print(fi)

    if max(d) > max_val:
        max_val = max(d)
        max_f = fi
print()
print()
print(max_val)
print(max_f)
