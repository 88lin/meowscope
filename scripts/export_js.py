"""Export the RandomForest to plain JS (tree ensemble as JSON)."""
import json
import os

import joblib
import numpy as np

OUT = os.path.dirname(__file__)
clf = joblib.load(os.path.join(OUT, "model.joblib"))


def export_tree(tree):
    t = tree.tree_
    def node(i):
        if t.children_left[i] == -1:
            return {"v": t.value[i][0].tolist()}
        return {"f": int(t.feature[i]), "t": round(float(t.threshold[i]), 5),
                "l": node(t.children_left[i]), "r": node(t.children_right[i])}
    return node(0)


forest = [export_tree(est) for est in clf.estimators_]
with open(os.path.join(OUT, "forest.json"), "w") as f:
    json.dump({"classes": ["brushing", "waiting_for_food", "isolation"], "trees": forest}, f)
print("trees:", len(forest), "size KB:", os.path.getsize(os.path.join(OUT, "forest.json")) // 1024)
