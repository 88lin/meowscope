"""Export the RandomForest to a compact JSON structure the browser can walk directly.

格式：内部节点 [特征下标, 阈值, 左子树, 右子树]，叶子节点 [三类票数]。
按数组长度区分（3 = 叶子，4 = 内部节点），比 {"f":..,"t":..,"l":..,"r":..}
的对象格式小一半以上。
"""
import json
import os

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public")
BUILD = os.path.join(ROOT, "build")
clf = joblib.load(os.path.join(BUILD, "model.joblib"))


def export_tree(tree):
    t = tree.tree_

    def node(i):
        if t.children_left[i] == -1:
            return [round(float(v), 4) for v in t.value[i][0].tolist()]
        return [int(t.feature[i]), round(float(t.threshold[i]), 5),
                node(t.children_left[i]), node(t.children_right[i])]

    return node(0)


forest = [export_tree(est) for est in clf.estimators_]
with open(os.path.join(OUT, "forest.json"), "w") as f:
    json.dump({"classes": ["brushing", "waiting_for_food", "isolation"],
               "n_features": int(clf.n_features_in_),
               "trees": forest}, f, separators=(",", ":"))
size = os.path.getsize(os.path.join(OUT, "forest.json"))
print("trees:", len(forest), "size KB:", size // 1024)
