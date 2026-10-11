"""Train a cat meow context classifier on the CatMeows dataset (Zenodo 4008297).

3 classes from filename prefix: B=brushing, F=waiting for food, I=isolation.
Features: MFCC + deltas + spectral stats. Model: RandomForest (n=440, small-data friendly).
Split: grouped by cat ID (no leakage across train/test).
Outputs: model.joblib, metrics.json, feature_config.json
"""
import json
import os
import re

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_score

DATA = os.environ.get("MEOWSCOPE_DATA") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 网页唯一的数据来源：只有 metrics.json / forest.json 需要进 public/
OUT = os.path.join(ROOT, "public")
# 中间产物（joblib、特征配置）放 build/，不随网站发布
BUILD = os.path.join(ROOT, "build")
os.makedirs(BUILD, exist_ok=True)

SR = 22050
N_MFCC = 20
# 150 棵、深度上限 10：与 400 棵无限深度的精度相同（见 README「模型体积」一节），
# 但导出的 JSON 只有约三分之一，网页加载更快。
N_TREES = 150
MAX_DEPTH = 10

CLASSES = ["B", "F", "I"]
CLASS_NAMES = {"B": "brushing", "F": "waiting_for_food", "I": "isolation"}


def _load_linear(path, sr=SR):
    """Load wav and resample with the same linear interpolation as the web
    (features.js) loader, so training and browser inference see identical audio."""
    import wave
    w = wave.open(path)
    ch, sw, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
    assert sw == 2, path
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(1)
    if rate == sr:
        return raw
    m = int(np.floor(len(raw) * sr / rate))
    p = np.arange(m) * rate / sr
    i0 = np.floor(p).astype(int)
    frac = p - i0
    return raw[i0] + (raw[np.minimum(i0 + 1, len(raw) - 1)] - raw[i0]) * frac


def yin_f0(y, sr=SR, fl=2048, hop=512, fmin=150, fmax=1200, thr=0.1):
    """features.js 里那套 YIN 的 numpy 版，逐行对应，保证训练/推理看到同一组音高特征。

    注意：librosa.pyin 的累积均值归一化与阈值搜索和这套实现不同，
    所以这里不能用 librosa.yin 代替，否则 JS 与 Python 会对不上。
    """
    tau_min, tau_max = int(sr / fmax), int(np.ceil(sr / fmin))
    pad = fl >> 1
    ypad = np.concatenate([np.zeros(pad), y, np.zeros(fl)])
    n_frames = 1 + len(y) // hop
    out = np.zeros(n_frames)
    taus = np.arange(tau_min, tau_max + 1)
    for f in range(n_frames):
        seg = ypad[f * hop:f * hop + fl]
        d = np.array([np.sum((seg[:fl - t] - seg[t:]) ** 2) for t in taus])
        run = np.cumsum(d)
        cmnd = np.where(run > 0, d * np.arange(1, len(d) + 1) / np.maximum(run, 1e-12), 1.0)
        hit = np.where(cmnd[1:-1] < thr)[0]
        if len(hit):
            k = int(hit[0]) + 1
            while k + 1 < len(cmnd) - 1 and cmnd[k + 1] < cmnd[k]:
                k += 1
            tau = float(taus[k])
        else:
            tau = float(taus[int(np.argmin(cmnd))])
        i = int(tau) - tau_min
        if 0 < i < len(cmnd) - 1:
            s0, s1, s2 = cmnd[i - 1], cmnd[i], cmnd[i + 1]
            den = 2 * (2 * s1 - s2 - s0)
            if abs(den) > 1e-12:
                tau += (s2 - s0) / den
        out[f] = sr / max(tau, 1e-9)
    return out


def f0_stats(y):
    """与 features.js extractF0Stats 完全一致：中位数、标准差、音高走向、
    10/90 分位、抖动。"""
    v = np.clip(yin_f0(y), 150, 1200)
    n = len(v)
    t = np.arange(n) / (SR / 512)
    slope = float(np.polyfit(t, v, 1)[0]) if n > 1 else 0.0
    jitter = float(np.std(np.diff(v))) if n > 1 else 0.0
    return np.array([np.median(v), np.std(v), slope,
                     np.percentile(v, 10), np.percentile(v, 90), jitter])


def extract(path):
    y = _load_linear(path)
    if len(y) < SR // 4:
        y = np.pad(y, (0, SR // 4 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    feat = np.concatenate([
        mfcc.mean(1), mfcc.std(1),
        d1.mean(1), d1.std(1),
        [cent.mean(), cent.std(), rolloff.mean(), zcr.mean(), zcr.std(),
         rms.mean(), rms.std(), rms.max()],
        [len(y) / SR],
        f0_stats(y),
    ])
    return feat


def main():
    files = [f for f in os.listdir(DATA) if f.endswith(".wav")]
    X, y, groups = [], [], []
    for f in sorted(files):
        m = re.match(r"^([BFI])_([A-Z]{3})\d+_", f)
        if not m:
            continue
        X.append(extract(os.path.join(DATA, f)))
        y.append(CLASSES.index(m.group(1)))
        groups.append(m.group(2))  # cat ID
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    print(f"{len(X)} samples, cats: {len(set(groups))}")

    clf = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy")
    print("grouped 5-fold CV accuracy:", scores, "mean:", scores.mean().round(3))

    # final fit on all data
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(BUILD, "model.joblib"))
    with open(os.path.join(BUILD, "feature_config.json"), "w") as f:
        json.dump({"sr": SR, "n_mfcc": N_MFCC}, f)

    # holdout report for documentation
    logo = LeaveOneGroupOut()
    preds = np.zeros_like(y)
    for tr, te in logo.split(X, y, groups):
        c = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
        c.fit(X[tr], y[tr])
        preds[te] = c.predict(X[te])
    names = [CLASS_NAMES[c] for c in CLASSES]
    rep = classification_report(y, preds, target_names=names, output_dict=True)
    cm = confusion_matrix(y, preds).tolist()

    # 诚实的对照组：只猜最多的那一类能拿多少分（isolation 占一半）
    majority = float(max(np.bincount(y)) / len(y))
    metrics = {
        "n_samples": int(len(X)),
        "n_cats": int(len(set(groups))),
        "n_features": int(X.shape[1]),
        "grouped_5fold_cv_accuracy": float(scores.mean()),
        "grouped_5fold_cv_macro_f1": float(f1_score(y, preds, average="macro")),
        "leave_one_cat_out_accuracy": float(accuracy_score(y, preds)),
        "leave_one_cat_out_macro_f1": float(f1_score(y, preds, average="macro")),
        "majority_class_accuracy": majority,
        "random_guess_accuracy": 1.0 / len(CLASSES),
        "per_class": rep,
        "confusion_matrix": cm,
        "labels": names,
        "evaluation_note": (
            "训练/测试按猫划分，同一只猫不会同时出现在两边。"
            "多数类基线 = 全猜 isolation；随机猜测 = 1/3。"
            "文献里的 90%+ 多为随机划分（同一只猫的录音被分到两边），与本数字不可直接比较。"
        ),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("LOSO accuracy:", round(metrics["leave_one_cat_out_accuracy"], 4),
          "macro-F1:", round(metrics["leave_one_cat_out_macro_f1"], 4))
    print("多数类基线:", round(majority, 4), " 随机猜测:", round(1 / len(CLASSES), 4))


if __name__ == "__main__":
    main()
