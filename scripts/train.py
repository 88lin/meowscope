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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score

DATA = os.path.join(os.path.dirname(__file__), "data", "dataset")
OUT = os.path.dirname(__file__)

SR = 22050
N_MFCC = 20

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
        d1.mean(1),
        [cent.mean(), cent.std(), rolloff.mean(), zcr.mean(), zcr.std(), rms.mean(), rms.max()],
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

    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy")
    print("grouped 5-fold CV accuracy:", scores, "mean:", scores.mean().round(3))

    # final fit on all data
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(OUT, "model.joblib"))
    with open(os.path.join(OUT, "feature_config.json"), "w") as f:
        json.dump({"sr": SR, "n_mfcc": N_MFCC}, f)

    # holdout report for documentation
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    preds = np.zeros_like(y)
    for tr, te in logo.split(X, y, groups):
        c = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1)
        c.fit(X[tr], y[tr])
        preds[te] = c.predict(X[te])
    rep = classification_report(y, preds, target_names=[CLASS_NAMES[c] for c in CLASSES], output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
    metrics = {
        "n_samples": int(len(X)),
        "n_cats": int(len(set(groups))),
        "grouped_5fold_cv_accuracy": float(scores.mean()),
        "leave_one_cat_out_accuracy": float(accuracy_score(y, preds)),
        "per_class": rep,
        "confusion_matrix": cm,
        "labels": [CLASS_NAMES[c] for c in CLASSES],
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("LOSO accuracy:", metrics["leave_one_cat_out_accuracy"])


if __name__ == "__main__":
    main()
