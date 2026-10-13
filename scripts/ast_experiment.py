"""AST embedding 路线实验（任务 2：模型跃升可行性验证）。

问题：README 论证过手工 MFCC 特征到顶（LOSO 60%），真正跃升要靠预训练音频表征
（Ghani 等 2023：冻结 Perch embedding + LR 在同类任务 0.74-0.92）。
本项目零依赖纯 JS 的约束下，浏览器跑不动 ViT——但如果离线 teacher 显著更强，
可以尝试把 teacher 的暗知识蒸馏进 MFCC student（浏览器可部署）。

协议与 scripts/train.py 完全一致：LOSO（留一猫）、线性重采样、同样分组。
对照：MFCC+RF 基线（应复现 ~60%）。
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset")
BUILD = os.path.join(ROOT, "build")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR_AST = 16000
CLASSES = ["B", "F", "I"]


def load_linear_16k(path):
    """CatMeows 是 8kHz 16-bit wav；线性插值重采样到 16k（AST 期望采样率）。
    与 train.py:_load_linear 同一套插值逻辑。"""
    import wave
    w = wave.open(path)
    ch, rate = w.getnchannels(), w.getframerate()
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(1)
    m = int(np.floor(len(raw) * SR_AST / rate))
    p = np.arange(m) * rate / SR_AST
    i0 = np.floor(p).astype(int)
    return raw[i0] + raw[np.minimum(i0 + 1, len(raw) - 1)] * (p - i0)


def ast_features(paths):
    """返回 (cls_embedding[n,768], mean_embedding[n,768], logits[n,527])。"""
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification

    # 权重已在 HF 缓存里；离线加载，避免镜像站个别请求无超时挂起
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID, local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()
    torch.set_num_threads(8)
    cls_all, mean_all, logit_all = [], [], []
    with torch.no_grad():
        for i, p in enumerate(paths):
            y = load_linear_16k(p)
            # AST 训练时用 10s 随机裁剪；推理直接用整段（数据集片段 1-5s）
            inputs = fe(y, sampling_rate=SR_AST, return_tensors="pt")
            out = model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1][0]          # [n_patches+1, 768]
            cls_all.append(h[0].numpy())
            mean_all.append(h[1:].mean(0).numpy())
            logit_all.append(out.logits[0].numpy())
            if (i + 1) % 50 == 0:
                print(f"  AST 推理 {i + 1}/{len(paths)}", flush=True)
    return np.array(cls_all), np.array(mean_all), np.array(logit_all)


def loso_eval(X, y, groups, make_clf, soft=None):
    """LOSO 评估；soft 非空时训练目标换成 teacher 软标签（蒸馏 student）。
    返回 (acc, macro_f1, 逐样本预测)，供配对 bootstrap 用。"""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut
    preds = np.zeros_like(y)
    softpreds = None if soft is None else np.zeros((len(y), len(CLASSES)))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        clf = make_clf()
        if soft is None:
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict(X[te])
        else:
            clf.fit(X[tr], soft[tr])
            # 回归器用 predict；分类器（若将来换回）用 predict_proba
            softpreds[te] = (clf.predict_proba(X[te]) if hasattr(clf, "predict_proba")
                             else clf.predict(X[te]))
    if soft is None:
        return accuracy_score(y, preds), f1_score(y, preds, average="macro"), preds
    # 软标签 student：硬判据取 argmax 供对照
    hard = softpreds.argmax(1)
    return accuracy_score(y, hard), f1_score(y, hard, average="macro"), hard


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from train import extract as mfcc_extract  # 复用线上同一条 MFCC 管线

    files = [f for f in os.listdir(DATA) if f.endswith(".wav")]
    paths, y, groups = [], [], []
    for f in sorted(files):
        m = re.match(r"^([BFI])_([A-Z]{3})\d+_", f)
        if not m:
            continue
        paths.append(os.path.join(DATA, f))
        y.append(CLASSES.index(m.group(1)))
        groups.append(m.group(2))
    y, groups = np.array(y), np.array(groups)
    print(f"{len(paths)} samples, cats: {len(set(groups))}", flush=True)

    # ---- 特征：MFCC（线上管线）+ AST embedding/logits ----
    cache = os.path.join(BUILD, "ast_features.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        cls, mean, logits = z["cls"], z["mean"], z["logits"]
        print("loaded cached AST features")
    else:
        cls, mean, logits = ast_features(paths)
        np.savez(cache, cls=cls, mean=mean, logits=logits)
    mfcc = np.array([mfcc_extract(p) for p in paths])
    print("MFCC:", mfcc.shape, "| AST cls:", cls.shape, "| logits:", logits.shape, flush=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def rf():
        return RandomForestClassifier(n_estimators=150, max_depth=10,
                                      min_samples_leaf=2, random_state=42, n_jobs=-1)

    def lr():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=1.0, random_state=42))

    results = {}
    preds_keep = {}
    for name, X, mk in [
        ("MFCC+RF（线上基线）", mfcc, rf),
        ("AST-CLS+LR", cls, lr),
        ("AST-CLS+RF", cls, rf),
        ("AST-mean+LR", mean, lr),
        ("AST-logits+LR", logits, lr),
        ("AST-logits+RF", logits, rf),
        ("MFCC+AST-CLS+LR", np.hstack([mfcc, cls]), lr),
        ("MFCC+AST-logits+LR", np.hstack([mfcc, logits]), lr),
    ]:
        acc, f1, preds = loso_eval(X, y, groups, mk)
        results[name] = {"loso_acc": round(float(acc), 4), "loso_macro_f1": round(float(f1), 4)}
        preds_keep[name] = preds
        print(f"{name:24s} LOSO acc={acc:.4f}  macroF1={f1:.4f}", flush=True)

    # ---- 蒸馏：teacher（AST-mean+LR，本实验最强配置）软标签 → student（MFCC+RF）----
    # teacher 在训练折内按猫做二次交叉验证，给训练样本打诚实的 out-of-fold 软标签；
    # 直接用 teacher 的训练折内预测会过自信（软标签失真），且违背 LOSO 精神。
    # RandomForestClassifier 不接受连续多输出标签，student 用回归器学三维概率。
    from sklearn.ensemble import RandomForestRegressor
    soft = np.zeros((len(y), len(CLASSES)))
    for tr, te in LeaveOneGroupOut().split(mfcc, y, groups):
        inner = np.zeros((len(tr), len(CLASSES)))
        for itr, ite in LeaveOneGroupOut().split(mean[tr], y[tr], groups[tr]):
            t = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
            t.fit(mean[tr][itr], y[tr][itr])
            inner[ite] = t.predict_proba(mean[tr][ite])
        soft[tr] = inner

    def rf_reg():
        return RandomForestRegressor(n_estimators=150, max_depth=10,
                                     min_samples_leaf=2, random_state=42, n_jobs=-1)

    acc, f1, dpreds = loso_eval(mfcc, y, groups, rf_reg, soft=soft)
    results["蒸馏 student（MFCC+RF）"] = {"loso_acc": round(float(acc), 4), "loso_macro_f1": round(float(f1), 4)}
    preds_keep["蒸馏 student（MFCC+RF）"] = dpreds
    print(f"{'蒸馏 student（MFCC+RF）':24s} LOSO acc={acc:.4f}  macroF1={f1:.4f}", flush=True)

    # ---- 配对 bootstrap：关键对比的差值置信区间（10000 次重采样）----
    # 单次 LOSO 差 2-3 个点接近噪声边缘，用配对 bootstrap 判断方向是否可信。
    rng = np.random.default_rng(42)
    n = len(y)

    def paired_boot(pa, pb):
        """pa/pb 为两套逐样本预测，返回 (pa-pb) 准确率差的 95% CI 与 P(差>0)。"""
        diffs = np.array([ (pa == y).astype(float) - (pb == y).astype(float) ])[0]
        idx = rng.integers(0, n, size=(10000, n))
        boot = diffs[idx].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4), round(float((boot > 0).mean()), 4)

    sig = {}
    for a, b in [
        ("AST-mean+LR", "MFCC+RF（线上基线）"),
        ("蒸馏 student（MFCC+RF）", "MFCC+RF（线上基线）"),
        ("蒸馏 student（MFCC+RF）", "AST-mean+LR"),
    ]:
        lo, hi, p = paired_boot(preds_keep[a], preds_keep[b])
        sig[f"{a} vs {b}"] = {"diff_ci95": [lo, hi], "p_boot_gt0": p}
        print(f"配对 bootstrap {a} vs {b}: 差值 95% CI [{lo:+.4f}, {hi:+.4f}]，P(更优)={p:.3f}", flush=True)
    results["_significance"] = sig

    # 多数类基线对照
    majority = float(np.bincount(y).max() / len(y))
    results["_majority_baseline"] = round(majority, 4)
    print(f"多数类基线: {majority:.4f}")

    out = os.path.join(BUILD, "ast_experiment.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
