#!/usr/bin/env python3
"""CI 闸门：JS extractFeatures 必须与 Python librosa 输出一致。

不需要 CatMeows 数据集——用 scripts/make_fixture.py 生成的确定性 wav。

Usage: python3 scripts/parity_check.py [wav] [tolerance]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WAV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tests", "fixture.wav")
# 容差说明：librosa 默认以 float32 计算 STFT，与 JS 的 float64 之间存在
# 约 1e-7~1e-6 量级的舍入差（实测中位 2.4e-7）。这是精度上限，不是实现差异。
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5

sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import librosa  # noqa: E402
from train import SR, N_MFCC, f0_stats  # noqa: E402


def py_features(path):
    y, _ = librosa.load(path, sr=SR)
    if len(y) < SR // 4:
        y = np.pad(y, (0, SR // 4 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    feat = np.concatenate([
        mfcc.mean(1), mfcc.std(1), d1.mean(1), d1.std(1),
        [cent.mean(), cent.std(), rolloff.mean(), zcr.mean(), zcr.std(),
         rms.mean(), rms.std(), rms.max()],
        [len(y) / SR],
    ])
    return feat, f0_stats(y)


def js_features(path):
    out = subprocess.run(
        ["node", os.path.join(HERE, "test_parity.cjs"), path],
        capture_output=True, text=True, check=True,
    ).stdout
    j = json.loads(out)
    return np.array(j["feat"]), np.array(j["f0"])


def main():
    if not os.path.exists(WAV):
        print(f"缺少测试音频，先生成：{WAV}")
        subprocess.run([sys.executable, os.path.join(HERE, "make_fixture.py"), WAV], check=True)
    pmfcc, ppitch = py_features(WAV)
    jmfcc, jpitch = js_features(WAV)
    ok = True
    if pmfcc.shape != jmfcc.shape:
        print(f"FAIL 维度不一致：Python {pmfcc.shape} vs JS {jmfcc.shape}")
        ok = False
    else:
        d = np.abs(pmfcc - jmfcc)
        print(f"MFCC/谱统计 {pmfcc.shape[0]} 维：最大绝对差 {d.max():.3e}，"
              f"中位 {np.median(d):.3e}（对照 librosa，残差来自其默认 float32）")
        ok &= d.max() <= TOL
    if ppitch.shape != jpitch.shape:
        print(f"FAIL 音高维度不一致：Python {ppitch.shape} vs JS {jpitch.shape}")
        ok = False
    else:
        d = np.abs(ppitch - jpitch)
        print(f"YIN 音高 {ppitch.shape[0]} 维：最大绝对差 {d.max():.3e}"
              f"（对照同算法 numpy 版，librosa.pyin 实现不同不能直接比）")
        ok &= d.max() <= TOL
    print(("PASS" if ok else "FAIL") + f" 容差 {TOL}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
