#!/usr/bin/env python3
"""生成确定性的测试用 wav（不依赖 CatMeows 数据集，供 CI 对拍用）。

合成一段"类猫叫"的信号：调频谐波 + 共振峰包络 + 轻噪声，
固定随机种子，任何机器上生成的文件都逐字节一致。

Usage: python3 scripts/make_fixture.py [out.wav]
"""
import math
import struct
import sys

SR = 22050


def synth(seconds=1.6, seed=20240601):
    n = int(SR * seconds)
    # 简单的线性同余伪随机，避免不同平台 numpy 版本差异
    state = seed
    def rnd():
        nonlocal state
        state = (1103515245 * state + 12345) % (1 << 31)
        return state / (1 << 31) * 2 - 1

    noise = [rnd() * 0.02 for _ in range(n)]
    out = []
    for i in range(n):
        t = i / SR
        # 基频：从 620Hz 滑到 480Hz 再微升，模拟一声喵的音高走向
        f0 = 620 - 140 * (t / seconds) + 40 * math.sin(2 * math.pi * 3 * t / seconds)
        # 振幅包络：快起慢落
        env = math.exp(-3.0 * max(0.0, t - 0.06)) * (1 - math.exp(-60 * t))
        v = 0.0
        for k, amp in enumerate((1.0, 0.55, 0.30, 0.16, 0.08)):
            v += amp * math.sin(2 * math.pi * f0 * (k + 1) * t)
        # 共振峰：两个随时间移动的带通（用简单的调制近似）
        v *= 1.0 + 0.25 * math.sin(2 * math.pi * 7 * t)
        out.append(0.35 * v * env + noise[i])
    peak = max(abs(x) for x in out) or 1.0
    return [int(max(-1.0, min(1.0, x / peak * 0.9)) * 32767) for x in out]


def write_wav(path, samples):
    raw = b"".join(struct.pack("<h", s) for s in samples)
    hdr = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
    data = b"data" + struct.pack("<I", len(raw))
    with open(path, "wb") as f:
        f.write(hdr + fmt + data + raw)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixture.wav"
    write_wav(out, synth())
    print("wrote", out)
