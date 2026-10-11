#!/usr/bin/env python3
"""Generate the site-level Open Graph share image (public/og-image.png, 1200x630).

Social platforms (WeChat/Telegram/X/Slack/Discord) crop link previews to roughly
1.91:1, so the card is drawn at 2x (2400x1260) and down-sampled for crisp text.

Fonts: needs a CJK font. Override with env vars:
    MEOWSCOPE_FONT_BOLD / MEOWSCOPE_FONT_REGULAR
Falls back to any Noto Sans CJK found on the system.

Usage:  python3 scripts/make_og_image.py
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont

# ---- palette (kept in sync with public/index.html :root) ----
BG = "#faf7f2"
INK = "#2b2620"
MUTED = "#8a8175"
LINE = "#e6dfd3"
CARD = "#ffffff"
ACCENT = "#d96a2b"
ACCENT_SOFT = "#f7e3d3"
OK = "#3d7a4e"

W, H = 1200, 630
S = 2  # supersample factor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "public", "og-image.png")


def find_font(*names):
    for n in names:
        for d in ("/usr/share/fonts", "/usr/local/share/fonts",
                  os.path.expanduser("~/.fonts")):
            for root, _dirs, files in os.walk(d):
                if n in files:
                    return os.path.join(root, n)
    return None


def load(bold):
    key = "MEOWSCOPE_FONT_BOLD" if bold else "MEOWSCOPE_FONT_REGULAR"
    p = os.environ.get(key)
    if not p:
        p = find_font("NotoSansCJKsc-Bold.otf" if bold else "NotoSansCJKsc-Regular.otf",
                      "NotoSansSC-Bold.otf" if bold else "NotoSansSC-Regular.otf")
    if not p or not os.path.exists(p):
        raise SystemExit("找不到中文字体，请设置 %s 指向一个 CJK 字体文件" % key)
    return p


FB = load(True)
FR = load(False)


def font(path, size):
    return ImageFont.truetype(path, int(size * S))


def text(d, xy, s, f, fill, spacing=0, anchor=None):
    """Draw text with manual letter-spacing (Pillow has no letterSpacing for CJK)."""
    x, y = xy[0] * S, xy[1] * S
    if spacing == 0:
        d.text((x, y), s, font=f, fill=fill, anchor=anchor)
        return x / S + d.textlength(s, font=f) / S
    cursor = x
    for ch in s:
        d.text((cursor, y), ch, font=f, fill=fill, anchor=anchor)
        cursor += d.textlength(ch, font=f) + spacing * S
    return (cursor - x) / S


def width(d, s, f, spacing=0):
    w = d.textlength(s, font=f)
    if spacing:
        w += spacing * S * max(0, len(s) - 1)
    return w / S


def rrect(d, box, r, fill=None, outline=None, width_=1):
    d.rounded_rectangle([v * S for v in box], radius=r * S, fill=fill,
                        outline=outline, width=int(width_ * S))


def cat_face(d, cx, cy, size, color=INK, eye="#ffffff"):
    """Minimal cat head: two triangular ears + round head + two eyes + nose."""
    cx, cy, size = cx * S, cy * S, size * S
    r = size / 2
    # ears: thin triangles poking out diagonally at 10 & 2 o'clock, base hidden by head
    for sx in (-1, 1):
        tip = (cx + sx * r * 1.00, cy - r * 1.02)
        inner = (cx + sx * r * 0.22, cy - r * 0.82)
        outer = (cx + sx * r * 0.90, cy - r * 0.14)
        d.polygon([tip, inner, outer], fill=color)
    # head
    d.ellipse([cx - r, cy - r * 0.94, cx + r, cy + r * 0.94], fill=color)
    # eyes
    er = r * 0.16
    for sx in (-1, 1):
        d.ellipse([cx + sx * r * 0.36 - er, cy - r * 0.22 - er,
                   cx + sx * r * 0.36 + er, cy - r * 0.22 + er], fill=eye)
    # nose
    d.polygon([(cx - r * 0.10, cy + r * 0.16), (cx + r * 0.10, cy + r * 0.16),
               (cx, cy + r * 0.32)], fill=eye)


def waveform(d, x0, y0, x1, y1, color=ACCENT, seed=7):
    rnd = random.Random(seed)
    n = 64
    cx0, cy0, cx1, cy1 = x0 * S, y0 * S, x1 * S, y1 * S
    mid = (cy0 + cy1) / 2
    amp = (cy1 - cy0) / 2
    bw = (cx1 - cx0) / n
    for i in range(n):
        # deterministic envelope: bursts of meows with quiet gaps
        env = abs(rnd.gauss(0.55, 0.35)) * (0.45 + 0.55 * abs(__import__("math").sin(i / 7.0)))
        h = max(2 * S, min(1.0, env) * amp)
        x = cx0 + i * bw
        d.rounded_rectangle([x, mid - h, x + bw * 0.55, mid + h],
                            radius=bw * 0.27, fill=color)


def main():
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)

    # soft decorative blob, top-right
    d.ellipse([880 * S, -190 * S, 1420 * S, 350 * S], fill=ACCENT_SOFT)
    # faint blob, bottom-left
    d.ellipse([-160 * S, 470 * S, 260 * S, 890 * S], fill="#f2ece1")

    # frame
    rrect(d, (24, 24, W - 24, H - 24), 30, outline=LINE, width_=2)

    # left accent bar
    d.rounded_rectangle([24 * S, 24 * S, 32 * S, (H - 24) * S],
                        radius=4 * S, fill=ACCENT)

    # ---------- brand row ----------
    cat_face(d, 104, 112, 52)
    f_brand = font(FB, 34)
    f_brand_lat = font(FB, 20)
    text(d, (144, 112), "喵语通", f_brand, INK)
    text(d, (144 + width(d, "喵语通", f_brand) + 14, 116), "MEOWSCOPE",
         f_brand_lat, MUTED, spacing=3)

    # ---------- kicker ----------
    f_kick = font(FB, 19)
    text(d, (88, 186), "纯浏览器端 · 音频零上传 · 开源 MIT", f_kick, ACCENT, spacing=3)

    # ---------- headline ----------
    f_h1 = font(FB, 88)
    text(d, (88, 232), "你的猫", f_h1, INK)
    text(d, (88, 330), "在说什么？", f_h1, ACCENT)

    # ---------- chips ----------
    f_chip = font(FB, 21)
    chips = [("被刷毛", ACCENT), ("等饭", OK), ("被隔离", "#6b6357")]
    cx = 88.0
    for label, color in chips:
        tw = width(d, label, f_chip)
        box_w = tw + 54
        rrect(d, (cx, 464, cx + box_w, 506), 21, fill=CARD, outline=LINE, width_=1)
        d.ellipse([(cx + 18) * S, 478 * S, (cx + 28) * S, 488 * S], fill=color)
        text(d, (cx + 37, 485), label, f_chip, INK)
        cx += box_w + 12

    # ---------- stat + url ----------
    f_stat = font(FB, 21)
    w1 = text(d, (88, 534), "按猫划分准确率 60.9%", f_stat, INK)
    f_stat2 = font(FR, 21)
    text(d, (88 + w1 + 10, 534), "· 人类对照组 26–41% · 数据集 CatMeows", f_stat2, MUTED)
    f_url = font(FR, 19)
    text(d, (88, 570), "88lin.github.io/meowscope", f_url, ACCENT)

    # ---------- right card: cat + waveform ----------
    rrect(d, (782, 120, 1122, 520), 34, fill=CARD, outline=LINE, width_=2)
    d.ellipse([834 * S, 148 * S, 1070 * S, 396 * S], fill=BG)
    cat_face(d, 952, 272, 168)
    waveform(d, 818, 412, 1086, 466)
    f_cap = font(FR, 17)
    tw = width(d, "22.05 kHz · 70 维特征 · 随机森林 150 棵", f_cap)
    text(d, (952 - tw / 2, 492), "22.05 kHz · 70 维特征 · 随机森林 150 棵", f_cap, MUTED)

    img = img.resize((W, H), Image.LANCZOS)
    img.save(OUT, "PNG", optimize=True)
    print("wrote", OUT, img.size, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
