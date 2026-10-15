"""SNS 共有用のカード画像（1200x630）を作る。キャラクターの絵は使わず、文字と吹き出しだけで構成する。

  python3 tools/make_og.py --out docs/og.png --params 71万 --steps "よちよち:あ、おち、おちい！" "いちばん:いいのだ！ボクもいきたいのだ！"
"""
import argparse

from PIL import Image, ImageDraw, ImageFont

BG, FG, MUTED, ZUNDA, SOFT, MOCHI, LINE = "#eef3e8", "#1d2a19", "#5f6e58", "#5f9c2f", "#dcebc9", "#ffffff", "#d3dcc9"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(path, size):
    return ImageFont.truetype(path, size, index=0)   # index 0 = JP


def bubble(d, x, y, text, f, fill, fg, outline=None, pad=(26, 16)):
    w = d.textlength(text, font=f)
    h = f.size
    d.rounded_rectangle([x, y, x + w + pad[0] * 2, y + h + pad[1] * 2], radius=26, fill=fill, outline=outline, width=2)
    d.text((x + pad[0], y + pad[1] - 4), text, font=f, fill=fg)
    return y + h + pad[1] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="71万")
    ap.add_argument("--steps", nargs="+", required=True, help="段階名:返事")
    ap.add_argument("--opener", default="こんにちは。")
    args = ap.parse_args()
    W, H = 1200, 630
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((72, 40), "幼生ずんだもん", font=font(FONT_B, 92), fill=FG)
    d.text((76, 158), f"{args.params}パラメータの手作りずんだもんLLM", font=font(FONT_R, 38), fill=MUTED)
    d.text((76, 206), "ブラウザでそのまま話せます", font=font(FONT_R, 38), fill=MUTED)
    fb = font(FONT_R, 34)
    y = 286
    tw = d.textlength(args.opener, font=fb) + 52
    bubble(d, W - 72 - tw, y, args.opener, fb, MOCHI, FG, outline=LINE)
    y += 86
    small = font(FONT_R, 24)
    for spec in args.steps[:2]:
        label, text = spec.split(":", 1)
        d.text((76, y), f"ずんだもん（{label}）", font=small, fill=MUTED)
        y = bubble(d, 72, y + 30, text, fb, SOFT, FG) + 12
    d.rectangle([0, H - 14, W, H], fill=ZUNDA)
    im.save(args.out, optimize=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
