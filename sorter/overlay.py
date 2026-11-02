"""動画用の画面合成（メイン映像 + 認識結果パネル + 字幕）。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H = 1280, 720
MAIN_W = 940
PANEL_BG = (24, 26, 30)
COLOR = {"red": (235, 70, 70), "green": (60, 200, 90), "blue": (80, 130, 255)}
NAME_JA = {"red": "赤", "green": "緑", "blue": "青"}

_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_B if bold else FONT, size, index=0)
    return _fonts[key]


def compose(main_rgb, top_rgb, dets, status, caption="", done=0, total=0, det_res=160, heat=None):
    """main_rgb: メイン映像, top_rgb: 真上カメラ画像, dets の画素座標は det_res 基準。"""
    canvas = Image.new("RGB", (W, H), PANEL_BG)
    canvas.paste(Image.fromarray(main_rgb[..., :3]).resize((MAIN_W, H)), (0, 0))
    d = ImageDraw.Draw(canvas)
    px = MAIN_W + 20
    d.text((px, 18), "真上カメラ + CNN の推定", font=font(20, True), fill=(235, 235, 235))

    top = Image.fromarray(top_rgb[..., :3])
    top = top.resize((300, 300))
    s = 300 / det_res
    td = ImageDraw.Draw(top)
    for det in dets:
        u, v = det.u * s, det.v * s
        r = 13
        col = COLOR[det.color]
        td.rectangle([u - r, v - r, u + r, v + r], outline=col, width=3)
        td.line([u - 4, v, u + 4, v], fill=(255, 255, 255), width=1)
        td.line([u, v - 4, u, v + 4], fill=(255, 255, 255), width=1)
        td.text((u - r, v - r - 17), f"{NAME_JA[det.color]} {det.score:.2f}", font=font(13, True), fill=col)
    canvas.paste(top, (px, 52))
    y = 52 + top.height + 12

    if heat is not None:
        d.text((px, y), "色ごとの中心ヒートマップ", font=font(15), fill=(200, 200, 200))
        y += 24
        hw = 92
        for i, c in enumerate(["red", "green", "blue"]):
            h = np.clip(heat[i] * 1.5, 0, 1) ** 0.6   # 見やすいように持ち上げる
            img = (h[..., None] * np.array(COLOR[c])[None, None]).astype(np.uint8)
            canvas.paste(Image.fromarray(img).resize((hw, hw), Image.NEAREST), (px + i * (hw + 12), y))
        y += hw + 14

    d.text((px, y), "状態", font=font(15), fill=(160, 160, 160))
    d.text((px, y + 22), status, font=font(19, True), fill=(255, 255, 255))
    if total:
        d.text((px, y + 56), f"仕分け済み  {done} / {total}", font=font(19, True), fill=(255, 215, 90))

    if caption:
        size = 28
        while size > 18 and d.textlength(caption, font=font(size, True)) > MAIN_W - 60:
            size -= 1
        f = font(size, True)
        tw = d.textlength(caption, font=f)
        x0 = (MAIN_W - tw) / 2
        d.rectangle([x0 - 16, H - 74, x0 + tw + 16, H - 26], fill=(0, 0, 0))
        d.text((x0, H - 70), caption, font=f, fill=(255, 255, 255))
    return np.asarray(canvas)


def title_card(lines, sub=None, size=(W, H)):
    """lines: [(text, font_size, bold), ...] を中央に並べた画面。"""
    canvas = Image.new("RGB", size, (20, 22, 28))
    d = ImageDraw.Draw(canvas)
    heights = [s + 18 for _, s, _ in lines]
    y = (size[1] - sum(heights) - (sub.shape[0] + 10 if sub is not None else 0)) / 2
    for (text, s, b), h in zip(lines, heights):
        f = font(s, b)
        d.text(((size[0] - d.textlength(text, font=f)) / 2, y), text, font=f, fill=(240, 240, 240))
        y += h
    if sub is not None:
        canvas.paste(Image.fromarray(sub), ((size[0] - sub.shape[1]) // 2, int(y) + 10))
    return np.asarray(canvas)
