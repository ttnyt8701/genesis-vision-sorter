"""比較用：色のしきい値 + 連結成分によるルールベース検出。"""
import cv2
import numpy as np

from .scene import CLASSES

# RGB のしきい値（シミュレータの標準照明で調整した値）
RULES = {
    "red": lambda r, g, b: (r > 150) & (g < 90) & (b < 90),
    "green": lambda r, g, b: (g > 120) & (r < 100) & (b < 120),
    "blue": lambda r, g, b: (b > 150) & (r < 100) & (g < 140),
}


def detect(image, min_area=20):
    img = image[..., :3].astype(np.int32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    dets = []
    for ci, c in enumerate(CLASSES):
        mask = RULES[c](r, g, b).astype(np.uint8)
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_area:
                continue
            # 画素の中心は +0.5 の位置（project() と座標系をそろえる）。向きは比較しないので 0
            dets.append((ci, float(cents[k][0]) + 0.5, float(cents[k][1]) + 0.5, 0.0, 1.0))
    return dets
