"""真上カメラの画像からブロックの位置と向きを推定する。"""
from dataclasses import dataclass

import torch

from .geometry import unproject_to_plane
from .model import decode, load_model, to_tensor
from .scene import CLASSES, CUBE


@dataclass
class Detection:
    color: str
    u: float          # 画素座標
    v: float
    x: float          # ロボット座標 [m]
    y: float
    yaw: float        # [rad]
    score: float


class CubeDetector:
    def __init__(self, model_path, K, E, thresh=0.35):
        self.net = load_model(model_path)
        self.K, self.E = K, E
        self.thresh = thresh
        torch.set_num_threads(2)

    def __call__(self, image):
        with torch.no_grad():
            out = self.net(to_tensor(image[None, ..., :3]))
        dets = []
        for c, u, v, yaw, s in decode(out, self.thresh)[0]:
            p = unproject_to_plane([[u, v]], self.K, self.E, CUBE)[0]   # ブロック上面の高さで交点を取る
            dets.append(Detection(CLASSES[c], u, v, p[0], p[1], yaw, s))
        self.last_heat = torch.sigmoid(out[0, :len(CLASSES)]).numpy()
        return dets
