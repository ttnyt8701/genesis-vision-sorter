"""ブロック検出用の小さな CNN（CenterNet 風）。

出力は入力の 1/4 解像度のマップ:
  0-2: 色ごとの中心ヒートマップ (red, green, blue)
  3-4: 中心のサブピクセルずれ (du, dv)
  5-6: 向き (cos 4θ, sin 4θ)  ※立方体は 90° 回転で同じ形なので 4θ で表す
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
N_CLS = 3


def conv(cin, cout, stride=1, dil=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dil, dilation=dil, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CubeNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.body = nn.Sequential(
            conv(3, w, 2), conv(w, w),              # 1/2
            conv(w, 2 * w, 2),                      # 1/4
            conv(2 * w, 2 * w, dil=2), conv(2 * w, 2 * w, dil=4), conv(2 * w, 2 * w))
        self.heat = nn.Conv2d(2 * w, N_CLS, 1)
        self.reg = nn.Conv2d(2 * w, 4, 1)
        nn.init.constant_(self.heat.bias, -2.2)   # 最初はほぼ「何もない」と出すようにしておく

    def forward(self, x):
        f = self.body(x)
        return torch.cat([self.heat(f), self.reg(f)], 1)


def to_tensor(images):
    """uint8 (N, H, W, 3) -> float (N, 3, H, W)"""
    x = torch.as_tensor(np.ascontiguousarray(images)).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def decode(out, thresh=0.35, k=20):
    """ネットワーク出力から [(cls, u, v, yaw, score), ...] を画像ごとに取り出す。"""
    heat = torch.sigmoid(out[:, :N_CLS])
    peak = (heat == F.max_pool2d(heat, 3, 1, 1)) & (heat > thresh)
    results = []
    for b in range(out.shape[0]):
        idx = peak[b].nonzero()
        scores = heat[b][peak[b]]
        order = scores.argsort(descending=True)[:k]
        dets = []
        for c, i, j in idx[order].tolist():
            du, dv, cs, sn = out[b, N_CLS:, i, j].tolist()
            u, v = (j + du) * STRIDE, (i + dv) * STRIDE
            yaw = np.arctan2(sn, cs) / 4.0
            dets.append((c, u, v, yaw, float(heat[b, c, i, j])))
        results.append(dets)
    return results


def load_model(path, device="cpu"):
    net = CubeNet()
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()
