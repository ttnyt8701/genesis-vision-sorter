"""CubeNet の学習。

python scripts/train.py --train data/train.npz --val data/val.npz --out models/cubenet.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sorter.model import N_CLS, STRIDE, CubeNet, decode, to_tensor  # noqa: E402


def load(path):
    d = np.load(path)
    off = np.r_[0, np.cumsum(d["counts"])]
    labels = [d["labels"][off[i]:off[i + 1]] for i in range(len(d["counts"]))]
    return d["images"], labels


def augment(img, lab, rng):
    """反転・90°回転と、明るさ・ノイズの変化。立方体の向きは 4θ で扱うので 90° 回転では変わらない。"""
    h, w = img.shape[:2]
    lab = lab.copy()
    if rng.random() < 0.5:
        img = img[:, ::-1]
        lab[:, 1] = w - lab[:, 1]
        lab[:, 3] = -lab[:, 3]
    if rng.random() < 0.5:
        img = img[::-1]
        lab[:, 2] = h - lab[:, 2]
        lab[:, 3] = -lab[:, 3]
    k = rng.integers(4)
    for _ in range(k):  # 反時計回りに 90° ずつ回す: (u, v) -> (v, w - u)
        img = np.rot90(img)
        lab[:, 1], lab[:, 2] = lab[:, 2].copy(), w - lab[:, 1]
    x = img.astype(np.float32)
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)          # 明るさ・コントラスト
    x = x * rng.uniform(0.92, 1.08, size=3)                       # 色味の少しのずれ
    x = x + rng.normal(0, rng.uniform(0, 8), x.shape)             # センサノイズ
    return np.clip(x, 0, 255).astype(np.uint8), lab


def make_target(lab, size, sigma=1.2):
    hs, ws = size[0] // STRIDE, size[1] // STRIDE
    heat = np.zeros((N_CLS, hs, ws), np.float32)
    reg = np.zeros((4, hs, ws), np.float32)
    mask = np.zeros((hs, ws), np.float32)
    yy, xx = np.mgrid[:hs, :ws]
    for c, u, v, yaw in lab:
        cu, cv = u / STRIDE, v / STRIDE
        j, i = int(cu), int(cv)
        if not (0 <= i < hs and 0 <= j < ws):
            continue
        g = np.exp(-((xx - cu + 0.5) ** 2 + (yy - cv + 0.5) ** 2) / (2 * sigma ** 2))
        heat[int(c)] = np.maximum(heat[int(c)], g)
        heat[int(c), i, j] = 1.0
        reg[:, i, j] = [cu - j, cv - i, np.cos(4 * yaw), np.sin(4 * yaw)]
        mask[i, j] = 1.0
    return heat, reg, mask


def focal_loss(logit, gt):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    neg_loss = -(torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


def batch(images, labels, idx, rng, aug):
    xs, hs, rs, ms = [], [], [], []
    for i in idx:
        img, lab = (augment(images[i], labels[i], rng) if aug else (images[i], labels[i]))
        h, r, m = make_target(lab, img.shape[:2])
        xs.append(np.ascontiguousarray(img)); hs.append(h); rs.append(r); ms.append(m)
    return to_tensor(np.stack(xs)), *(torch.from_numpy(np.stack(a)) for a in (hs, rs, ms))


def evaluate(net, images, labels, px=4.0):
    """中心のずれが px 画素以内で色も合っていれば正解とみなす。"""
    net.eval()
    tp = fp = fn = 0
    errs, yaw_errs = [], []
    with torch.no_grad():
        for s in range(0, len(images), 100):
            dets = decode(net(to_tensor(images[s:s + 100])))
            for lab, det in zip(labels[s:s + 100], dets):
                used = set()
                for c, u, v, yaw in lab:
                    best, bd = None, px
                    for k, (dc, du, dv, dyaw, _) in enumerate(det):
                        d = np.hypot(du - u, dv - v)
                        if k not in used and dc == c and d < bd:
                            best, bd = k, d
                    if best is None:
                        fn += 1
                    else:
                        used.add(best); tp += 1; errs.append(bd)
                        e = (det[best][3] - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
                        yaw_errs.append(abs(e))
                fp += len(det) - len(used)
    net.train()
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                px_err=float(np.mean(errs)) if errs else float("nan"),
                yaw_err_deg=float(np.degrees(np.mean(yaw_errs))) if yaw_errs else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.npz")
    ap.add_argument("--val", default="data/val.npz")
    ap.add_argument("--out", default="models/cubenet.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    rng = np.random.default_rng(0)
    tr_img, tr_lab = load(args.train)
    va_img, va_lab = load(args.val)
    net = CubeNet()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr_img) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(len(tr_img))
        tot = 0.0
        for s in range(0, len(perm) - args.bs + 1, args.bs):
            x, heat, reg, mask = batch(tr_img, tr_lab, perm[s:s + args.bs], rng, aug=True)
            out = net(x)
            l_heat = focal_loss(out[:, :N_CLS], heat)
            m = mask.unsqueeze(1)
            n = m.sum().clamp(min=1)
            l_off = (F.l1_loss(out[:, N_CLS:N_CLS + 2], reg[:, :2], reduction="none") * m).sum() / n
            l_yaw = (F.l1_loss(out[:, N_CLS + 2:], reg[:, 2:], reduction="none") * m).sum() / n
            loss = l_heat + l_off + 0.5 * l_yaw
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        res = evaluate(net, va_img, va_lab)
        res.update(epoch=ep + 1, loss=tot / (len(perm) // args.bs), sec=round(time.time() - t0))
        log.append(res)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
        torch.save(net.state_dict(), args.out)
    np.save(os.path.splitext(args.out)[0] + "_log.npy", log)


if __name__ == "__main__":
    main()
