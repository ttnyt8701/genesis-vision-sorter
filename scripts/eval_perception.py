"""検証データで CNN と色しきい値ルールを比べる。画像の条件（暗い・明るい・ノイズ等）を変えて試す。

python scripts/eval_perception.py --model models/cubenet.pt --val data/val.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sorter import baseline  # noqa: E402
from sorter.model import decode, load_model, to_tensor  # noqa: E402
from train import load  # noqa: E402

CONDITIONS = {
    "標準": lambda x, r: x,
    "暗い (x0.6)": lambda x, r: x * 0.6,
    "明るい (x1.3+20)": lambda x, r: x * 1.3 + 20,
    "ノイズ (σ=15)": lambda x, r: x + r.normal(0, 15, x.shape),
    "色かぶり (青み)": lambda x, r: x * np.array([0.85, 0.95, 1.2]),
}


def match(labels, dets, px=4.0):
    tp = fp = fn = 0
    errs = []
    for lab, det in zip(labels, dets):
        used = set()
        for c, u, v, _ in lab:
            best, bd = None, px
            for k, (dc, du, dv, _y, _s) in enumerate(det):
                d = np.hypot(du - u, dv - v)
                if k not in used and dc == c and d < bd:
                    best, bd = k, d
            if best is None:
                fn += 1
            else:
                used.add(best); tp += 1; errs.append(bd)
        fp += len(det) - len(used)
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                px_err=float(np.mean(errs)) if errs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cubenet.pt")
    ap.add_argument("--val", default="data/val.npz")
    ap.add_argument("--out", default="results/perception.json")
    args = ap.parse_args()
    net = load_model(args.model)
    images, labels = load(args.val)
    res = {}
    for name, f in CONDITIONS.items():
        rng = np.random.default_rng(0)
        imgs = np.clip(f(images.astype(np.float32), rng), 0, 255).astype(np.uint8)
        with torch.no_grad():
            cnn = sum((decode(net(to_tensor(imgs[s:s + 100]))) for s in range(0, len(imgs), 100)), [])
        rule = [baseline.detect(im) for im in imgs]
        res[name] = {"CNN": match(labels, cnn), "色しきい値": match(labels, rule)}
        print(name, json.dumps(res[name], ensure_ascii=False))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
