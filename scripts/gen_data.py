"""シミュレータで真上カメラの画像と正解（各ブロックの画素位置・向き）を作る。

python scripts/gen_data.py --n 3000 --out data/train.npz --seed 0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import genesis as gs  # noqa: E402

from sorter.geometry import camera_matrices, project  # noqa: E402
from sorter.scene import (CLASSES, CUBE, TOP_CAM_LOOKAT, TOP_CAM_POS, TOP_CAM_UP,  # noqa: E402
                          apply_layout, build_scene, random_layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/train.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cam-jitter", type=float, default=1.0, help="カメラ位置のずらし幅の倍率")
    args = ap.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    ss = build_scene(n_per_class=2, n_distractor=2, with_view_cam=False)
    rng = np.random.default_rng(args.seed)
    images, labels = [], []
    for i in range(args.n):
        layout = random_layout(rng, n_per_class=(0, 2), n_distractor=(0, 2))
        apply_layout(ss, layout)
        # カメラの取り付け誤差を想定して位置と注視点を少しずらす
        j = args.cam_jitter
        pos = np.array(TOP_CAM_POS) + rng.normal(0, [0.01 * j, 0.01 * j, 0.02 * j])
        look = np.array(TOP_CAM_LOOKAT) + rng.normal(0, 0.01 * j, 3)
        ss.top_cam.set_pose(pos=pos, lookat=look, up=TOP_CAM_UP)
        ss.scene.step()
        img = ss.top_cam.render()[0]
        K, E = camera_matrices(ss.top_cam)
        objs = []
        for ci, c in enumerate(CLASSES):
            for x, y, yaw in layout[c]:
                uv = project([[x, y, CUBE]], K, E)[0]
                objs.append((ci, uv[0], uv[1], yaw))
        images.append(img)
        labels.append(np.array(objs, np.float32).reshape(-1, 4))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{args.n}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    counts = np.array([len(l) for l in labels])
    np.savez_compressed(args.out, images=np.stack(images), labels=np.concatenate(labels), counts=counts)
    print("saved", args.out, np.stack(images).shape, counts.sum(), "objects")


if __name__ == "__main__":
    main()
