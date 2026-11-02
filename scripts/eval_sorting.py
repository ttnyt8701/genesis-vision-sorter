"""CNN の推定だけを使って仕分けを最後まで行い、正しいトレイに入った割合を数える。

python scripts/eval_sorting.py --episodes 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import genesis as gs  # noqa: E402

from sorter.geometry import camera_matrices  # noqa: E402
from sorter.perception import CubeDetector  # noqa: E402
from sorter.runner import SortingRun, count_sorted  # noqa: E402
from sorter.scene import HOME_QPOS, apply_layout, build_scene, random_layout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cubenet.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="results/sorting.json")
    args = ap.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    ss = build_scene(n_per_class=2, n_distractor=2, with_view_cam=False)
    det = CubeDetector(args.model, *camera_matrices(ss.top_cam))
    rows = []
    for ep in range(args.episodes):
        rng = np.random.default_rng(args.seed + ep)
        layout = random_layout(rng, n_per_class=(1, 2), n_distractor=(0, 2))
        ss.franka.set_qpos(HOME_QPOS)
        ss.franka.control_dofs_position(HOME_QPOS)
        apply_layout(ss, layout)
        for _ in range(30):
            ss.scene.step()
        t = time.time()
        run = SortingRun(ss, det)
        picks = run.run()
        ok, total, moved = count_sorted(ss, layout)
        rows.append(dict(episode=ep, cubes=total, sorted=ok, picks=picks, yellow_moved=moved,
                         sec=round(time.time() - t, 1)))
        print(rows[-1], flush=True)
    s = dict(cubes=sum(r["cubes"] for r in rows), sorted=sum(r["sorted"] for r in rows),
             perfect_episodes=sum(r["cubes"] == r["sorted"] for r in rows), episodes=len(rows),
             extra_picks=sum(r["picks"] - r["cubes"] for r in rows),
             yellow_moved=sum(r["yellow_moved"] for r in rows))
    print("SUMMARY", s)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(summary=s, episodes=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
