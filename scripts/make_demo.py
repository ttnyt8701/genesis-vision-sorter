"""デモ動画を作る（タイトル → 学習データの説明 → 仕分け 3 本 → 結果）。

python scripts/make_demo.py --out video/demo.mp4
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import genesis as gs  # noqa: E402

from sorter.geometry import camera_matrices  # noqa: E402
from sorter.overlay import H, W, compose, title_card  # noqa: E402
from sorter.perception import CubeDetector  # noqa: E402
from sorter.runner import SortingRun, count_sorted  # noqa: E402
from sorter.scene import (CLASSES, CUBE, HOME_QPOS, WORK_X, WORK_Y, apply_layout,  # noqa: E402
                          build_scene, random_layout)
from sorter.video import VideoWriter  # noqa: E402

FPS = 25
EVERY = 4          # dt=0.01 なので 4 ステップごとに 1 枚で等速になる


class Recorder:
    def __init__(self, ss, writer):
        self.ss, self.w = ss, writer
        self.state = dict(status="", dets=[], done=0, total=0, heat=None, caption="", img_fx=None)
        self.n = 0
        self.caption_left = 0

    def say(self, text, sec):
        self.state["caption"] = text
        self.caption_left = int(sec * FPS)

    def frame(self):
        self.n += 1
        if self.n % EVERY:
            return
        st = self.state
        main = self.ss.view_cam.render()[0]
        top = self.ss.top_display.render()[0]
        if st["img_fx"] is not None:
            top = st["img_fx"](top)
        if self.caption_left > 0:
            self.caption_left -= 1
        elif st["caption"]:
            st["caption"] = ""
        self.w.write(compose(main, top, st["dets"], st["status"], st["caption"], st["done"], st["total"],
                             heat=st["heat"]))

    def hold(self, ss, steps):
        for _ in range(steps):
            ss.scene.step()
            self.frame()


class DarkDetector:
    """カメラ画像を暗くしてから CNN に渡す（露出不足の想定）。"""

    def __init__(self, det, gain=0.6):
        self.det, self.gain = det, gain

    def __call__(self, img):
        out = self.det(dark(img, self.gain))
        self.last_heat = self.det.last_heat
        return out


def dark(img, gain=0.6):
    return np.clip(img[..., :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)


def still(writer, img, sec):
    for _ in range(int(sec * FPS)):
        writer.write(img)


def training_card(data_path):
    from PIL import Image, ImageDraw
    d = np.load(data_path)
    imgs, labs, cnt = d["images"], d["labels"], d["counts"]
    off = np.r_[0, np.cumsum(cnt)]
    tiles = []
    for i in range(8):
        im = Image.fromarray(imgs[i]).resize((180, 180))
        dr = ImageDraw.Draw(im)
        for c, u, v, yaw in labs[off[i]:off[i + 1]]:
            u, v = u * 180 / 160, v * 180 / 160
            dr.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 255, 255), width=2)
        tiles.append(np.asarray(im))
    gap = lambda n: np.full((180, n, 3), 20, np.uint8)
    rows = [np.hstack(sum(([t, gap(8)] for t in tiles[k:k + 4]), [])[:-1]) for k in (0, 4)]
    grid = np.vstack([rows[0], np.full((8, rows[0].shape[1], 3), 20, np.uint8), rows[1]])
    return title_card([("AI 部分：ブロックの位置と向きを推定する CNN", 34, True),
                       ("シミュレータでブロックの配置をランダムに変え、真上カメラの画像 4000 枚と正解を自動で作成", 22, False),
                       ("色ごとの中心ヒートマップ・中心のずれ・向き (4θ) を出力する小さな CNN を CPU で学習", 22, False)],
                      sub=grid)


def result_card(sort_json, perc_json):
    lines = [("結果", 40, True)]
    if os.path.exists(sort_json):
        s = json.load(open(sort_json))["summary"]
        lines.append((f"ランダム配置 {s['episodes']} 回：{s['cubes']} 個中 {s['sorted']} 個を正しいトレイへ"
                      f"（黄色の誤操作 {s['yellow_moved']} 回）", 26, False))
    if os.path.exists(perc_json):
        p = json.load(open(perc_json))
        lines.append(("検出の再現率（検証画像 500 枚）", 26, True))
        for k, v in p.items():
            lines.append((f"{k}：CNN {v['CNN']['recall'] * 100:.0f}%  /  色しきい値 {v['色しきい値']['recall'] * 100:.0f}%",
                          24, False))
    return title_card(lines)


def episode(ss, rec, det, layout, captions=(), disturb=False, img_fx=None):
    ss.franka.set_qpos(HOME_QPOS)
    ss.franka.control_dofs_position(HOME_QPOS)
    apply_layout(ss, layout)
    rec.state.update(dets=[], heat=None, done=0, total=0, status="待機", img_fx=img_fx)
    for text, sec in captions:
        rec.say(text, sec)
    rec.hold(ss, 50)

    def after_place(n):
        if not disturb or n != 2:
            return
        # 残っているブロックを 1 つ、人が手で動かしたことにして別の場所へ移す
        for c in CLASSES:
            for i, (x, y, _) in enumerate(layout[c]):
                p = np.asarray(ss.cubes[c][i].get_pos().cpu())
                if abs(p[0] - x) < 0.02 and abs(p[1] - y) < 0.02:
                    goal = np.array([np.mean(WORK_X) + 0.02, WORK_Y[1] - 0.04 if y < 0 else WORK_Y[0] + 0.04])
                    rec.say("途中でブロックを動かす（外乱）→ 撮り直して位置を更新", 6)
                    rec.state["status"] = "外乱：ブロックが動かされた"
                    for k in range(1, 61):
                        q = p[:2] + (goal - p[:2]) * (k / 60)
                        ss.cubes[c][i].set_pos(np.array([q[0], q[1], CUBE / 2 + 0.03 * np.sin(np.pi * k / 60)]))
                        ss.cubes[c][i].zero_all_dofs_velocity()
                        ss.scene.step()
                        rec.frame()
                    rec.hold(ss, 30)
                    return

    run = SortingRun(ss, det, rec)
    run.run(after_place=after_place)
    ok, total, moved = count_sorted(ss, layout)
    rec.state["status"] = f"完了：{ok}/{total} 個を正しく仕分け"
    rec.say(f"{ok} / {total} 個を正しいトレイへ", 3)
    rec.hold(ss, 80)
    return ok, total, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cubenet.pt")
    ap.add_argument("--data", default="data/train.npz")
    ap.add_argument("--out", default="video/demo.mp4")
    args = ap.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    ss = build_scene(n_per_class=2, n_distractor=2, top_display_res=(320, 320))
    det = CubeDetector(args.model, *camera_matrices(ss.top_cam))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = VideoWriter(args.out, (W, H), fps=FPS)
    rec = Recorder(ss, vw)

    still(vw, title_card([("色と向きを画像から推定して仕分けるロボットアーム", 40, True),
                          ("Physical AI 応用編1 最終課題", 26, False),
                          ("Genesis シミュレータ / Franka Panda / 自作 CNN（等速再生）", 22, False)]), 4)
    still(vw, training_card(args.data), 7)

    log = []
    rng = np.random.default_rng(7)
    lay = random_layout(rng, n_per_class=(2, 2), n_distractor=(1, 1))
    log.append(episode(ss, rec, det, lay, captions=[("色を見分けて、同じ色のトレイへ運ぶ。黄色は対象外なので無視", 6)]))

    lay = random_layout(np.random.default_rng(21), n_per_class=(1, 2), n_distractor=(2, 2))
    log.append(episode(ss, rec, det, lay, disturb=True,
                       captions=[("1 個置くたびにカメラで撮り直して、残りのブロックを認識し直す", 6)]))

    lay = random_layout(np.random.default_rng(5), n_per_class=(1, 1), n_distractor=(1, 1))
    log.append(episode(ss, rec, DarkDetector(det), lay, img_fx=dark,
                       captions=[("カメラ画像を暗くした条件（明るさ 60%）。色しきい値では検出できない", 6)]))

    still(vw, result_card("results/sorting.json", "results/perception.json"), 7)
    vw.close()
    print("episodes", log, "frames", vw.frames, "sec", vw.frames / FPS)


if __name__ == "__main__":
    main()
