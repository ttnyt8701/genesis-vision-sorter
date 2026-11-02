"""撮影 → 認識 → 1 個運ぶ → 撮り直す、を繰り返す仕分けループ。"""
import numpy as np

from .controller import PickPlace
from .overlay import NAME_JA
from .scene import CLASSES, TRAY_POS, TRAY_SIZE


class NullRecorder:
    def __init__(self):
        self.state = {}

    def frame(self):
        pass

    def hold(self, ss, steps):
        for _ in range(steps):
            ss.scene.step()


def pick_order(dets):
    # ロボットに近いものから取る（遠くのブロックを取るときに手前を引っかけにくい）
    return sorted(dets, key=lambda d: np.hypot(d.x, d.y))


class SortingRun:
    def __init__(self, ss, detector, recorder=None):
        self.ss = ss
        self.detector = detector
        self.rec = recorder or NullRecorder()
        self.pp = PickPlace(ss, on_step=self.rec.frame, on_phase=self._phase)
        self.target = None

    def _phase(self, name):
        if self.target is None:
            return
        c = NAME_JA[self.target.color]
        self.rec.state["status"] = f"{c}ブロックを掴む" if name == "pick" else f"{c}のトレイへ置く"

    def observe(self):
        img = self.ss.top_cam.render()[0]
        dets = self.detector(img)
        self.rec.state.update(dets=dets, heat=self.detector.last_heat)
        return dets

    def run(self, max_iter=12, after_place=None):
        st = self.rec.state
        st.update(status="撮影して認識", done=0, total=0)
        slots = {c: 0 for c in CLASSES}
        placed = 0
        for it in range(max_iter):
            st["status"] = "撮影して認識"
            dets = self.observe()
            if it == 0:
                st["total"] = len(dets)
            self.rec.hold(self.ss, 40)
            if not dets:
                break
            self.target = pick_order(dets)[0]
            self.pp.pick_and_place((self.target.x, self.target.y), self.target.yaw, self.target.color,
                                   slots[self.target.color])
            slots[self.target.color] += 1
            placed += 1
            st["done"] = placed
            if after_place:
                after_place(placed)
            self.target = None
            st["status"] = "待機姿勢へ戻る"
            self.pp.home(70)
        st["status"] = "完了"
        st["dets"] = []
        return placed


def count_sorted(ss, layout):
    """正しいトレイに入ったブロック数と、触ってはいけない黄色が動いた数を返す。"""
    ok, total = 0, 0
    for c in CLASSES:
        for i, _ in enumerate(layout.get(c, [])):
            p = np.asarray(ss.cubes[c][i].get_pos().cpu())
            ok += bool(np.all(np.abs(p[:2] - TRAY_POS[c]) < TRAY_SIZE / 2) and p[2] < 0.06)
            total += 1
    moved = 0
    for i, (x, y, _) in enumerate(layout.get("yellow", [])):
        p = np.asarray(ss.cubes["yellow"][i].get_pos().cpu())
        moved += bool(np.hypot(p[0] - x, p[1] - y) > 0.01)
    return ok, total, moved
