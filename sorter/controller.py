"""逆運動学 (IK) によるピック&プレース制御。"""
import numpy as np

from .scene import CUBE, HOME_QPOS, TRAY_POS

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # グリッパを真下に向ける姿勢 (w, x, y, z)
HAND_OFFSET = 0.105                           # hand リンク原点から指先中央までの距離 [m]
OPEN = 0.04          # 指の開き [m]
GRIP_FORCE = -8.0    # 閉じるときの力 [N]。強すぎると指の形のせいでブロックが滑り出る


def grasp_quat(yaw):
    """真下向きのまま z 軸まわりに yaw 回転した姿勢。"""
    rz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return quat_mul(rz, DOWN_QUAT)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class PickPlace:
    """ss.scene を進めながらアームを動かす。on_step は毎ステップ後、on_phase は動作の区切りで呼ばれる。"""

    def __init__(self, ss, on_step=None, on_phase=None):
        self.ss, self.franka = ss, ss.franka
        self.hand = self.franka.get_link("hand")
        self.arm_dofs = np.arange(7)
        self.finger_dofs = np.arange(7, 9)
        self.on_step = on_step or (lambda: None)
        self.on_phase = on_phase or (lambda name: None)
        self.q = HOME_QPOS.copy()
        self.grip_closed = False

    def step(self, n=1):
        for _ in range(n):
            self.ss.scene.step()
            self.on_step()

    def _apply(self, q):
        self.franka.control_dofs_position(q[:7], self.arm_dofs)
        if self.grip_closed:
            self.franka.control_dofs_force(np.array([GRIP_FORCE, GRIP_FORCE]), self.finger_dofs)
        else:
            self.franka.control_dofs_position(np.array([OPEN, OPEN]), self.finger_dofs)

    def ik(self, pos, quat, samples=20):
        q = self.franka.inverse_kinematics(link=self.hand, pos=np.asarray(pos), quat=np.asarray(quat),
                                           init_qpos=self.q, max_samples=samples)
        return np.asarray(q.cpu() if hasattr(q, "cpu") else q, dtype=float)

    def move_joint(self, q_goal, steps):
        """関節空間で滑らかに補間して移動する。"""
        q0 = self.q.copy()
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)
            self._apply(q0 + (q_goal - q0) * s)
            self.step()
        self.q = q_goal.copy()

    def move_line(self, p0, p1, quat, steps):
        """手先を直線的に動かす（IK を細かく解く）。"""
        for i in range(1, steps + 1):
            s = i / steps
            s = s * s * (3 - 2 * s)
            # 直線移動中は現在の姿勢から解き、解が別の枝へ飛ばないようにする
            q = self.ik(p0 + (p1 - p0) * s, quat, samples=1)
            self._apply(q)
            self.q = q
            self.step()

    def gripper(self, close, steps=40):
        self.grip_closed = close
        self._apply(self.q)
        self.step(steps)

    def home(self, steps=80):
        self.grip_closed = False
        self.move_joint(HOME_QPOS.copy(), steps)

    def pick_and_place(self, xy, yaw, color, place_slot=0):
        """(x, y) のブロックを掴み、color のトレイへ置く。"""
        quat = grasp_quat(yaw)
        hover = np.array([xy[0], xy[1], 0.25])
        grasp = np.array([xy[0], xy[1], CUBE / 2 + HAND_OFFSET - 0.01])
        self.on_phase("pick")
        self.gripper(False, steps=1)
        self.move_joint(self.ik(hover, quat), 90)
        self.move_line(hover, grasp, quat, 60)
        self._apply(self.q)
        self.step(25)                      # 降下後に静止するのを待ってから閉じる
        self.gripper(True, 45)
        self.move_line(grasp, hover, quat, 60)

        self.on_phase("place")
        t = TRAY_POS[color]
        # 同じトレイに複数置くときは少しずらす
        off = np.array([[-0.03, -0.03], [0.03, 0.03], [-0.03, 0.03], [0.03, -0.03]])[place_slot % 4]
        tq = quat                          # 運搬中は手首を回さない（回すと IK が大回りしてブロックを飛ばす）
        above = np.array([t[0] + off[0], t[1] + off[1], 0.22])
        drop = np.array([above[0], above[1], CUBE / 2 + HAND_OFFSET + 0.04])
        self.move_line(hover, above, tq, 110)
        self.move_line(above, drop, tq, 40)
        self._apply(self.q)
        self.step(15)
        self.gripper(False, 30)
        self.move_line(drop, above, tq, 40)
