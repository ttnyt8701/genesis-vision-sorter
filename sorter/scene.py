"""Genesis のシーン構築（Franka、ブロック、トレイ、カメラ）。

Genesis は build 後にエンティティを追加できないため、ブロックは最大数ぶん作っておき、
使わないものはカメラの視野外（PARK_POS）に退避させる。
"""
from dataclasses import dataclass, field

import numpy as np
import genesis as gs

# 色クラス。CNN の出力チャンネル順と一致させる
CLASSES = ["red", "green", "blue"]
RGB = {
    "red": (0.85, 0.15, 0.15),
    "green": (0.15, 0.7, 0.25),
    "blue": (0.15, 0.3, 0.85),
    "yellow": (0.9, 0.8, 0.1),   # 仕分け対象外の「妨害物」
}

CUBE = 0.04                      # ブロック一辺 [m]
TABLE_Z = 0.0                    # 床面
# ブロックが置かれる作業領域（ロボット座標, m）
WORK_X = (0.45, 0.70)
WORK_Y = (-0.20, 0.20)
# 色別トレイの中心
TRAY_POS = {
    "red": np.array([0.30, 0.40]),
    "green": np.array([0.47, 0.40]),
    "blue": np.array([0.64, 0.40]),
}
TRAY_SIZE = 0.14
PARK_POS = np.array([-1.5, -1.5, 0.02])  # 未使用ブロックの退避先

# 真上から見下ろす認識用カメラ（作業領域を写す）
TOP_CAM_POS = (0.575, 0.0, 1.10)
TOP_CAM_LOOKAT = (0.575, 0.0, 0.0)
TOP_CAM_UP = (1.0, 0.0, 0.0)     # 画像の上方向 = ロボット前方 (+x)
TOP_CAM_FOV = 26
TOP_RES = (160, 160)

# 動画用の斜め俯瞰カメラ
VIEW_CAM_POS = (1.38, -0.72, 1.0)
VIEW_CAM_LOOKAT = (0.40, 0.02, 0.12)


@dataclass
class SorterScene:
    scene: object
    franka: object
    cubes: dict = field(default_factory=dict)      # color -> [entity, ...]
    top_cam: object = None
    view_cam: object = None
    top_display: object = None


def _tray(scene, center, color):
    """薄い底板と 4 枚の壁で浅いトレイを作る（固定物体）。"""
    cx, cy = center
    s, t, h = TRAY_SIZE, 0.008, 0.03
    surf = gs.surfaces.Default(color=color)
    parts = [((cx, cy, 0.003), (s, s, 0.006)),
             ((cx + s / 2, cy, h / 2), (t, s, h)), ((cx - s / 2, cy, h / 2), (t, s, h)),
             ((cx, cy + s / 2, h / 2), (s, t, h)), ((cx, cy - s / 2, h / 2), (s, t, h))]
    for pos, size in parts:
        scene.add_entity(gs.morphs.Box(pos=pos, size=size, fixed=True), surface=surf)


def build_scene(n_per_class=2, n_distractor=1, view_res=(940, 720), with_view_cam=True, top_display_res=None, dt=0.01):
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=dt),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.35, 0.35, 0.35)),
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.82, 0.8, 0.76)))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    for name, c in TRAY_POS.items():
        _tray(scene, c, RGB[name])

    cubes = {}
    for name in CLASSES + ["yellow"]:
        n = n_distractor if name == "yellow" else n_per_class
        cubes[name] = []
        for i in range(n):
            p = PARK_POS + np.array([0.1 * len(cubes) , 0.1 * i, 0])
            cubes[name].append(scene.add_entity(
                gs.morphs.Box(size=(CUBE, CUBE, CUBE), pos=tuple(p)),
                material=gs.materials.Rigid(friction=1.5, rho=500),
                surface=gs.surfaces.Default(color=RGB[name])))

    top_cam = scene.add_camera(res=TOP_RES, pos=TOP_CAM_POS, lookat=TOP_CAM_LOOKAT,
                               up=TOP_CAM_UP, fov=TOP_CAM_FOV)
    top_display = None
    if top_display_res:   # 動画表示用に同じ位置の高解像度カメラを置く
        top_display = scene.add_camera(res=top_display_res, pos=TOP_CAM_POS, lookat=TOP_CAM_LOOKAT,
                                       up=TOP_CAM_UP, fov=TOP_CAM_FOV)
    view_cam = None
    if with_view_cam:
        view_cam = scene.add_camera(res=view_res, pos=VIEW_CAM_POS, lookat=VIEW_CAM_LOOKAT, fov=38)
    scene.build()
    setup_franka(franka)
    return SorterScene(scene, franka, cubes, top_cam, view_cam, top_display)


# 待機姿勢。アームを -y 側へ逃がして、真上のカメラに写らないようにしている
HOME_QPOS = np.array([-1.57, -0.6, 0.0, -2.3, 0.0, 1.75, 0.785, 0.04, 0.04])


def setup_franka(franka):
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    franka.set_qpos(HOME_QPOS)
    franka.control_dofs_position(HOME_QPOS)


def random_layout(rng, n_per_class=(1, 2), n_distractor=(0, 1), min_dist=0.075):
    """作業領域内に重ならないようブロックを配置するレイアウトを生成する。

    戻り値: {color: [(x, y, yaw), ...]}
    """
    counts = {c: rng.integers(n_per_class[0], n_per_class[1] + 1) for c in CLASSES}
    counts["yellow"] = rng.integers(n_distractor[0], n_distractor[1] + 1)
    placed, layout = [], {}
    for c, n in counts.items():
        layout[c] = []
        for _ in range(n):
            for _try in range(200):
                p = np.array([rng.uniform(*WORK_X), rng.uniform(*WORK_Y)])
                if all(np.linalg.norm(p - q) > min_dist for q in placed):
                    break
            placed.append(p)
            layout[c].append((p[0], p[1], rng.uniform(-np.pi / 4, np.pi / 4)))
    return layout


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def apply_layout(ss: SorterScene, layout):
    """レイアウトどおりにブロックを置き、残りを退避させる。"""
    for c, ents in ss.cubes.items():
        poses = layout.get(c, [])
        for i, e in enumerate(ents):
            if i < len(poses):
                x, y, yaw = poses[i]
                e.set_pos(np.array([x, y, CUBE / 2 + 0.001]))
                e.set_quat(yaw_quat(yaw))
            else:
                e.set_pos(PARK_POS + np.array([0.1 * CLASSES.index(c) if c in CLASSES else 0.4, 0.1 * i, 0]))
                e.set_quat(np.array([1.0, 0, 0, 0]))
            e.zero_all_dofs_velocity()
