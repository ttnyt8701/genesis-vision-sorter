"""カメラ座標とワールド座標の変換（OpenCV 形式の K, extrinsics を使う）。"""
import numpy as np


def project(points, K, E):
    """ワールド座標 (N, 3) を画素座標 (N, 2) に変換する。"""
    p = np.c_[np.asarray(points, float), np.ones(len(points))] @ E.T
    uv = p[:, :3] @ K.T
    return uv[:, :2] / uv[:, 2:3]


def unproject_to_plane(uv, K, E, z):
    """画素座標 (N, 2) を通る視線と平面 Z=z の交点を返す。"""
    uv = np.atleast_2d(uv)
    cam2world = np.linalg.inv(E)
    origin = cam2world[:3, 3]
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    dirs = rays @ cam2world[:3, :3].T
    t = (z - origin[2]) / dirs[:, 2]
    return origin + dirs * t[:, None]


def camera_matrices(cam):
    return np.array(cam.intrinsics, float), np.array(cam.extrinsics, float)
