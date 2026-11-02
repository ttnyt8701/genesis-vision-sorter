"""ffmpeg へフレームを流し込んで MP4 を書き出す簡易ライタ。"""
import subprocess

import numpy as np


class VideoWriter:
    def __init__(self, path, size, fps=30, crf=20):
        w, h = size
        self.size = (w, h)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
             "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, frame):
        frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        assert frame.shape[1] == self.size[0] and frame.shape[0] == self.size[1], frame.shape
        self.proc.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
