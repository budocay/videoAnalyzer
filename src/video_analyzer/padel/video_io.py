"""Full-rate frame streaming through ffmpeg (rawvideo pipe → numpy).

For reading *every* frame, multithreaded software decode is much faster than VideoToolbox
(1080p H.264, 15 min: 15 s vs 95 s — per-frame GPU download dominates). VideoToolbox stays
the choice for sparse extraction and HDR tone mapping (see extract.py).
"""
import subprocess
from collections.abc import Iterator

import numpy as np

from ..extract import VideoInfo


def stream(info: VideoInfo, width: int, height: int, start: float | None = None,
           end: float | None = None, fps: float | None = None,
           gray: bool = False) -> Iterator[np.ndarray]:
    """Yield HxWx3 uint8 RGB (or HxW gray) frames. Output size is forced to width x height."""
    cmd = ["ffmpeg", "-v", "error", "-threads", "0"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(info.path)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    vf = [f"fps={fps}"] if fps else []
    vf.append(f"scale={width}:{height}:flags=bilinear")
    pix = "gray" if gray else "rgb24"
    cmd += ["-map", "0:v:0", "-vf", ",".join(vf), "-pix_fmt", pix, "-f", "rawvideo", "-"]
    size = width * height * (1 if gray else 3)
    shape = (height, width) if gray else (height, width, 3)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size * 4)
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(shape)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()
