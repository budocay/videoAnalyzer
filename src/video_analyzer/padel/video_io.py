"""Full-rate frame streaming through ffmpeg (rawvideo pipe → numpy).

For reading *every* frame, multithreaded software decode is much faster than VideoToolbox
(1080p H.264, 15 min: 15 s vs 95 s — per-frame GPU download dominates). VideoToolbox stays
the choice for sparse extraction and HDR tone mapping (see extract.py).
"""
import queue
import subprocess
import threading
from collections.abc import Iterator

import numpy as np

from ..extract import VideoInfo

PREFETCH = 16  # decoded frames queued ahead of the consumer (16 x 11 MB at 1440p)


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
    shape = (height, width) if gray else (height, width, 3)
    # bufsize=0 + readinto the frame array: a large Python buffer on a Windows pipe caps 1440p at 20 img/s
    # (95 img/s unbuffered). A reader thread (readinto releases the GIL) overlaps decode with the consumer.
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    q: queue.Queue = queue.Queue(maxsize=PREFETCH)
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                frame = np.empty(shape, np.uint8)
                mv, got = memoryview(frame).cast("B"), 0
                while got < len(mv):
                    k = p.stdout.readinto(mv[got:])
                    if not k:
                        return
                    got += k
                q.put(frame)
        except (OSError, ValueError):  # pipe closed by the consumer
            pass
        finally:
            q.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        while (frame := q.get()) is not None:
            yield frame
    finally:
        stop.set()
        p.kill()
        while th.is_alive():  # unblock a reader waiting on a full queue
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass
        p.stdout.close()
        p.wait()
