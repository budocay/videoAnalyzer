"""Players: YOLO11-pose at full frame rate on rally shots, court positions, 4-player tracking.

- Detector: ultralytics YOLO11n-pose on MPS at imgsz=1920 (at 1280 the far players, ~100 px
  tall in 1080p, are missed). ~60 ms/frame on an M1 Max while MLX shares the GPU.
- Court position = midpoint of the ankles (bbox bottom-centre if ankles are unsure),
  projected with the court homography. Only people standing inside the court are kept.
- Tracking: players never cross the net during a rally, so each side (far y<10, near y>10) has
  2 slots matched frame to frame by Hungarian assignment on court positions.
- Team colour: mean RGB of the torso polygon (shoulders→hips); clustered into 2 teams later.

COCO keypoints: 0 nose, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 15/16 ankles.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..extract import VideoInfo
from .court import Court
from .rules import COURT_LENGTH, COURT_WIDTH, NET_Y
from .video_io import stream

POSE_MODEL = "yolo11n-pose.pt"
IMGSZ = 1920
BATCH = 8
# Pose is sampled at <= 30 img/s whatever the source: every per-frame setting downstream (wrist speed per frame and
# its smash/bandeja thresholds, ±2-frame contact height, classifier window of ±15 frames, smoothing) was tuned on the
# ~30 img/s Paris final. A 60 img/s source would halve per-frame speeds and the classifier's time span, and cost 2x.
POSE_FPS = 30.0


def pose_fps(info: VideoInfo) -> float:
    return POSE_FPS if info.native_fps > POSE_FPS * 1.05 else info.native_fps
MARGIN = 0.6  # metres outside the lines still counted as "on court" (players lean on the glass)
MAX_JUMP = 2.5  # metres a player can move between two frames before the slot is considered lost


@dataclass
class ShotTracks:
    """Arrays indexed [frame, slot]; slots 0,1 = far side, 2,3 = near side."""
    shot_id: int
    t0: float
    fps: float
    pos: np.ndarray    # (T, 4, 2) court metres, NaN when missing
    kpts: np.ndarray   # (T, 4, 17, 3) image px + confidence
    color: np.ndarray  # (4, 3) mean torso RGB over the shot

    def save(self, path: Path) -> None:
        np.savez_compressed(path, shot_id=self.shot_id, t0=self.t0, fps=self.fps,
                            pos=self.pos, kpts=self.kpts, color=self.color)

    @classmethod
    def load(cls, path: Path) -> "ShotTracks":
        d = np.load(path)
        return cls(int(d["shot_id"]), float(d["t0"]), float(d["fps"]), d["pos"], d["kpts"], d["color"])


def _feet(k: np.ndarray, box: np.ndarray) -> np.ndarray:
    ank = k[[15, 16]]
    good = ank[:, 2] > 0.4
    if good.any():
        return ank[good, :2].mean(axis=0)
    return np.array([(box[0] + box[2]) / 2, box[3]])


def _torso_color(frame: np.ndarray, k: np.ndarray) -> np.ndarray | None:
    pts = k[[5, 6, 12, 11]]
    if (pts[:, 2] < 0.4).any():
        return None
    x0, y0 = pts[:, :2].min(axis=0).astype(int)
    x1, y1 = pts[:, :2].max(axis=0).astype(int)
    # inner 60 % of the shoulder-hip box, avoids arms and background
    dx, dy = (x1 - x0) // 5, (y1 - y0) // 5
    patch = frame[max(0, y0 + dy):y1 - dy, max(0, x0 + dx):x1 - dx]
    if patch.size < 30:
        return None
    return patch.reshape(-1, 3).mean(axis=0)


class PoseTracker:
    def __init__(self, court: Court, model_dir: Path):
        from ultralytics import YOLO

        from ..backend import torch_device
        from ..console import setup_streams

        setup_streams()  # ultralytics forces stdout to UTF-8 on Windows: garbled accents through a pipe

        self.court = court
        self.model = YOLO(str(model_dir / POSE_MODEL))
        self.device = torch_device()  # cuda, mps or cpu

    def _detect(self, frames: list[np.ndarray], up: float = 1.0):
        """frames may be downscaled from court.frame_size; `up` maps their pixels back to it."""
        res = self.model.predict(frames, imgsz=IMGSZ, device=self.device, verbose=False, conf=0.25)
        out = []
        for frame, r in zip(frames, res):
            if r.keypoints is None or len(r.boxes) == 0:
                out.append([])
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy()  # (N,17,3)
            cols = [_torso_color(frame, k) for k in kp]
            boxes *= up
            kp[..., :2] *= up
            feet = np.array([_feet(k, b) for k, b in zip(kp, boxes)])
            cpos = self.court.to_court(feet)
            dets = []
            for p, k, c in zip(cpos, kp, cols):
                if -MARGIN <= p[0] <= COURT_WIDTH + MARGIN and -MARGIN <= p[1] <= COURT_LENGTH + MARGIN:
                    dets.append((p, k, c))
            out.append(dets)
        return out

    def track_shot(self, info: VideoInfo, shot_id: int, start: float, end: float) -> ShotTracks:
        fps = pose_fps(info)
        w, h = self.court.frame_size
        # ffmpeg downscales to the detector size itself: YOLO would do it anyway, single-threaded on the CPU
        # (6.8 ms/img at 1440p), and the pipe carries 44 % less. Keypoints are mapped back to w x h.
        up = max(1.0, max(w, h) / IMGSZ)
        sw, sh = round(w / up / 2) * 2, round(h / up / 2) * 2
        frames = stream(info, sw, sh, start=start, end=end, fps=fps if fps != info.native_fps else None)
        pos, kpts, cols = [], [], [[] for _ in range(4)]
        prev = np.full((4, 2), np.nan)
        batch = []

        def flush():
            for dets in self._detect(batch, w / sw):
                p_t = np.full((4, 2), np.nan)
                k_t = np.zeros((4, 17, 3))
                for side, slots in ((0, (0, 1)), (1, (2, 3))):
                    cand = [d for d in dets if (d[0][1] < NET_Y) == (side == 0)]
                    cand = sorted(cand, key=lambda d: -d[1][:, 2].mean())[:2]
                    if not cand:
                        continue
                    ref = prev[list(slots)]
                    if np.isnan(ref).all():
                        # initialise by x order: slot a = left in image, slot b = right
                        order = sorted(range(len(cand)), key=lambda i: cand[i][0][0])
                        assign = [(slots[j], order[j]) for j in range(len(order))]
                    else:
                        cost = np.zeros((2, len(cand)))
                        for i, s in enumerate(slots):
                            for j, d in enumerate(cand):
                                cost[i, j] = (np.linalg.norm(ref[i] - d[0]) if not np.isnan(ref[i]).any()
                                              else MAX_JUMP)
                        r_, c_ = linear_sum_assignment(cost)
                        assign = [(slots[i], j) for i, j in zip(r_, c_)]
                    for s, j in assign:
                        p, k, c = cand[j]
                        if not np.isnan(prev[s]).any() and np.linalg.norm(prev[s] - p) > MAX_JUMP:
                            continue
                        p_t[s], k_t[s] = p, k
                        if c is not None:
                            cols[s].append(c)
                for s in range(4):
                    if not np.isnan(p_t[s]).any():
                        prev[s] = p_t[s]
                pos.append(p_t)
                kpts.append(k_t)
            batch.clear()

        for f in frames:
            batch.append(f)
            if len(batch) == BATCH:
                flush()
        if batch:
            flush()
        color = np.array([np.median(c, axis=0) if c else [np.nan] * 3 for c in cols])
        return ShotTracks(shot_id, start, fps, np.array(pos), np.array(kpts), color)


def track_rallies(info: VideoInfo, court: Court, shots, cache: Path, model_dir: Path, log=print) -> list[ShotTracks]:
    out_dir = cache / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    tracks = []
    rallies = [s for s in shots if s.rally]
    total = sum(s.duration for s in rallies)
    fps = pose_fps(info)
    stale = [p for p in out_dir.glob("shot*.npz") if abs(ShotTracks.load(p).fps - fps) > 0.01]
    if stale:  # tracks from before POSE_FPS: frame indices of cached hits no longer match
        log(f"  {len(stale)} pistes à {ShotTracks.load(stale[0]).fps:g} img/s recalculées à {fps:g} img/s")
        for p in stale + list(cache.glob("hits*.json")):
            p.unlink()
    done_s, t_all = 0.0, time.perf_counter()
    computed = 0.0
    for k, s in enumerate(rallies, 1):
        path = out_dir / f"shot{s.id:04d}.npz"
        if path.exists():
            tracks.append(ShotTracks.load(path))
            done_s += s.duration
            continue
        tracker = tracker or PoseTracker(court, model_dir)
        t = time.perf_counter()
        tr = tracker.track_shot(info, s.id, s.start, s.end)
        tr.save(path)
        tracks.append(tr)
        done_s += s.duration
        computed += s.duration
        seen = np.isfinite(tr.pos[..., 0]).mean(axis=0)
        el = time.perf_counter() - t_all
        dt = time.perf_counter() - t
        log(f"  [{k}/{len(rallies)}] plan {s.id} {s.duration:.1f}s → {len(tr.pos)} images en "
            f"{dt:.1f}s ({len(tr.pos) / max(dt, 1e-6):.0f} img/s) · présence joueurs {np.round(seen, 2).tolist()} · "
            f"reste ~{el / computed * (total - done_s) / 60:.1f} min")
    (cache / "tracks" / "index.json").write_text(json.dumps([t.shot_id for t in tracks]), encoding="utf-8")
    return tracks
