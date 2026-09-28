"""--dry-run support: frame/segment counts and a rough memory estimate, without loading a model."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .extract import VideoInfo, fit, frame_times
from .timeline import plan_segments

# Fallback weight sizes (GB, from the HF API) when the model is not in the local cache yet.
KNOWN_WEIGHTS_GB = {
    "mlx-community/Qwen3.5-9B-MLX-8bit": 10.45,
    "mlx-community/Qwen3.5-4B-MLX-8bit": 5.16,
    "mlx-community/Qwen3-VL-8B-Instruct-8bit": 9.87,
    "mlx-community/whisper-large-v3-turbo": 1.61,
}
# Empirical peak-memory overhead on top of weights. Calibrated on the M1 Max runs (see README).
BASE_OVERHEAD_GB = 1.1
GB_PER_1K_PROMPT_TOKENS = 0.22


@dataclass(frozen=True)
class Plan:
    n_frames: int
    n_segments: int
    frames_per_segment: int
    frame_size: tuple[int, int]
    image_tokens: int
    prompt_tokens_per_segment: int
    weights_gb: float
    whisper_gb: float
    est_peak_gb: float


def _snapshot(repo: str) -> Path | None:
    from huggingface_hub.constants import HF_HUB_CACHE

    base = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots"
    snaps = sorted(base.glob("*")) if base.exists() else []
    return snaps[-1] if snaps else None


def weights_gb(repo: str) -> float:
    snap = _snapshot(repo)
    if snap is not None:
        files = list(snap.glob("*.safetensors"))
        if files:
            return sum(f.stat().st_size for f in files) / 1e9  # stat() follows the xet symlinks
    return KNOWN_WEIGHTS_GB.get(repo, float("nan"))


def image_tokens(repo: str, width: int, height: int) -> int:
    """Qwen-VL family: one token per (patch*merge)^2 pixels; smart_resize rounds each side to that multiple."""
    patch, merge = 16, 2
    snap = _snapshot(repo)
    if snap is not None and (snap / "preprocessor_config.json").exists():
        pc = json.loads((snap / "preprocessor_config.json").read_text(encoding="utf-8"))
        patch, merge = pc.get("patch_size", patch), pc.get("merge_size", merge)
    f = patch * merge
    return max(1, round(height / f)) * max(1, round(width / f))


def make_plan(info: VideoInfo, vlm: str, whisper: str, fps: float, segment: float,
              size: int, audio: bool) -> Plan:
    n_frames = len(frame_times(info.duration, fps))
    segs = plan_segments(info.duration, segment)
    width, height = fit(info.width, info.height, size)
    tok = image_tokens(vlm, width, height)
    per_seg = math.ceil(segment * fps)
    prompt_tokens = per_seg * (tok + 2) + 150  # +2 = vision_start/end, 150 = text
    w = weights_gb(vlm)
    return Plan(
        n_frames=n_frames,
        n_segments=len(segs),
        frames_per_segment=per_seg,
        frame_size=(width, height),
        image_tokens=tok,
        prompt_tokens_per_segment=prompt_tokens,
        weights_gb=w,
        whisper_gb=weights_gb(whisper) if audio and info.has_audio else 0.0,
        est_peak_gb=w + BASE_OVERHEAD_GB + GB_PER_1K_PROMPT_TOKENS * prompt_tokens / 1000,
    )
