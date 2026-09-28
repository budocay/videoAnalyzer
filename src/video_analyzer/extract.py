"""ffmpeg stage: probe, timestamped frames, mono 16 kHz audio.

Frame timestamps are computed here (index / fps), never by a model.

Frame extraction tries VideoToolbox first (hardware decode + scale + HDR→SDR tone mapping
via scale_vt), then falls back to plain software decoding. Homebrew ffmpeg has no zscale /
libplacebo, so the software path cannot tone-map: HLG still looks acceptable, PQ looks washed out.
"""
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10 / Dolby Vision), HLG


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int  # displayed size, i.e. after rotation
    height: int
    native_fps: float
    has_audio: bool
    rotation: int = 0  # display-matrix rotation in degrees, as reported by ffprobe
    pix_fmt: str = ""
    codec: str = ""
    hdr: bool = False


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    path: Path


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise FFmpegError(f"{' '.join(cmd)}\n{r.stderr[-1500:]}")
    return r.stdout


def _duration(data: dict, v: dict) -> float:
    for src in (data.get("format", {}), v):
        try:
            d = float(src.get("duration", "nan"))
        except ValueError:
            continue
        if d == d and d > 0:
            return d
    raise FFmpegError("Durée introuvable (flux live ou fichier tronqué ?)")


def probe(video: Path) -> VideoInfo:
    out = _run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
                "r_frame_rate,pix_fmt,color_transfer,duration:stream_disposition=attached_pic"
                ":stream_side_data=rotation",
                "-of", "json", str(video)])
    data = json.loads(out)
    streams = data.get("streams", [])
    # First real video stream; skip cover art (attached_pic) found in audio files.
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise FFmpegError(f"Pas de flux vidéo dans {video}")

    def rate(key):
        num, _, den = v.get(key, "0/1").partition("/")
        return float(num) / float(den) if den and float(den) else 0.0

    rotation = next((int(sd["rotation"]) for sd in v.get("side_data_list", []) if "rotation" in sd), 0)
    width, height = int(v["width"]), int(v["height"])
    if rotation % 180:
        width, height = height, width
    return VideoInfo(
        path=video,
        duration=_duration(data, v),
        width=width,
        height=height,
        native_fps=rate("avg_frame_rate") or rate("r_frame_rate"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        rotation=rotation,
        pix_fmt=v.get("pix_fmt", ""),
        codec=v.get("codec_name", ""),
        hdr=v.get("color_transfer") in HDR_TRANSFERS,
    )


def video_key(video: Path) -> str:
    """Cheap, stable identity: name + size + hash of the first and last MiB."""
    size = video.stat().st_size
    h = hashlib.sha1(f"{video.name}:{size}".encode())
    with video.open("rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
    return f"{video.stem}-{h.hexdigest()[:12]}"


def frame_times(duration: float, fps: float) -> list[float]:
    """Timestamps ffmpeg's fps filter produces: k / fps for grid points before the end."""
    n = max(1, int(duration * fps + 0.5))
    return [k / fps for k in range(n)]


def fit(width: int, height: int, size: int) -> tuple[int, int]:
    """Scale so the longest side is `size` (never upscale), both sides even."""
    s = min(1.0, size / max(width, height))
    return max(2, round(width * s / 2) * 2), max(2, round(height * s / 2) * 2)


def _transpose(rotation: int) -> list[str]:
    """Same mapping as ffmpeg's autorotate (fftools get_rotation / insert filters)."""
    theta = (-rotation) % 360
    if abs(theta - 90) < 1:
        return ["transpose=clock"]
    if abs(theta - 180) < 1:
        return ["hflip", "vflip"]
    if abs(theta - 270) < 1:
        return ["transpose=cclock"]
    return []


def _hw_cmd(info: VideoInfo, fps: float, size: int, pattern: str) -> list[str] | None:
    if sys.platform != "darwin":  # VideoToolbox is macOS-only; other systems use the software path
        return None
    if info.pix_fmt in ("yuv420p", "yuvj420p", "nv12"):
        dl = "nv12"
    elif info.pix_fmt in ("yuv420p10le", "p010le"):
        dl = "p010le"
    else:  # 4:2:2 / 4:4:4 / RGB / exotic: let software handle it
        return None
    dw, dh = fit(info.width, info.height, size)
    sw, sh = (dh, dw) if info.rotation % 180 else (dw, dh)  # scale_vt runs before rotation
    color = ":color_matrix=bt709:color_primaries=bt709:color_transfer=bt709" if info.hdr else ""
    vf = [f"fps={fps}", f"scale_vt=w={sw}:h={sh}{color}", "hwdownload", f"format={dl}",
          "format=yuvj420p", *_transpose(info.rotation)]
    return ["ffmpeg", "-v", "error", "-y", "-noautorotate",
            "-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld",
            "-i", str(info.path), "-map", "0:v:0", "-vf", ",".join(vf), "-q:v", "3", pattern]


def _sw_cmd(info: VideoInfo, fps: float, size: int, pattern: str) -> list[str]:
    dw, dh = fit(info.width, info.height, size)  # autorotate applies before -vf
    return ["ffmpeg", "-v", "error", "-y", "-i", str(info.path), "-map", "0:v:0",
            "-vf", f"fps={fps},scale={dw}:{dh},format=yuvj420p", "-q:v", "3", pattern]


def extract_frames(info: VideoInfo, outdir: Path, fps: float, size: int) -> tuple[list[Frame], str]:
    """Extract frames at `fps`, longest side `size`. Reuses a completed `outdir`.
    Returns (frames, method) with method in {"videotoolbox", "software", "cache"}."""
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / ".done"
    method = "cache"
    if not done.exists():
        pattern = str(outdir / "f_%06d.jpg")
        hw = _hw_cmd(info, fps, size, pattern)
        method = "software"
        if hw is not None:
            try:
                _run(hw)
                method = "videotoolbox"
            except FFmpegError:
                pass
        if method == "software":
            for old in outdir.glob("f_*.jpg"):
                old.unlink()
            _run(_sw_cmd(info, fps, size, pattern))
        done.write_text(method, encoding="utf-8")
    elif done.read_text(encoding="utf-8"):
        method = f"cache ({done.read_text(encoding="utf-8")})"
    paths = sorted(outdir.glob("f_*.jpg"))
    if not paths:
        raise FFmpegError(f"Aucune frame extraite de {info.path}")
    return [Frame(index=i, t=round(i / fps, 3), path=p) for i, p in enumerate(paths)], method


def extract_audio(video: Path, out: Path) -> Path:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.wav")
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
              "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", str(tmp)])
        tmp.rename(out)
    return out
