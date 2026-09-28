"""Compute backend selection: MLX on Apple Silicon, portable (Ollama + Whisper + PyTorch) elsewhere.

- "mlx"      : macOS arm64 with mlx installed — mlx-vlm, mlx-whisper, Silero VAD from mlx-audio.
- "portable" : Windows / Linux / Intel Mac — VLM served by a local Ollama (qwen3-vl, which uses NVIDIA
               CUDA or AMD ROCm by itself), Silero VAD from faster-whisper, and Whisper on:
                 NVIDIA → faster-whisper (CTranslate2 CUDA),
                 AMD with a ROCm PyTorch → transformers Whisper on the GPU (CTranslate2 has no AMD support),
                 otherwise → faster-whisper CPU int8.
Pose (ultralytics) runs on torch "cuda" (NVIDIA, or AMD through ROCm/HIP), MPS or CPU.
Override with VIDEO_ANALYZER_BACKEND=mlx|portable.
"""
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


@dataclass(frozen=True)
class GPU:
    vendor: str  # "nvidia", "amd" or ""
    name: str
    vram_gb: float


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def name() -> str:
    forced = os.environ.get("VIDEO_ANALYZER_BACKEND", "").lower()
    if forced in ("mlx", "portable"):
        return forced
    return "mlx" if is_apple_silicon() and importlib.util.find_spec("mlx") else "portable"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia() -> GPU | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    try:
        gpu_name, mem = out.splitlines()[0].rsplit(",", 1)
        return GPU("nvidia", gpu_name.strip(), float(mem) / 1024)
    except (ValueError, IndexError):
        return None


def _amd_linux() -> GPU | None:
    best = None
    for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
        try:
            if (dev / "vendor").read_text().strip() != "0x1002":
                continue
            vram = int((dev / "mem_info_vram_total").read_text()) / 2**30
        except (OSError, ValueError):
            continue
        if best is None or vram > best.vram_gb:  # skip the Ryzen iGPU: keep the biggest VRAM
            best = GPU("amd", "AMD Radeon", vram)
    return best


def _amd_windows() -> GPU | None:
    # WMI's AdapterRAM is a 32-bit field capped at 4 GB; the driver's registry key has the real size.
    ps = ("Get-ItemProperty -Path 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\"
          "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue | "
          "Where-Object { $_.DriverDesc -match 'Radeon' } | "
          "ForEach-Object { \"$($_.DriverDesc)|$($_.'HardwareInformation.qwMemorySize')\" }")
    best = None
    for line in _run(["powershell", "-NoProfile", "-Command", ps]).splitlines():
        gpu_name, _, mem = line.strip().partition("|")
        try:
            vram = int(mem) / 2**30
        except ValueError:
            vram = 0.0
        if gpu_name and (best is None or vram > best.vram_gb):
            best = GPU("amd", gpu_name, vram)
    return best


@lru_cache(maxsize=1)
def gpu() -> GPU:
    """Most capable discrete GPU (NVIDIA first), without importing torch."""
    found = _nvidia()
    if found is None and sys.platform.startswith("linux"):
        found = _amd_linux()
    if found is None and os.name == "nt":
        found = _amd_windows()
    return found or GPU("", "", 0.0)


def gpu_vram_gb() -> float:
    return gpu().vram_gb


def torch_device() -> str:
    """torch device name: "cuda" also covers AMD GPUs with a ROCm (HIP) build of PyTorch."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_accel() -> str:
    """Human-readable torch acceleration: "ROCm", "CUDA", "MPS" or "CPU"."""
    import torch

    if torch.cuda.is_available():
        return "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    return "MPS" if torch.backends.mps.is_available() else "CPU"


def whisper_engine() -> str:
    """"mlx", "faster-whisper" (NVIDIA CUDA or CPU) or "transformers" (AMD GPU through ROCm PyTorch)."""
    forced = os.environ.get("VIDEO_ANALYZER_WHISPER", "").lower()
    if forced in ("mlx", "faster-whisper", "transformers"):
        return forced
    if name() == "mlx":
        return "mlx"
    if gpu().vendor == "amd" and importlib.util.find_spec("transformers"):
        try:
            if torch_accel() == "ROCm":
                return "transformers"
        except ImportError:
            pass
    return "faster-whisper"


def describe() -> str:
    if name() == "mlx":
        return "mlx (Apple Silicon, GPU Metal)"
    g = gpu()
    card = f"{g.name} {g.vram_gb:.0f} Go" if g.vendor else "pas de GPU NVIDIA/AMD détecté (CPU)"
    return f"portable (Ollama) · {card}"
