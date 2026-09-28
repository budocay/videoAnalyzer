"""Compute backend selection: MLX on Apple Silicon, portable (Ollama + faster-whisper + PyTorch) elsewhere.

- "mlx"      : macOS arm64 with mlx installed — mlx-vlm, mlx-whisper, Silero VAD from mlx-audio.
- "portable" : Windows / Linux / Intel Mac — VLM served by a local Ollama (qwen3-vl), transcription
               and VAD by faster-whisper (CTranslate2, CUDA if available else CPU int8).
Pose (ultralytics) runs on CUDA, MPS or CPU in both cases.
Override with VIDEO_ANALYZER_BACKEND=mlx|portable.
"""
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def name() -> str:
    forced = os.environ.get("VIDEO_ANALYZER_BACKEND", "").lower()
    if forced in ("mlx", "portable"):
        return forced
    return "mlx" if is_apple_silicon() and importlib.util.find_spec("mlx") else "portable"


@lru_cache(maxsize=1)
def nvidia_vram_gb() -> float:
    """Total VRAM of the first NVIDIA GPU (0 if none), without importing torch."""
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0.0


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe() -> str:
    b = name()
    if b == "mlx":
        return "mlx (Apple Silicon, GPU Metal)"
    vram = nvidia_vram_gb()
    gpu = f"GPU NVIDIA {vram:.0f} Go" if vram else "pas de GPU NVIDIA détecté (CPU)"
    return f"portable (Ollama + faster-whisper) · {gpu}"
