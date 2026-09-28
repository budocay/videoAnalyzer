"""Keep runs offline: once every model is in the HF cache, forbid Hub network calls.

huggingface_hub reads HF_HUB_OFFLINE at import time, so this must run before anything
imports it (mlx_vlm, mlx_whisper, mlx_audio). Hence no huggingface_hub import here.
"""
import os
from pathlib import Path


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
    return Path(home) / "hub"


def is_cached(repo: str) -> bool:
    if Path(repo).is_dir():  # local model folder
        return True
    snaps = hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots"
    weights = ("*.safetensors", "model.bin", "*.npz")  # MLX / HF, CTranslate2 (faster-whisper), MLX npz
    return snaps.exists() and any(any(s.glob(w)) for s in snaps.glob("*") for w in weights)


def go_offline_if_cached(repos: list[str]) -> list[str]:
    """Set HF_HUB_OFFLINE=1 when all repos are cached. Returns the missing ones."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    missing = [r for r in repos if not is_cached(r)]
    if not missing:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return missing


def required_repos(vlm: str, whisper: str | None) -> list[str]:
    """Hugging Face repos the run needs on the active backend. Ollama models are handled by Ollama."""
    from . import backend
    from .transcribe import VAD_REPO, whisper_repo

    repos = []
    if backend.name() == "mlx":
        repos.append(vlm)
        if whisper:
            repos += [whisper, VAD_REPO]
    elif whisper:
        repos.append(whisper_repo(whisper))  # faster-whisper bundles its VAD
    return repos
