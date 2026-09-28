"""Defaults and model presets. Repo names were verified on huggingface.co/mlx-community."""
from dataclasses import dataclass
from pathlib import Path

VLM_PRESETS = {
    "9b": "mlx-community/Qwen3.5-9B-MLX-8bit",
    "4b": "mlx-community/Qwen3.5-4B-MLX-8bit",
    "qwen3vl": "mlx-community/Qwen3-VL-8B-Instruct-8bit",
}
DEFAULT_VLM = VLM_PRESETS["9b"]
DEFAULT_WHISPER = "mlx-community/whisper-large-v3-turbo"

DEFAULT_FPS = 1.0
DEFAULT_SEGMENT = 5.0
DEFAULT_SIZE = 640  # longest side of the frames sent to the VLM
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "video-analyzer"

SEGMENT_MAX_TOKENS = 400
SUMMARY_MAX_TOKENS = 1200
# Above this many segments, the summary is built in two passes (chunk summaries, then global).
SUMMARY_CHUNK_SEGMENTS = 40

SEGMENT_PROMPT = (
    "Voici {n} images extraites d'une vidéo, dans l'ordre chronologique, "
    "de {t0} à {t1} (une image toutes les {step:.2f} s).\n"
    "{context}"
    "Décris factuellement ce qui se passe sur ce segment : lieu, personnes, "
    "actions et mouvements (y compris ceux de la caméra), changements notables d'une image à l'autre. "
    "3 à 5 phrases, en français, en texte brut. Décris uniquement ce qui est visible : "
    "pas de supposition sur l'intention, le contexte ou le type de lieu s'il n'est pas évident."
)

SUMMARY_PROMPT = (
    "Voici la timeline d'une vidéo : pour chaque segment, ce qui est vu et ce "
    "qui est entendu.\n\n{timeline}\n\n"
    "Fais un résumé en français, en Markdown sobre (titres ### et listes, sans emoji ni tableau), "
    "sans phrase d'introduction, avec trois parties : contexte général, déroulé chronologique "
    "en quelques points, moments notables avec leur timecode. Reste fidèle à la timeline : "
    "n'ajoute aucune interprétation qui n'y figure pas."
)

CHUNK_SUMMARY_PROMPT = (
    "Voici un extrait de la timeline d'une vidéo, de {t0} à {t1} : pour chaque segment, "
    "ce qui est vu et ce qui est entendu.\n\n{timeline}\n\n"
    "Résume cet extrait en français en 5 à 8 phrases factuelles, en gardant les timecodes "
    "des moments notables."
)


# Portable backend (Ollama tags, verified on ollama.com/library/qwen3-vl): 8b-instruct = 6.1 GB (Q4).
OLLAMA_PRESETS = {"9b": "qwen3-vl:8b-instruct", "qwen3vl": "qwen3-vl:8b-instruct", "4b": "qwen3-vl:4b-instruct"}
PORTABLE_WHISPER = "large-v3-turbo"  # faster-whisper alias → mobiuslabsgmbh/faster-whisper-large-v3-turbo


SETTINGS_FILE = Path(__file__).resolve().parents[2] / "settings.json"  # written by scripts/bootstrap.py


def default_vlm_choice() -> str | None:
    """VLM chosen at install time (VIDEO_ANALYZER_VLM, else settings.json), if any."""
    import json
    import os

    if os.environ.get("VIDEO_ANALYZER_VLM"):
        return os.environ["VIDEO_ANALYZER_VLM"]
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("vlm")
    except (OSError, ValueError):
        return None


def resolve_vlm(name: str | None = None) -> str:
    """Preset alias (4b, 9b, qwen3vl) or explicit id → model id for the active backend.
    Default: the install-time choice; else 9b on Apple Silicon; elsewhere 8b with ≥ 8 GB of NVIDIA
    VRAM, else 4b."""
    from . import backend

    name = name or default_vlm_choice()

    if backend.name() == "mlx":
        return VLM_PRESETS.get((name or "9b").lower(), name or DEFAULT_VLM)
    if name is None:
        name = "9b" if backend.gpu_vram_gb() >= 8 else "4b"
    if name.startswith("mlx-community/"):  # MLX repo given on a non-MLX machine: closest Ollama model
        name = "4b" if "4B" in name else "9b"
    return OLLAMA_PRESETS.get(name.lower(), name)


@dataclass(frozen=True)
class RunConfig:
    vlm: str = DEFAULT_VLM
    whisper: str = DEFAULT_WHISPER
    fps: float = DEFAULT_FPS
    segment: float = DEFAULT_SEGMENT
    size: int = DEFAULT_SIZE
    lang: str | None = None
    audio: bool = True
    summary: bool = False
