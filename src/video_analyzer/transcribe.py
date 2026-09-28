"""Audio stage: Silero VAD + Whisper, with an on-disk cache.

Backends: mlx (Silero from mlx-audio + mlx-whisper) or portable (faster-whisper, whose package
bundles Silero VAD as ONNX; CUDA if available, else CPU int8).

Whisper large-v3-turbo hallucinates on audio without speech ("Thank you.", ".", subtitle
credits) and reports no_speech_prob≈0 there, so its own confidence is useless. We gate it
with a VAD: no speech → Whisper is skipped; otherwise only utterances that overlap detected
speech are kept, and timestamps are clamped to the audio duration.
"""
import json
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from .memory import free_mlx

VAD_REPO = "mlx-community/silero-vad"
MIN_OVERLAP_S = 0.2  # an utterance needs at least this much detected speech inside it
CACHE_VERSION = 2


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AudioResult:
    utterances: list[Utterance]
    speech: list[tuple[float, float]]  # VAD regions, seconds
    language: str | None
    from_cache: bool = False


def cache_path(cache_dir: Path, repo: str, lang: str | None) -> Path:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", repo)
    return cache_dir / f"transcript_v{CACHE_VERSION}_{slug}_{lang or 'auto'}.json"


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def filter_utterances(raw: list[dict], speech: list[tuple[float, float]], duration: float) -> list[Utterance]:
    """Keep utterances with text, clamped to [0, duration], overlapping VAD speech enough."""
    out = []
    for s in raw:
        text = s["text"].strip()
        if not re.search(r"\w", text):
            continue
        start, end = max(0.0, s["start"]), min(duration, s["end"])
        if end <= start:
            continue
        overlap = sum(max(0.0, min(end, e) - max(start, b)) for b, e in speech)
        if overlap < min(MIN_OVERLAP_S, (end - start) / 2):
            continue
        out.append(Utterance(round(start, 2), round(end, 2), text))
    return out


def whisper_repo(repo: str) -> str:
    """HF repo actually downloaded for `repo` on the active backend (for offline checks)."""
    from . import backend

    if backend.name() == "mlx":
        return repo
    from .config import PORTABLE_WHISPER

    return {"large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}.get(
        _portable_name(repo), PORTABLE_WHISPER)


def _portable_name(repo: str) -> str:
    from .config import PORTABLE_WHISPER

    # "mlx-community/whisper-large-v3-turbo" → "large-v3-turbo"; faster-whisper names pass through
    return repo.split("/whisper-")[-1] if "/whisper-" in repo else (repo or PORTABLE_WHISPER)


def _vad(wav: Path) -> list[tuple[float, float]]:
    from . import backend

    if backend.name() == "mlx":
        from mlx_audio.vad import load

        model = load(VAD_REPO)
        regions = model.get_speech_timestamps(str(wav), return_seconds=True)
        del model
        return [(round(r["start"], 2), round(r["end"], 2)) for r in regions]
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(str(wav), sampling_rate=16000)
    return [(round(r["start"] / 16000, 2), round(r["end"] / 16000, 2))
            for r in get_speech_timestamps(audio, VadOptions())]


def _whisper(wav: Path, repo: str, lang: str | None) -> tuple[list[dict], str | None]:
    """Raw segments [{start, end, text}] and detected language."""
    from . import backend

    if backend.name() == "mlx":
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        # transcribe() returns {"text", "segments": [{"start", "end", "text", ...}], "language"}
        r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, language=lang)
        segs, language = r.get("segments", []), r.get("language", lang)
        # mlx_whisper keeps the model in a class attribute; drop it or ~1.6 GB stays resident.
        ModelHolder.model = None
        ModelHolder.model_path = None
        return segs, language
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(_portable_name(repo), device="auto", compute_type="auto")
        segments, info = model.transcribe(str(wav), language=lang)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    except Exception:  # noqa: BLE001 — CUDA present but cuBLAS/cuDNN missing or broken: fall back to CPU
        model = WhisperModel(_portable_name(repo), device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(wav), language=lang)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    del model
    return segs, info.language


def transcribe(audio: Path, repo: str, lang: str | None, cache_file: Path) -> AudioResult:
    """Frees Whisper's MLX memory before returning."""
    if cache_file.exists():
        d = json.loads(cache_file.read_text(encoding="utf-8"))
        return AudioResult([Utterance(**u) for u in d["utterances"]],
                           [tuple(r) for r in d["speech"]], d["language"], from_cache=True)

    speech = _vad(audio)
    utterances, language = [], lang
    if speech:
        raw, language = _whisper(audio, repo, lang)
        utterances = filter_utterances(raw, speech, wav_duration(audio))
    free_mlx()

    result = AudioResult(utterances, speech, language)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"utterances": [asdict(u) for u in utterances],
                                      "speech": speech, "language": language},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return result
