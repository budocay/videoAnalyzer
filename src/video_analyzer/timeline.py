"""Pure logic: segmentation, seen/heard fusion, JSON schema, Markdown export.

No model and no ffmpeg here, so everything is unit-testable.
"""
from dataclasses import asdict, dataclass, field
from typing import Any

from .extract import Frame
from .transcribe import Utterance

SCHEMA_VERSION = "1.0"


@dataclass
class Segment:
    id: int
    start: float
    end: float
    frames: list[Frame] = field(default_factory=list)
    heard: str = ""
    context: str = ""
    seen: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def plan_segments(duration: float, segment: float) -> list[Segment]:
    """Fixed windows [k*segment, min((k+1)*segment, duration)). Last one may be shorter."""
    if duration <= 0 or segment <= 0:
        return []
    segs, k = [], 0
    while k * segment < duration - 1e-6:
        start = round(k * segment, 3)
        segs.append(Segment(id=k, start=start, end=round(min(start + segment, duration), 3)))
        k += 1
    return segs


def assign_frames(segments: list[Segment], frames: list[Frame]) -> None:
    """Put each frame in the window containing its timestamp; overflow goes to the last one."""
    if not segments:
        return
    size = segments[0].end - segments[0].start
    for f in frames:
        idx = min(int(f.t // size), len(segments) - 1) if size > 0 else 0
        segments[idx].frames.append(f)


def assign_transcript(segments: list[Segment], utterances: list[Utterance]) -> None:
    """heard: each utterance goes to exactly one segment (by midpoint), for a non-redundant timeline.
    context: every utterance overlapping the window, fed to the VLM prompt."""
    for seg in segments:
        seg.context = " ".join(u.text for u in utterances if u.start < seg.end and u.end > seg.start)
    buckets: dict[int, list[str]] = {}
    for u in utterances:
        mid = (u.start + u.end) / 2
        seg = next((s for s in segments if s.start <= mid < s.end), segments[-1] if segments else None)
        if seg is not None:
            buckets.setdefault(seg.id, []).append(u.text)
    for seg in segments:
        seg.heard = " ".join(buckets.get(seg.id, []))


def build_document(*, video: dict, config: dict, models: dict, transcript: list[Utterance],
                   segments: list[Segment], summary: str | None, timings: dict,
                   speech: list[tuple[float, float]] | None = None, language: str | None = None) -> dict:
    """Stable, versioned output. Phase 2 modules add their data under `extensions`."""
    return {
        "schema_version": SCHEMA_VERSION,
        "video": video,
        "config": config,
        "models": models,
        "audio": {
            "language": language,
            "speech": [{"start": b, "end": e} for b, e in speech or []],
        },
        "transcript": [asdict(u) for u in transcript],
        "segments": [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "frames": [{"index": f.index, "t": f.t} for f in s.frames],
                "seen": s.seen,
                "heard": s.heard,
                "stats": s.stats,
            }
            for s in segments
        ],
        "summary": summary,
        "timings": timings,
        "extensions": {},
    }


def to_markdown(doc: dict) -> str:
    v, c = doc["video"], doc["config"]
    md = [f"# Analyse — {v['name']}", "",
          f"Durée : {fmt(v['duration'])} · Modèle : `{doc['models']['vlm']}` · "
          f"{c['fps']:g} fps · segments de {c['segment']:g} s", ""]
    if doc.get("summary"):
        md += ["## Résumé", "", doc["summary"], ""]
    md += ["## Timeline", ""]
    for s in doc["segments"]:
        md += [f"### {fmt(s['start'])} – {fmt(s['end'])}", ""]
        if s["seen"]:
            md += [f"**Vu :** {s['seen']}", ""]
        if s["heard"]:
            md += [f"**Entendu :** {s['heard']}", ""]
    return "\n".join(md)


def timeline_text(segments: list[Segment]) -> str:
    """Compact text given to the summary prompt."""
    return "\n".join(
        f"[{fmt(s.start)}–{fmt(s.end)}] VU : {s.seen or '—'}"
        + (f" | ENTENDU : {s.heard}" if s.heard else "")
        for s in segments)
