"""Orchestration: extract → transcribe → describe → export.

Each stage reads/writes plain data (Frame, Utterance, Segment) and caches under
<cache_root>/<video_key>/, so a future `pose` stage can plug in on the same frames
and write into the document's `extensions`.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

from . import config as C
from .extract import VideoInfo, extract_audio, extract_frames, video_key
from .memory import peak_gb, reset_peak
from .timeline import (assign_frames, assign_transcript, build_document, fmt, plan_segments,
                       timeline_text, to_markdown)
from .transcribe import VAD_REPO, cache_path, transcribe


def log(msg: str) -> None:
    print(msg, flush=True)


def _gb(x: float) -> str:
    """Peak memory for logs; empty when unknown (Ollama backend does not report it)."""
    return f"pic {x:.2f} Go · " if x == x and x > 0 else ""


def summarize(vlm, segments) -> str:
    """One pass for short videos; for long ones, summarize chunks first so the prompt stays small."""
    n = C.SUMMARY_CHUNK_SEGMENTS
    if len(segments) <= n:
        return vlm.ask(C.SUMMARY_PROMPT.format(timeline=timeline_text(segments)),
                       max_tokens=C.SUMMARY_MAX_TOKENS).text
    parts = []
    chunks = [segments[i:i + n] for i in range(0, len(segments), n)]
    for k, chunk in enumerate(chunks, 1):
        t0, t1 = fmt(chunk[0].start), fmt(chunk[-1].end)
        a = vlm.ask(C.CHUNK_SUMMARY_PROMPT.format(t0=t0, t1=t1, timeline=timeline_text(chunk)),
                    max_tokens=500)
        parts.append(f"[{t0}–{t1}] {a.text}")
        log(f"  résumé partiel {k}/{len(chunks)} ({a.seconds:.1f} s)")
    return vlm.ask(C.SUMMARY_PROMPT.format(timeline="\n\n".join(parts)),
                   max_tokens=C.SUMMARY_MAX_TOKENS).text


def run(info: VideoInfo, cfg: C.RunConfig, out_dir: Path, cache_root: Path) -> tuple[Path, Path]:
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    cache = cache_root / video_key(info.path)

    # 1. Transcript first, then free Whisper before the VLM loads.
    transcript, speech, language = [], [], None
    if cfg.audio and info.has_audio:
        t = time.perf_counter()
        reset_peak()
        wav = extract_audio(info.path, cache / "audio_16k.wav")
        audio = transcribe(wav, cfg.whisper, cfg.lang, cache_path(cache, cfg.whisper, cfg.lang))
        transcript, speech, language = audio.utterances, audio.speech, audio.language
        timings["transcribe_s"] = round(time.perf_counter() - t, 2)
        src = "cache" if audio.from_cache else (_gb(peak_gb()).rstrip(" ·") or "calculé")
        talk = sum(e - b for b, e in speech)
        log(f"• Transcript : {len(transcript)} énoncés, parole {talk:.1f} s, langue {language or '—'} "
            f"({timings['transcribe_s']:.1f} s, {src})")
    else:
        log("• Transcript : ignoré (pas d'audio ou --no-audio)")

    # 2. Frames + segmentation.
    t = time.perf_counter()
    frames, method = extract_frames(info, cache / f"frames_fps{cfg.fps:g}_s{cfg.size}", cfg.fps, cfg.size)
    segments = plan_segments(info.duration, cfg.segment)
    assign_frames(segments, frames)
    assign_transcript(segments, transcript)
    segments = [s for s in segments if s.frames]
    timings["extract_s"] = round(time.perf_counter() - t, 2)
    log(f"• {len(frames)} frames, {len(segments)} segments ({timings['extract_s']:.1f} s, {method})")
    if info.hdr and method == "software":
        log("  ⚠ vidéo HDR décodée sans tone-mapping (VideoToolbox indisponible) : couleurs délavées possibles")

    # 3. VLM, one call per segment.
    from .vision import load_vlm

    reset_peak()
    log(f"• Chargement {cfg.vlm}…")
    vlm = load_vlm(cfg.vlm)
    timings["vlm_load_s"] = round(vlm.load_seconds, 2)
    log(f"  chargé en {vlm.load_seconds:.1f} s")

    t_vlm = time.perf_counter()
    for k, seg in enumerate(segments, 1):
        ctx = f"Paroles entendues sur ce segment : « {seg.context} »\n" if seg.context else ""
        prompt = C.SEGMENT_PROMPT.format(n=len(seg.frames), t0=fmt(seg.start), t1=fmt(seg.end),
                                         step=1 / cfg.fps, context=ctx)
        a = vlm.ask(prompt, [f.path for f in seg.frames], max_tokens=C.SEGMENT_MAX_TOKENS)
        seg.seen = a.text
        seg.stats = {"seconds": round(a.seconds, 2), "prompt_tokens": a.prompt_tokens,
                     "generation_tokens": a.generation_tokens, "peak_gb": round(a.peak_gb, 2),
                     "finish_reason": a.finish_reason}
        elapsed = time.perf_counter() - t_vlm
        eta = elapsed / k * (len(segments) - k)
        log(f"  [{k}/{len(segments)}] {fmt(seg.start)}–{fmt(seg.end)} "
            f"{a.seconds:.1f} s · {a.prompt_tokens} tok in / {a.generation_tokens} out · "
            f"{_gb(a.peak_gb)}écoulé {elapsed:.0f} s · reste ~{eta:.0f} s")
    timings["vlm_segments_s"] = round(time.perf_counter() - t_vlm, 2)

    # 4. Optional summary, text-only.
    summary = None
    if cfg.summary:
        t = time.perf_counter()
        summary = summarize(vlm, segments)
        timings["summary_s"] = round(time.perf_counter() - t, 2)
        log(f"• Résumé : {timings['summary_s']:.1f} s")
    vlm_peak = peak_gb()
    vlm.close()

    # 5. Export.
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    doc = build_document(
        video={"path": str(info.path.resolve()), "name": info.path.name, "key": cache.name,
               "duration": info.duration, "width": info.width, "height": info.height,
               "native_fps": info.native_fps, "has_audio": info.has_audio,
               "rotation": info.rotation, "codec": info.codec, "pix_fmt": info.pix_fmt,
               "hdr": info.hdr},
        config={**{k: v for k, v in asdict(cfg).items() if k not in ("vlm", "whisper")},
                "frame_extraction": method},
        models={"vlm": cfg.vlm, "whisper": cfg.whisper if transcript else None,
                "vad": VAD_REPO if cfg.audio and info.has_audio else None},
        transcript=transcript, speech=speech, language=language, segments=segments, summary=summary,
        timings={**timings, "vlm_peak_gb": round(vlm_peak, 2)},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    json_path = out_dir / f"{stem}_analyse.json"
    md_path = out_dir / f"{stem}_analyse.md"
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(doc), encoding="utf-8")
    log(f"✔ {md_path}\n✔ {json_path}\n  total {timings['total_s']:.0f} s"
        + (f" · pic VLM {vlm_peak:.2f} Go" if vlm_peak == vlm_peak and vlm_peak > 0 else ""))
    return md_path, json_path
