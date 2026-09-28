#!/usr/bin/env python3
"""
video_analyze.py — Analyse vidéo 100 % locale sur Apple Silicon (MLX)

Pipeline :
  ffmpeg (frames horodatées + audio 16 kHz)
    → mlx-whisper (transcript horodaté)
    → mlx-vlm / Qwen (description visuelle par segment de N secondes)
    → timeline Markdown + JSON (+ résumé global optionnel)

Install :
  brew install ffmpeg
  python3 -m venv .venv && source .venv/bin/activate
  pip install -U mlx-vlm mlx-whisper

Usage :
  python video_analyze.py ma_video.mp4
  python video_analyze.py padel.mp4 --fps 2 --segment 4 --summary
  python video_analyze.py clip.mov --model mlx-community/Qwen3.5-4B-MLX-8bit --lang fr
"""
import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Sur 32 Go : 9B en 8-bit = sweet spot. Si le repo n'existe pas sous ce nom,
# vérifie sur https://huggingface.co/mlx-community (ou passe le 4B en fallback).
DEFAULT_VLM = "mlx-community/Qwen3.5-9B-MLX-8bit"
DEFAULT_WHISPER = "mlx-community/whisper-large-v3-turbo"

SEGMENT_PROMPT = (
    "Voici {n} images extraites d'une vidéo, dans l'ordre chronologique, "
    "de {t0} à {t1} (une image toutes les {step:.2f} s).\n"
    "{context}"
    "Décris factuellement ce qui se passe sur ce segment : lieu, personnes, "
    "actions et mouvements, changements notables d'une image à l'autre. "
    "3 à 5 phrases, en français, sans spéculer au-delà de ce qui est visible."
)

SUMMARY_PROMPT = (
    "Voici la timeline d'une vidéo : pour chaque segment, ce qui est vu et ce "
    "qui est entendu.\n\n{timeline}\n\n"
    "Fais un résumé structuré en français : contexte général, déroulé "
    "chronologique en quelques points, moments notables avec leur timecode."
)


# ---------------------------------------------------------------- utils

def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.exit(f"Commande échouée : {' '.join(cmd)}\n{r.stderr[-1500:]}")


def probe_duration(video: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def has_audio(video: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True)
    return bool(r.stdout.strip())


def free_mlx():
    gc.collect()
    try:
        import mlx.core as mx
        clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
        clear()
    except Exception:
        pass


def clean(text: str) -> str:
    # Qwen3.5 est hybride (thinking) : on vire un éventuel bloc de raisonnement.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


# ---------------------------------------------------------------- étapes

def extract_frames(video: Path, outdir: Path, fps: float, width: int):
    run(["ffmpeg", "-y", "-i", str(video),
         "-vf", f"fps={fps},scale={width}:-2", "-q:v", "3",
         str(outdir / "f_%06d.jpg")])
    frames = sorted(outdir.glob("f_*.jpg"))
    # Horodatage calculé par nous, pas par le modèle (plus fiable en MLX).
    return [(i / fps, p) for i, p in enumerate(frames)]


def extract_audio(video: Path, out: Path):
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out)])


def transcribe(audio: Path, repo: str, lang):
    import mlx_whisper
    r = mlx_whisper.transcribe(str(audio), path_or_hf_repo=repo, language=lang)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in r.get("segments", []) if s["text"].strip()]


class VLM:
    def __init__(self, repo: str):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        from mlx_vlm.prompt_utils import apply_chat_template
        self.model, self.processor = load(repo)
        self.config = load_config(repo)
        self._tpl = apply_chat_template

    def ask(self, prompt: str, images=None, max_tokens: int = 400) -> str:
        from mlx_vlm import generate
        images = [str(p) for p in (images or [])]
        formatted = self._tpl(self.processor, self.config, prompt, num_images=len(images))
        out = generate(self.model, self.processor, formatted,
                       image=images or None, max_tokens=max_tokens, verbose=False)
        return clean(getattr(out, "text", out))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Analyse vidéo locale (MLX)")
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default=DEFAULT_VLM, help="repo HF du VLM (format MLX)")
    ap.add_argument("--whisper-model", default=DEFAULT_WHISPER)
    ap.add_argument("--fps", type=float, default=1.0, help="frames/s échantillonnées")
    ap.add_argument("--segment", type=float, default=5.0, help="durée d'un segment (s)")
    ap.add_argument("--width", type=int, default=640, help="largeur des frames (px)")
    ap.add_argument("--lang", default=None, help="langue audio (fr, en…), auto si absent")
    ap.add_argument("--no-audio", action="store_true", help="ignorer la transcription")
    ap.add_argument("--summary", action="store_true", help="résumé global en fin")
    ap.add_argument("--out", type=Path, default=None, help="dossier de sortie")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg introuvable : brew install ffmpeg")
    if not args.video.exists():
        sys.exit(f"Fichier introuvable : {args.video}")

    out_dir = args.out or args.video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    duration = probe_duration(args.video)
    print(f"▶ {args.video.name} — {fmt(duration)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Transcript (avant de charger le VLM, pour ne pas cumuler en RAM)
        transcript = []
        if not args.no_audio and has_audio(args.video):
            t = time.time()
            print("• Transcription…", end="", flush=True)
            wav = tmp / "audio.wav"
            extract_audio(args.video, wav)
            transcript = transcribe(wav, args.whisper_model, args.lang)
            print(f" {len(transcript)} segments ({time.time() - t:.0f} s)")
            free_mlx()

        # 2. Frames
        frames = extract_frames(args.video, tmp, args.fps, args.width)
        print(f"• {len(frames)} frames extraites")

        groups: dict[int, list] = {}
        for ts, p in frames:
            groups.setdefault(int(ts // args.segment), []).append((ts, p))

        # 3. VLM segment par segment
        t = time.time()
        print(f"• Chargement {args.model}…")
        vlm = VLM(args.model)
        print(f"  chargé en {time.time() - t:.0f} s")

        timeline = []
        for k, idx in enumerate(sorted(groups)):
            t0 = idx * args.segment
            t1 = min(t0 + args.segment, duration)
            heard = " ".join(s["text"] for s in transcript
                             if s["start"] < t1 and s["end"] > t0)
            context = f"Paroles entendues sur ce segment : « {heard} »\n" if heard else ""
            prompt = SEGMENT_PROMPT.format(n=len(groups[idx]), t0=fmt(t0), t1=fmt(t1),
                                           step=1 / args.fps, context=context)
            ts = time.time()
            seen = vlm.ask(prompt, [p for _, p in groups[idx]])
            print(f"  [{k + 1}/{len(groups)}] {fmt(t0)}–{fmt(t1)} ({time.time() - ts:.0f} s)")
            timeline.append({"start": t0, "end": t1, "seen": seen, "heard": heard})

        # 4. Résumé global optionnel
        summary = None
        if args.summary:
            print("• Résumé global…")
            tl_txt = "\n".join(
                f"[{fmt(s['start'])}–{fmt(s['end'])}] VU : {s['seen']}"
                + (f" | ENTENDU : {s['heard']}" if s["heard"] else "")
                for s in timeline)
            summary = vlm.ask(SUMMARY_PROMPT.format(timeline=tl_txt), max_tokens=1200)

    # 5. Sorties
    data = {"video": str(args.video), "duration": duration, "model": args.model,
            "fps": args.fps, "segment": args.segment,
            "transcript": transcript, "timeline": timeline, "summary": summary}
    json_path = out_dir / f"{stem}_analyse.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# Analyse — {args.video.name}", "",
          f"Durée : {fmt(duration)} · Modèle : `{args.model}` · "
          f"{args.fps:g} fps · segments de {args.segment:g} s", ""]
    if summary:
        md += ["## Résumé", "", summary, ""]
    md += ["## Timeline", ""]
    for s in timeline:
        md += [f"### {fmt(s['start'])} – {fmt(s['end'])}", "", f"**Vu :** {s['seen']}", ""]
        if s["heard"]:
            md += [f"**Entendu :** {s['heard']}", ""]
    md_path = out_dir / f"{stem}_analyse.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"✔ {md_path}\n✔ {json_path}")


if __name__ == "__main__":
    main()
