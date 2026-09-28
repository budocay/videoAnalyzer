import argparse
import shutil
import sys
from pathlib import Path

from . import config as C
from .extract import FFmpegError, probe
from .timeline import fmt


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="video-analyzer",
                                 description="Analyse vidéo 100 % locale (ffmpeg + MLX)")
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default=None,
                    help=f"VLM : alias {'/'.join(C.VLM_PRESETS)}, repo HF MLX ou tag Ollama "
                         "(défaut : 9b sur Mac, selon le GPU ailleurs)")
    ap.add_argument("--whisper-model", default=C.DEFAULT_WHISPER)
    ap.add_argument("--fps", type=float, default=C.DEFAULT_FPS, help="frames/s échantillonnées")
    ap.add_argument("--segment", type=float, default=C.DEFAULT_SEGMENT, help="durée d'un segment (s)")
    ap.add_argument("--size", type=int, default=C.DEFAULT_SIZE,
                    help="côté long des frames envoyées au VLM (px), portrait ou paysage")
    ap.add_argument("--lang", default=None, help="langue audio (fr, en…), auto si absent")
    ap.add_argument("--no-audio", action="store_true", help="ignorer la transcription")
    ap.add_argument("--summary", action="store_true", help="résumé global en fin")
    ap.add_argument("--out", type=Path, default=None, help="dossier de sortie (défaut : celui de la vidéo)")
    ap.add_argument("--cache-dir", type=Path, default=C.DEFAULT_CACHE_DIR)
    ap.add_argument("--dry-run", action="store_true", help="affiche le plan sans charger de modèle")
    return ap.parse_args(argv)


def parse_padel_args(argv) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="video-analyzer padel",
                                 description="Analyse complète d'un match de padel (100 % local)")
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default=None, help="VLM (tri des plans, score, coups, résumé) ; défaut selon la machine")
    ap.add_argument("--no-vlm-strokes", action="store_true",
                    help="types de coups par la pose seule (plus rapide, moins fiable)")
    ap.add_argument("--no-summary", action="store_true", help="pas de résumé rédigé")
    ap.add_argument("--out", type=Path, default=None, help="dossier de sortie (défaut : celui de la vidéo)")
    ap.add_argument("--cache-dir", type=Path, default=C.DEFAULT_CACHE_DIR)
    return ap.parse_args(argv)


def main_padel(argv) -> int:
    import os

    args = parse_padel_args(argv)
    if not args.video.is_file():
        sys.exit(f"Fichier introuvable : {args.video}")
    info = probe(args.video)
    print(f"▶ {info.path.name} — {fmt(info.duration)} · {info.width}x{info.height} @ {info.native_fps:.2f} fps")
    vlm = C.resolve_vlm(args.model)
    model_dir = args.cache_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    from .hub import go_offline_if_cached, required_repos
    from .padel.players import POSE_MODEL

    missing = go_offline_if_cached(required_repos(vlm, None))
    if (model_dir / POSE_MODEL).exists():
        os.environ.setdefault("YOLO_OFFLINE", "1")
    else:
        missing.append(POSE_MODEL)
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(model_dir / POSE_MODEL)
    if missing:
        print(f"• Téléchargement nécessaire (une seule fois) : {', '.join(missing)}")
    from .padel.pipeline import run as run_padel

    run_padel(info, args.out or args.video.parent, args.cache_dir, vlm, model_dir,
              refine_strokes=not args.no_vlm_strokes, summary=not args.no_summary)
    return 0


PADEL_TOOLS = ("annotate", "import-labels", "eval", "train")


def main_padel_tools(argv) -> int:
    ap = argparse.ArgumentParser(prog="video-analyzer padel",
                                 description="Boucle d'apprentissage : annoter → importer → évaluer → entraîner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate", help="génère la page d'annotation des frappes et l'ouvre")
    a.add_argument("video", type=Path)
    a.add_argument("-n", type=int, default=150, help="nombre de frappes à proposer (défaut 150)")
    a.add_argument("--out", type=Path, default=None, help="dossier de la page (défaut : data/annotation/<vidéo>)")
    a.add_argument("--no-open", action="store_true")
    i = sub.add_parser("import-labels", help="importe le JSON exporté par la page d'annotation")
    i.add_argument("file", type=Path)
    i.add_argument("--annotator", default="user")
    e = sub.add_parser("eval", help="précision de chaque méthode sur les frappes annotées")
    e.add_argument("video", type=Path)
    sub.add_parser("train", help="entraîne le classifieur de coups sur toutes les annotations")
    for p_ in (a, i, e):
        p_.add_argument("--cache-dir", type=Path, default=C.DEFAULT_CACHE_DIR)
    args = ap.parse_args(argv)
    cache = getattr(args, "cache_dir", C.DEFAULT_CACHE_DIR)

    from .extract import video_key
    from .padel import dataset as DS
    from .padel import learn as L

    if args.cmd == "annotate":
        info = probe(args.video)
        key = video_key(info.path)
        hits, tracks = DS.load_hits(key, cache)
        # hits labelled by a human are skipped; bootstrap labels (annotator "claude") are offered again
        already = {k for k, v in DS.load_labels(key).items() if v["annotator"] == "user"}
        pick = DS.select_for_annotation(hits, args.n, already)
        out = args.out or DS.DATA_DIR / "annotation" / key
        page = DS.export_page(info, pick, tracks, out)
        print(f"✔ {page} ({len(pick)} frappes, {len(already)} déjà annotées)")
        if not args.no_open:
            import webbrowser

            webbrowser.open(page.resolve().as_uri())
    elif args.cmd == "import-labels":
        path, counts = DS.import_labels(args.file, annotator=args.annotator)
        refs = L.build_references(sorted((DS.DATA_DIR / "annotation").glob("*")))
        print(f"✔ {path} · {sum(counts.values())} annotations : {dict(counts)} · {len(refs)} images de référence")
    elif args.cmd == "eval":
        info = probe(args.video)
        path, res = L.evaluate(video_key(info.path), cache)
        print(path.read_text(encoding="utf-8"))
    elif args.cmd == "train":
        path, rep = L.train_all(C.DEFAULT_CACHE_DIR)
        print((DS.DATA_DIR / "eval" / "entrainement.md").read_text(encoding="utf-8"))
        print(f"✔ modèle : {path}")
    return 0


def doctor() -> int:
    """Environment check used by the installers. Exit code 1 if something blocking is missing."""
    import os
    import platform
    import subprocess

    from . import backend
    from .hub import is_cached, required_repos
    from .padel.players import POSE_MODEL

    ok = True

    def line(good: bool, label: str, detail: str, blocking: bool = True):
        nonlocal ok
        ok &= good or not blocking
        print(f"{'OK ' if good else ('ERR' if blocking else '-- ')} {label:<14} {detail}")

    print(f"video-analyzer · Python {platform.python_version()} · {platform.system()} {platform.machine()}")
    line(True, "moteur", backend.describe())
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        ver = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout.split("\n")[0] if path else ""
        line(bool(path), tool, ver or "introuvable (voir README, installation)")
    g = backend.gpu()
    if backend.name() != "mlx":
        line(bool(g.vendor), "carte graphique",
             f"{g.name} · {g.vram_gb:.0f} Go" if g.vendor else "aucune carte NVIDIA/AMD : tout sur le processeur",
             blocking=False)
    try:
        accel = backend.torch_accel()
        gpu_unused = g.vendor in ("nvidia", "amd") and accel == "CPU"
        hint = {"nvidia": " — relancer l'installateur (PyTorch CUDA)",
                "amd": " — pilote Adrenalin ≥ 26.2.2 + Windows 11, ou groupes render/video sous Linux"}
        line(not gpu_unused, "pose (torch)", f"accélération {accel}" + (hint[g.vendor] if gpu_unused else ""),
             blocking=False)
    except ImportError:
        line(False, "pose (torch)", "PyTorch absent : pip install -e .")
    engine = backend.whisper_engine()
    where = {"mlx": "GPU Apple", "transformers": "GPU AMD (ROCm)",
             "faster-whisper": "GPU NVIDIA si disponible, sinon processeur"}[engine]
    line(True, "transcription", f"{engine} · {where}")
    vlm = C.resolve_vlm(None)
    if backend.name() == "portable":
        from .vision import OllamaVLM

        try:
            OllamaVLM(vlm, timeout=5).close()
            line(True, "VLM (Ollama)", f"{vlm} prêt")
        except RuntimeError as e:
            line(False, "VLM (Ollama)", str(e))
    for repo in required_repos(vlm, C.DEFAULT_WHISPER):
        line(is_cached(repo), "modèle HF", repo + ("" if is_cached(repo) else " (sera téléchargé au 1er lancement)"),
             blocking=False)
    yolo = C.DEFAULT_CACHE_DIR / "models" / POSE_MODEL
    line(yolo.exists(), "pose (YOLO)", str(yolo) if yolo.exists() else f"{POSE_MODEL} (téléchargé au 1er `padel`)",
         blocking=False)
    print("\nTout est prêt." if ok else "\nÀ corriger avant de lancer une analyse (lignes ERR).")
    return 0 if ok else 1


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "doctor":
        return doctor()
    if argv and argv[0] == "padel":
        if len(argv) > 1 and argv[1] in PADEL_TOOLS:
            return main_padel_tools(argv[1:])
        return main_padel(argv[1:])
    args = parse_args(argv)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe introuvable : brew install ffmpeg")
    if not args.video.is_file():
        sys.exit(f"Fichier introuvable : {args.video}")
    if args.fps <= 0 or args.segment <= 0 or args.size < 64:
        sys.exit("--fps et --segment doivent être > 0, --size >= 64")

    cfg = C.RunConfig(vlm=C.resolve_vlm(args.model), whisper=args.whisper_model, fps=args.fps,
                      segment=args.segment, size=args.size, lang=args.lang,
                      audio=not args.no_audio, summary=args.summary)
    try:
        info = probe(args.video)
    except FFmpegError as e:
        sys.exit(str(e))
    print(f"▶ {info.path.name} — {fmt(info.duration)} · {info.width}x{info.height} "
          f"@ {info.native_fps:.2f} fps · {info.codec} {info.pix_fmt}"
          f"{' HDR' if info.hdr else ''}{f' rot {info.rotation}°' if info.rotation else ''}"
          f" · audio={'oui' if info.has_audio else 'non'}")

    if args.dry_run:
        from .plan import make_plan
        p = make_plan(info, cfg.vlm, cfg.whisper, cfg.fps, cfg.segment, cfg.size, cfg.audio)
        print(f"Plan : {p.n_frames} frames {p.frame_size[0]}x{p.frame_size[1]}, "
              f"{p.n_segments} segments de {cfg.segment:g} s (~{p.frames_per_segment} frames/segment)\n"
              f"VLM  : {cfg.vlm} · poids {p.weights_gb:.2f} Go\n"
              f"       ~{p.image_tokens} tokens/image, ~{p.prompt_tokens_per_segment} tokens de prompt/segment\n"
              f"       pic mémoire estimé ~{p.est_peak_gb:.1f} Go (estimation grossière)\n"
              f"Audio: {cfg.whisper if p.whisper_gb else 'aucun'}"
              + (f" · poids {p.whisper_gb:.2f} Go (libéré avant le VLM)" if p.whisper_gb else ""))
        return 0

    from .hub import go_offline_if_cached, required_repos

    missing = go_offline_if_cached(required_repos(cfg.vlm, cfg.whisper if cfg.audio and info.has_audio else None))
    if missing:
        print(f"• Téléchargement nécessaire (une seule fois) : {', '.join(missing)}")

    from .pipeline import run
    try:
        run(info, cfg, args.out or args.video.parent, args.cache_dir)
    except FFmpegError as e:
        sys.exit(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
