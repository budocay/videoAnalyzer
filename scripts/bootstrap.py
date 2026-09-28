#!/usr/bin/env python3
"""Cross-platform installer for video-analyzer (standard library only).

Called by install.sh (macOS / Linux) and install.cmd / scripts/install.ps1 (Windows) once Python,
ffmpeg and (outside Apple Silicon) Ollama are present. Steps:
  1. .venv in the project folder (reused if healthy)
  2. PyTorch build matching the machine (CUDA on NVIDIA, CPU otherwise), then `pip install -e .[dev]`
  3. models: MLX repos on Apple Silicon, or `ollama pull` + faster-whisper elsewhere; YOLO pose weights
  4. `video-analyzer doctor` + unit tests
  5. --demo: generates a short synthetic video and analyses it end to end

Usage: python scripts/bootstrap.py [--demo] [--model 4b|9b] [--skip-models] [--video FILE]
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
WIN = os.name == "nt"
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def say(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def run(cmd: list, **kw) -> None:
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def venv_bin(name: str) -> Path:
    return VENV / ("Scripts" if WIN else "bin") / (name + (".exe" if WIN else ""))


def nvidia_vram_gb() -> float:
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0.0


def ensure_venv() -> Path:
    py = venv_bin("python")
    if py.exists():
        ok = subprocess.run([str(py), "-c", "import sys; sys.exit(sys.version_info < (3, 11))"]).returncode == 0
        if ok:
            print(f"   environnement existant réutilisé : {VENV}")
            return py
        say("environnement .venv trop ancien : recréation")
        shutil.rmtree(VENV)
    say(f"création de l'environnement Python ({sys.executable}, {platform.python_version()})")
    run([sys.executable, "-m", "venv", VENV])
    return py


def install_packages(py: Path, vram: float) -> None:
    pip = [py, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "wheel", "setuptools"])
    if not APPLE_SILICON and sys.platform != "darwin":
        # Windows' default PyPI torch is CPU-only; Linux's bundles CUDA (~2.5 GB) even without a GPU.
        if vram:
            if WIN:
                say(f"GPU NVIDIA ({vram:.0f} Go) : PyTorch avec CUDA")
                run(pip + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"])
        else:
            say("pas de GPU NVIDIA : PyTorch CPU")
            run(pip + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    say("installation de video-analyzer et de ses dépendances")
    run(pip + ["install", "-e", f"{ROOT}[dev]"])
    if vram and sys.platform != "darwin":
        # A CPU-only torch already in the venv satisfies pip: check CUDA really works, else force the CUDA build.
        cuda = subprocess.run([str(py), "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"])
        if cuda.returncode != 0:
            say("PyTorch ne voit pas le GPU : réinstallation de la version CUDA")
            index = ["--index-url", "https://download.pytorch.org/whl/cu128"] if WIN else []
            run(pip + ["install", "--force-reinstall", "torch", "torchvision"] + index)


def ollama_up(url: str = "http://127.0.0.1:11434") -> bool:
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=3):
            return True
    except OSError:
        return False


def ensure_ollama() -> bool:
    if ollama_up():
        return True
    exe = shutil.which("ollama")
    if not exe:
        print("   Ollama introuvable : relance l'installateur de ton système (install.sh / install.cmd)")
        return False
    say("démarrage du service Ollama")
    flags = {"creationflags": 0x00000008 | 0x00000200} if WIN else {"start_new_session": True}  # detached
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **flags)
    for _ in range(30):
        time.sleep(1)
        if ollama_up():
            return True
    print("   Ollama ne répond pas sur 127.0.0.1:11434 : lance l'application Ollama puis relance ce script")
    return False


def download_models(py: Path, model: str | None, vram: float) -> None:
    cache = Path.home() / ".cache" / "video-analyzer" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    if APPLE_SILICON:
        repos = {"9b": "mlx-community/Qwen3.5-9B-MLX-8bit", "4b": "mlx-community/Qwen3.5-4B-MLX-8bit"}
        vlm = repos[model or "9b"]
        say(f"modèles MLX : {vlm} + whisper-large-v3-turbo + silero-vad (≈ 12 Go la première fois)")
        code = ("from huggingface_hub import snapshot_download as d\n"
                f"for r in {[vlm, 'mlx-community/whisper-large-v3-turbo', 'mlx-community/silero-vad']!r}:\n"
                "    print('   ', r); d(r)")
        run([py, "-c", code])
    else:
        tag = {"9b": "qwen3-vl:8b-instruct", "4b": "qwen3-vl:4b-instruct"}[model or ("9b" if vram >= 8 else "4b")]
        if ensure_ollama():
            say(f"modèle VLM Ollama : {tag} ({'6.1' if '8b' in tag else '3.3'} Go)")
            run([shutil.which("ollama"), "pull", tag])
        say("modèle de transcription faster-whisper large-v3-turbo (≈ 1.6 Go)")
        run([py, "-c", "from huggingface_hub import snapshot_download as d; "
                       "d('mobiuslabsgmbh/faster-whisper-large-v3-turbo')"])
    shipped = ROOT / "models"  # trained stroke classifier shipped by scripts/package.py
    for name in ("stroke_clf.pt", "stroke_clf.json"):
        if (shipped / name).exists() and not (cache / name).exists():
            shutil.copy(shipped / name, cache / name)
            print(f"   classifieur de coups installé : {name}")
    say("poids de pose YOLO11n (6 Mo)")
    run([py, "-c", "from pathlib import Path; from ultralytics.utils.downloads import attempt_download_asset; "
                   f"attempt_download_asset(Path(r'{cache / 'yolo11n-pose.pt'}'))"])


def make_demo_video(path: Path) -> None:
    """10 s: a red square moving on white, then a colour test chart. No speech (VAD path)."""
    run(["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=white:s=1280x720:d=5:r=25",
         "-f", "lavfi", "-i", "color=c=red:s=200x200:d=5:r=25",
         "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=5:r=25",
         "-f", "lavfi", "-i", "sine=f=440:d=10",
         "-filter_complex", "[0][1]overlay=x='t*200':y=260[a];[a][2]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", path])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="analyse une courte vidéo de démonstration à la fin")
    ap.add_argument("--model", choices=["4b", "9b"], default=None,
                    help="taille du VLM (défaut : 9b sur Mac Apple Silicon ou GPU ≥ 8 Go, sinon 4b)")
    ap.add_argument("--skip-models", action="store_true", help="ne télécharge pas les modèles maintenant")
    ap.add_argument("--video", type=Path, default=None, help="analyse cette vidéo à la fin de l'installation")
    args = ap.parse_args()

    if sys.version_info < (3, 11):
        sys.exit(f"Python ≥ 3.11 requis (trouvé {platform.python_version()})")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} introuvable dans le PATH : relance l'installateur de ton système")

    vram = nvidia_vram_gb()
    engine = "MLX (Apple Silicon)" if APPLE_SILICON else ("Ollama + CUDA" if vram else "Ollama + CPU")
    print(f"video-analyzer · {platform.system()} {platform.machine()} · Python {platform.python_version()} · "
          f"moteur {engine}" + (f" · GPU {vram:.0f} Go" if vram else ""))

    if args.model:  # remembered for every later run (read by video_analyzer.config)
        import json
        (ROOT / "settings.json").write_text(json.dumps({"vlm": args.model}, indent=1) + "\n", encoding="utf-8")
    py = ensure_venv()
    install_packages(py, vram)
    if not args.skip_models:
        download_models(py, args.model, vram)

    cli = venv_bin("video-analyzer")
    env = {**os.environ, "PYTHONUTF8": "1"}
    say("diagnostic")
    doctor_ok = subprocess.run([cli, "doctor"], env=env).returncode == 0
    say("tests unitaires")
    subprocess.run([py, "-m", "pytest", "-q", str(ROOT / "tests")], env=env, cwd=ROOT)

    target = args.video
    if args.demo and not target:
        target = ROOT / "samples" / "demo.mp4"
        target.parent.mkdir(exist_ok=True)
        say("vidéo de démonstration")
        make_demo_video(target)
    if target and doctor_ok:
        say(f"analyse de {target.name}")
        model = ["--model", args.model] if args.model else []
        run([cli, target, "--summary", "--out", ROOT / "samples" / "out"] + model, env=env)

    launcher = "va.cmd" if WIN else "./va.sh"
    print(f"""
Installation terminée{'' if doctor_ok else ' avec des points à corriger (voir le diagnostic ci-dessus)'}.

Utilisation (depuis {ROOT}) :
  {launcher} ma_video.mp4 --lang fr --summary       description vu / entendu
  {launcher} padel match.mp4                         analyse complète d'un match de padel
  {launcher} doctor                                  vérifier l'installation
""")
    return 0 if doctor_ok else 1


if __name__ == "__main__":
    sys.exit(main())
