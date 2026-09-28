#!/usr/bin/env python3
"""Cross-platform installer for video-analyzer (standard library only).

Called by install.sh (macOS / Linux) and install.cmd / scripts/install.ps1 (Windows) once Python,
ffmpeg and (outside Apple Silicon) Ollama are present. Steps:
  1. .venv in the project folder (reused if healthy)
  2. PyTorch build matching the machine (CUDA on NVIDIA, ROCm on AMD, CPU otherwise), then `pip install -e .[dev]`
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


# AMD's PyTorch for Windows (ROCm 7.2.1, Windows 11, Python 3.12, Adrenalin driver >= 26.2.2).
# URLs from rocm.docs.amd.com (install PyTorch on Radeon, Windows), checked to exist on repo.radeon.com.
AMD_WIN = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
AMD_WIN_ROCM = [AMD_WIN + f for f in ("rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
                                      "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                                      "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
                                      "rocm-7.2.1.tar.gz")]
AMD_WIN_TORCH = [AMD_WIN + f for f in ("torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                                       "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl")]
AMD_LINUX_INDEX = "https://download.pytorch.org/whl/rocm7.0"


def _out(cmd: list) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_gpu() -> tuple[str, str, float]:
    """(vendor, name, VRAM in GB) of the most capable discrete GPU; ("", "", 0) if none.
    Same logic as video_analyzer.backend.gpu(), duplicated because the package is not installed yet."""
    if shutil.which("nvidia-smi"):
        try:
            name, mem = _out(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"]).splitlines()[0].rsplit(",", 1)
            return "nvidia", name.strip(), float(mem) / 1024
        except (ValueError, IndexError):
            pass
    best = ("", "", 0.0)
    if sys.platform.startswith("linux"):
        for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
            try:
                if (dev / "vendor").read_text().strip() == "0x1002":
                    vram = int((dev / "mem_info_vram_total").read_text()) / 2**30
                    if vram > best[2]:
                        best = ("amd", "AMD Radeon", vram)
            except (OSError, ValueError):
                continue
    elif WIN:
        ps = ("Get-ItemProperty -Path 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\"
              "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue | "
              "Where-Object { $_.DriverDesc -match 'Radeon' } | "
              "ForEach-Object { \"$($_.DriverDesc)|$($_.'HardwareInformation.qwMemorySize')\" }")
        for line in _out(["powershell", "-NoProfile", "-Command", ps]).splitlines():
            name, _, mem = line.strip().partition("|")
            try:
                vram = int(mem) / 2**30
            except ValueError:
                vram = 0.0
            if name and (vram > best[2] or not best[0]):
                best = ("amd", name, vram)
    return best


def ensure_venv(need_312: bool = False) -> Path:
    """Reuse .venv if its Python is recent enough (exactly 3.12 when AMD's Windows wheels require it)."""
    py = venv_bin("python")
    check = "sys.version_info[:2] == (3, 12)" if need_312 else "sys.version_info >= (3, 11)"
    if py.exists():
        if subprocess.run([str(py), "-c", f"import sys; sys.exit(0 if {check} else 1)"]).returncode == 0:
            print(f"   environnement existant réutilisé : {VENV}")
            return py
        say("environnement .venv incompatible (version de Python) : recréation")
        shutil.rmtree(VENV)
    say(f"création de l'environnement Python ({sys.executable}, {platform.python_version()})")
    run([sys.executable, "-m", "venv", VENV])
    return py


def install_packages(py: Path, gpu: tuple[str, str, float]) -> None:
    vendor, gpu_name, vram = gpu
    pip = [py, "-m", "pip"]
    run(pip + ["install", "--upgrade", "pip", "wheel", "setuptools"])
    # torch goes in first so that ultralytics (installed with the project) keeps this build.
    if not APPLE_SILICON and sys.platform != "darwin":
        if vendor == "nvidia":
            if WIN:  # Windows' default PyPI torch is CPU-only; Linux's already bundles CUDA
                say(f"{gpu_name} ({vram:.0f} Go) : PyTorch CUDA")
                run(pip + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"])
        elif vendor == "amd" and WIN:
            if sys.version_info[:2] == (3, 12):
                say(f"{gpu_name} ({vram:.0f} Go) : PyTorch ROCm 7.2.1 pour Windows (paquets AMD, ~2 Go)")
                run(pip + ["install", "--no-cache-dir"] + AMD_WIN_ROCM)
                run(pip + ["install", "--no-cache-dir"] + AMD_WIN_TORCH)
            else:
                say("carte AMD détectée mais Python n'est pas en 3.12 (exigé par AMD) : PyTorch CPU")
                run(pip + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        elif vendor == "amd":
            say(f"carte AMD ({vram:.0f} Go) : PyTorch ROCm 7.0 (Linux)")
            run(pip + ["install", "torch", "torchvision", "--index-url", AMD_LINUX_INDEX])
        else:
            say("pas de carte NVIDIA ni AMD détectée : PyTorch CPU")
            run(pip + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    say("installation de video-analyzer et de ses dépendances")
    run(pip + ["install", "-e", f"{ROOT}[dev]"])
    if vendor in ("nvidia", "amd") and sys.platform != "darwin":
        # A torch already in the venv satisfies pip even if it is the wrong build: check the GPU is usable.
        ok = subprocess.run([str(py), "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"])
        if ok.returncode != 0 and vendor == "nvidia":
            say("PyTorch ne voit pas le GPU : réinstallation de la version CUDA")
            index = ["--index-url", "https://download.pytorch.org/whl/cu128"] if WIN else []
            run(pip + ["install", "--force-reinstall", "torch", "torchvision"] + index)
        elif ok.returncode != 0:
            print("   ⚠ PyTorch ne voit pas la carte AMD : la pose et la transcription tourneront sur le processeur.\n"
                  "     Windows : pilote AMD Adrenalin 26.2.2 ou plus récent et Windows 11 requis.\n"
                  "     Linux : utilisateur dans les groupes 'render' et 'video' (sudo usermod -aG render,video $USER).")


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


def download_models(py: Path, model: str | None, vram: float, vendor: str = "") -> None:
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
        # AMD: Whisper runs through transformers + ROCm PyTorch (CTranslate2 has no AMD backend)
        repo = "openai/whisper-large-v3-turbo" if vendor == "amd" else "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        say(f"modèle de transcription {repo} (≈ 1.6 Go)")
        run([py, "-c", f"from huggingface_hub import snapshot_download as d; d('{repo}')"])
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

    gpu = ("", "", 0.0) if APPLE_SILICON else detect_gpu()
    vendor, gpu_name, vram = gpu
    engine = ("MLX (Apple Silicon)" if APPLE_SILICON else
              {"nvidia": "Ollama + CUDA", "amd": "Ollama + ROCm"}.get(vendor, "Ollama + CPU"))
    print(f"video-analyzer · {platform.system()} {platform.machine()} · Python {platform.python_version()} · "
          f"moteur {engine}" + (f" · {gpu_name} {vram:.0f} Go" if vendor else ""))

    if args.model:  # remembered for every later run (read by video_analyzer.config)
        import json
        (ROOT / "settings.json").write_text(json.dumps({"vlm": args.model}, indent=1) + "\n", encoding="utf-8")
    py = ensure_venv(need_312=WIN and vendor == "amd")
    install_packages(py, gpu)
    if not args.skip_models:
        download_models(py, args.model, vram, vendor)

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
