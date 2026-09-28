#!/usr/bin/env bash
# video-analyzer — installation macOS / Linux.
#   ./install.sh            installe tout (prérequis, environnement Python, modèles) et vérifie
#   ./install.sh --demo     … puis analyse une courte vidéo de démonstration
#   ./install.sh --help     options (--model 4b|9b, --skip-models, --video FICHIER)
set -euo pipefail
cd "$(dirname "$0")"

OS="$(uname -s)"
ARCH="$(uname -m)"
have() { command -v "$1" >/dev/null 2>&1; }
info() { printf '\n==> %s\n' "$*"; }

# Python >= 3.11 already on the machine?
find_python() {
  for p in python3.13 python3.12 python3.11 python3; do
    if have "$p" && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      echo "$p"; return 0
    fi
  done
  return 1
}

if [ "$OS" = "Darwin" ]; then
  if ! have brew; then
    echo "Homebrew est requis : installe-le depuis https://brew.sh puis relance ./install.sh"
    exit 1
  fi
  have ffmpeg || { info "installation de ffmpeg"; brew install ffmpeg; }
  PY="$(find_python)" || { info "installation de Python 3.12"; brew install python@3.12; PY="$(brew --prefix)/bin/python3.12"; }
  if [ "$ARCH" != "arm64" ]; then  # Intel Mac: no MLX, the VLM runs in Ollama
    have ollama || { info "installation d'Ollama"; brew install ollama; }
    pgrep -x ollama >/dev/null || brew services start ollama || true
  fi

elif [ "$OS" = "Linux" ]; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  pkgs_needed=0
  have ffmpeg || pkgs_needed=1
  find_python >/dev/null || pkgs_needed=1
  if [ $pkgs_needed -eq 1 ]; then
    info "installation de ffmpeg et Python (mot de passe administrateur demandé)"
    if have apt-get; then
      $SUDO apt-get update
      $SUDO apt-get install -y ffmpeg python3 python3-venv python3-pip
      find_python >/dev/null || $SUDO apt-get install -y python3.11 python3.11-venv  # Ubuntu 22.04 ships 3.10
    elif have dnf; then
      $SUDO dnf install -y ffmpeg python3 python3-pip || $SUDO dnf install -y ffmpeg-free python3 python3-pip
    elif have pacman; then
      $SUDO pacman -Sy --noconfirm ffmpeg python python-pip
    elif have zypper; then
      $SUDO zypper install -y ffmpeg python311 python311-pip
    else
      echo "Gestionnaire de paquets non reconnu : installe ffmpeg et Python >= 3.11, puis relance."
      exit 1
    fi
  fi
  if ! have ollama; then
    info "installation d'Ollama (installeur officiel ollama.com)"
    curl -fsSL https://ollama.com/install.sh | sh
  fi
else
  echo "Système non géré par ce script ($OS). Sous Windows, lance install.cmd."
  exit 1
fi

PY="$(find_python)" || { echo "Python >= 3.11 introuvable après installation."; exit 1; }
chmod +x va.sh 2>/dev/null || true
exec "$PY" scripts/bootstrap.py "$@"
