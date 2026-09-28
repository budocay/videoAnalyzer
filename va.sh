#!/usr/bin/env bash
# video-analyzer — lanceur macOS / Linux : ./va.sh ma_video.mp4 --summary · ./va.sh padel match.mp4 · ./va.sh doctor
export PYTHONUTF8=1
exec "$(dirname "$0")/.venv/bin/video-analyzer" "$@"
