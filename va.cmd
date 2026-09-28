@echo off
rem video-analyzer - lanceur Windows : va.cmd ma_video.mp4 --summary / va.cmd padel match.mp4 / va.cmd doctor
set PYTHONUTF8=1
"%~dp0.venv\Scripts\video-analyzer.exe" %*
