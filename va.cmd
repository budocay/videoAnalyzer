@echo off
rem video-analyzer - lanceur Windows : va.cmd ma_video.mp4 --summary / va.cmd padel match.mp4 / va.cmd doctor
set PYTHONUTF8=1
rem terminal ouvert avant l'installation : ffmpeg (winget) pas encore dans le PATH de cette session
where ffprobe >nul 2>nul || set "PATH=%PATH%;%LOCALAPPDATA%\Microsoft\WinGet\Links"
"%~dp0.venv\Scripts\video-analyzer.exe" %*
