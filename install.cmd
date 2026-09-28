@echo off
rem video-analyzer - installation Windows : double-clic, ou "install.cmd --demo" dans un terminal.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
set RC=%ERRORLEVEL%
echo.
pause
exit /b %RC%
