# video-analyzer — installation Windows (appelé par install.cmd).
#   install.cmd            installe tout (Python, ffmpeg, Ollama, environnement, modèles) et vérifie
#   install.cmd --demo     ... puis analyse une courte vidéo de démonstration
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Info($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
function Winget-Install($id, $label) {
    Info "installation de $label"
    winget install -e --id $id --silent --accept-source-agreements --accept-package-agreements
    Refresh-Path
}

if (-not (Have winget)) {
    Write-Host "winget est requis (application 'App Installer' du Microsoft Store), puis relance install.cmd."
    exit 1
}

# Python >= 3.11 via the py launcher, else install 3.12
$py = $null
foreach ($v in @("3.13", "3.12", "3.11")) {
    if (Have py) {
        try {
            $null = & py "-$v" -c "import sys" 2>&1
            if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
        } catch { }
    }
}
if (-not $py) {
    Winget-Install "Python.Python.3.12" "Python 3.12"
    $py = @("py", "-3.12")
}

if (-not (Have ffmpeg)) { Winget-Install "Gyan.FFmpeg" "ffmpeg" }
if (-not (Have ollama)) { Winget-Install "Ollama.Ollama" "Ollama" }

# winget links for portable apps (ffmpeg) land in the user's Links folder
$links = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
if ((Test-Path $links) -and ($env:Path -notlike "*$links*")) { $env:Path += ";$links" }

if (-not (Have ffmpeg)) {
    Write-Host "ffmpeg installé mais pas encore visible : ferme cette fenêtre et relance install.cmd."
    exit 1
}

$env:PYTHONUTF8 = "1"
& $py[0] $py[1] scripts\bootstrap.py @args
exit $LASTEXITCODE
