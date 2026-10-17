# Coinbase Trader — Backend Startup Script
# venv check, log dirs, then main.py. main.py's startup guard refuses an occupied port;
# pass -Replace to ask it to replace a running instance it can verify as its own.

param([switch]$Replace)

$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv   = Join-Path $root ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$pip    = Join-Path $venv "Scripts\pip.exe"
$req    = Join-Path $root "backend\requirements.txt"
$logDir = Join-Path $root "backend\logs"

Write-Host ""
Write-Host "  Coinbase Trader — Backend" -ForegroundColor Cyan
Write-Host "  =============================" -ForegroundColor Cyan

# 1. Create logs directory
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
    Write-Host "  Created backend\logs\" -ForegroundColor Gray
}

# 2. Create virtualenv if missing
if (-not (Test-Path $python)) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venv
}

# 3. Install / upgrade requirements (silent unless there are changes)
Write-Host "  Checking Python dependencies..." -ForegroundColor Yellow
& $pip install -q -r $req

# 4. Ensure .env exists
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "  WARNING: .env not found — creating from template." -ForegroundColor Red
    Copy-Item (Join-Path $root ".env.example") $envFile
    Write-Host "  Edit .env with your Coinbase CDP API keys before trading." -ForegroundColor Yellow
}

# 5. Start backend (the guard in main.py decides whether the port may be used)
Write-Host "  Starting backend on http://localhost:8001 ..." -ForegroundColor Green
Write-Host "  Logs → backend\logs\backend.log" -ForegroundColor Gray
Set-Location (Join-Path $root "backend")
$mainArgs = @("main.py")
if ($Replace) { $mainArgs += @("--mode", "replace") }
& $python @mainArgs
