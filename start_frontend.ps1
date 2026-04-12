# Coinbase Trader — Frontend Startup Script
# Installs npm packages if needed, then starts Vite dev server on port 5174

$ErrorActionPreference = "Stop"
$root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root "frontend"

Write-Host ""
Write-Host "  Coinbase Trader — Frontend" -ForegroundColor Cyan
Write-Host "  ==============================" -ForegroundColor Cyan

Set-Location $frontend

# Install node_modules if missing
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "  Installing npm packages (first run)..." -ForegroundColor Yellow
    npm install
}

# Ensure NODE_TLS_REJECT_UNAUTHORIZED allows self-signed cert proxying
$env:NODE_TLS_REJECT_UNAUTHORIZED = "0"

Write-Host "  Starting frontend on http://localhost:5174 ..." -ForegroundColor Green
npm run dev
