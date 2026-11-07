# OMEGA Neural Network Engine Dashboard Launcher for Windows PowerShell
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "     LAUNCHING OMEGA NEURAL NETWORK ENGINE DASHBOARD    " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check Python environment
if (Test-Path "$ScriptDir\..\.venv\Scripts\Activate.ps1") {
    Write-Host "[Environment] Activating workspace virtualenv..." -ForegroundColor Green
    & "$ScriptDir\..\.venv\Scripts\Activate.ps1"
}

Write-Host "[Dashboard] Starting server on http://127.0.0.1:8500 ..." -ForegroundColor Yellow
Write-Host "[UI] Open your browser at http://127.0.0.1:8500" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate the dashboard server.`n" -ForegroundColor Gray

python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8500 --reload
