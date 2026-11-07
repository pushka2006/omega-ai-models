# OMEGA MODEL 2 Server Launcher for Windows PowerShell
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "         LAUNCHING OMEGA MODEL 2 AI ASSISTANT         " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check Python environment
if (Test-Path "$ScriptDir\..\.venv\Scripts\Activate.ps1") {
    Write-Host "[Environment] Activating workspace virtualenv..." -ForegroundColor Green
    & "$ScriptDir\..\.venv\Scripts\Activate.ps1"
}

Write-Host "[Server] Starting FastAPI backend on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Write-Host "[UI] Open your browser at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "[Docs] Interactive API documentation at http://127.0.0.1:8000/docs" -ForegroundColor Gray
Write-Host "Press Ctrl+C to terminate the server.`n" -ForegroundColor Gray

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

