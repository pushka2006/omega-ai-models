# OMEGA MODEL 2 Terminal CLI Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (Test-Path "$ScriptDir\..\.venv\Scripts\Activate.ps1") {
    & "$ScriptDir\..\.venv\Scripts\Activate.ps1"
}

python cli/omega_cli.py

