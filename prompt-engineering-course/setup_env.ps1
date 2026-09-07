# setup_env.ps1 - create a local Python venv and install requirements
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# or, if your session already allows scripts:
#   .\setup_env.ps1

$ErrorActionPreference = "Stop"

# Move to the directory this script lives in
Set-Location -Path $PSScriptRoot

$venv = ".venv"

if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment in $venv ..."
    py -3 -m venv $venv
} else {
    Write-Host "Virtual environment $venv already exists, reusing it."
}

$python = Join-Path $venv "Scripts\python.exe"

Write-Host "Upgrading pip ..."
& $python -m pip install --upgrade pip

Write-Host "Installing packages from requirements.txt ..."
& $python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host "    .\$venv\Scripts\Activate.ps1"
