# deploy/dev.ps1
# Author: Noah Rix
# Starts the FastAPI backend with uvicorn on port 8000 for local development.
# The API reloads automatically on file changes.
# Uses the .venv Python in the project root.
#
# Usage: .\deploy\dev.ps1

$RootPath = (Resolve-Path "$PSScriptRoot\..").Path
$python   = Join-Path $RootPath ".venv\Scripts\python.exe"

$sep = "=" * 70
Write-Host $sep -ForegroundColor Cyan
Write-Host "  IMAGE BACKGROUND REMOVER API - LOCAL DEV MODE" -ForegroundColor Cyan
Write-Host $sep -ForegroundColor Cyan
Write-Host ""
Write-Host "  API : http://localhost:8000                  (uvicorn + FastAPI)" -ForegroundColor Green
Write-Host "  Docs: http://localhost:8000/docs             (Swagger UI)"        -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host $sep -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $python)) {
    Write-Host "  [WARN] .venv not found at $python — falling back to system python" -ForegroundColor Yellow
    $python = "python"
}

Set-Location $RootPath
& $python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
