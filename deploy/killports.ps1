# deploy/killports.ps1
# Author: Noah Rix
# Kills any process currently listening on port 8000.
# Run this if uvicorn fails to start because the port is already in use.
#
# Usage: .\deploy\killports.ps1

$Port = 8000

Write-Host "Checking for processes on port $Port..." -ForegroundColor Cyan

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

if (-not $connections) {
    Write-Host "  No process found on port $Port." -ForegroundColor Green
    exit 0
}

$pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($pid in $pids) {
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "  Killing PID $pid ($($proc.Name)) on port $Port..." -ForegroundColor Yellow
        Stop-Process -Id $pid -Force
        Write-Host "  [OK] Killed PID $pid" -ForegroundColor Green
    }
}

Write-Host "Port $Port is now free." -ForegroundColor Green
