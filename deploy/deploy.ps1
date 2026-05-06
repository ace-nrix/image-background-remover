# deploy/deploy.ps1
# Author: Noah Rix
# Uploads all API source files to Databricks Workspace and deploys the
# Databricks App for image-background-remover.
#
# Prerequisites:
#   - Databricks CLI installed and authenticated (profile: nrix)
#   - App already created: databricks apps create image-background-remover
#
# Usage:
#   .\deploy\deploy.ps1

$sep = "=" * 70

# ── Config ────────────────────────────────────────────────────────────────────
$AppName    = "image-background-remover"
$RemotePath = "/Workspace/ML_ai_squad/nrix/$AppName"
$LocalPath  = (Resolve-Path "$PSScriptRoot\..").Path
$Profile    = "nrix"

# File extensions to upload
$UploadExtensions = @('.py', '.txt', '.yaml', '.yml', '.json', '.sh')

# Directories to skip
$ExcludeDirs = @('__pycache__', '.venv', '.git', 'deploy')

# ── Step 1: Create workspace directory ───────────────────────────────────────
Write-Host $sep -ForegroundColor Cyan
Write-Host "STEP 1 - Creating workspace directory" -ForegroundColor Cyan
Write-Host $sep -ForegroundColor Cyan

databricks workspace mkdirs $RemotePath --profile $Profile
Write-Host "  [OK] Directories ready" -ForegroundColor Green

# ── Step 2: Upload source files ───────────────────────────────────────────────
Write-Host ""
Write-Host $sep -ForegroundColor Cyan
Write-Host "STEP 2 - Uploading source files from $LocalPath" -ForegroundColor Cyan
Write-Host $sep -ForegroundColor Cyan

$allFiles = Get-ChildItem -Path $LocalPath -Recurse -File | Where-Object {
    $_.Extension -in $UploadExtensions -and
    -not ($ExcludeDirs | Where-Object { $_.FullName -match "\\$_\\" -or $_.FullName -match "\\$_$" })
}

foreach ($f in $allFiles) {
    $rel        = $f.FullName.Substring($LocalPath.Length + 1).Replace('\', '/')
    $remoteDest = "$RemotePath/$rel"

    $parts     = $remoteDest -split '/'
    $remoteDir = ($parts[0..($parts.Count - 2)]) -join '/'
    databricks workspace mkdirs $remoteDir --profile $Profile | Out-Null

    databricks workspace import $remoteDest --file $f.FullName --format RAW --overwrite --profile $Profile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK]   $rel" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $rel" -ForegroundColor Red
        exit 1
    }
}

Write-Host "  [OK] All files uploaded" -ForegroundColor Green

# ── Step 3: Deploy Databricks App ─────────────────────────────────────────────
Write-Host ""
Write-Host $sep -ForegroundColor Cyan
Write-Host "STEP 3 - Deploying Databricks App: $AppName" -ForegroundColor Cyan
Write-Host $sep -ForegroundColor Cyan

databricks apps deploy $AppName --source-code-path $RemotePath --profile $Profile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Deployment failed"
    exit 1
}

Write-Host ""
Write-Host $sep -ForegroundColor Green
Write-Host "DEPLOY COMPLETE" -ForegroundColor Green
Write-Host $sep -ForegroundColor Green
Write-Host ""

databricks apps get $AppName --profile $Profile
