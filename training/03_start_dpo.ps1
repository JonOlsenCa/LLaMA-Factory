# VGPT2 v3 Stage 2: Start DPO Training
# =====================================
# Run this after SFT completes successfully
#
# Prerequisites: Stage 1 SFT must be complete
# Expected time: 30-60 minutes

# Get the repo root (parent of training folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Change to repo root
Set-Location $RepoRoot

Write-Host "Starting VGPT2 v3 Stage 2: DPO Training" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Repo root: $RepoRoot"
Write-Host ""

# Verify SFT adapter exists
$sftAdapter = Join-Path $RepoRoot "saves\vgpt2_v3\sft\adapter_model.safetensors"
if (-not (Test-Path $sftAdapter)) {
    Write-Host "ERROR: SFT adapter not found at $sftAdapter" -ForegroundColor Red
    Write-Host "Complete Stage 1 SFT training first!" -ForegroundColor Red
    exit 1
}

Write-Host "SFT adapter found. Starting DPO..." -ForegroundColor Cyan
Write-Host ""
Write-Host "DPO pairs: 1,427"
Write-Host "Epochs: 2"
Write-Host "Expected time: 30-60 minutes"
Write-Host ""

$configPath = Join-Path $RepoRoot "automation\configs\vgpt2_v3\stage2_dpo.yaml"

llamafactory-cli train $configPath

