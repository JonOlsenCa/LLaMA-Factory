# VGPT2 v3 Stage 1: Start SFT Training
# =====================================
# Run this to start supervised fine-tuning from scratch
#
# Expected time: 8-12 hours
# First checkpoint: ~20-30 minutes (step 500)
# Saves every 500 steps, keeps last 5 checkpoints

# Get the repo root (parent of training folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Change to repo root
Set-Location $RepoRoot

Write-Host "Starting VGPT2 v3 Stage 1: SFT Training" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Repo root: $RepoRoot"
Write-Host "Records: 68,435"
Write-Host "Epochs: 3"
Write-Host "Save interval: Every 500 steps (~20-30 min)"
Write-Host ""
Write-Host "First checkpoint will be at: $RepoRoot\saves\vgpt2_v3\sft\checkpoint-500"
Write-Host ""

$configPath = Join-Path $RepoRoot "automation\configs\vgpt2_v3\stage1_sft.yaml"

llamafactory-cli train $configPath

