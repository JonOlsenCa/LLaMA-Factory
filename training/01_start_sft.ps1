# VGPT2 v3 Stage 1: Start SFT Training
# =====================================
# Run this to start supervised fine-tuning from scratch
#
# Expected time: 8-12 hours
# First checkpoint: ~20-30 minutes (step 500)
# Saves every 500 steps, keeps last 5 checkpoints
#
# USAGE: Run as script file, not by pasting!
#   .\training\01_start_sft.ps1
#   Or: C:\Github\LLM_fine-tuning\training\01_start_sft.ps1

$RepoRoot = "C:\Github\LLM_fine-tuning"

# Change to repo root
Set-Location $RepoRoot

Write-Host "Starting VGPT2 v3 Stage 1: SFT Training" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Repo root: $RepoRoot"
Write-Host "Records: 68,435"
Write-Host "Epochs: 3"
Write-Host "Save interval: Every 250 steps (~10-15 min)"
Write-Host ""
Write-Host "First checkpoint will be at: $RepoRoot\saves\vgpt2_v3\sft\checkpoint-250"
Write-Host ""

# Activate virtual environment
Write-Host "Activating venv: $RepoRoot\venv" -ForegroundColor Cyan
& "$RepoRoot\venv\Scripts\Activate.ps1"

$configPath = "$RepoRoot\automation\configs\vgpt2_v3\stage1_sft.yaml"

Write-Host "Running: llamafactory-cli train $configPath" -ForegroundColor Cyan
Write-Host ""

& "$RepoRoot\venv\Scripts\llamafactory-cli.exe" train $configPath

