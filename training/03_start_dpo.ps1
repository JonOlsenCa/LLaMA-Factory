# VGPT2 v3 Stage 2: Start DPO Training
# =====================================
# Run this after SFT completes successfully
#
# Prerequisites: Stage 1 SFT must be complete
# Expected time: 30-60 minutes
#
# USAGE: Run as script file, not by pasting!
#   .\training\03_start_dpo.ps1

$RepoRoot = "C:\Github\LLM_fine-tuning"

# Change to repo root
Set-Location $RepoRoot

Write-Host "Starting VGPT2 v3 Stage 2: DPO Training" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Repo root: $RepoRoot"
Write-Host ""

# Verify SFT adapter exists
$sftAdapter = "$RepoRoot\saves\vgpt2_v3\sft\adapter_model.safetensors"
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

# Activate virtual environment
Write-Host "Activating venv: $RepoRoot\venv" -ForegroundColor Cyan
& "$RepoRoot\venv\Scripts\Activate.ps1"

$configPath = "$RepoRoot\automation\configs\vgpt2_v3\stage2_dpo.yaml"

& "$RepoRoot\venv\Scripts\llamafactory-cli.exe" train $configPath

