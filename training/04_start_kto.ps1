# VGPT2 v3 Stage 3: Start KTO Training
# =====================================
# Run this after DPO completes successfully
#
# Prerequisites: Stage 1 SFT and Stage 2 DPO must be complete
# Expected time: 15-30 minutes
#
# USAGE: Run as script file, not by pasting!
#   .\training\04_start_kto.ps1

$RepoRoot = "C:\Github\LLM_fine-tuning"

# Change to repo root
Set-Location $RepoRoot

Write-Host "Starting VGPT2 v3 Stage 3: KTO Training" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Repo root: $RepoRoot"
Write-Host ""

# Verify DPO adapter exists
$dpoAdapter = "$RepoRoot\saves\vgpt2_v3\dpo\adapter_model.safetensors"
if (-not (Test-Path $dpoAdapter)) {
    Write-Host "ERROR: DPO adapter not found at $dpoAdapter" -ForegroundColor Red
    Write-Host "Complete Stage 2 DPO training first!" -ForegroundColor Red
    exit 1
}

Write-Host "DPO adapter found. Starting KTO..." -ForegroundColor Cyan
Write-Host ""
Write-Host "KTO records: 1,420"
Write-Host "Epochs: 1"
Write-Host "Expected time: 15-30 minutes"
Write-Host ""
Write-Host "Final model will be saved to: $RepoRoot\saves\vgpt2_v3\final"
Write-Host ""

# Activate virtual environment
Write-Host "Activating venv: $RepoRoot\venv" -ForegroundColor Cyan
& "$RepoRoot\venv\Scripts\Activate.ps1"

$configPath = "$RepoRoot\automation\configs\vgpt2_v3\stage3_kto.yaml"

& "$RepoRoot\venv\Scripts\llamafactory-cli.exe" train $configPath

