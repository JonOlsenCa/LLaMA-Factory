# VGPT2 v3 Stage 2 v2: DPO with Hallucination Focus
# ==================================================
# Run this to retrain DPO with hallucination-focused dataset
#
# Prerequisites: Stage 1 SFT must be complete
# Expected time: ~45 minutes
#
# USAGE: Run as script file, not by pasting!
#   .\training\03b_start_dpo_v2.ps1

$RepoRoot = "C:\Github\LLM_fine-tuning"

# Change to repo root
Set-Location $RepoRoot

Write-Host "Starting VGPT2 v3 Stage 2 v2: Hallucination-Focused DPO" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
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

# Verify new dataset exists
$datasetPath = "$RepoRoot\data\vgpt2_v3_dpo_v2.json"
if (-not (Test-Path $datasetPath)) {
    Write-Host "ERROR: DPO v2 dataset not found at $datasetPath" -ForegroundColor Red
    Write-Host "Run scripts/merge_dpo_datasets.py first!" -ForegroundColor Red
    exit 1
}

Write-Host "SFT adapter found. Starting DPO v2 (Hallucination Focus)..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Dataset: vgpt2_v3_dpo_v2.json"
Write-Host "DPO pairs: 2,584"
Write-Host "  - Hallucination rejection: 83%"
Write-Host "  - Quality/style: 17%"
Write-Host "Epochs: 2"
Write-Host "Expected time: ~45 minutes"
Write-Host ""

# Activate virtual environment
Write-Host "Activating venv: $RepoRoot\venv" -ForegroundColor Cyan
& "$RepoRoot\venv\Scripts\Activate.ps1"

$configPath = "$RepoRoot\automation\configs\vgpt2_v3\stage2_dpo_v2.yaml"

Write-Host "Starting training with config: $configPath" -ForegroundColor Cyan
Write-Host ""

& "$RepoRoot\venv\Scripts\llamafactory-cli.exe" train $configPath

