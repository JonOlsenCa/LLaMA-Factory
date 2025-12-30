# VGPT2 v3 Stage 1: Resume SFT Training
# ======================================
# Run this to resume SFT training from the last checkpoint
#
# This modifies the config temporarily to enable resume
#
# USAGE: Run as script file, not by pasting!
#   .\training\02_resume_sft.ps1

$RepoRoot = "C:\Github\LLM_fine-tuning"

# Change to repo root
Set-Location $RepoRoot

Write-Host "Resuming VGPT2 v3 Stage 1: SFT Training" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "Repo root: $RepoRoot"
Write-Host ""

# Check for existing checkpoints
$checkpointDir = "$RepoRoot\saves\vgpt2_v3\sft"
$checkpoints = Get-ChildItem -Path $checkpointDir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue | Sort-Object Name

if ($checkpoints) {
    Write-Host "Found checkpoints:" -ForegroundColor Cyan
    $checkpoints | ForEach-Object { Write-Host "  - $($_.Name)" }
    Write-Host ""
    Write-Host "Resuming from latest checkpoint..." -ForegroundColor Green
} else {
    Write-Host "No checkpoints found! Run 01_start_sft.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating venv: $RepoRoot\venv" -ForegroundColor Cyan
& "$RepoRoot\venv\Scripts\Activate.ps1"

# Create a temporary config with resume enabled
$configPath = "$RepoRoot\automation\configs\vgpt2_v3\stage1_sft.yaml"
$tempConfigPath = "$RepoRoot\automation\configs\vgpt2_v3\stage1_sft_resume.yaml"

$content = Get-Content $configPath -Raw
$content = $content -replace "resume_from_checkpoint: false", "resume_from_checkpoint: true"
$content | Set-Content $tempConfigPath

Write-Host "Using temporary config: $tempConfigPath" -ForegroundColor Cyan
Write-Host ""

& "$RepoRoot\venv\Scripts\llamafactory-cli.exe" train $tempConfigPath

# Clean up temp config
Remove-Item $tempConfigPath -ErrorAction SilentlyContinue

