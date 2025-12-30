# VGPT2 v3 Validation
# ====================
# Run validation tests against a trained model
#
# Usage:
#   .\05_validate.ps1              # Validate final model
#   .\05_validate.ps1 -Stage sft   # Validate SFT model
#   .\05_validate.ps1 -Quick       # Quick test (fewer questions)
#
# USAGE: Run as script file, not by pasting!

param(
    [ValidateSet("sft", "dpo", "final")]
    [string]$Stage = "final",
    [switch]$Quick
)

$RepoRoot = "C:\Github\LLM_fine-tuning"

# Change to repo root
Set-Location $RepoRoot

Write-Host "VGPT2 v3 Validation" -ForegroundColor Green
Write-Host "===================" -ForegroundColor Green
Write-Host ""
Write-Host "Repo root: $RepoRoot"
Write-Host ""

# Activate virtual environment
Write-Host "Activating venv: $RepoRoot\venv" -ForegroundColor Cyan
& "$RepoRoot\venv\Scripts\Activate.ps1"

$modelPath = switch ($Stage) {
    "sft"   { "$RepoRoot\saves\vgpt2_v3\sft" }
    "dpo"   { "$RepoRoot\saves\vgpt2_v3\dpo" }
    "final" { "$RepoRoot\saves\vgpt2_v3\final" }
}

if (-not (Test-Path $modelPath)) {
    Write-Host "ERROR: Model not found at $modelPath" -ForegroundColor Red
    exit 1
}

Write-Host "Validating model: $modelPath" -ForegroundColor Cyan
Write-Host ""

$scriptPath = "$RepoRoot\scripts\vgpt2_v3\run_validation.py"
$pyArgs = @($scriptPath, "--model", $modelPath)
if ($Quick) {
    $pyArgs += "--quick"
    Write-Host "Running quick validation..." -ForegroundColor Yellow
} else {
    Write-Host "Running full validation..." -ForegroundColor Yellow
}

python @pyArgs

