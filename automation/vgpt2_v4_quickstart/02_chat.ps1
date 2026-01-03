# ==============================================================================
# VGPT2 v4 Interactive Chat
# ==============================================================================
#
# WHAT THIS DOES:
#   Opens an interactive chat with your fine-tuned VGPT2 v4 model
#   to test SQL generation capabilities
#
# EXPECTED BEHAVIOR:
#   - Type a question about Vista data
#   - Model generates SQL with explanation
#   - Type 'exit' or Ctrl+C to quit
#
# USAGE (copy/paste from C:\Users\olsen or anywhere):
#   & "C:\Github\LLM_fine-tuning\automation\vgpt2_v4_quickstart\02_chat.ps1"
#
# ==============================================================================

$ErrorActionPreference = "Stop"

# Configuration - ABSOLUTE PATHS (no reliance on PATH or activation)
$PROJECT_ROOT = "C:\Github\LLM_fine-tuning"
$PYTHON_EXE = "C:\Github\LLM_fine-tuning\venv\Scripts\python.exe"
$MODEL_PATH = "defog/llama-3-sqlcoder-8b"
$ADAPTER_PATH = "saves/vgpt2_v4/sft_optimized"
$TEMPLATE = "llama3"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VGPT2 v4 Interactive Chat                " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project
Set-Location $PROJECT_ROOT

# Verify venv Python exists
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "ERROR: venv Python not found at $PYTHON_EXE" -ForegroundColor Red
    exit 1
}

# Check if adapter exists
$adapterFullPath = Join-Path $PROJECT_ROOT $ADAPTER_PATH
if (-not (Test-Path $adapterFullPath)) {
    Write-Host "WARNING: No trained adapter found at $ADAPTER_PATH" -ForegroundColor Yellow
    Write-Host "Running with base SQLCoder model only..." -ForegroundColor Yellow
    Write-Host ""

    # Run without adapter using VENV PYTHON DIRECTLY
    Write-Host "Starting chat with base model..." -ForegroundColor Green
    Write-Host "Type 'exit' or Ctrl+C to quit" -ForegroundColor Gray
    Write-Host ""

    & $PYTHON_EXE -m llamafactory.chat.cli `
        --model_name_or_path $MODEL_PATH `
        --template $TEMPLATE `
        --trust_remote_code true
}
else {
    Write-Host "Model: $MODEL_PATH" -ForegroundColor White
    Write-Host "Adapter: $ADAPTER_PATH" -ForegroundColor White
    Write-Host ""
    Write-Host "Starting chat..." -ForegroundColor Green
    Write-Host "Type 'exit' or Ctrl+C to quit" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Example prompts:" -ForegroundColor Yellow
    Write-Host '  "Show all open AP invoices for vendor 12345"' -ForegroundColor Gray
    Write-Host '  "Get job cost variance by phase for job 1234-001"' -ForegroundColor Gray
    Write-Host '  "List employees with timecard hours this month"' -ForegroundColor Gray
    Write-Host ""

    # Run with adapter using VENV PYTHON DIRECTLY
    & $PYTHON_EXE -m llamafactory.chat.cli `
        --model_name_or_path $MODEL_PATH `
        --adapter_name_or_path $ADAPTER_PATH `
        --template $TEMPLATE `
        --finetuning_type lora `
        --trust_remote_code true
}

