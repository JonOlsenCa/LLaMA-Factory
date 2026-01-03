# ==============================================================================
# VGPT2 v4 Training Script
# ==============================================================================
#
# WHAT THIS DOES:
#   Fine-tunes defog/llama-3-sqlcoder-8b on Vista SQL examples using LoRA
#
# EXPECTED OUTPUT:
#   - Model checkpoint: saves/vgpt2_v4/sft_optimized/
#   - Training logs: W&B dashboard
#   - Duration: ~30-45 minutes
#
# USAGE (copy/paste from C:\Users\olsen or anywhere):
#   & "C:\Github\LLM_fine-tuning\automation\vgpt2_v4_quickstart\01_train.ps1"
#
# ==============================================================================

$ErrorActionPreference = "Stop"

# Configuration - ABSOLUTE PATHS (no reliance on PATH or activation)
$PROJECT_ROOT = "C:\Github\LLM_fine-tuning"
$PYTHON_EXE = "C:\Github\LLM_fine-tuning\venv\Scripts\python.exe"
$CONFIG_FILE = "automation/configs/vgpt2_v4/stage1_sft.yaml"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VGPT2 v4 Training - SQLCoder Fine-Tune   " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Navigate to project
Write-Host "[1/4] Navigating to project directory..." -ForegroundColor Yellow
Set-Location $PROJECT_ROOT
Write-Host "      Current directory: $(Get-Location)" -ForegroundColor Gray

# Step 2: Verify venv Python exists
Write-Host "[2/4] Verifying Python virtual environment..." -ForegroundColor Yellow
if (Test-Path $PYTHON_EXE) {
    $pythonVersion = & $PYTHON_EXE --version 2>&1
    Write-Host "      Python: $PYTHON_EXE" -ForegroundColor Gray
    Write-Host "      Version: $pythonVersion" -ForegroundColor Gray
} else {
    Write-Host "      ERROR: venv Python not found at $PYTHON_EXE" -ForegroundColor Red
    Write-Host "      Run: python -m venv venv" -ForegroundColor Red
    Write-Host "      Then: .\venv\Scripts\Activate.ps1" -ForegroundColor Red
    Write-Host "      Then: pip install -e .[torch,metrics]" -ForegroundColor Red
    exit 1
}

# Step 3: Verify config exists
Write-Host "[3/4] Verifying training configuration..." -ForegroundColor Yellow
$configPath = Join-Path $PROJECT_ROOT $CONFIG_FILE
if (Test-Path $configPath) {
    Write-Host "      Config found: $CONFIG_FILE" -ForegroundColor Gray
} else {
    Write-Host "      ERROR: Config not found at $configPath" -ForegroundColor Red
    exit 1
}

# Step 4: Pre-flight checks
Write-Host "[4/4] Running pre-flight checks..." -ForegroundColor Yellow

# Check GPU
$gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if ($gpuInfo) {
    Write-Host "      GPU: $gpuInfo" -ForegroundColor Gray
} else {
    Write-Host "      WARNING: nvidia-smi not found or no GPU detected" -ForegroundColor Yellow
}

# Check VRAM availability
$vramFree = nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
if ($vramFree -and [int]$vramFree -lt 40000) {
    Write-Host "      WARNING: Only $($vramFree)MB VRAM free. Training needs ~40GB." -ForegroundColor Yellow
    Write-Host "      Close other GPU applications if training fails." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Starting Training...                      " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Config: $CONFIG_FILE" -ForegroundColor White
Write-Host "Output: saves/vgpt2_v4/sft_optimized/" -ForegroundColor White
Write-Host ""
Write-Host "Training will begin in 3 seconds... (Ctrl+C to cancel)" -ForegroundColor Gray
Start-Sleep -Seconds 3

# Run training using VENV PYTHON DIRECTLY (not relying on PATH)
try {
    & $PYTHON_EXE -m llamafactory.cli train $CONFIG_FILE

    if ($LASTEXITCODE -ne 0) {
        throw "Training exited with code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Training Complete!                        " -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "  1. Test the model:   & '$PROJECT_ROOT\automation\vgpt2_v4_quickstart\02_chat.ps1'" -ForegroundColor Gray
    Write-Host "  2. Run evaluation:   & '$PROJECT_ROOT\automation\vgpt2_v4_quickstart\03_evaluate.ps1'" -ForegroundColor Gray
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  Training Failed                           " -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common fixes:" -ForegroundColor Yellow
    Write-Host "  - OOM: Reduce per_device_train_batch_size in config" -ForegroundColor Gray
    Write-Host "  - Flash Attn: pip install flash-attn --no-build-isolation" -ForegroundColor Gray
    Write-Host "  - Or change flash_attn: fa2 to flash_attn: sdpa in config" -ForegroundColor Gray
    exit 1
}

