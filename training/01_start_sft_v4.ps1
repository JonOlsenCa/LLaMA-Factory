# VGPT2 v4 Stage 1: SFT Training Script
# =======================================
# SQLCoder base model + V4 SQLCoder-style dataset
#
# Usage:
#   .\training\01_start_sft_v4.ps1
#   .\training\01_start_sft_v4.ps1 -DryRun  # Preview config without training

param(
    [switch]$DryRun,
    [switch]$Resume
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Paths (absolute so the script works from any cwd)
$PythonExe = Join-Path $RepoRoot ".venv/Scripts/python.exe"
$CliExe = Join-Path $RepoRoot ".venv/Scripts/llamafactory-cli.exe"
$CONFIG_FILE = Join-Path $RepoRoot "automation/configs/vgpt2_v4/stage1_sft.yaml"
$OUTPUT_DIR = Join-Path $RepoRoot "saves/vgpt2_v4/sft"
$DATASET_PATH = Join-Path $RepoRoot "data/vgpt2_v4_sft_expanded_clean.json"
$ExpandScript = Join-Path $RepoRoot "scripts/expand_v4_training_data.py"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VGPT2 v4 Training: SQLCoder SFT" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

# Optional: Speed up HF downloads and silence cache warnings (hf_xet)
# Install once: pip install "huggingface_hub[hf_xet]"
# If symlink warning persists on Windows, set: setx HF_HUB_DISABLE_SYMLINKS_WARNING 1

# Check CUDA
$cudaCheck = & $PythonExe -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaCheck -ne "True") {
    Write-Host "ERROR: CUDA not available!" -ForegroundColor Red
    Write-Host "Run (inside .venv): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" -ForegroundColor Yellow
    exit 1
}
Write-Host "  CUDA: Available" -ForegroundColor Green

# Check GPU
$gpuName = & $PythonExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
Write-Host "  GPU: $gpuName" -ForegroundColor Green

# Check GPU memory
$gpuMem = & $PythonExe -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')" 2>$null
Write-Host "  VRAM: $gpuMem" -ForegroundColor Green

# Check LLaMA Factory
try {
    $version = & $CliExe version 2>$null
    Write-Host "  LLaMA Factory: $version" -ForegroundColor Green
} catch {
    Write-Host "ERROR: LLaMA Factory not found!" -ForegroundColor Red
    exit 1
}

# Check config file
Write-Host ""
Write-Host "[2/5] Checking configuration..." -ForegroundColor Yellow
if (-not (Test-Path $CONFIG_FILE)) {
    Write-Host "ERROR: Config file not found: $CONFIG_FILE" -ForegroundColor Red
    exit 1
}
Write-Host "  Config: $CONFIG_FILE" -ForegroundColor Green

# Parse config for key settings
$config = Get-Content $CONFIG_FILE -Raw
if ($config -match "model_name_or_path:\s*(.+)") { 
    Write-Host "  Base Model: $($Matches[1].Trim())" -ForegroundColor Green 
}
if ($config -match "dataset:\s*(\S+)") { 
    Write-Host "  Dataset: $($Matches[1])" -ForegroundColor Green 
}
if ($config -match "num_train_epochs:\s*(\S+)") { 
    Write-Host "  Epochs: $($Matches[1])" -ForegroundColor Green 
}
if ($config -match "lora_rank:\s*(\S+)") { 
    Write-Host "  LoRA Rank: $($Matches[1])" -ForegroundColor Green 
}

# Check dataset
Write-Host ""
Write-Host "[3/5] Checking dataset..." -ForegroundColor Yellow
if (-not (Test-Path $DATASET_PATH)) {
    Write-Host "WARNING: Expanded dataset not found, generating..." -ForegroundColor Yellow
    & $PythonExe $ExpandScript
}

$datasetSize = & $PythonExe -c "import json, pathlib; p = pathlib.Path('$DATASET_PATH'); print(len(json.load(open(p, encoding='utf-8'))))" 2>$null
Write-Host "  Dataset size: $datasetSize examples" -ForegroundColor Green

# Check/create output directory
Write-Host ""
Write-Host "[4/5] Preparing output directory..." -ForegroundColor Yellow
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null
    Write-Host "  Created: $OUTPUT_DIR" -ForegroundColor Green
} else {
    $checkpoints = Get-ChildItem -Path $OUTPUT_DIR -Filter "checkpoint-*" -Directory 2>$null
    if ($checkpoints) {
        Write-Host "  Found $($checkpoints.Count) existing checkpoint(s)" -ForegroundColor Yellow
        if (-not $Resume) {
            Write-Host "  Starting fresh (use -Resume to continue)" -ForegroundColor Yellow
        }
    }
}

# Dry run check
if ($DryRun) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  DRY RUN - No training started" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To start training, run without -DryRun:" -ForegroundColor Yellow
    Write-Host "  .\training\01_start_sft_v4.ps1" -ForegroundColor White
    exit 0
}

# Start training
Write-Host ""
Write-Host "[5/5] Starting training..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Training Started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Monitor with:" -ForegroundColor Cyan
Write-Host "  nvidia-smi -l 1" -ForegroundColor White
Write-Host "  Get-Content $OUTPUT_DIR/trainer_log.jsonl -Tail 10 -Wait" -ForegroundColor White
Write-Host ""

# Build command args (array to avoid Invoke-Expression parsing issues)
$cmdArgs = @("train", "$CONFIG_FILE")
if ($Resume) {
    # Find latest checkpoint
    $latestCheckpoint = Get-ChildItem -Path $OUTPUT_DIR -Filter "checkpoint-*" -Directory | 
        Sort-Object Name -Descending | 
        Select-Object -First 1
    if ($latestCheckpoint) {
        Write-Host "Resuming from: $($latestCheckpoint.FullName)" -ForegroundColor Yellow
        $cmdArgs += @("--resume_from_checkpoint", "$($latestCheckpoint.FullName)")
    }
}

# Execute training
Write-Host "Command: $CliExe $($cmdArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& $CliExe @cmdArgs

# Training complete
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Training Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Validate: $PythonExe $RepoRoot\scripts\vgpt2_v4\probe.py --model $OUTPUT_DIR --output $RepoRoot\output\probe_sft.json" -ForegroundColor White
Write-Host "  2. Chat test: $CliExe chat --model_name_or_path $OUTPUT_DIR --template llama3 --adapter_name_or_path $OUTPUT_DIR" -ForegroundColor White
Write-Host "  3. If needed, run DPO: pwsh -File $RepoRoot\training\02_start_dpo_v4.ps1" -ForegroundColor White
