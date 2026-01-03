# ==============================================================================
# VGPT2 v4 Environment Setup
# ==============================================================================
#
# WHAT THIS DOES:
#   1. VERIFIES all required dependencies are installed
#   2. Only installs missing packages (if any)
#   3. Fails fast if critical requirements are missing
#
# USAGE (copy/paste from C:\Users\olsen or anywhere):
#   & "C:\Github\LLM_fine-tuning\automation\vgpt2_v4_quickstart\00_setup.ps1"
#
# ==============================================================================

$ErrorActionPreference = "Stop"

# Configuration - ABSOLUTE PATHS
$PROJECT_ROOT = "C:\Github\LLM_fine-tuning"
$PYTHON_EXE = "C:\Github\LLM_fine-tuning\venv\Scripts\python.exe"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VGPT2 v4 Environment Verification        " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project
Set-Location $PROJECT_ROOT

# Step 1: Check venv exists
Write-Host "[1/2] Checking virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "      FAIL: venv not found at $PYTHON_EXE" -ForegroundColor Red
    Write-Host ""
    Write-Host "      Run these commands to create it:" -ForegroundColor Yellow
    Write-Host "        cd $PROJECT_ROOT" -ForegroundColor Gray
    Write-Host "        py -3.12 -m venv venv" -ForegroundColor Gray
    Write-Host "        .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host "        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Gray
    Write-Host "        pip install -e `".[torch,metrics]`"" -ForegroundColor Gray
    Write-Host "        pip install wheel wandb tensorboard" -ForegroundColor Gray
    Write-Host "        pip install https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl" -ForegroundColor Gray
    exit 1
}
$pythonVersion = & $PYTHON_EXE --version 2>&1
Write-Host "      OK  Python: $pythonVersion" -ForegroundColor Green

# Step 2: Verify all packages
Write-Host "[2/2] Verifying packages..." -ForegroundColor Yellow

$verifyScript = @"
import sys

failed = []
warnings = []

# Check Python version
import platform
py_ver = platform.python_version()
if not py_ver.startswith('3.12'):
    warnings.append(f'Python {py_ver} (3.12.x recommended for flash-attn)')
else:
    print(f'  OK  Python {py_ver}')

# Check PyTorch version
try:
    import torch
    if torch.__version__.startswith('2.5.1'):
        print(f'  OK  PyTorch {torch.__version__}')
    else:
        warnings.append(f'PyTorch {torch.__version__} (2.5.1+cu124 recommended)')
except ImportError:
    failed.append('PyTorch - pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124')

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f'  OK  CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})')
    else:
        failed.append('CUDA not available - check GPU drivers')
except:
    failed.append('CUDA check failed')

# Check core packages
core_packages = [
    ('transformers', 'Transformers'),
    ('peft', 'PEFT (LoRA)'),
    ('datasets', 'Datasets'),
    ('accelerate', 'Accelerate'),
]
for module, name in core_packages:
    try:
        __import__(module)
        print(f'  OK  {name}')
    except ImportError:
        failed.append(f'{name} - pip install -e \".[torch,metrics]\"')

# Check optional packages
optional = [('wandb', 'W&B'), ('tensorboard', 'TensorBoard')]
for module, name in optional:
    try:
        __import__(module)
        print(f'  OK  {name}')
    except ImportError:
        warnings.append(f'{name} missing - pip install {module}')

# Check flash attention
try:
    import flash_attn
    print(f'  OK  Flash Attention {flash_attn.__version__}')
except ImportError:
    warnings.append('Flash Attention missing (will use SDPA fallback)')

# Summary
print('')
if failed:
    print('FAILED - Missing required packages:')
    for f in failed:
        print(f'  ✗ {f}')
    sys.exit(1)
elif warnings:
    print('READY with warnings:')
    for w in warnings:
        print(f'  ⚠ {w}')
    sys.exit(0)
else:
    print('ALL CHECKS PASSED')
    sys.exit(0)
"@

& $PYTHON_EXE -c $verifyScript
$verifyResult = $LASTEXITCODE

Write-Host ""
if ($verifyResult -eq 0) {
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Environment Ready!                       " -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Start training:" -ForegroundColor White
    Write-Host '  & "C:\Github\LLM_fine-tuning\automation\vgpt2_v4_quickstart\01_train.ps1"' -ForegroundColor Cyan
} else {
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  Environment NOT Ready                    " -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix the issues above, then run this script again." -ForegroundColor Yellow
    exit 1
}
Write-Host ""

