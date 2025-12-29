# ============================================================================
# Start VGPT2 Training in External Terminal
# ============================================================================
# ALWAYS use this script to start training! Never run training directly in
# VS Code's integrated terminal - it will randomly send Ctrl+C and kill
# your training mid-run.
#
# Usage:
#   .\scripts\start_training.ps1
#   .\scripts\start_training.ps1 -Resume     # Resume from checkpoint
#   .\scripts\start_training.ps1 -Config "custom_config.yaml"
# ============================================================================

param(
    [string]$Config = "automation/configs/vgpt2_lora_sft.yaml",
    [switch]$Resume
)

$ProjectRoot = "C:\Github\LLM_fine-tuning"
$VenvActivate = "$ProjectRoot\venv\Scripts\activate.ps1"
$ConfigPath = "$ProjectRoot\$Config"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VGPT2 Training Launcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check config exists
if (-not (Test-Path $ConfigPath)) {
    Write-Error "Config not found: $ConfigPath"
    exit 1
}

Write-Host "Config: $Config" -ForegroundColor White
Write-Host "Resume: $Resume" -ForegroundColor White
Write-Host ""

# If resume, check that resume_from_checkpoint is enabled
if ($Resume) {
    $configContent = Get-Content $ConfigPath -Raw
    if ($configContent -match "^#\s*resume_from_checkpoint") {
        Write-Host "WARNING: resume_from_checkpoint is commented out in config!" -ForegroundColor Yellow
        Write-Host "Edit $Config and uncomment: resume_from_checkpoint: true" -ForegroundColor Yellow
        Write-Host ""
        $response = Read-Host "Continue anyway? (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            exit 1
        }
    }
}

# Build command
$TrainCommand = "llamafactory-cli train '$ConfigPath'"

$FullCommand = @"
cd '$ProjectRoot'
& '$VenvActivate'
Write-Host '============================================' -ForegroundColor Cyan
Write-Host '  VGPT2 Training Started' -ForegroundColor Cyan
Write-Host '  DO NOT CLOSE THIS WINDOW!' -ForegroundColor Red
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
$TrainCommand
Write-Host ''
Write-Host 'Training complete!' -ForegroundColor Green
Write-Host 'Press any key to close...' -ForegroundColor Yellow
`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
"@

Write-Host "Launching training in external terminal..." -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: Do not close the new terminal window!" -ForegroundColor Red
Write-Host ""

# Start new PowerShell window with admin title
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FullCommand

Write-Host "Training launched. Monitor progress in the new window." -ForegroundColor Green
Write-Host ""
Write-Host "To monitor resources, run in another terminal:" -ForegroundColor Cyan
Write-Host "  python scripts/resource_monitor.py" -ForegroundColor White

