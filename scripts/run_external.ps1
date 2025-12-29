# ============================================================================
# Run Script in External Terminal (Outside VS Code)
# ============================================================================
# VS Code's integrated terminal randomly sends Ctrl+C interrupts which kills
# long-running scripts. This script launches commands in a separate PowerShell
# window that is immune to VS Code's interference.
#
# Usage:
#   .\scripts\run_external.ps1 -Script "test_vgpt2.py"
#   .\scripts\run_external.ps1 -Script "resource_monitor.py"
#   .\scripts\run_external.ps1 -Command "llamafactory-cli train automation/configs/vgpt2_lora_sft.yaml"
# ============================================================================

param(
    [string]$Script,
    [string]$Command,
    [switch]$KeepOpen = $true
)

$ProjectRoot = "C:\Github\LLM_fine-tuning"
$VenvActivate = "$ProjectRoot\venv\Scripts\activate.ps1"

if ($Script) {
    $ScriptPath = "$ProjectRoot\scripts\$Script"
    if (-not (Test-Path $ScriptPath)) {
        Write-Error "Script not found: $ScriptPath"
        exit 1
    }
    $FullCommand = "cd '$ProjectRoot'; & '$VenvActivate'; python '$ScriptPath'"
} elseif ($Command) {
    $FullCommand = "cd '$ProjectRoot'; & '$VenvActivate'; $Command"
} else {
    Write-Host "Usage:" -ForegroundColor Cyan
    Write-Host "  .\scripts\run_external.ps1 -Script 'test_vgpt2.py'" -ForegroundColor White
    Write-Host "  .\scripts\run_external.ps1 -Command 'llamafactory-cli webui'" -ForegroundColor White
    exit 0
}

if ($KeepOpen) {
    $FullCommand += "; Write-Host ''; Write-Host 'Press any key to close...' -ForegroundColor Yellow; `$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')"
}

Write-Host "Launching external terminal..." -ForegroundColor Green
Write-Host "Command: $FullCommand" -ForegroundColor DarkGray

# Start new PowerShell window
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FullCommand

Write-Host "External terminal launched. Check the new window." -ForegroundColor Green

