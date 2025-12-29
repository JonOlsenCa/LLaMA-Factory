# ============================================================================
# Kill Orphaned Python/Training Processes
# ============================================================================
# Use this script to clean up orphaned processes after VS Code randomly
# kills terminals with Ctrl+C, leaving background processes running.
#
# Usage:
#   .\scripts\kill_orphans.ps1           # Interactive mode (asks before killing)
#   .\scripts\kill_orphans.ps1 -Force    # Kill all without asking
#   .\scripts\kill_orphans.ps1 -List     # Just list processes, don't kill
# ============================================================================

param(
    [switch]$Force,
    [switch]$List
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Orphaned Process Cleanup Tool" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Processes to look for
$ProcessNames = @(
    "python",
    "llamafactory*",
    "nvidia-smi"
)

$FoundProcesses = @()

foreach ($name in $ProcessNames) {
    $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
    if ($procs) {
        foreach ($proc in $procs) {
            # Skip system processes
            if ($proc.Path -and $proc.Path -notlike "*WindowsApps*") {
                $FoundProcesses += $proc
            }
        }
    }
}

if ($FoundProcesses.Count -eq 0) {
    Write-Host "No orphaned processes found." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($FoundProcesses.Count) potentially orphaned process(es):" -ForegroundColor Yellow
Write-Host ""

foreach ($proc in $FoundProcesses) {
    $cpu = [math]::Round($proc.CPU, 1)
    $mem = [math]::Round($proc.WorkingSet64 / 1MB, 0)
    Write-Host "  PID: $($proc.Id) | Name: $($proc.ProcessName) | CPU: ${cpu}s | RAM: ${mem}MB" -ForegroundColor White
    if ($proc.Path) {
        Write-Host "       Path: $($proc.Path)" -ForegroundColor DarkGray
    }
}
Write-Host ""

if ($List) {
    Write-Host "List mode - no processes killed." -ForegroundColor Cyan
    exit 0
}

if ($Force) {
    Write-Host "Force mode - killing all processes..." -ForegroundColor Red
    foreach ($proc in $FoundProcesses) {
        try {
            Stop-Process -Id $proc.Id -Force
            Write-Host "  Killed PID $($proc.Id)" -ForegroundColor Green
        } catch {
            Write-Host "  Failed to kill PID $($proc.Id): $_" -ForegroundColor Red
        }
    }
} else {
    $response = Read-Host "Kill these processes? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        foreach ($proc in $FoundProcesses) {
            try {
                Stop-Process -Id $proc.Id -Force
                Write-Host "  Killed PID $($proc.Id)" -ForegroundColor Green
            } catch {
                Write-Host "  Failed to kill PID $($proc.Id): $_" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green

