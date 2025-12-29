# Checkpoint Monitor
# ==================
# Run this in a separate terminal to watch for new checkpoints
#
# Shows:
# - All saved checkpoints
# - Time since last checkpoint
# - Estimated progress

param(
    [string]$Stage = "sft",
    [int]$IntervalSeconds = 30
)

# Get the repo root (parent of training folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

$outputDir = Join-Path $RepoRoot "saves\vgpt2_v3\$Stage"

Write-Host "Checkpoint Monitor - $Stage" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"
Write-Host "Watching: $outputDir"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

$lastCheckpoint = $null

while ($true) {
    Clear-Host
    Write-Host "Checkpoint Monitor - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "=============================================" -ForegroundColor Cyan
    Write-Host ""

    if (-not (Test-Path $outputDir)) {
        Write-Host "Output directory not yet created..." -ForegroundColor Yellow
        Write-Host "Training may not have started."
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    
    $checkpoints = Get-ChildItem -Path $outputDir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue | 
                   Sort-Object { [int]($_.Name -replace "checkpoint-", "") }
    
    if ($checkpoints) {
        Write-Host "Checkpoints found:" -ForegroundColor Green
        Write-Host ""
        
        foreach ($cp in $checkpoints) {
            $step = [int]($cp.Name -replace "checkpoint-", "")
            $time = $cp.LastWriteTime
            $age = [math]::Round(((Get-Date) - $time).TotalMinutes, 1)
            
            $sizeBytes = (Get-ChildItem -Path $cp.FullName -Recurse | Measure-Object -Property Length -Sum).Sum
            $sizeMB = [math]::Round($sizeBytes / 1MB, 1)
            
            Write-Host "  $($cp.Name) - $sizeMB MB - $age min ago" -ForegroundColor White
        }
        
        $latest = $checkpoints[-1]
        $latestStep = [int]($latest.Name -replace "checkpoint-", "")
        
        # Estimate progress (assuming ~12,800 total steps for SFT)
        $totalSteps = switch ($Stage) {
            "sft" { 12800 }
            "dpo" { 180 }
            "kto" { 90 }
            default { 12800 }
        }
        
        $progress = [math]::Round(($latestStep / $totalSteps) * 100, 1)
        
        Write-Host ""
        Write-Host "Latest: checkpoint-$latestStep" -ForegroundColor Cyan
        Write-Host "Progress: ~$progress% ($latestStep / $totalSteps steps)" -ForegroundColor Cyan
        
        if ($latestStep -ne $lastCheckpoint) {
            Write-Host ""
            Write-Host "NEW CHECKPOINT SAVED!" -ForegroundColor Green
            $lastCheckpoint = $latestStep
        }
    } else {
        Write-Host "No checkpoints yet..." -ForegroundColor Yellow
        Write-Host "First checkpoint expected at step 500"
    }
    
    # Check for trainer_state.json for more accurate progress
    $trainerState = Join-Path $outputDir "trainer_state.json"
    if (Test-Path $trainerState) {
        $state = Get-Content $trainerState | ConvertFrom-Json
        if ($state.global_step) {
            Write-Host ""
            Write-Host "Current step: $($state.global_step)" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "Refreshing every $IntervalSeconds seconds..." -ForegroundColor DarkGray
    
    Start-Sleep -Seconds $IntervalSeconds
}

