# GPU Monitoring Script
# =====================
# Run this in a separate terminal to monitor GPU during training
#
# Updates every 5 seconds with:
# - GPU utilization
# - Memory usage
# - Temperature
# - Power draw

param(
    [int]$IntervalSeconds = 5
)

# Get the repo root (parent of training folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host "GPU Monitor - Press Ctrl+C to stop" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"
Write-Host ""

while ($true) {
    Clear-Host
    Write-Host "GPU Monitor - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Get GPU stats
    $gpuInfo = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    $parts = $gpuInfo -split ","
    
    $name = $parts[0].Trim()
    $util = [int]$parts[1].Trim()
    $memUsed = [int]$parts[2].Trim()
    $memTotal = [int]$parts[3].Trim()
    $temp = [int]$parts[4].Trim()
    $power = [float]$parts[5].Trim()
    
    $memPercent = [math]::Round(($memUsed / $memTotal) * 100, 1)
    
    Write-Host "GPU: $name" -ForegroundColor White
    Write-Host ""
    
    # Utilization bar
    $utilBar = ("█" * [math]::Floor($util / 5)) + ("░" * (20 - [math]::Floor($util / 5)))
    $utilColor = if ($util -gt 80) { "Green" } elseif ($util -gt 50) { "Yellow" } else { "Red" }
    Write-Host "Utilization: [$utilBar] $util%" -ForegroundColor $utilColor
    
    # Memory bar
    $memBar = ("█" * [math]::Floor($memPercent / 5)) + ("░" * (20 - [math]::Floor($memPercent / 5)))
    $memColor = if ($memPercent -lt 90) { "Green" } elseif ($memPercent -lt 95) { "Yellow" } else { "Red" }
    Write-Host "Memory:      [$memBar] $memUsed / $memTotal MiB ($memPercent%)" -ForegroundColor $memColor
    
    # Temperature
    $tempColor = if ($temp -lt 70) { "Green" } elseif ($temp -lt 80) { "Yellow" } else { "Red" }
    Write-Host "Temperature: $temp°C" -ForegroundColor $tempColor
    
    # Power
    Write-Host "Power Draw:  $power W" -ForegroundColor White
    
    Write-Host ""
    Write-Host "Refreshing every $IntervalSeconds seconds..." -ForegroundColor DarkGray
    
    Start-Sleep -Seconds $IntervalSeconds
}

