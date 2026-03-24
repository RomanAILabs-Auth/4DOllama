# Rebuild 4dollama from source and copy to %USERPROFILE%\.4dollama\bin (same target as install.ps1).
# Stops a running `4dollama serve` first so the exe is not locked.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\rebuild-install.ps1
param(
    [switch]$NoStopServe,
    [switch]$TryStartServe
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$destDir = Join-Path $env:USERPROFILE ".4dollama\bin"
$destExe = Join-Path $destDir "4dollama.exe"

if (-not $NoStopServe) {
    Get-CimInstance Win32_Process -Filter "Name = '4dollama.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.CommandLine -match '\bserve\b') {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped 4dollama serve PID $($_.ProcessId)"
        }
    }
    Start-Sleep -Milliseconds 500
}

if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Error "Go not on PATH."
}

$env:CGO_ENABLED = if ($env:CGO_ENABLED) { $env:CGO_ENABLED } else { "0" }
Write-Host "go build (CGO_ENABLED=$($env:CGO_ENABLED)) ..."
go build -o 4dollama.exe ./cmd/4dollama
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $destDir | Out-Null
Copy-Item -Force (Join-Path $root "4dollama.exe") $destExe
Unblock-File -Path $destExe -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $PSScriptRoot "fourdollama-go.cmd") (Join-Path $destDir "fourdollama-go.cmd")
Copy-Item -Force (Join-Path $PSScriptRoot "fourdollama-go.ps1") (Join-Path $destDir "fourdollama-go.ps1")
Unblock-File -Path (Join-Path $destDir "fourdollama-go.cmd") -ErrorAction SilentlyContinue
Unblock-File -Path (Join-Path $destDir "fourdollama-go.ps1") -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable("FOURD_REPO", $root, "User")
$env:FOURD_REPO = $root
Write-Host "Installed: $destExe (+ fourdollama-go.*, FOURD_REPO updated)"
$ErrorActionPreference = "Continue"
try {
    & $destExe version
} catch {
    Write-Host "exe blocked; version via go run:" -ForegroundColor Cyan
    Push-Location $root
    go run ./cmd/4dollama version
    Pop-Location
}
$ErrorActionPreference = "Stop"

if ($TryStartServe) {
    Write-Host "Starting serve in background..."
    try {
        Start-Process -FilePath $destExe -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop
    } catch {
        try {
            $goExe = (Get-Command go).Source
            Start-Process -FilePath $goExe -ArgumentList @("run", "./cmd/4dollama", "serve") -WorkingDirectory $root -WindowStyle Hidden
        } catch {
            Write-Warning "Could not start serve: $($_.Exception.Message)"
            Write-Host "Run: fourdollama-go.cmd serve"
        }
    }
} else {
    Write-Host "Start API: 4dollama serve   OR   fourdollama-go.cmd serve"
}
