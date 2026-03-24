# Remove 4DOllama user install (PATH, env vars, binary). Models kept unless -PurgeModels.
# Run: powershell -ExecutionPolicy Bypass -File .\scripts\uninstall.ps1
param(
    [switch]$PurgeModels,
    [switch]$PurgeData
)

$ErrorActionPreference = "Stop"
$installDir = Join-Path $env:USERPROFILE ".4dollama\bin"
$exe = Join-Path $installDir "4dollama.exe"
$target = [System.EnvironmentVariableTarget]::User

Write-Host "Stopping 4dollama (all instances, unlock binary)..."
Get-CimInstance Win32_Process -Filter "Name = '4dollama.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "  stopped 4dollama PID $($_.ProcessId)"
}
Get-CimInstance Win32_Process -Filter "Name = 'go.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.CommandLine -match '4dollama' -and $_.CommandLine -match '\bserve\b') {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "  stopped go serve PID $($_.ProcessId)"
    }
}
Start-Sleep -Milliseconds 400

if (Test-Path $exe) {
    Remove-Item -Force $exe
    Write-Host "Removed $exe"
}
$goCmd = Join-Path $installDir "fourdollama-go.cmd"
$goPs1 = Join-Path $installDir "fourdollama-go.ps1"
if (Test-Path $goCmd) { Remove-Item -Force $goCmd; Write-Host "Removed $goCmd" }
if (Test-Path $goPs1) { Remove-Item -Force $goPs1; Write-Host "Removed $goPs1" }
if (Test-Path $installDir) {
    $left = @(Get-ChildItem -Force $installDir -ErrorAction SilentlyContinue)
    if ($left.Count -eq 0) {
        Remove-Item -Force $installDir -ErrorAction SilentlyContinue
    }
}

$userPath = [Environment]::GetEnvironmentVariable("Path", $target)
if ($userPath) {
    $parts = $userPath -split ';' | Where-Object { $_ -and ($_ -ne $installDir) }
    $newPath = ($parts -join ';').TrimEnd(';')
    [Environment]::SetEnvironmentVariable("Path", $newPath, $target)
    Write-Host "Removed from user PATH: $installDir"
}

$vars = @(
    "FOURD_SHARE_OLLAMA", "FOURD_MODELS", "FOURD_DEFAULT_MODEL",
    "FOURD_PORT", "OLLAMA_HOST", "FOURD_GPU", "FOURD_INFERENCE", "FOURD_REPO"
)
foreach ($v in $vars) {
    [Environment]::SetEnvironmentVariable($v, $null, $target)
}
$om = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", $target)
$defaultOllama = Join-Path $env:USERPROFILE ".ollama\models"
if ($om -eq $defaultOllama) {
    [Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $null, $target)
    Write-Host "Cleared user OLLAMA_MODELS (was default ~/.ollama/models)."
}

if ($PurgeModels) {
    $m = Join-Path $env:USERPROFILE ".4dollama\models"
    if (Test-Path $m) {
        Remove-Item -Recurse -Force $m
        Write-Host "Removed $m"
    }
}

if ($PurgeData) {
    $d = Join-Path $env:USERPROFILE ".4dollama"
    if (Test-Path $d) {
        Remove-Item -Recurse -Force $d
        Write-Host "Removed $d"
    }
}

Write-Host ""
Write-Host "4DOllama user install removed. Open a new terminal (PATH refresh)." -ForegroundColor Cyan
Write-Host "Reinstall: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1"
