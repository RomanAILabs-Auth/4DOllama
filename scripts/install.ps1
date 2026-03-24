# One-click install: Rust four_d_engine + Go CLI, CPU/GPU env, optional qwen2.5 pull, background serve.
# Run from repo root: powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
param(
    [switch]$InstallGo,
    [switch]$SkipCargo,
    [switch]$SkipBackup
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path (Join-Path $root "go.mod"))) {
    Write-Error "go.mod not found - run from the repo root."
}

if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    if ($InstallGo -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        winget install -e --id GoLang.Go --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    }
}
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Error "Go not on PATH. Install Go or run: install.ps1 -InstallGo"
}
$script:GoExe = (Get-Command go).Source

$manifest = Join-Path $root "4d-engine\Cargo.toml"
$cargoOk = $false
if ($SkipCargo) {
    Write-Host "SkipCargo: skipping Rust build (use after installing VS Build Tools + C++, or to save time)."
} elseif (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host "cargo build --release --manifest-path $manifest"
    cargo build --release --manifest-path $manifest
    if ($LASTEXITCODE -eq 0) {
        $cargoOk = $true
    } else {
        Write-Warning "Rust release build failed (install Visual Studio Build Tools with C++ for link.exe, or use GNU Rust). Continuing with CGO_ENABLED=0."
    }
} else {
    Write-Warning "cargo not found - skipping Rust build (Go will use CGO_ENABLED=0)."
}

$env:CGO_ENABLED = if ($cargoOk) { "1" } else { "0" }
$buildExe = Join-Path $root "4dollama.exe"
Write-Host "go build -o 4dollama.exe ./cmd/4dollama (CGO_ENABLED=$($env:CGO_ENABLED))"
& $script:GoExe build -o $buildExe .\cmd\4dollama
if ($LASTEXITCODE -ne 0 -and $env:CGO_ENABLED -eq "1") {
    Write-Warning "CGO build failed - retrying with CGO_ENABLED=0"
    $env:CGO_ENABLED = "0"
    & $script:GoExe build -o $buildExe .\cmd\4dollama
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Unblock-File -Path $buildExe -ErrorAction SilentlyContinue

Write-Host "Stopping any previous 4dollama (unlock binary for copy)..."
Get-CimInstance Win32_Process -Filter "Name = '4dollama.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 800

$installDir = Join-Path $env:USERPROFILE ".4dollama\bin"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
$destExe = Join-Path $installDir "4dollama.exe"
$script:destExe = $destExe

if (-not $SkipBackup) {
    $backupFiles = @("4dollama.exe", "fourdollama-go.cmd", "fourdollama-go.ps1")
    $anyPrev = $false
    foreach ($f in $backupFiles) {
        if (Test-Path (Join-Path $installDir $f)) { $anyPrev = $true; break }
    }
    if ($anyPrev) {
        $backupRoot = Join-Path $env:USERPROFILE ".4dollama\backup"
        New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupDir = Join-Path $backupRoot $stamp
        New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
        foreach ($f in $backupFiles) {
            $p = Join-Path $installDir $f
            if (Test-Path $p) {
                Copy-Item -LiteralPath $p -Destination (Join-Path $backupDir $f) -Force
            }
        }
        Write-Host "Backed up previous install to: $backupDir" -ForegroundColor DarkCyan
    }
}

$copied = $false
for ($try = 0; $try -lt 6; $try++) {
    try {
        Copy-Item -Force $buildExe $destExe -ErrorAction Stop
        $copied = $true
        break
    } catch {
        if ($try -lt 5) {
            Get-CimInstance Win32_Process -Filter "Name = '4dollama.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
            Start-Sleep -Milliseconds (400 + $try * 200)
        } else {
            throw
        }
    }
}
if (-not $copied) { exit 1 }
Unblock-File -Path "$env:USERPROFILE\.4dollama\bin\4dollama.exe" -ErrorAction SilentlyContinue

Copy-Item -Force (Join-Path $PSScriptRoot "fourdollama-go.cmd") (Join-Path $installDir "fourdollama-go.cmd")
Copy-Item -Force (Join-Path $PSScriptRoot "fourdollama-go.ps1") (Join-Path $installDir "fourdollama-go.ps1")
Unblock-File -Path (Join-Path $installDir "fourdollama-go.cmd") -ErrorAction SilentlyContinue
Unblock-File -Path (Join-Path $installDir "fourdollama-go.ps1") -ErrorAction SilentlyContinue

$target = [System.EnvironmentVariableTarget]::User
$userPath = [Environment]::GetEnvironmentVariable("Path", $target)
if ($userPath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$installDir;$userPath", $target)
    Write-Host "Added to user PATH: $installDir"
}

$ollamaModels = Join-Path $env:USERPROFILE ".ollama\models"
$modelsDir = Join-Path $env:USERPROFILE ".4dollama\models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

[Environment]::SetEnvironmentVariable("FOURD_SHARE_OLLAMA", "true", $target)
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", $ollamaModels, $target)
[Environment]::SetEnvironmentVariable("FOURD_MODELS", $modelsDir, $target)
[Environment]::SetEnvironmentVariable("FOURD_DEFAULT_MODEL", "qwen2.5", $target)
[Environment]::SetEnvironmentVariable("FOURD_PORT", "13373", $target)
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "http://127.0.0.1:11434", $target)
[Environment]::SetEnvironmentVariable("FOURD_INFERENCE", "ollama", $target)
[Environment]::SetEnvironmentVariable("FOURD_REPO", $root, $target)

$hasAccel = $false
if ($env:CUDA_PATH) { $hasAccel = $true }
try {
    $vc = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    foreach ($v in $vc) {
        if ($v.Name -match "NVIDIA|AMD Radeon") { $hasAccel = $true }
    }
} catch {}

if (-not $hasAccel) {
    [Environment]::SetEnvironmentVariable("FOURD_GPU", "cpu", $target)
    $env:FOURD_GPU = "cpu"
    Write-Host "FOURD_GPU=cpu (no discrete GPU / CUDA path detected - full 4D stack on CPU)."
} else {
    [Environment]::SetEnvironmentVariable("FOURD_GPU", $null, $target)
    Remove-Item Env:\FOURD_GPU -ErrorAction SilentlyContinue
    Write-Host "GPU-friendly host detected - FOURD_GPU unset (use FOURD_GPU=cpu to force CPU-only)."
}

# Keep Machine PATH (e.g. Program Files\Go\bin) — do not replace session PATH with User-only.
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$freshUserPath = [Environment]::GetEnvironmentVariable("Path", $target)
$env:Path = "$installDir;$machinePath;$freshUserPath"
$env:FOURD_MODELS = $modelsDir
$env:OLLAMA_MODELS = $ollamaModels
$env:FOURD_SHARE_OLLAMA = "true"
$env:FOURD_PORT = "13373"
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:FOURD_INFERENCE = "ollama"
$env:FOURD_REPO = $root

function script:Invoke-FourdMain {
    param([string[]]$CommandArgs)
    $saveEA = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $script:destExe @CommandArgs 2>&1 | ForEach-Object { Write-Host $_ }
        return
    } catch {
        Write-Host "Application Control blocked 4dollama.exe; using go run from FOURD_REPO=$script:root" -ForegroundColor Cyan
    } finally {
        $ErrorActionPreference = $saveEA
    }
    Push-Location $script:root
    try {
        & $script:GoExe run ./cmd/4dollama @CommandArgs
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Pulling qwen2.5 (best-effort; requires network)..."
Invoke-FourdMain -CommandArgs @("pull", "qwen2.5")

Write-Host "Starting 4dollama serve in background..."
$p = $null
try {
    $p = Start-Process -FilePath $destExe -ArgumentList "serve" -WindowStyle Hidden -PassThru -ErrorAction Stop
} catch {
    Write-Host "exe blocked; starting serve via go run ..." -ForegroundColor Cyan
    try {
        $p = Start-Process -FilePath $script:GoExe -ArgumentList @("run", "./cmd/4dollama", "serve") -WorkingDirectory $root -WindowStyle Hidden -PassThru
    } catch {
        Write-Warning "Could not start serve in background: $($_.Exception.Message)"
        Write-Host "  Run: fourdollama-go.cmd serve   OR   fourdollama-go.ps1 serve"
    }
}
$base = "http://127.0.0.1:13373"
$ok = $false
if ($null -ne $p) {
    for ($i = 0; $i -lt 80; $i++) {
        Start-Sleep -Milliseconds 150
        try {
            $r = Invoke-WebRequest -Uri "$base/healthz" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch {}
    }
}
if (-not $ok) {
    Write-Warning "Serve did not respond on $base yet - run: 4dollama serve   OR   fourdollama-go.cmd serve"
} else {
    Write-Host "Serve is up on $base (PID $($p.Id))."
}

Write-Host ""
Write-Host "4DOllama is ready! Works on CPU or GPU." -ForegroundColor Green
Write-Host "  Normal: 4dollama run qwen2.5" -ForegroundColor Green
Write-Host "  If Application Control blocks 4dollama.exe, use (on PATH):" -ForegroundColor Yellow
Write-Host "    fourdollama-go.cmd run qwen2.5" -ForegroundColor Yellow
Write-Host "    fourdollama-go.ps1 run qwen2.5" -ForegroundColor Yellow
Write-Host "  FOURD_REPO=$root" -ForegroundColor Gray
Write-Host "  doctor: 4dollama doctor  - open a new terminal if PATH was just updated."
Write-Host ""
Invoke-FourdMain -CommandArgs @("version")
