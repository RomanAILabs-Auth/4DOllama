#Requires -Version 5.1
<#
.SYNOPSIS
  Install r4d on your Windows user PATH and set R4D_PKG_ROOT so you can run r4d from any directory.

.DESCRIPTION
  - go install ./cmd/r4 ./cmd/r4d ./cmd/roma4d
  - Adds the *actual* install directory to your *user* PATH:
      * If go env GOBIN is set (common with conda / venv e.g. quantum_win\Scripts), that folder is used.
      * Otherwise %GOPATH%\bin is used.
  - Set user environment variable R4D_PKG_ROOT to this repo root (folder containing roma4d.toml)
  - Optionally set R4D_GNU_ROOT when MSYS2 MinGW is present (Clang fallback only)

  Default Windows linker: Zig (`zig cc` compiles .ll + links rt/roma4d_rt.c; no separate MinGW install).
  Install: winget install Zig.Zig  — or https://ziglang.org/download/ — ensure zig.exe is on PATH (or set R4D_ZIG).
  If Zig is missing, Roma4D falls back to LLVM clang + MinGW (R4D_GNU_ROOT / MSYS2).

  After running, open a NEW PowerShell (or VS Code terminal) so PATH and env vars reload.

.EXAMPLE
  cd C:\path\to\4DEngine\roma4d
  .\scripts\Install-R4dUserEnvironment.ps1
#>
$ErrorActionPreference = "Stop"

$romaRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$toml = Join-Path $romaRoot "roma4d.toml"
if (-not (Test-Path $toml)) {
    Write-Error "Expected roma4d.toml at $romaRoot - run this script from the Roma4D repo (scripts lives under roma4d\scripts)."
}

Set-Location $romaRoot
Write-Host "Building r4 / r4d / roma4d from $romaRoot" -ForegroundColor Cyan
# Embed repo path in the binaries so `r4d C:\anywhere\file.r4d` works without cd into this clone.
$embedPath = ($romaRoot -replace '\\', '/')
$embedX = "github.com/RomanAILabs-Auth/Roma4D/internal/cli.EmbeddedPkgRoot=$embedPath"
go install -ldflags "-X $embedX" ./cmd/r4 ./cmd/r4d ./cmd/roma4d

$gopath = (go env GOPATH).Trim()
if ([string]::IsNullOrWhiteSpace($gopath)) {
    Write-Error "go env GOPATH is empty"
}
$gobinEnv = (go env GOBIN).Trim()
if ([string]::IsNullOrWhiteSpace($gobinEnv)) {
    $installBin = Join-Path $gopath "bin"
} else {
    $installBin = [System.IO.Path]::GetFullPath($gobinEnv)
}
if (-not (Test-Path $installBin)) {
    New-Item -ItemType Directory -Path $installBin -Force | Out-Null
}
$r4dInstalled = Join-Path $installBin "r4d.exe"
if (Test-Path $r4dInstalled) {
    Write-Host "Installed r4d.exe -> $r4dInstalled" -ForegroundColor Green
} else {
    Write-Warning "r4d.exe not found at $r4dInstalled - run from repo root and check go env GOBIN / GOPATH."
}

function Normalize-Dir {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return "" }
    $t = $Path.Trim()
    try {
        $full = [System.IO.Path]::GetFullPath($t)
    } catch {
        $full = $t
    }
    return $full.TrimEnd([char[]]@('\', '/'))
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($null -eq $userPath) { $userPath = "" }

$installBinNorm = Normalize-Dir $installBin
$already = $false
foreach ($segment in ($userPath -split ";")) {
    $sn = Normalize-Dir $segment
    if ($sn -ne "" -and $sn -ieq $installBinNorm) {
        $already = $true
        break
    }
}

if (-not $already) {
    if ($userPath -eq "") {
        $newPath = $installBin
    } else {
        $newPath = $userPath + ";" + $installBin
    }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Appended user PATH: $installBin" -ForegroundColor Green
    if (-not [string]::IsNullOrWhiteSpace($gobinEnv)) {
        Write-Host "(Using go env GOBIN - conda/venv installs go here, not GOPATH\bin.)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "User PATH already contains: $installBin" -ForegroundColor DarkGray
}

[Environment]::SetEnvironmentVariable("R4D_PKG_ROOT", $romaRoot, "User")
Write-Host "Set user R4D_PKG_ROOT=$romaRoot" -ForegroundColor Green

# MinGW prefix for Clang (-target *-windows-gnu): adds --gcc-toolchain + GCC builtin includes (mm_malloc.h).
$mingwCandidates = @(
    "C:\msys64\ucrt64",
    "C:\msys64\mingw64",
    "C:\msys64\clang64"
)
foreach ($m in $mingwCandidates) {
    $libGcc = Join-Path $m "lib\gcc"
    if (Test-Path $libGcc) {
        [Environment]::SetEnvironmentVariable("R4D_GNU_ROOT", $m, "User")
        Write-Host "Set user R4D_GNU_ROOT=$m (Clang + MinGW headers)" -ForegroundColor Green
        break
    }
}

$demoPath = Join-Path $romaRoot "demos\cosmic_genesis.r4d"
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  SETUP FINISHED - you only need to remember ONE thing:" -ForegroundColor Green
Write-Host ""
Write-Host "      r4d  path\to\yourfile.r4d" -ForegroundColor White
Write-Host ""
Write-Host "  Examples:" -ForegroundColor Green
Write-Host ('      r4d "' + $demoPath + '"') -ForegroundColor White
Write-Host "      r4d myfile.r4d                    (when you are already in that folder)" -ForegroundColor White
Write-Host ""
Write-Host "  Close this window. Open a BRAND NEW PowerShell, then type the line above." -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
if (-not (Get-Command zig -ErrorAction SilentlyContinue)) {
    Write-Host "Optional: install Zig for easier Windows builds: https://ziglang.org/download/" -ForegroundColor DarkYellow
    Write-Host ""
}
