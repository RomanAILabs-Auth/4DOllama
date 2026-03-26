#Requires -Version 5.1
<#
.SYNOPSIS
  Install r4d on your Windows user PATH and set R4D_PKG_ROOT so you can run r4d from any directory.

.DESCRIPTION
  - go install ./cmd/r4d ./cmd/roma4d into %GOPATH%\bin
  - Append that bin directory to your *user* PATH if missing
  - Set user environment variable R4D_PKG_ROOT to this repo root (folder containing roma4d.toml)

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
go install ./cmd/r4 ./cmd/r4d ./cmd/roma4d

$gopath = (go env GOPATH).Trim()
if ([string]::IsNullOrWhiteSpace($gopath)) {
    Write-Error "go env GOPATH is empty"
}
$goBin = Join-Path $gopath "bin"
if (-not (Test-Path $goBin)) {
    New-Item -ItemType Directory -Path $goBin -Force | Out-Null
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

$goBinNorm = Normalize-Dir $goBin
$already = $false
foreach ($segment in ($userPath -split ";")) {
    $sn = Normalize-Dir $segment
    if ($sn -ne "" -and $sn -ieq $goBinNorm) {
        $already = $true
        break
    }
}

if (-not $already) {
    $newPath = if ($userPath -eq "") { $goBin } else { "$userPath;$goBin" }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Appended user PATH: $goBin" -ForegroundColor Green
} else {
    Write-Host "User PATH already contains: $goBin" -ForegroundColor DarkGray
}

[Environment]::SetEnvironmentVariable("R4D_PKG_ROOT", $romaRoot, "User")
Write-Host "Set user R4D_PKG_ROOT=$romaRoot" -ForegroundColor Green

$demoPath = Join-Path $romaRoot "demos\cosmic_genesis.r4s"
Write-Host ""
Write-Host "Done. Open a NEW terminal, then:" -ForegroundColor Yellow
Write-Host "  r4 version"
Write-Host ('  r4 run ' + $demoPath)
Write-Host ""
Write-Host "From repo folder: cd ...\roma4d ; r4 run demos\cosmic_genesis.r4s"
Write-Host "You can keep .r4s files anywhere; imports resolve against R4D_PKG_ROOT. (.roma4d still works.)"
Write-Host ""
