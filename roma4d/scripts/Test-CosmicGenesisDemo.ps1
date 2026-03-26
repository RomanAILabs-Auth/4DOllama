#Requires -Version 5.1
<#
.SYNOPSIS
  Regression test: ensure demos/cosmic_genesis.r4d can be written with absolute paths and r4d run succeeds.

.EXAMPLE
  cd C:\path\to\4DEngine\roma4d
  .\scripts\Test-CosmicGenesisDemo.ps1
#>
$ErrorActionPreference = "Stop"

$romaRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$demosPath = Join-Path $romaRoot "demos"
$outFile = Join-Path $demosPath "cosmic_genesis.r4d"

if (!(Test-Path $demosPath)) {
    New-Item -ItemType Directory -Path $demosPath -Force | Out-Null
    Write-Host "Created demos folder" -ForegroundColor Green
}

if (!(Test-Path $outFile)) {
    Write-Error "Missing $outFile (expected committed demo in repo)."
}

# Same as WriteAllText(..., UTF-8): round-trip through absolute path to catch cwd/BOM issues.
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$body = [System.IO.File]::ReadAllText($outFile, $utf8NoBom)
[System.IO.File]::WriteAllText($outFile, $body, $utf8NoBom)

Write-Host "Round-tripped UTF-8 (no BOM): $outFile" -ForegroundColor Cyan

Set-Location $romaRoot
$r4d = Get-Command r4d -ErrorAction SilentlyContinue
if (-not $r4d) {
    Write-Error "r4d not on PATH. Run .\scripts\Install-R4dUserEnvironment.ps1 and open a new terminal."
}
& r4d run "demos\cosmic_genesis.r4d"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "Test-CosmicGenesisDemo: OK" -ForegroundColor Green
