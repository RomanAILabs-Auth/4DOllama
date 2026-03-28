#Requires -Version 5.1
# Canonical entry: src/cli/main.r4d -> romanai.r4d + r4d/romanai_main.r4d (host `romanai run`).
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$src = Join-Path $root "src\cli\main.r4d"
$rom = Join-Path $root "romanai.r4d"
$dstDir = Join-Path $root "r4d"
$dst = Join-Path $dstDir "romanai_main.r4d"
if (-not (Test-Path $src)) { Write-Error "Missing $src" }
Copy-Item -Path $src -Destination $rom -Force
New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
$body = Get-Content -Path $src -Raw
$header = "# RomanAI kernel: r4d/romanai_main.r4d (synced from src/cli/main.r4d via scripts\Sync-RomanAIKernel.ps1)`r`n`r`n"
Set-Content -Path $dst -Value ($header + $body) -NoNewline
Write-Host "Wrote $rom and $dst" -ForegroundColor Green
