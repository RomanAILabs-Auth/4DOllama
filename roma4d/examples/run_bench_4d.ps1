# Run Roma4D + Python + Rust informal benchmarks from the roma4d repo root.
# Usage (from repo root):  .\examples\run_bench_4d.ps1

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== r4d (Bench_4d.r4d) ===" -ForegroundColor Cyan
r4d run examples\Bench_4d.r4d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== python (bench_4d.py) ===" -ForegroundColor Cyan
python examples\bench_4d.py

Write-Host "`n=== rust (bench_4d.rs) ===" -ForegroundColor Cyan
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "rustc not on PATH; install Rust from https://rustup.rs" -ForegroundColor Yellow
    exit 0
}

$out = Join-Path $env:TEMP "bench_4d_rust.exe"
Remove-Item $out -ErrorAction SilentlyContinue

# Default host (MSVC on typical Windows rustup install)
$rustLog = Join-Path $env:TEMP "bench_4d_rustc.log"
& rustc examples\bench_4d.rs -O -o $out 2>&1 | Tee-Object -FilePath $rustLog
if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) {
    & $out
    exit 0
}

# MinGW target (matches Roma4D's clang -target gnu); requires stdlib for that triple
Write-Host "Default rustc failed (see $rustLog). Trying x86_64-pc-windows-gnu ..." -ForegroundColor DarkYellow
& rustc examples\bench_4d.rs -O -o $out --target x86_64-pc-windows-gnu 2>&1 | Tee-Object -FilePath $rustLog
if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) {
    & $out
    exit 0
}

Write-Host @"

Rust build still failed. Do ONE of the following:

  [GNU / matches Roma4D + MinGW]
    rustup target add x86_64-pc-windows-gnu
    rustc examples\bench_4d.rs -O -o `"$out`" --target x86_64-pc-windows-gnu
    & `"$out`"

  [MSVC]
    Install 'Visual Studio Build Tools' with the C++ workload (provides link.exe), then:
    rustc examples\bench_4d.rs -O -o `"$out`"
    & `"$out`"

Last rustc output saved to: $rustLog
"@ -ForegroundColor Yellow

exit 1
