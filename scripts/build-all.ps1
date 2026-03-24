$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Push-Location "$root\4d-engine"
cargo build --release
Pop-Location
$env:CGO_ENABLED = "1"
$rel = Join-Path $root "4d-engine\target\release"
if ($env:LD_LIBRARY_PATH) {
    $env:LD_LIBRARY_PATH = "$rel;$env:LD_LIBRARY_PATH"
} else {
    $env:LD_LIBRARY_PATH = $rel
}
go build -o "$root\4dollama.exe" .\cmd\4dollama
