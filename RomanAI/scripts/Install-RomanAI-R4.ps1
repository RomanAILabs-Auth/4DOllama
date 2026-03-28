#Requires -Version 5.1
# Installs r4 and r4d from RomanAI\roma4d into Go bin (overwrites older builds).
$RomanAIRoot = Split-Path $PSScriptRoot -Parent
$Roma4dRoot = Join-Path $RomanAIRoot "roma4d"
if (-not (Test-Path -LiteralPath $Roma4dRoot)) { throw "Missing $Roma4dRoot" }
Push-Location $Roma4dRoot
try {
    go install ./cmd/r4 ./cmd/r4d ./cmd/romanai
    $bin = go env GOPATH
    Write-Host "Installed r4.exe, r4d.exe, and romanai.exe to: $(Join-Path $bin 'bin')" -ForegroundColor Green
    Write-Host "Ensure that folder is on PATH. romanai uses go run from this roma4d tree (not a stale r4d)." -ForegroundColor DarkGray
}
finally {
    Pop-Location
}
