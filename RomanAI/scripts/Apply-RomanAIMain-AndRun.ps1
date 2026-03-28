#Requires -Version 5.1
# Writes src/cli/main.r4d (Pass 4 quant bridge + interactive CLI), syncs kernel copies, builds, runs once with empty stdin (exits REPL).
$ErrorActionPreference = 'Stop'
$Dest = 'C:\Users\Asus\Desktop\4DEngine\RomanAI\src\cli\main.r4d'
$Repo = 'C:\Users\Asus\Desktop\4DEngine\RomanAI'
$R4dDir = Join-Path $Repo 'roma4d'
$B64Path = Join-Path $PSScriptRoot 'main.r4d.b64.txt'
if (-not (Test-Path $B64Path)) {
    Write-Error "Missing payload: $B64Path (regenerate with: [Convert]::ToBase64String([IO.File]::ReadAllBytes('$Dest')) | Set-Content ...)"
}
$B64 = (Get-Content -LiteralPath $B64Path -Raw) -replace '\s', ''
$bytes = [Convert]::FromBase64String($B64)
[System.IO.File]::WriteAllBytes($Dest, $bytes)
Write-Host "Wrote $Dest" -ForegroundColor Green
Push-Location $Repo
try {
    .\scripts\Sync-RomanAIKernel.ps1
}
finally {
    Pop-Location
}
Push-Location $R4dDir
try {
    go run ./cmd/r4d --strict build $Dest
    $exe = Join-Path $R4dDir 'main.exe'
    if (Test-Path $exe) {
        Write-Host "Running engine (empty line ends session)..." -ForegroundColor Cyan
        '' | & $exe
    }
}
finally {
    Pop-Location
}
