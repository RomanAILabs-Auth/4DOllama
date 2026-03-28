#Requires -Version 5.1
# RomanAI host launcher: always uses bundled RomanAI\roma4d (ignores stale PATH r4d).
param(
    [Parameter(Position = 0)][string]$Command = '',
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
$RomanAIRoot = Split-Path $PSScriptRoot -Parent
$Roma4dRoot = Join-Path $RomanAIRoot 'roma4d'
$Kernel = Join-Path $RomanAIRoot 'r4d\romanai_main.r4d'

if (-not (Test-Path -LiteralPath $Roma4dRoot)) {
    Write-Error "Missing roma4d toolchain: $Roma4dRoot"
}
if (-not (Test-Path -LiteralPath $Kernel)) {
    Write-Error "Kernel missing: $Kernel - run .\scripts\Sync-RomanAIKernel.ps1"
}

$userCwd = (Get-Location).Path

function Invoke-R4dOnKernel {
    param(
        [string]$CliModel = '',
        [string]$Gguf = '',
        [string]$Prompt = ''
    )
    if ($CliModel -ne '') {
        [Environment]::SetEnvironmentVariable('ROMANAI_CLI_MODEL', $CliModel, 'Process')
        $env:ROMANAI_CLI_MODEL = $CliModel
    }
    if ($Gguf -ne '') {
        [Environment]::SetEnvironmentVariable('ROMANAI_GGUF', $Gguf, 'Process')
        $env:ROMANAI_GGUF = $Gguf
    }
    [Environment]::SetEnvironmentVariable('ROMANAI_PROMPT', $Prompt, 'Process')
    $env:ROMANAI_PROMPT = $Prompt

    Push-Location $Roma4dRoot
    try {
        $kAbs = (Resolve-Path -LiteralPath $Kernel).Path
        & go run ./cmd/r4d run $kAbs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

switch -Regex ($Command) {
    '^run$' {
        $modelArg = if ($Rest.Count -ge 1) { $Rest[0] } else { '' }
        if (-not $modelArg) {
            Write-Error 'Usage: .\scripts\romanai.ps1 run path\to\model.gguf [prompt words...]'
        }
        $promptText = 'run'
        if ($Rest.Count -ge 2) {
            $promptText = ($Rest[1..($Rest.Count - 1)] -join ' ')
        }
        $full = $modelArg
        if (-not [System.IO.Path]::IsPathRooted($full)) {
            $full = Join-Path $userCwd $full
        }
        $full = [System.IO.Path]::GetFullPath($full)
        if (-not (Test-Path -LiteralPath $full)) {
            Write-Error "romanai run: GGUF not found: $full"
        }
        $ext = [System.IO.Path]::GetExtension($full).ToLowerInvariant()
        if ($ext -ne '.gguf') {
            Write-Warning "Expected .gguf extension (got $ext). Check the path."
        }
        $env:R4D_EXPERT_INTERACTIVE = '0'
        Invoke-R4dOnKernel -CliModel $full -Gguf $full -Prompt $promptText
    }
    '^chat$' {
        $env:R4D_EXPERT_INTERACTIVE = '0'
        Invoke-R4dOnKernel -Prompt 'chat'
    }
    '^version$' {
        $env:R4D_EXPERT_INTERACTIVE = '0'
        Invoke-R4dOnKernel -Prompt 'version'
    }
    '^list$' {
        $env:R4D_EXPERT_INTERACTIVE = '0'
        Invoke-R4dOnKernel -Prompt 'list'
    }
    default {
        Write-Host 'RomanAI launcher (bundled roma4d)'
        Write-Host ''
        Write-Host '  .\scripts\romanai.ps1 run  path\to\model.gguf [prompt...]'
        Write-Host '  .\scripts\romanai.ps1 chat'
        Write-Host '  .\scripts\romanai.ps1 version'
        Write-Host '  .\scripts\romanai.ps1 list'
        Write-Host ''
        Write-Host 'Or from RomanAI folder: .\romanai.cmd run ...'
        Write-Host ''
        Write-Host 'Optional: install r4d to Go bin:'
        Write-Host ('  cd ' + [char]34 + $Roma4dRoot + [char]34)
        Write-Host '  go install ./cmd/r4d ./cmd/r4'
        Write-Host ''
        exit 1
    }
}
