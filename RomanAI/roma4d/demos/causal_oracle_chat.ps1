#Requires -Version 5.1
<#
.SYNOPSIS
  Interactive Causal Oracle chat — Ollama (qwen2.5) with Roma4D / 4D spacetime context.

.DESCRIPTION
  Loops: you type questions about past, present, future, collisions, rotors, worldtubes,
  spacetime regions, etc. Exits on /bye, /exit, /quit, or empty line.

  Prerequisites: ollama serve, ollama pull qwen2.5

  Usage (from repo root Roma4D):
    .\demos\causal_oracle_chat.ps1
    .\demos\causal_oracle_chat.ps1 -Model llama3.2
    $env:OLLAMA_HOST = "http://127.0.0.1:11434"; .\demos\causal_oracle_chat.ps1
#>
param(
    [string] $Model = "qwen2.5",
    [string] $OllamaHost = $(if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST.TrimEnd('/') } else { "http://127.0.0.1:11434" })
)

$ErrorActionPreference = "Stop"
$chatUrl = "$OllamaHost/api/chat"

# Single-quoted here-string: no $ or @ parsing issues (e.g. value @ t, spacetime:, inner quotes).
$systemPreamble = @'
You are the Roma4D Causal Oracle - a concise expert on four-dimensional spacetime programming and geometric algebra in a research language called Roma4D.

Context you may reference when helpful:
- Native types: vec4, rotor, multivector; geometric product (*), outer (^), inner (|) on 4D objects.
- Structure-of-arrays (SoA) fields and linear ownership for high-performance layouts.
- Keyword par for structured parallel loops over columns / lists; spacetime: regions for compile-time temporal staging.
- time coordinate t and expressions like value @ t for temporal views (compile-time story today).
- timetravel_borrow(rotor) ties to MIR chrono-read / backward-cone narrative in demos.
- Simulations often use a list[vec4] worldtube evolved with rotors and par for.

Answer in clear, friendly technical prose. If the user asks about "the simulation", assume a particle / worldtube picture unless they specify otherwise. Keep answers focused; use short paragraphs or bullets for long questions.

If asked something outside 4D / Roma4D / physics intuition, answer briefly but you may still connect ideas when natural.
'@.Trim()

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "  ROMA4D CAUSAL ORACLE  (Ollama chat)" -ForegroundColor Cyan
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host "  Endpoint: $chatUrl"
Write-Host "  Model:    $Model"
Write-Host ""
Write-Host "  Type your question (past / present / future / rotors / collisions / SoA / par / spacetime)."
Write-Host "  Commands: /help  /bye  /exit  /quit  (or empty line to exit)"
Write-Host "  ============================================================" -ForegroundColor Cyan
Write-Host ""

# Probe Ollama
try {
    $null = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -Method Get -TimeoutSec 5
} catch {
    Write-Host "Cannot reach Ollama at $OllamaHost - start it with:  ollama serve" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor DarkRed
    exit 1
}

$messages = [System.Collections.ArrayList]@()
[void]$messages.Add(@{ role = "system"; content = $systemPreamble })

function Show-Help {
    Write-Host ""
    Write-Host "  Example prompts:" -ForegroundColor DarkYellow
    Write-Host "    - What does timetravel_borrow mean in the Roma4D demos?"
    Write-Host "    - How could I explain a collision at t=42 in a worldtube picture?"
    Write-Host "    - What changes if we tweak velocity before t=0?"
    Write-Host "    - Compare SoA vs AoS for a rotor swarm."
    Write-Host "    - What is a spacetime: region doing at compile time?"
    Write-Host ""
}

while ($true) {
    $line = Read-Host "you"
    if ($null -eq $line) { break }
    $line = $line.Trim()
    if ($line -eq "") { break }
    switch -Regex ($line) {
        "^/(bye|exit|quit)$" { Write-Host "Bye." -ForegroundColor Cyan; exit 0 }
        "^/help$" { Show-Help; continue }
    }

    [void]$messages.Add(@{ role = "user"; content = $line })

    $payload = [ordered]@{
        model    = $Model
        messages = @($messages.ToArray())
        stream   = $false
    }

    $json = $payload | ConvertTo-Json -Depth 12 -Compress

    try {
        $resp = Invoke-RestMethod -Uri $chatUrl -Method Post -Body $json -ContentType "application/json; charset=utf-8" -TimeoutSec 120
        $content = $resp.message.content
        if (-not $content) { $content = "(no content in response)" }
        [void]$messages.Add(@{ role = "assistant"; content = $content })
        Write-Host ""
        Write-Host "oracle> " -NoNewline -ForegroundColor Green
        Write-Host $content
        Write-Host ""
    } catch {
        Write-Host ""
        Write-Host "Request failed: $($_.Exception.Message)" -ForegroundColor Red
        if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message -ForegroundColor DarkRed }
        # Remove last user message so retry does not duplicate
        if ($messages.Count -gt 1) { [void]$messages.RemoveAt($messages.Count - 1) }
        Write-Host ""
    }
}

Write-Host "Bye." -ForegroundColor Cyan
