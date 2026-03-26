# Save-4DOllamaWorkspace.ps1
# Copyright RomanAILabs - Daniel Harding
# Christ is King.
#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$Source = $PSScriptRoot
$Dest = "C:\Users\Asus\Desktop\RomanAILabs\4DOllama"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Copy-Item -Path (Join-Path $Source "*") -Destination $Dest -Recurse -Force
Write-Host "Copied 4DOllama -> $Dest" -ForegroundColor Green
