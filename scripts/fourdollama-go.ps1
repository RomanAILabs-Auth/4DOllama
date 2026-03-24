# Run 4dollama via "go run" when 4dollama.exe is blocked by WDAC / AppLocker.
# Requires user env FOURD_REPO (set by install.ps1) = your 4DEngine clone path.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Remaining
)
$repo = [Environment]::GetEnvironmentVariable("FOURD_REPO", "User")
if ([string]::IsNullOrWhiteSpace($repo) -or -not (Test-Path (Join-Path $repo "go.mod"))) {
    Write-Error "FOURD_REPO is missing or invalid. Re-run scripts\install.ps1 from your 4DEngine clone."
    exit 1
}
Push-Location $repo
go run ./cmd/4dollama @Remaining
$ec = $LASTEXITCODE
Pop-Location
exit $ec
