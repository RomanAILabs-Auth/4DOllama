# Install r4d and roma4d into GOBIN / GOPATH\bin; embeds this repo path so r4d can run .r4d files from any cwd.
Set-Location $PSScriptRoot
$embedPath = ((Resolve-Path $PSScriptRoot).Path -replace '\\', '/')
$embedX = "github.com/RomanAILabs-Auth/Roma4D/internal/cli.EmbeddedPkgRoot=$embedPath"
go install -ldflags "-X $embedX" ./cmd/r4 ./cmd/r4d ./cmd/roma4d
Write-Host "Installed r4, r4d, and roma4d — ensure your Go bin directory is on PATH."
Write-Host "Windows: for user PATH + R4D_PKG_ROOT (run r4d from any folder) run:"
Write-Host "  .\scripts\Install-R4dUserEnvironment.ps1"
