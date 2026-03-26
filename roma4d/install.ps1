# Install r4d and roma4d into GOBIN / GOPATH\bin (same as: go install ./cmd/r4d ./cmd/roma4d)
Set-Location $PSScriptRoot
go install ./cmd/r4d ./cmd/roma4d
Write-Host "Installed r4d and roma4d — ensure your Go bin directory is on PATH."
Write-Host "Windows: for user PATH + R4D_PKG_ROOT (run r4d from any folder) run:"
Write-Host "  .\scripts\Install-R4dUserEnvironment.ps1"
