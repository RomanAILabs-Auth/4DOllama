# Install r4d and roma4d into GOBIN / GOPATH\bin (same as: go install ./cmd/r4d ./cmd/roma4d)
Set-Location $PSScriptRoot
go install ./cmd/r4d ./cmd/roma4d
Write-Host "Installed r4d and roma4d — ensure your Go bin directory is on PATH."
