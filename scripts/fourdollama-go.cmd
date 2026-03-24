@echo off
setlocal EnableExtensions
if "%FOURD_REPO%"=="" (
  echo FOURD_REPO is not set. Run install.ps1 from your 4DEngine folder ^(it records the clone path^).
  exit /b 1
)
if not exist "%FOURD_REPO%\go.mod" (
  echo FOURD_REPO points to a folder without go.mod: %FOURD_REPO%
  echo Move your clone or set FOURD_REPO to the correct 4DEngine path, then try again.
  exit /b 1
)
pushd "%FOURD_REPO%"
go run ./cmd/4dollama %*
set "EC=%ERRORLEVEL%"
popd
exit /b %EC%
