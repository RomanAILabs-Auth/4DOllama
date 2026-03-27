@echo off
REM Double-click or run from repo root: installs Roma4D (r4d). See roma4d\docs\Install_Guide.md
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Install-Roma4d.ps1"
if errorlevel 1 (
  echo.
  echo Install failed. See roma4d\docs\Install_Guide.md section 4.
  pause
  exit /b 1
)
echo.
echo Success. Close this window, open a NEW terminal, then run: r4d version
pause
