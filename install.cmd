@echo off
REM 4DEngine — installs 4DOllama (4dollama CLI + serve) then Roma4D (r4d).
REM Double-click this file from the repo root, or run: install.cmd
cd /d "%~dp0"
echo.
echo  4DEngine installer: 4DOllama + Roma4D
echo  Full guide: docs\INSTALL_4DOLLAMA.md
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Install-Repo.ps1" %*
if errorlevel 1 (
  echo.
  echo Install failed. See docs\INSTALL_4DOLLAMA.md
  pause
  exit /b 1
)
echo.
echo Success. Close this window, open a NEW terminal, then:
echo   4dollama doctor
echo   4dollama run qwen2.5
echo   (Roma4D)  cd roma4d ^&^& r4d version
pause
