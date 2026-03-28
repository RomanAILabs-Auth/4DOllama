@echo off
REM Double-click or: r4d.cmd myfile.r4d   — same as r4d.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0r4d.ps1" %*
