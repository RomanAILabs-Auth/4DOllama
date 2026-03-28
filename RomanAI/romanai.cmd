@echo off
REM Use bundled RomanAI\roma4d (avoids stale global r4d on PATH).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\romanai.ps1" %*
