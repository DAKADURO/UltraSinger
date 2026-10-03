@echo off
:: Open the UltraSinger window (no console)
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "src\UltraSingerGui.py"
