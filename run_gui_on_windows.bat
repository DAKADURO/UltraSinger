@echo off
:: Open the UltraSinger window (no console)
cd /d "%~dp0"

:: The window only needs tkinter. Start the base python directly: the launcher of the virtual
:: environment cannot find it on some systems ("No Python at ...").
set "PYHOME="
for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"home" ".venv\pyvenv.cfg"') do set "PYHOME=%%b"
if defined PYHOME set "PYHOME=%PYHOME:~1%"
if defined PYHOME if exist "%PYHOME%\pythonw.exe" (
    start "" "%PYHOME%\pythonw.exe" "src\UltraSingerGui.py"
    exit /b
)
start "" ".venv\Scripts\pythonw.exe" "src\UltraSingerGui.py"
