@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 goto missing

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 goto missing

if not exist .venv\Scripts\python.exe (
  echo First run: creating a virtual environment...
  python -m venv .venv
  if errorlevel 1 goto failed
)

REM A venv Python alone does not prove an interrupted install is usable.
.venv\Scripts\python.exe -m pip check >nul 2>&1
if errorlevel 1 goto repair
.venv\Scripts\python.exe -m labelnoiseaudit --help >nul 2>&1
if errorlevel 1 goto repair
goto run

:repair
echo Installing or repairing pinned packages...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -e . --no-deps
if errorlevel 1 goto failed

:run
.venv\Scripts\python.exe -m labelnoiseaudit
exit /b %errorlevel%

:missing
echo LabelNoiseAudit needs Python 3.11 or newer.
echo Python was not found, or it is too old.
echo Download it from https://www.python.org/downloads/ and then double-click this launcher again.
pause
exit /b 1

:failed
echo Setup failed. See the messages above.
pause
exit /b 1
