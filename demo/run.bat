@echo off
cd /d "%~dp0"

rem ===== DroneCheck demo: one-click setup + run =====

rem 1) find Python
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
    echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

rem 2) create a local virtual environment (first run only)
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY% -m venv .venv || (echo Could not create the virtual environment. & pause & exit /b 1)
)

rem 3) install the requirements (first run only, can take a few minutes)
if not exist ".venv\installed.txt" (
    echo Installing requirements...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
    echo ok> ".venv\installed.txt"
)

rem 4) check the model
if not exist "..\model\drone_check.pth" (
    echo Model not found: model\drone_check.pth
    echo Train it with notebook\DroneCheck.ipynb and save it first.
    pause
    exit /b 1
)

rem 5) start the server and open the browser
echo Starting DroneCheck demo on http://localhost:47613 ...
".venv\Scripts\python.exe" server.py
pause
