@echo off
setlocal
cd /d "%~dp0"

echo ========================================================================
echo          CS2 DEATHMATCH BOT - WINDOWS LAUNCHER
echo ========================================================================

REM Find Python executable: prefer .venv\Scripts\python.exe if available
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

REM Test if python is accessible
%PYTHON_CMD% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in virtualenv or PATH!
    echo Please install Python 3.10+ and add it to your PATH, or run setup:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Run launcher with pre-flight doctor and pass all arguments
%PYTHON_CMD% run.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Launcher] Bot stopped or failed with exit code %ERRORLEVEL%.
)
