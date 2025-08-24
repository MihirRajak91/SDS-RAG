@echo off
REM SDS-RAG Application Starter for Windows
REM This script launches the SDS-RAG Streamlit application

echo ===============================================
echo    SDS-RAG: Financial Document Analysis
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and try again
    pause
    exit /b 1
)

REM Check if Poetry is available
poetry --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Poetry not found. Using pip instead.
    echo Installing dependencies with pip...
    pip install -r requirements.txt
    python app.py
) else (
    echo Using Poetry to run the application...
    poetry run python run_app.py
)

pause