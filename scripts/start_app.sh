#!/bin/bash
# SDS-RAG Application Starter for Unix/Linux
# This script launches the SDS-RAG Streamlit application

echo "==============================================="
echo "   SDS-RAG: Financial Document Analysis"
echo "==============================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.10+ and try again"
    exit 1
fi

# Check if Poetry is available
if command -v poetry &> /dev/null; then
    echo "Using Poetry to run the application..."
    poetry run python run_app.py
else
    echo "WARNING: Poetry not found. Using pip instead."
    echo "Installing dependencies with pip..."
    pip3 install -r requirements.txt
    python3 run_app.py
fi