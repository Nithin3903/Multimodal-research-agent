@echo off
title Multimodal Research Agent

echo ============================================================
echo  MULTIMODAL RESEARCH AGENT — STARTING
echo ============================================================
echo.

:: Check venv exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at .venv\
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Check Ollama
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Ollama not found in PATH.
    echo Make sure Ollama is installed and running.
    echo Download from: https://ollama.com
    echo.
)

:: Start FastAPI backend in a new window
echo [1/2] Starting FastAPI backend on http://127.0.0.1:8000 ...
start "Research Agent — Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload"

:: Wait a moment for the backend to start
timeout /t 3 /nobreak >nul

:: Start Vite frontend in a new window
echo [2/2] Starting React frontend on http://127.0.0.1:5173 ...
start "Research Agent — Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================
echo  Both services are starting in separate windows.
echo.
echo  Backend  : http://127.0.0.1:8000
echo  Frontend : http://127.0.0.1:5173
echo  API Docs : http://127.0.0.1:8000/docs
echo ============================================================
echo.
echo  Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul
start http://127.0.0.1:5173
