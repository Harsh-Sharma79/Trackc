@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create and install the Python environment first. See README.md.
  pause
  exit /b 1
)
if not exist "frontend\node_modules" (
  call npm --prefix frontend install || exit /b 1
)
start "Speech Evaluation API" cmd /k "set PYTHONPATH=%CD%\backend&& .venv\Scripts\python.exe -m uvicorn api:app --app-dir backend --host 127.0.0.1 --port 8000"
start "Speech Evaluation Frontend" cmd /k "npm --prefix frontend run dev"
echo Started backend and frontend. Open the Vite URL shown in the frontend terminal.
