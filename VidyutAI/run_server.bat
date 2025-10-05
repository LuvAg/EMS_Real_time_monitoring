@echo off
REM Run FastAPI uvicorn server for ingestion
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
pause
