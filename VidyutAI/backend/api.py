from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
from typing import Dict, Any
from .ingest_manager import IngestManager

app = FastAPI(title="VidyutAI IoT Ingest API")
manager = IngestManager()

@app.post('/ingest')
async def ingest_http(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    ok = await manager.push(payload)
    if not ok:
        raise HTTPException(status_code=503, detail="Ingest queue full")
    return JSONResponse({"status": "accepted"})

# Simple status endpoint
@app.get('/status')
async def status():
    return manager.stats()

# WebSocket endpoint for streaming ingestion
@app.websocket('/ws/ingest')
async def websocket_ingest(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                import json
                payload = json.loads(data)
            except Exception:
                await ws.send_text('{"error":"invalid json"}')
                continue
            ok = await manager.push(payload)
            if not ok:
                await ws.send_text('{"error":"queue full"}')
            else:
                await ws.send_text('{"status":"accepted"}')
    except WebSocketDisconnect:
        return

# Expose manager for programmatic access
@app.on_event("startup")
async def startup_event():
    # nothing for now
    pass

@app.on_event("shutdown")
async def shutdown_event():
    # nothing for now
    pass
