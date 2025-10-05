import asyncio
import json
import time
from typing import Callable, Dict, Any, List
from collections import deque
from datetime import datetime

# Small in-memory ingestion manager designed for real-time IoT streams.
# It supports multiple protocols (HTTP POST, WebSocket push, and local function hooks)
# and provides a single async queue that downstream consumers (like EnergyManagementSystem)
# can await on. This keeps APIs synchronized and integrated.

class IngestManager:
    def __init__(self, max_queue_size: int = 10000):
        # Use asyncio.Queue for async consumers
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        # Keep a short history for debug / replay
        self.history: deque = deque(maxlen=1000)
        # Registered callbacks for synchronous hooks
        self.sync_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        # Statistics
        self.total_received = 0
        self.start_time = time.time()

    async def push(self, payload: Dict[str, Any]):
        """Async push into the ingestion queue."""
        payload = self._normalize_payload(payload)
        try:
            await self.queue.put(payload)
            self.history.append((datetime.utcnow().isoformat(), payload))
            self.total_received += 1
            # Call sync callbacks without blocking main loop
            for cb in list(self.sync_callbacks):
                try:
                    cb(payload)
                except Exception:
                    # Swallow callback errors to avoid breaking ingestion
                    pass
            return True
        except asyncio.QueueFull:
            # Drop or handle overflow - currently drop and return False
            return False

    def push_nowait(self, payload: Dict[str, Any]) -> bool:
        """Try to push without awaiting. Returns True if accepted, False if full."""
        payload = self._normalize_payload(payload)
        try:
            self.queue.put_nowait(payload)
            self.history.append((datetime.utcnow().isoformat(), payload))
            self.total_received += 1
            for cb in list(self.sync_callbacks):
                try:
                    cb(payload)
                except Exception:
                    pass
            return True
        except asyncio.QueueFull:
            return False

    def register_callback(self, cb: Callable[[Dict[str, Any]], None]):
        """Register a synchronous callback that will be called on each message."""
        self.sync_callbacks.append(cb)

    async def get(self, timeout: float = None) -> Dict[str, Any]:
        """Get next message from queue. If timeout provided, wait up to timeout seconds."""
        if timeout:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        else:
            return await self.queue.get()

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure timestamps and canonical keys
        p = dict(payload)
        if "timestamp" not in p:
            p["timestamp"] = datetime.utcnow().isoformat()
        # Flatten common fields and accept multiple naming variants
        for k in ("device_id", "deviceId", "device", "id"):
            if k in p and "deviceId" not in p:
                p["deviceId"] = p.pop(k)
        return p

    def stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        return {
            "total_received": self.total_received,
            "queued": self.queue.qsize(),
            "uptime_seconds": uptime,
        }


# Utility: background task to drain queue and forward to a consumer function
async def forward_to_consumer(manager: IngestManager, consumer: Callable[[Dict[str, Any]], Any]):
    """Continuously forward incoming messages to provided consumer callable.
    Consumer may be sync or async; if sync it will be run in default loop executor.
    """
    loop = asyncio.get_running_loop()
    loop = asyncio.get_running_loop()
    while True:
        msg = await manager.get()
        try:
            result = consumer(msg)
            if asyncio.iscoroutine(result):
                await result
            else:
                # run sync consumer in executor to avoid blocking
                await loop.run_in_executor(None, lambda: consumer(msg))
        except Exception:
            # Log in real app; here we continue
            pass
