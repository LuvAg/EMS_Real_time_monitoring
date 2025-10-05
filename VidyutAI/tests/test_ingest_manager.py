import asyncio
import pytest
from backend.ingest_manager import IngestManager

@pytest.mark.asyncio
async def test_push_and_get():
    mgr = IngestManager(max_queue_size=10)
    payload = {"device_id":"dev1","temp":25}
    ok = await mgr.push(payload)
    assert ok is True
    msg = await mgr.get()
    assert msg["deviceId"] == "dev1"
    assert "timestamp" in msg

@pytest.mark.asyncio
async def test_queue_full():
    mgr = IngestManager(max_queue_size=2)
    await mgr.push({"device_id":"a"})
    await mgr.push({"device_id":"b"})
    # Next push should block or raise; manager.push returns False when full
    # Simulate by filling queue and attempting third
    res = await mgr.push({"device_id":"c"})
    assert res is False
