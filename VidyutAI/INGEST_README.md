# IoT Ingest Backend

This project contains a small ingestion backend implemented with FastAPI and
an in-memory async queue for real-time IoT data. The `IngestManager` accepts
messages and allows synchronous callbacks so existing systems (like
`EnergyManagementSystem`) may update state immediately.

Quick start

1. Install dependencies:

   python -m pip install -r d:\VidyutAI_main\VidyutAI\requirements.txt

2. Start server:

   python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

3. In another terminal wire the EMS to the ingest manager (optional):

   python scripts\wire_ingest_to_ems.py

4. Send sample messages:

   python scripts\send_sample_iot.py

Notes

- For production use replace in-memory queue with Redis/Kafka for durability.
- Add authentication and rate-limiting to the /ingest endpoints.
