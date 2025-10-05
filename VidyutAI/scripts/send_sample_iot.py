"""Send a few sample IoT messages to the ingestion HTTP endpoint."""
import requests
import json

URL = "http://localhost:8000/ingest"

sample_messages = [
    {"device_id": "BAT_1_1_1", "soc": 54.2, "temperature": 30.1},
    {"device_id": "INV_1_1_1", "efficiency": 92.1, "current_load": 45.0},
    {"device_id": "SOLAR_1_1_1", "current_generation": 8.3}
]

for msg in sample_messages:
    r = requests.post(URL, json=msg)
    print(r.status_code, r.text)
