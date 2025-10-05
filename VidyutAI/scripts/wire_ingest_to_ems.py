"""Example: wire backend.manager to EnergyManagementSystem.

Run this after starting the FastAPI app (which creates backend.manager). This script
imports the manager from the package and attaches EMS to it so incoming HTTP/WebSocket
messages immediately update EMS internal state.
"""
import time
import sys
sys.path.append(r"d:\VidyutAI_main\VidyutAI")

try:
    from backend import manager
except Exception as e:
    print("Could not import backend.manager:", e)
    raise SystemExit(1)

from energy_management_system import EnergyManagementSystem

if manager is None:
    print("Ingest manager not available. Make sure backend.api has been imported by uvicorn.")
    raise SystemExit(1)

ems = EnergyManagementSystem()
ems.register_ingest_manager(manager)

print("EMS registered with ingest manager. Listening for messages...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping wire script")
