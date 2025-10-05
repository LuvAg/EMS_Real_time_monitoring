"""Backend package for IoT ingestion.

Expose the FastAPI `app` and the module-level `manager` if available so
other parts of the system can import `backend.app` and `backend.manager`.
This import is done inside a try/except to avoid hard crashes when FastAPI
or other optional dependencies are not installed in test environments.
"""

try:
	from .api import app, manager  # type: ignore
except Exception:
	app = None
	manager = None

try:
	from .ingest_manager import IngestManager  # type: ignore
except Exception:
	IngestManager = None

__all__ = ["app", "manager", "IngestManager"]
