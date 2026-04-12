"""
Pytest fixtures shared across all test modules.
"""
import asyncio
import os
import sys
import pytest

# ── Make backend importable without installing ─────────────────────────────────
BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# ── Patch .env so tests never need real credentials ───────────────────────────
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test-org/apiKeys/test-key")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nMHQCAQEEIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABoAoGCCqGSM49\nAwEHoWQDYgAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=\n-----END EC PRIVATE KEY-----")
os.environ.setdefault("APP_API_KEY",              "test-api-key-fixture")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")


@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary SQLite path; set DATABASE_URL for the session."""
    db = str(tmp_path / "test.db")
    os.environ["DATABASE_URL"] = db
    yield db
    os.environ.pop("DATABASE_URL", None)


@pytest.fixture
async def init_db(tmp_db):
    """Initialise the database schema in the tmp_db."""
    import importlib
    import database
    # Reload so DB_PATH picks up the tmp_db environment variable
    importlib.reload(database)
    await database.init_db()
    yield database
