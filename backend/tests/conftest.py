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


@pytest.fixture(autouse=True)
def _redirect_cnn_dataset_cache(tmp_path_factory, monkeypatch):
    """Safety net: prevent any test from writing to the real CNN dataset cache.

    Background (#173): tests that call `agent.train_on_history(...)` without
    monkeypatching `_DATASET_CACHE_PATH` silently overwrite the production
    cache file at `backend/cnn_dataset_cache.pt`. This corrupted real
    candle data with synthetic 6-product fixture data on 2026-05-03,
    causing 11 consecutive trains to run on junk samples.

    Autouse here so EVERY test gets the redirect, even ones that don't
    realise they touch the cache (e.g. via deep import side-effects).
    """
    import agents.cnn_agent as ca
    tmp_dir = tmp_path_factory.mktemp("cnn_cache_isolated")
    monkeypatch.setattr(
        ca, "_DATASET_CACHE_PATH", str(tmp_dir / "dataset_cache.pt")
    )
    yield


@pytest.fixture(autouse=True)
def _redirect_database_path(tmp_path_factory, monkeypatch):
    """Safety net: prevent any test from writing to the production coinbase.db.

    Background (#176): tests that call `agent.train_on_history(...)` end at
    `agents/cnn_agent.py:2881 await database.save_training_session(result)`,
    which writes via `database.DB_PATH` (set at module import from
    `config.database_url`). The existing `tmp_db` fixture only sets
    DATABASE_URL — it does NOT monkeypatch `database.DB_PATH`, so already-
    imported `database.DB_PATH` keeps its production value unless
    `init_db` reloads the module.

    Autouse here so EVERY test gets the redirect. Tests that explicitly
    use `tmp_db` + `init_db` will reload `database` and pick up their
    own DATABASE_URL — that wins over this autouse monkeypatch.
    """
    import database
    tmp_dir = tmp_path_factory.mktemp("coinbase_db_isolated")
    monkeypatch.setattr(database, "DB_PATH", str(tmp_dir / "coinbase.db"))
    yield


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
