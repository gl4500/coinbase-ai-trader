"""Safety net: tests must never write to the production coinbase.db.

Background (#176 / sibling of #173): the venv pytest run on 2026-05-03
appended 11 junk rows (515-525) to `cnn_training_sessions` in production
`coinbase.db`. Tests calling `agent.train_on_history(...)` end at
`agents/cnn_agent.py:2881 await database.save_training_session(result)` —
which writes via `database.DB_PATH` set at module import from
`config.database_url`. The existing `tmp_db` fixture sets DATABASE_URL but
does NOT monkeypatch the already-imported `database.DB_PATH`, so the
real path is still used unless the test explicitly opts in.

This test asserts that during any pytest session, `database.DB_PATH` does
NOT resolve to the real production file. Enforced via an autouse fixture
in conftest.py.
"""
import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

import database  # noqa: E402


_PROD_DB = os.path.abspath(os.path.join(BACKEND, "coinbase.db"))


def test_db_path_is_redirected_during_tests():
    """database.DB_PATH must NOT point at the real production sqlite file."""
    current = os.path.abspath(database.DB_PATH)
    assert current != _PROD_DB, (
        f"database.DB_PATH still points to production DB!\n"
        f"  current = {current}\n"
        f"  prod    = {_PROD_DB}\n"
        f"Tests calling save_training_session() will pollute the live DB."
    )


def test_db_path_is_outside_backend_dir():
    """Sanity: redirected DB lives outside the backend/ tree (typically tmp)."""
    current = os.path.abspath(database.DB_PATH)
    backend_abs = os.path.abspath(BACKEND)
    try:
        common = os.path.commonpath([current, backend_abs])
    except ValueError:
        common = ""
    assert common != backend_abs, (
        f"database.DB_PATH must not be under backend/: {current}"
    )
