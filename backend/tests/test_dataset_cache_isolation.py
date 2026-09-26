"""Safety net: tests must never write to the production CNN dataset cache.

Background: prior to #173, several tests (test_cnn_agent.py:425, 453, 1001)
generated synthetic 6-product fixtures (COIN0-USD..COIN5-USD) and called
`agent.train_on_history(...)`. Because none of those tests monkeypatched
`_DATASET_CACHE_PATH`, training silently wrote a 3.2 MB junk fixture
(486 samples on synthetic candles with `start=1_700_000_000 + i*3600`) over
the real cache at `backend/cnn_dataset_cache.pt`. The next 11 production
trains (ids 504-514) ran on that junk and produced near-random AUCs.

This test asserts that during any pytest session, `_DATASET_CACHE_PATH`
does NOT resolve to the real production path. Enforced via an autouse
fixture in conftest.py.
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

import agents.cnn_agent as ca  # noqa: E402

_PROD_CACHE = os.path.abspath(os.path.join(BACKEND, "cnn_dataset_cache.pt"))


def test_dataset_cache_path_is_redirected_during_tests():
    """The cache path must NOT point at the real production file."""
    current = os.path.abspath(ca._DATASET_CACHE_PATH)
    assert current != _PROD_CACHE, (
        f"_DATASET_CACHE_PATH still points to production cache!\n"
        f"  current = {current}\n"
        f"  prod    = {_PROD_CACHE}\n"
        f"Tests calling train_on_history() will overwrite the real cache."
    )


def test_dataset_cache_path_is_writable_tmp_location():
    """Sanity: redirected path lives under the OS temp tree (not backend/)."""
    current = os.path.abspath(ca._DATASET_CACHE_PATH)
    assert os.path.commonpath([current, BACKEND]) != os.path.abspath(BACKEND), (
        f"_DATASET_CACHE_PATH must not be under backend/: {current}"
    )
