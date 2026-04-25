"""
Startup-sequence tests for the Coinbase Trader.

Verifies that:
  - Tech start-delay constant is sane (≤ 30s)
"""
import importlib


def test_tech_start_delay_is_sane():
    """_TECH_START_DELAY in main.py must be ≤ 30s so users see signals quickly."""
    m = importlib.import_module("main")
    delay = getattr(m, "_TECH_START_DELAY", None)
    assert delay is not None,         "_TECH_START_DELAY missing from main.py"
    assert isinstance(delay, int),    f"_TECH_START_DELAY must be int, got {type(delay)}"
    assert 0 < delay <= 30,           f"_TECH_START_DELAY {delay}s out of range (0, 30]"
