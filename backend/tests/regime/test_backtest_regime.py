import os
import sys

_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

from tools.regime.backtest_regime import apply_scaling, compare, metrics


def test_apply_scaling_defaults_to_one():
    trades = [{"pnl": 10.0, "usd_open": 100.0, "opened_at": "2025-01-01T00:00:00"}]
    out = apply_scaling(trades, {})  # no regime row -> scalar 1.0
    assert out[0]["scaled_pnl"] == 10.0


def test_apply_scaling_uses_date_scalar():
    trades = [{"pnl": -20.0, "usd_open": 100.0, "opened_at": "2025-11-30T12:00:00"}]
    out = apply_scaling(trades, {"2025-11-30": 0.5})
    assert out[0]["scaled_pnl"] == -10.0  # loss halved by risk-off scaling


def test_metrics_shapes():
    m = metrics([1.0, -2.0, 3.0])
    assert m["total"] == 2.0
    assert "sharpe" in m and "max_drawdown" in m


def test_compare_flags_improvement():
    # Regime halves the big losers, keeps winners -> should HELP.
    trades = [
        {"pnl": 10.0, "usd_open": 100.0, "opened_at": "2025-01-01T00:00:00"},
        {"pnl": -40.0, "usd_open": 100.0, "opened_at": "2025-11-30T00:00:00"},
        {"pnl": 12.0, "usd_open": 100.0, "opened_at": "2025-02-01T00:00:00"},
    ]
    scal = {"2025-11-30": 0.4}  # risk-off on the loser's day
    res = compare(trades, scal)
    assert res["scaled"]["total"] > res["baseline"]["total"]
    assert res["verdict"] in ("HELPS", "NO")


def test_compare_is_inconclusive_when_scalar_is_constant():
    """A scalar that never varies cannot discriminate — uniform shrinkage is
    not evidence that regime scaling helps or hurts, so the verdict must say
    so rather than reporting an arithmetic NO."""
    trades = [
        {"pnl": 10.0, "usd_open": 100.0, "opened_at": "2026-05-01T00:00:00"},
        {"pnl": -40.0, "usd_open": 100.0, "opened_at": "2026-06-01T00:00:00"},
        {"pnl": 12.0, "usd_open": 100.0, "opened_at": "2026-07-01T00:00:00"},
    ]
    scal = {"2026-05-01": 0.8, "2026-06-01": 0.8, "2026-07-01": 0.8}
    res = compare(trades, scal)
    assert res["verdict"] == "INCONCLUSIVE"
    assert res["scalar_stats"]["stdev"] < 1e-12  # constant, up to float noise


def test_compare_reports_scalar_coverage():
    trades = [
        {"pnl": 10.0, "usd_open": 100.0, "opened_at": "2026-05-01T00:00:00"},
        {"pnl": -5.0, "usd_open": 100.0, "opened_at": "2026-06-01T00:00:00"},
    ]
    res = compare(trades, {"2026-05-01": 0.5})  # only one of two days covered
    stats = res["scalar_stats"]
    assert stats["matched"] == 1
    assert stats["coverage"] == 0.5
    assert stats["min"] == 0.5 and stats["max"] == 1.0


def test_compare_is_inconclusive_when_scalar_never_protects():
    """A window where the scalar only ever levers UP does not test the layer's
    protective half. Scoring it NO would kill a gate that was never exercised."""
    trades = [
        {"pnl": 10.0, "usd_open": 100.0, "opened_at": "2026-05-01T00:00:00"},
        {"pnl": -40.0, "usd_open": 100.0, "opened_at": "2026-06-01T00:00:00"},
        {"pnl": 12.0, "usd_open": 100.0, "opened_at": "2026-07-01T00:00:00"},
    ]
    scal = {"2026-05-01": 1.05, "2026-06-01": 1.24, "2026-07-01": 1.10}
    res = compare(trades, scal)
    assert res["verdict"] == "INCONCLUSIVE"
    assert res["scalar_stats"]["protective_days"] == 0
    assert "protect" in res["reason"]


def test_compare_scores_a_window_that_does_protect():
    trades = [
        {"pnl": 10.0, "usd_open": 100.0, "opened_at": "2026-05-01T00:00:00"},
        {"pnl": -40.0, "usd_open": 100.0, "opened_at": "2026-06-01T00:00:00"},
        {"pnl": 12.0, "usd_open": 100.0, "opened_at": "2026-07-01T00:00:00"},
    ]
    scal = {"2026-05-01": 1.05, "2026-06-01": 0.5, "2026-07-01": 1.10}
    res = compare(trades, scal)
    assert res["verdict"] in ("HELPS", "NO")
    assert res["scalar_stats"]["protective_days"] == 1
