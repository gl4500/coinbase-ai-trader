import numpy as np
import pandas as pd
import pytest

from tools.slow_trend import screen as S
from tools.slow_trend.sim import Product

FINE = Product(1e-8, 1e-8, 1.0)
PRODUCTS = {"BTC-USD": FINE, "ETH-USD": FINE}
CONSTRAINTS = {p: {"base_increment": 1e-8, "base_min": 1e-8, "quote_min": 1.0} for p in PRODUCTS}


def _cal(prices, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="D")
    p = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({"open": p.shift(1).fillna(p.iloc[0]), "close": p})


def _raw(first, last, drop=()):
    days = pd.date_range(first, last, freq="D")
    p = np.linspace(100, 300, len(days))
    epoch_s = ((days - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")
    df = pd.DataFrame(
        {"start": epoch_s, "open": np.r_[p[0], p[:-1]], "close": p, "volume": 1.0, "page": 0}
    )
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    keep = ~days.strftime("%Y-%m-%d").isin(list(drop))
    return df[keep].reset_index(drop=True)


def test_period_runs_all_scenarios_and_comparators():
    up = np.linspace(100, 300, 400)
    out = S.run_period(
        {"BTC-USD": _cal(up), "ETH-USD": _cal(up)}, "2020-05-01", "2021-01-31", PRODUCTS
    )
    p0 = out["scenarios"]["P0"]
    assert set(out["scenarios"]) == {"P0", "S10", "S25", "D1", "SM"}
    assert {
        "trend",
        "buy_hold",
        "dca52",
        "cash",
        "passes",
        "ci_vs_bh",
        "ci_vs_dca52",
        "ci_vs_cash",
    } <= set(p0)
    assert p0["cash"]["terminal_value"] == 1000.0 and p0["passes"]["affected"] is False
    assert p0["trend"]["round_trips"] == 0
    assert set(p0["trend"]["per_sleeve"]) == {"BTC-USD", "ETH-USD"}


def test_pre_period_long_does_not_carry_in():
    up = np.linspace(100, 300, 400)
    btc, eth = _cal(up), _cal(up)
    for c in (btc, eth):
        c.loc["2020-04-30", ["open", "close"]] = np.nan  # invalidates the next 100 windows
    out = S.run_period({"BTC-USD": btc, "ETH-USD": eth}, "2020-05-01", "2020-06-30", PRODUCTS)
    assert out["scenarios"]["P0"]["trend"]["entries"] == 0
    assert out["diagnostics"]["BTC-USD"]["suppressed_decision_days"] == 61


def test_dev_start_is_first_common_valid_sma_day():
    btc = _cal(np.linspace(100, 300, 300))
    eth = _cal(np.linspace(100, 300, 290), start="2020-01-11")
    cal = {"BTC-USD": btc, "ETH-USD": eth.reindex(btc.index)}
    assert S.dev_start_from(cal, 100, "2020-12-31") == "2020-04-19"  # Jan 11 + 99 days
    btc.loc["2020-02-01", "close"] = np.nan
    assert S.dev_start_from(cal, 100, "2020-12-31") == "2020-05-11"  # first full window after gap


def test_dev_start_none_when_history_starts_too_late():
    cal = {p: _cal(np.linspace(100, 300, 50)) for p in PRODUCTS}
    assert S.dev_start_from(cal, 100, "2020-12-31") is None


def test_evaluate_end_to_end_on_monotone_rise_kills():
    raw = {p: _raw("2024-01-01", "2026-10-02") for p in PRODUCTS}
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["dev_start"] == "2024-04-09"
    assert rep["results"]["dev"]["scenarios"]["P0"]["passes"] == {
        "G1": True,
        "G2": False,
        "pass": False,
        "affected": False,
    }
    assert rep["verdict"] == {"verdict": "KILL", "reason": "primary_failed"}


def test_evaluate_empty_input_is_inadequate():
    raw = {p: _raw("2024-01-01", "2026-10-02").iloc[0:0] for p in PRODUCTS}
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["verdict"]["reason"] == "data"
    assert rep["data_checks"]["inadequate_because"] == "no_aligned_rows"


def test_evaluate_all_misaligned_input_is_inadequate():
    raw = {
        p: _raw("2024-01-01", "2026-10-02").assign(start=lambda d: d["start"] + 3600)
        for p in PRODUCTS
    }
    assert S.evaluate(raw, CONSTRAINTS)["data_checks"]["inadequate_because"] == "no_aligned_rows"


def test_evaluate_coverage_starting_at_block_start_is_inadequate():
    raw = {p: _raw("2025-04-14", "2026-10-02") for p in PRODUCTS}
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["verdict"]["reason"] == "data"
    assert rep["data_checks"]["inadequate_because"] == "no_development_coverage"


def test_unused_early_conflict_is_excluded_not_a_crash():
    early = _raw("2023-06-01", "2026-10-02")
    dup = early.iloc[[3]].copy()
    dup["close"] += 1.0
    dup["high"] += 1.0
    raw = {"BTC-USD": pd.concat([early, dup]), "ETH-USD": _raw("2024-01-01", "2026-10-02")}
    rep = S.evaluate(raw, CONSTRAINTS)  # BTC's 2023 rows precede common_first and are unused
    assert rep["data_checks"]["common_first"] == "2024-01-01"
    assert rep["results"] is not None


def test_malformed_hourly_overlap_is_a_diagnostic(tmp_path):
    bad = tmp_path / "BTC-USD.parquet"
    bad.write_text("not a parquet file")
    out = S.safe_overlap(bad, _raw("2024-01-01", "2024-03-01"), "2024-01-01")
    assert out["status"] == "error" and out["sha256"].startswith("sha256:")
    assert S.safe_overlap(
        tmp_path / "missing.parquet", _raw("2024-01-01", "2024-03-01"), "2024-01-01"
    ) == {"status": "absent"}


def test_unreadable_hourly_input_is_a_diagnostic_without_a_hash(tmp_path):
    unreadable = tmp_path / "BTC-USD.parquet"
    unreadable.mkdir()  # exists, but read_bytes() raises
    out = S.safe_overlap(unreadable, _raw("2024-01-01", "2024-03-01"), "2024-01-01")
    assert out["status"] == "error" and out["sha256"] is None


def test_source_digest_tracks_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    before = S.source_digest([f], root=tmp_path)
    assert before == S.source_digest([f], root=tmp_path)
    f.write_text("x = 2\n")
    assert S.source_digest([f], root=tmp_path) != before


def test_replay_label_distinguishes_corrected_computation():
    entries = [
        {"experiment_id": "E", "status": "started", "attempt": 1, "source_sha256": "s1"},
        {"experiment_id": "E", "status": "completed", "attempt": 1, "source_sha256": "s1"},
    ]
    assert S.replay_label(entries, "E", "s1") == ("replay", 1)
    assert S.replay_label(entries, "E", "s2") == ("corrected_replay", 1)


def test_missing_terminal_close_is_inadequate():
    raw = {
        "BTC-USD": _raw("2024-01-01", "2026-10-02"),
        "ETH-USD": _raw("2024-01-01", "2026-10-02", drop=("2026-10-02",)),
    }
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["verdict"] == {"verdict": "INCONCLUSIVE", "reason": "data"}
    assert rep["data_checks"]["terminal"]["ETH-USD"]["2026-10-02"] is False


def test_conflicting_duplicate_is_inadequate_not_crash():
    btc = _raw("2024-01-01", "2026-10-02")
    dup = btc.iloc[[500]].copy()
    dup["close"] += 1.0
    dup["high"] += 1.0
    raw = {"BTC-USD": pd.concat([btc, dup]), "ETH-USD": _raw("2024-01-01", "2026-10-02")}
    assert S.evaluate(raw, CONSTRAINTS)["verdict"]["reason"] == "data"


def test_refuses_snapshot_overwrite(tmp_path):
    S.ensure_new_snapshot(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="new preregistration"):
        S.ensure_new_snapshot(tmp_path)


def test_experiment_identity_ignores_head_but_not_prereg_or_snapshot():
    base = S.experiment_identity("prereg", "snap")
    assert base == S.experiment_identity("prereg", "snap")
    assert (
        len({base, S.experiment_identity("x", "snap"), S.experiment_identity("prereg", "x")}) == 3
    )


def _e(exp, status, attempt=1, head="h1"):
    return {"experiment_id": exp, "status": status, "attempt": attempt, "head": head}


def test_first_attempt():
    assert S.run_mode([], "E") == "first"


def test_failure_then_new_commit_is_still_a_retry():
    entries = [_e("E", "started", 1, "h1"), _e("E", "failed", 1, "h1")]
    assert S.run_mode(entries, "E") == "retry"  # HEAD h2 is irrelevant to the permission
    assert S.last_attempt(entries, "E") == 1


def test_unfinished_start_is_a_retry():
    assert S.run_mode([_e("E", "started")], "E") == "retry"


def test_completion_then_new_commit_refuses_without_replay():
    entries = [_e("E", "started"), _e("E", "completed")]
    with pytest.raises(RuntimeError, match="replay"):
        S.run_mode(entries, "E")
    assert S.run_mode(entries, "E", replay=True) == "replay"


def test_new_preregistration_must_be_explicit():
    entries = [_e("E", "started"), _e("E", "completed")]
    with pytest.raises(RuntimeError, match="new-preregistration"):
        S.run_mode(entries, "F")
    assert S.run_mode(entries, "F", new_prereg=True) == "new_preregistration"


def test_digests_ignore_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"A = 1\nB = 2\n")
    crlf.write_bytes(b"A = 1\r\nB = 2\r\n")
    assert S.text_digest(lf) == S.text_digest(crlf)
    for name, body in (("u", b"x = 1\n"), ("w", b"x = 1\r\n")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "m.py").write_bytes(body)
    assert S.source_digest([tmp_path / "u" / "m.py"], root=tmp_path / "u") == S.source_digest(
        [tmp_path / "w" / "m.py"], root=tmp_path / "w"
    )
