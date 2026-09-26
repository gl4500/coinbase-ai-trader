"""Diagnostics must not mix label versions or inflate denominators.

Spec: docs/specs/2026-09-26-outcome-label-contract.md
Audit: docs/audits/2026-09-26-strategy-audit-report.md section 1

The audit's complaint was that a dashboard number presented as a four-hour
predictive score pooled labels resolved a mean 45.75 h late, and that a
confidence-decile table was presented as calibration against a target the model
was never trained on. These tests pin both.
"""

import os
import sqlite3
import sys
from pathlib import Path

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from services import diagnostics as d  # noqa: E402
from services import outcome_labels as ol  # noqa: E402

_NOW = 1_800_000_000.0
_MATURED = _NOW - 10_000  # target_time in the past
_FUTURE = _NOW + 10_000  # target_time still ahead


def _seed(tmp_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(tmp_path / "d.db")
    con.executescript(
        """
        CREATE TABLE signal_outcomes (
            source TEXT, side TEXT, confidence REAL, pct_change REAL,
            signed_return REAL, outcome TEXT, created_at TEXT,
            label_version INTEGER, target_time REAL, check_after REAL,
            unresolved_reason TEXT);
        CREATE TABLE trades (agent TEXT, product_id TEXT, pnl REAL, pct_pnl REAL,
            hold_secs REAL, trigger_close TEXT, opened_at TEXT, closed_at TEXT);
        CREATE TABLE cnn_scans (product_id TEXT, side TEXT, model_prob REAL,
            regime TEXT, scanned_at TEXT);
        """
    )
    return con


def _row(outcome, *, version=2, ret=0.02, conf=0.9, target=_MATURED, reason=None):
    return (
        "CNN",
        "BUY",
        conf,
        ret,
        ret,
        outcome,
        "2026-08-08T00:00:00+00:00",
        version,
        target,
        target,
        reason,
    )


def _insert(con, rows):
    con.executemany("INSERT INTO signal_outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()


# ── Label versions must not be pooled ─────────────────────────────────────────


def test_legacy_rows_are_excluded_from_the_headline_numbers(tmp_path):
    """Version-1 rows carry the delayed-resolution defect; they must not feed
    the current-version accuracy figure."""
    con = _seed(tmp_path)
    _insert(
        con,
        [
            _row("WIN", version=2),
            _row("LOSS", version=2),
            # Legacy: would push precision to 3/4 if pooled.
            _row("WIN", version=None),
            _row("WIN", version=None),
        ],
    )
    out = d.signal_edge(con, cutoff=None, now=_NOW)
    assert out["label_version"] == ol.LABEL_VERSION
    assert out["n"] == 2
    assert out["wins"] == 1
    assert out["precision"] == 0.5


def test_legacy_rows_are_reported_separately_not_discarded(tmp_path):
    con = _seed(tmp_path)
    _insert(
        con,
        [
            _row("WIN", version=2),
            _row("WIN", version=None),
            _row("LOSS", version=None),
            _row("NEUTRAL", version=None),
        ],
    )
    out = d.signal_edge(con, cutoff=None, now=_NOW)
    assert out["legacy"]["n"] == 3
    assert out["legacy"]["wins"] == 1
    assert out["legacy"]["label_version"] == 1


# ── Denominators ──────────────────────────────────────────────────────────────


def test_unavailable_rows_are_excluded_from_the_accuracy_denominator(tmp_path):
    con = _seed(tmp_path)
    _insert(
        con,
        [
            _row("WIN"),
            _row("LOSS"),
            _row("UNAVAILABLE", ret=None, reason="missing_exit_candle"),
            _row("UNAVAILABLE", ret=None, reason="missing_entry_candle"),
        ],
    )
    out = d.signal_edge(con, cutoff=None, now=_NOW)
    assert out["n"] == 2
    assert out["precision"] == 0.5
    assert out["counts"]["unavailable"] == 2


def test_unmatured_rows_are_excluded_from_the_denominator(tmp_path):
    con = _seed(tmp_path)
    _insert(
        con,
        [
            _row("WIN"),
            _row(None, ret=None, target=_FUTURE),  # not yet measurable
        ],
    )
    out = d.signal_edge(con, cutoff=None, now=_NOW)
    assert out["n"] == 1
    assert out["counts"]["eligible"] == 1
    assert out["counts"]["matured"] == 1


def test_counts_cover_eligible_matured_unresolved_unavailable(tmp_path):
    con = _seed(tmp_path)
    _insert(
        con,
        [
            _row("WIN"),
            _row("LOSS"),
            _row(None, ret=None),  # eligible, unresolved
            _row("UNAVAILABLE", ret=None, reason="missing_exit_candle"),
            _row(None, ret=None, target=_FUTURE),  # not eligible yet
        ],
    )
    counts = d.signal_edge(con, cutoff=None, now=_NOW)["counts"]
    assert counts["eligible"] == 4  # excludes the future-dated row
    assert counts["matured"] == 2
    assert counts["unresolved"] == 1
    assert counts["unavailable"] == 1


# ── Calibration may not be claimed against a mismatched target ────────────────


def test_calibration_is_suppressed_with_an_explicit_reason(tmp_path):
    """The models train on a path-dependent triple-barrier target; this label is
    an endpoint return. Confidence buckets are not calibration."""
    con = _seed(tmp_path)
    _insert(con, [_row("WIN", conf=0.9), _row("LOSS", conf=0.2)])
    out = d.signal_edge(con, cutoff=None, now=_NOW)
    assert out["calibration"] == []
    assert out["calibration_available"] is False
    assert "target" in out["calibration_suppressed_reason"].lower()


def test_confidence_buckets_are_still_reported_as_descriptive(tmp_path):
    con = _seed(tmp_path)
    _insert(con, [_row("WIN", conf=0.9), _row("LOSS", conf=0.9), _row("WIN", conf=0.2)])
    buckets = d.signal_edge(con, cutoff=None, now=_NOW)["confidence_buckets"]
    by = {b["bucket"]: b for b in buckets}
    assert by[0.9]["n"] == 2
    assert by[0.9]["win_rate"] == 0.5
    assert by[0.2]["n"] == 1


# ── Label accuracy is not profitability ───────────────────────────────────────


def test_signal_edge_states_it_is_not_profitability(tmp_path):
    con = _seed(tmp_path)
    _insert(con, [_row("WIN")])
    out = d.signal_edge(con, cutoff=None, now=_NOW)
    assert out["is_profitability"] is False
    assert out["return_units"] == "fraction"


def test_return_uses_the_versioned_signed_return_column(tmp_path):
    """Not pct_change, which legacy rows populated under the old definition."""
    con = _seed(tmp_path)
    rows = [
        (
            "CNN",
            "BUY",
            0.9,
            999.0,
            0.04,
            "WIN",
            "2026-08-08T00:00:00+00:00",
            2,
            _MATURED,
            _MATURED,
            None,
        ),
    ]
    _insert(con, rows)
    out = d.signal_edge(con, cutoff=None, now=_NOW)
    assert out["e_return"] == 0.04
