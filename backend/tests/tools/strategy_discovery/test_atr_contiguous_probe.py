"""The contiguous-window diagnostic: does the SELECTION do what it says?

The variant arithmetic is anchored to production in `test_atr_causality_probe.py`. What needs
testing here is the population: exactly which entries are retained, that every exclusion is
counted under the right reason, that a gapped frame is USABLE (unlike the full-frame scanner,
which must refuse it), and that selection never looks at an outcome.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.atr_contiguous_probe import (
    CAVEATS,
    SELECTION_RULE,
    run_contiguous_probe,
    scan_contiguous_windows,
)

_BAR = 3_600_000
_H = 24


def _frame(rows: int = 400, *, seed: int = 3, atr: float = 0.05, label: float = 0.05):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=rows))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.015, size=rows)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.015, size=rows)))
    return pd.DataFrame(
        {
            "ts": np.arange(rows, dtype="int64") * _BAR,
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": high,
            "low": low,
            "close": close,
            "atr14_pct": np.full(rows, atr),
            f"label_h{_H}": [
                float(label) if row + _H < rows else float("nan") for row in range(rows)
            ],
        }
    )


def test_a_gapped_frame_is_usable_here_unlike_the_full_frame_scanner():
    """The entire reason this module exists. `atr_causality_report.scan_frame` must refuse a
    gapped frame; this one must scan the clean windows inside it."""
    from tools.strategy_discovery.atr_causality_report import scan_frame

    frame = _frame()
    ts = frame["ts"].to_numpy(dtype="int64").copy()
    ts[201:] += _BAR
    frame["ts"] = ts

    assert scan_frame(frame, product_id="GAP-USD", horizon=_H).skipped_reason is not None
    scan = scan_contiguous_windows(frame, product_id="GAP-USD", horizon=_H)
    assert scan.skipped_reason is None
    assert scan.retained_entries > 300, "a single hole must not disqualify the whole frame"


def test_the_excluded_window_count_is_exactly_the_windows_covering_the_hole():
    """One hole after row 200, horizon 24: exactly rows 177..200 are excluded for that reason."""
    frame = _frame()
    ts = frame["ts"].to_numpy(dtype="int64").copy()
    ts[201:] += _BAR
    frame["ts"] = ts

    scan = scan_contiguous_windows(frame, product_id="GAP-USD", horizon=_H, max_entries=None)
    assert scan.excluded_window_has_gap == 24
    assert (
        scan.retained_entries
        + scan.excluded_label_not_finite
        + scan.excluded_window_incomplete
        + scan.excluded_window_has_gap
        == scan.positions_considered
    ), "every considered position must land in exactly one bucket"


def test_a_clean_frame_excludes_only_the_unlabelled_tail():
    scan = scan_contiguous_windows(_frame(), product_id="CLEAN-USD", horizon=_H, max_entries=None)
    assert scan.excluded_window_has_gap == 0
    assert scan.excluded_label_not_finite == _H, "the last 24 rows carry no label"
    assert scan.retained_entries == 400 - _H
    assert scan.retained_first_ts == 0
    assert scan.retained_last_ts == (400 - _H - 1) * _BAR


def test_a_reversed_clock_is_still_fatal_even_though_gaps_are_tolerated():
    """Tolerating sparsity is not tolerating incoherence."""
    frame = _frame()
    frame["ts"] = frame["ts"].to_numpy(dtype="int64")[::-1]
    scan = scan_contiguous_windows(frame, product_id="REV-USD", horizon=_H)
    assert scan.skipped_reason is not None and "strictly increasing" in scan.skipped_reason
    assert scan.retained_entries == 0


def test_selection_ignores_outcomes_entirely():
    """Codex required that selection never depend on an outcome. Changing every label VALUE while
    keeping finiteness must not move a single retained entry."""
    base = scan_contiguous_windows(_frame(label=0.05), product_id="A-USD", horizon=_H)
    flipped = scan_contiguous_windows(_frame(label=-0.90), product_id="A-USD", horizon=_H)

    assert flipped.retained_entries == base.retained_entries
    assert flipped.excluded_window_has_gap == base.excluded_window_has_gap
    assert flipped.excluded_label_not_finite == base.excluded_label_not_finite
    assert flipped.retained_first_ts == base.retained_first_ts


def test_legacy_matches_stored_is_reported_and_can_be_zero():
    """The provenance anchor. These synthetic labels are a constant, so the recomputation will
    NOT match them -- and the report must say so rather than implying the baseline is the
    published label."""
    scan = scan_contiguous_windows(_frame(), product_id="A-USD", horizon=_H)
    assert scan.retained_entries > 0
    assert scan.legacy_matches_stored < scan.retained_entries, (
        "a constant synthetic label cannot equal the simulated PnL; if this ever matches, the "
        "fixture has become degenerate"
    )


def test_legacy_matches_stored_when_the_label_really_is_the_recomputation():
    """The other side: when stored labels ARE the legacy output, the anchor reaches 100%."""
    from tools.strategy_discovery.atr_causality_probe import LEGACY, simulate_variant
    from tools.strategy_discovery.atr_contiguous_probe import DEFAULT_CONFIG

    frame = _frame()
    args = dict(
        opens=frame["open"].to_numpy(dtype="float64"),
        closes=frame["close"].to_numpy(dtype="float64"),
        highs=frame["high"].to_numpy(dtype="float64"),
        lows=frame["low"].to_numpy(dtype="float64"),
        atr_pcts=frame["atr14_pct"].to_numpy(dtype="float64"),
        config=DEFAULT_CONFIG,
    )
    recomputed = [
        simulate_variant(entry_idx=row, horizon=_H, spec=LEGACY, **args).pnl
        for row in range(len(frame))
    ]
    frame[f"label_h{_H}"] = recomputed

    scan = scan_contiguous_windows(frame, product_id="EXACT-USD", horizon=_H)
    assert scan.retained_entries > 300
    assert scan.legacy_matches_stored == scan.retained_entries


def test_the_run_persists_the_rule_and_its_caveats(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for pid in ("AAA-USD", "BBB-USD"):
        pq.write_table(
            pa.Table.from_pandas(_frame(), preserve_index=False), frames / f"{pid}.parquet"
        )

    payload = run_contiguous_probe(frames, tmp_path / "out", horizons=(_H,), max_entries=200)
    assert payload["status"] == "ok"
    assert payload["selection_rule"] == SELECTION_RULE
    assert payload["caveats"] == list(CAVEATS)
    assert any("RECURSIVE smoothing" in c for c in payload["caveats"]), (
        "the ATR caveat must say recursive, not a finite 14-bar window"
    )
    assert any("NOT representative" in c for c in payload["caveats"])
    written = json.loads(
        (tmp_path / "out" / "contiguous_window_report.json").read_text(encoding="utf-8")
    )
    assert written["selection_rule"] == SELECTION_RULE


def test_the_run_refuses_to_write_under_its_inputs_and_leaves_them_untouched(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    pq.write_table(pa.Table.from_pandas(_frame(), preserve_index=False), frames / "AAA-USD.parquet")
    before = (frames / "AAA-USD.parquet").read_bytes()

    with pytest.raises(ValueError, match="or inside it"):
        run_contiguous_probe(frames, frames / "nested", horizons=(_H,))

    run_contiguous_probe(frames, tmp_path / "out", horizons=(_H,), max_entries=100)
    assert (frames / "AAA-USD.parquet").read_bytes() == before


def test_no_data_when_nothing_satisfies_the_rule(tmp_path):
    frame = _frame()
    frame[f"label_h{_H}"] = float("nan")
    frames = tmp_path / "frames"
    frames.mkdir()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), frames / "NONE-USD.parquet")

    payload = run_contiguous_probe(frames, tmp_path / "out", horizons=(_H,))
    assert payload["status"] == "no_data"
    assert "satisfied the selection rule" in payload["no_data_reason"]
    assert payload["retained_entries_total"] == 0
