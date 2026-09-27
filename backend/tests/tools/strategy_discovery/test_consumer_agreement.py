"""Both consumers, one published fixture, one answer.

Before this integration the miner measured occupancy on a wall clock while the portfolio replay
measured it on a different wall clock, and nothing compared them. Each is now individually tested
against the endpoints, which leaves the failure this effort is actually about unguarded: the two
paths DRIFTING APART again. A change to either basis would keep both modules internally consistent
and their own suites green.

**What each test here actually covers**, stated precisely because the first version of this
docstring overclaimed and Codex review `9eade26d` caught it:

* `test_the_miner_agrees_with_the_shared_eligibility_helper` compares the miner against
  `eligibility_boundaries`. The miner side travels the full production route -- parquet on disk,
  sidecar discovery, loader validation, retention, device placement -- so it catches a miner
  regression. It does **not** execute the replay, so on its own it cannot catch a replay
  regression. The original version of this file had only that test while claiming both.
* `test_the_replay_accepts_the_source_rows_the_endpoints_imply` executes `simulate_portfolio` and
  asserts the accepted source-row sequence against a greedy walk derived from the fixture
  parameters alone, using no production helper. That is the half that catches a replay regression,
  and reverting the replay to horizon timing must break it.

Both are needed. Neither is sufficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.tools.strategy_discovery.endpoint_publication_fixture import (
    publish_endpoints_for_frame,
)

_BAR = 3_600_000
_HORIZON = 24
_ROWS = 2000
_PID = "AGREE-USD"

# Exits earlier than entry + horizon, so a clock and the records cannot coincide. With every
# exit at the horizon the two agree by construction and this test would prove nothing.
_EARLY = 3
# An interior unlabelled row, so retained POSITIONS and source IDS diverge. Without it, id and
# position are the same number and a path that confused them would still pass.
#
# 12, not 11 (Codex fe422a90): on a stride of 3 the accepted rows are 0, 3, 6, 9, 12, ... so a
# hole at 11 is never an eligible entry and the REPLAY test could not observe it being skipped.
# At 12 the hole lands on a row that would otherwise have been accepted, so the accepted
# sequence itself changes and the skip becomes observable.
_HOLE = 12


def _label_for(row: int) -> float:
    """A label unique to its source row, invertible by `_row_from_label`."""
    return (row + 1) / 100000.0


def _row_from_label(value: float) -> int:
    return int(round(value * 100000.0)) - 1


def _published_frame(tmp_path):
    from tools.strategy_discovery.mine_profiles import _FEATURE_COLUMNS

    rng = np.random.default_rng(19)
    starts = (np.arange(_ROWS, dtype="int64") * _BAR).tolist()
    columns = {
        "ts": starts,
        "source_row_id": list(range(_ROWS)),
        "close": [1.0] * _ROWS,
        "high": [1.0] * _ROWS,
        "low": [1.0] * _ROWS,
    }
    for name in _FEATURE_COLUMNS:
        columns[name] = (
            [0.06] * _ROWS if name == "atr14_pct" else rng.uniform(0.0, 1.0, size=_ROWS).tolist()
        )
    # DISTINCT per row, so `realized_pnl` (which is the record's own label_value) identifies
    # which source row the replay realized. Small and positive so equity stays sane.
    columns[f"label_h{_HORIZON}"] = [
        float("nan") if (row == _HOLE or row + _HORIZON >= _ROWS) else _label_for(row)
        for row in range(_ROWS)
    ]
    frame = pd.DataFrame(columns)
    parquet_path = tmp_path / f"{_PID}.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet_path)
    publish_endpoints_for_frame(
        tmp_path,
        _PID,
        frame,
        horizons=[_HORIZON],
        exit_row_for=lambda row, h: row + min(_EARLY, h),
    )
    return frame, parquet_path


def _miner_eligibility(monkeypatch, parquet_path):
    """The vector the miner actually fits on, taken from the full production path."""
    from tools.strategy_discovery import mine_profiles as miner

    class Captured(Exception):
        pass

    seen: dict = {}

    def capture(features, labels, next_eligible, **kwargs):
        seen["vector"] = next_eligible.cpu().numpy().copy()
        raise Captured

    monkeypatch.setattr(miner, "fit_tree", capture)
    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon(
            pid=_PID, horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )
    assert "vector" in seen, "the miner never reached fit_tree, so nothing was compared"
    return seen["vector"]


def _load_validated(frame, tmp_path):
    """The validated records, loaded exactly as a consumer loads them."""
    import json

    from tools.strategy_discovery.endpoint_consumers import load_validated_endpoints

    sidecar = json.loads((tmp_path / f"{_PID}.endpoints.json").read_text(encoding="utf-8"))
    return load_validated_endpoints(
        tmp_path / "endpoints" / _PID, frame=frame, sidecar=sidecar, product_id=_PID
    )


def _replay_eligibility(frame, tmp_path):
    """The boundaries the shared helper produces from the validated records.

    NOTE: this is the HELPER, not the replay. `simulate_portfolio` is executed in its own test.
    """
    from tools.strategy_discovery.endpoint_consumers import eligibility_boundaries

    validated = _load_validated(frame, tmp_path)
    retained = frame[np.isfinite(frame[f"label_h{_HORIZON}"].to_numpy(dtype="float64"))]
    return eligibility_boundaries(validated, retained["source_row_id"], horizon=_HORIZON)


def test_the_miner_agrees_with_the_shared_eligibility_helper(tmp_path, monkeypatch):
    frame, parquet_path = _published_frame(tmp_path)

    mined = _miner_eligibility(monkeypatch, parquet_path)
    replayed = _replay_eligibility(frame, tmp_path)

    np.testing.assert_array_equal(mined, replayed)


def test_that_agreement_is_not_something_any_vector_would_satisfy(tmp_path, monkeypatch):
    """The guard on the guard. If the shared answer happened to equal what a wall clock
    produces, the test above would pass while both consumers used a clock."""
    import torch

    from tools.strategy_discovery.profit_split import build_next_eligible

    frame, parquet_path = _published_frame(tmp_path)
    agreed = _replay_eligibility(frame, tmp_path)

    retained = frame[np.isfinite(frame[f"label_h{_HORIZON}"].to_numpy(dtype="float64"))]
    clock = build_next_eligible(
        torch.tensor(retained["ts"].to_numpy(dtype="int64")), horizon_bars=_HORIZON
    ).numpy()

    assert not np.array_equal(agreed, clock), (
        "the endpoint-derived and clock-derived vectors are identical on this fixture, so the "
        "agreement test could not distinguish the two bases"
    )
    # And the interior hole really does separate positions from source ids.
    ids = retained["source_row_id"].to_numpy(dtype="int64")
    assert ids[_HOLE] != _HOLE, (
        f"source id at position {_HOLE} is {ids[_HOLE]}, so the unlabelled row did not shift "
        f"the mapping and an id/position confusion would go undetected"
    )


def test_the_replay_accepts_the_source_rows_the_endpoints_imply(tmp_path, monkeypatch):
    """Codex 9eade26d. Executes the actual replay and checks WHICH rows it traded.

    The expectation is an independent greedy walk over the fixture's own parameters -- exits at
    `entry + _EARLY`, one interior unlabelled row, a cap of one slot -- and touches no production
    helper. Occupancy is half-open `[entry, exit)`, so the exit row is itself eligible and the
    next accepted row is the first with `id >= previous_exit_id`.

    Falsification, and this is the half the earlier version of this file could not do: reverting
    the replay to horizon timing makes the accepted sequence start `0, 24, 48, ...` instead of
    `0, 3, 6, ...`, and the first assertion fails.
    """
    from tests.tools.strategy_discovery.rule_fixtures import machine_rule_fixture
    from tools.strategy_discovery.portfolio_sim import simulate_portfolio
    from tools.strategy_discovery.profile_loader import LoadedProfile

    frame, _ = _published_frame(tmp_path)
    validated = _load_validated(frame, tmp_path)

    profile = LoadedProfile(
        pid=_PID,
        horizon=_HORIZON,
        leaf_id=0,
        rule_path="price_over_ema20 > -1.0",  # always true, so every retained row is a candidate
        machine_rule=machine_rule_fixture("price_over_ema20 > -1.0"),
        cumulative_profit_raw=0.07,
        cumulative_profit_deflated=0.05,
        deflation_pp=0.02,
        win_rate=0.6,
        avg_win=0.08,
        avg_loss=-0.04,
        max_dd=0.2,
        sortino=1.2,
        trade_count=30,
        n_folds_passed_q0=5,
        chosen_depth=5,
        chosen_min_leaf=50,
    )

    metrics, telemetry = simulate_portfolio(
        [profile],
        cap=1,
        pid_features={_PID: frame},
        endpoints_by_pid={_PID: validated},
        bar_duration_ms=_BAR,
    )

    realized_rows = [
        _row_from_label(row.realized_pnl) for row in telemetry if row.realized_pnl is not None
    ]

    # Independent expectation: a greedy walk over the published exits, no helper involved.
    retained_ids = [row for row in range(_ROWS - _HORIZON) if row != _HOLE]
    expected_rows = []
    free_from = 0
    for row_id in retained_ids:
        if row_id < free_from:
            continue
        expected_rows.append(row_id)
        free_from = row_id + _EARLY

    assert realized_rows, "the replay realized nothing, so this test asserted nothing"
    assert realized_rows == expected_rows
    assert metrics.trade_count == len(expected_rows)
    # The discriminating prefix: endpoint exits give a stride of 3, the horizon clock gives 24.
    # And row 12 -- which a clean stride of 3 WOULD have accepted -- is absent because it carries
    # no label, so the run steps to 13 instead. That transition is what makes the interior hole
    # observable in the replay rather than only in the helper comparison.
    assert realized_rows[:6] == [0, 3, 6, 9, 13, 16]
    assert _HOLE not in realized_rows, (
        f"row {_HOLE} has no label and no endpoint, so the replay must not have traded it"
    )
