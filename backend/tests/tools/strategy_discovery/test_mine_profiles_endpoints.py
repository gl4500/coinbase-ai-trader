"""The miner reads eligibility from published endpoints, or it refuses to mine.

Mining measured occupancy on `searchsorted(ts, ts + horizon * 3_600_000)` -- a wall-clock
instant resolved to a position -- while the producer recorded the row the exit rule actually
reached. Every test here is about that disagreement, and each one is written so that it fails
if the miner falls back to the clock.

The load-bearing fixture choice: `exit_row_for` publishes exits EARLIER than
`entry + horizon`. A default fixture cannot distinguish the two sources, because when every
exit is the horizon exit the clock and the endpoints agree by construction -- which is exactly
how a wired-looking integration can be vacuous.
"""

from __future__ import annotations

from pathlib import Path

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
# 2000 because the fold gates (5 outer x 3 inner, min training rows) must be SATISFIED for
# the run to reach `fit_tree`. At 300 rows the miner returned early with an insufficient-history
# warning, so the capture never fired and two tests failed for the wrong reason.
_ROWS = 2000

# Anything earlier than entry + _HORIZON. Chosen small so the endpoint vector and the clock
# vector cannot coincide at any row.
_EARLY_EXIT_BARS = 2


def _frame(rows: int = _ROWS, *, horizon: int = _HORIZON, label: float = 0.05):
    """A Phase 2 frame shaped as `labels.py` writes it, including `source_row_id`."""
    from tools.strategy_discovery.mine_profiles import _FEATURE_COLUMNS

    rng = np.random.default_rng(7)
    starts = (np.arange(rows, dtype="int64") * _BAR).tolist()
    columns = {
        "ts": starts,
        "source_row_id": list(range(rows)),
        "close": [1.0] * rows,
        "high": [1.0] * rows,
        "low": [1.0] * rows,
    }
    for name in _FEATURE_COLUMNS:
        columns[name] = (
            [0.06] * rows if name == "atr14_pct" else rng.uniform(0.0, 1.0, size=rows).tolist()
        )
    # The producer cannot label a row whose exit bar does not exist, so the tail is NaN.
    columns[f"label_h{horizon}"] = [
        float(label) if row + horizon < rows else float("nan") for row in range(rows)
    ]
    return pd.DataFrame(columns)


def _write_published(tmp_path: Path, pid: str = "END-USD", *, early: bool, frame=None) -> Path:
    """Write the parquet AND its publication into one directory, as Phase 2 does."""
    frame = _frame() if frame is None else frame
    parquet_path = tmp_path / f"{pid}.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet_path)
    publish_endpoints_for_frame(
        tmp_path,
        pid,
        frame,
        horizons=[_HORIZON],
        exit_row_for=(lambda row, _h: row + _EARLY_EXIT_BARS) if early else None,
    )
    return parquet_path


def _capture_eligibility(monkeypatch, miner):
    """Stop the miner at `fit_tree` and hand back the `next_eligible` it was given."""

    class Captured(Exception):
        pass

    seen: dict = {}

    def capture(features, labels, next_eligible, **kwargs):
        seen["next_eligible"] = next_eligible.cpu().numpy().copy()
        raise Captured

    monkeypatch.setattr(miner, "fit_tree", capture)
    return Captured, seen


def test_eligibility_comes_from_the_endpoints_where_an_early_exit_disagrees_with_the_clock(
    tmp_path, monkeypatch
):
    """The whole point. Every published exit is `entry + 2`, while the clock says
    `entry + 24`, so the two vectors cannot agree at any row and the assertion identifies
    which source was used rather than merely that some vector arrived."""
    from tools.strategy_discovery import mine_profiles as miner

    parquet_path = _write_published(tmp_path, early=True)
    Captured, seen = _capture_eligibility(monkeypatch, miner)

    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon(
            pid="END-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )

    retained = _ROWS - _HORIZON  # rows 0..275 carry a finite label
    expected = np.minimum(np.arange(retained) + _EARLY_EXIT_BARS, retained)
    np.testing.assert_array_equal(seen["next_eligible"], expected)

    # Non-vacuity: the clock would have produced something else entirely.
    clock = np.minimum(np.arange(retained) + _HORIZON, retained)
    assert not np.array_equal(expected, clock), (
        "fixture is vacuous -- the endpoint and clock vectors agree, so this test could not "
        "tell which source the miner used"
    )


def test_a_missing_publication_stops_the_miner_instead_of_falling_back_to_a_clock(
    tmp_path, monkeypatch
):
    """A frame with no publication must not be mined on horizon arithmetic. Silent fallback
    is the failure this whole contract exists to remove, and it would be invisible: the run
    would simply report profiles computed on a different occupancy basis."""
    from tools.strategy_discovery import mine_profiles as miner
    from tools.strategy_discovery import profit_split

    frame = _frame()
    parquet_path = tmp_path / "BARE-USD.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("the miner fell back to the wall clock")

    # Patched at its source module, not via a miner alias: the miner no longer imports the
    # clock baseline at all, and patching the definition catches a call through any path.
    monkeypatch.setattr(profit_split, "build_next_eligible", forbidden)

    with pytest.raises(FileNotFoundError, match="BARE-USD.endpoints.json"):
        miner.mine_profiles_for_pid_horizon(
            pid="BARE-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )


def test_the_publication_is_validated_against_the_full_frame_not_the_filtered_one(
    tmp_path, monkeypatch
):
    """`source_row_id` must still be the original ordinals `0..n-1` when the binding is
    checked, so validation happens BEFORE the finite-label filter. Validating the filtered
    frame would recompute a different `data_id` and could never match the publication."""
    from tools.strategy_discovery import mine_profiles as miner

    parquet_path = _write_published(tmp_path, early=True)
    real_loader = miner.load_validated_endpoints
    seen: dict = {}

    def spy(directory, *, frame, sidecar, product_id):
        seen["rows"] = len(frame)
        seen["ordinals"] = frame["source_row_id"].tolist()
        return real_loader(directory, frame=frame, sidecar=sidecar, product_id=product_id)

    monkeypatch.setattr(miner, "load_validated_endpoints", spy)
    Captured, _ = _capture_eligibility(monkeypatch, miner)

    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon(
            pid="END-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )

    assert seen["rows"] == _ROWS, (
        f"the loader saw {seen['rows']} rows; the filtered frame has {_ROWS - _HORIZON}, so a "
        f"smaller number means the binding was checked against the wrong frame"
    )
    assert seen["ordinals"] == list(range(_ROWS))


def test_an_infinite_label_is_not_retained_so_expectations_and_retention_agree(
    tmp_path, monkeypatch
):
    """`dropna` keeps +/-inf -- executed: `pd.Series([0.1, nan, inf]).dropna()` keeps inf.
    Retention must use `isfinite`, matching how the publication chose its rows, or an
    inf-labelled row is retained with no endpoint and fails as a spurious MissingEndpoint."""
    from tools.strategy_discovery import mine_profiles as miner

    frame = _frame()
    frame.loc[5, f"label_h{_HORIZON}"] = float("inf")
    parquet_path = _write_published(tmp_path, early=True, frame=frame)
    Captured, seen = _capture_eligibility(monkeypatch, miner)

    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon(
            pid="END-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )

    # One row fewer than the all-finite case, and no MissingEndpoint was raised.
    assert len(seen["next_eligible"]) == _ROWS - _HORIZON - 1


def test_purge_and_embargo_stay_on_the_configured_horizon_not_the_observed_holding(
    tmp_path, monkeypatch
):
    """Every published record here holds 2 bars, but the label still looks 24 bars ahead, so
    the causal purge must remain the CONFIGURED horizon. Deriving embargo from observed
    holding times would shrink the purge to 2 and readmit leakage the folds exist to prevent."""
    from tools.strategy_discovery import mine_profiles as miner

    parquet_path = _write_published(tmp_path, early=True)
    seen: dict = {}
    real_outer = miner.outer_folds

    class Captured(Exception):
        pass

    def spy(n, *, n_folds, embargo_bars):
        seen["embargo_bars"] = embargo_bars
        raise Captured

    monkeypatch.setattr(miner, "outer_folds", spy)

    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon(
            pid="END-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )

    assert seen["embargo_bars"] == _HORIZON, (
        f"embargo came back as {seen['embargo_bars']}; observed holding is "
        f"{_EARLY_EXIT_BARS} bars and must never become the purge width"
    )
    assert real_outer is not spy


def test_a_publication_that_fails_validation_stops_the_miner(tmp_path, monkeypatch):
    """A corrupt or mismatched artifact must raise, not be skipped. Skipping would mine the
    product on a different basis while still emitting profiles that look ordinary."""
    import json

    from tools.strategy_discovery import mine_profiles as miner
    from tools.strategy_discovery import profit_split

    parquet_path = _write_published(tmp_path, early=True)
    sidecar_path = tmp_path / "END-USD.endpoints.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["data_id"] = "sha256:" + "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("the miner fell back to the wall clock")

    # Patched at its source module, not via a miner alias: the miner no longer imports the
    # clock baseline at all, and patching the definition catches a call through any path.
    monkeypatch.setattr(profit_split, "build_next_eligible", forbidden)

    with pytest.raises(ValueError):
        miner.mine_profiles_for_pid_horizon(
            pid="END-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )


def test_a_declared_multi_horizon_publication_still_mines_one_horizon(tmp_path, monkeypatch):
    """Codex 01a768bd. The publication declares horizons 1 AND 24; this run mines only 24.

    The trap it guards: expectations built for just the mined horizon are rejected on the
    first record of any other horizon, because `require_complete_coverage=False` relaxes
    missing expected keys and never permits UNEXPECTED records. Verified by execution
    earlier -- both flag values rejected a single-horizon expectation set against a
    multi-horizon dataset.
    """
    from tools.strategy_discovery import mine_profiles as miner

    frame = _frame()
    # A finite h1 label wherever its exit row exists, so horizon 1 really is populated.
    # Finite only where the PUBLISHED exit row exists. Keying this off horizon 1 alone
    # produced entry 1998 -> exit 2000 and the fixture's impossible-record guard fired.
    frame["label_h1"] = [
        0.02 if row + _EARLY_EXIT_BARS < _ROWS else float("nan") for row in range(_ROWS)
    ]
    parquet_path = tmp_path / "MULTI-USD.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet_path)
    publish_endpoints_for_frame(
        tmp_path,
        "MULTI-USD",
        frame,
        horizons=[1, _HORIZON],
        # min(bars, h) because `bars_held` may not exceed min(horizon, cap) -- real validation
        # rejected a horizon-1 record claiming a 2-bar hold. So h1 gets its horizon exit and
        # h24 gets the early stop, which is the combination the test needs anyway.
        exit_row_for=lambda row, h: row + min(_EARLY_EXIT_BARS, h),
    )

    Captured, seen = _capture_eligibility(monkeypatch, miner)
    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon(
            pid="MULTI-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )

    assert "next_eligible" in seen, "capture never fired, so this test asserted nothing"
    retained = _ROWS - _HORIZON
    np.testing.assert_array_equal(
        seen["next_eligible"], np.minimum(np.arange(retained) + _EARLY_EXIT_BARS, retained)
    )


def test_an_interior_filtered_row_shifts_the_retained_positions_exactly(tmp_path, monkeypatch):
    """Codex 01a768bd: assert the MAPPING, not the length.

    Row 5 carries no label, so it is published no endpoint and retained by nothing. Every
    later source id therefore sits one position lower than its id. An exit row is a source
    ID and must be resolved through the retained ids; treating it as a position would put
    entry 4's boundary at 6 instead of 5, which a length check cannot see.
    """
    from tools.strategy_discovery import mine_profiles as miner

    frame = _frame()
    frame.loc[5, f"label_h{_HORIZON}"] = float("nan")
    parquet_path = _write_published(tmp_path, pid="HOLE-USD", early=True, frame=frame)

    Captured, seen = _capture_eligibility(monkeypatch, miner)
    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon(
            pid="HOLE-USD", horizon=_HORIZON, parquet_path=parquet_path, device="cpu"
        )

    assert "next_eligible" in seen, "capture never fired, so this test asserted nothing"
    retained_ids = np.array(
        [row for row in range(_ROWS - _HORIZON) if row != 5]
        + [row for row in range(_ROWS - _HORIZON, _ROWS) if False],
        dtype="int64",
    )
    expected = np.searchsorted(retained_ids, retained_ids + _EARLY_EXIT_BARS, side="left")
    np.testing.assert_array_equal(seen["next_eligible"], expected)

    # The discriminating row: entry id 4 exits at source id 6, which is retained POSITION 5.
    position_of_id_4 = int(np.searchsorted(retained_ids, 4))
    assert seen["next_eligible"][position_of_id_4] == 5
    assert expected[position_of_id_4] != 6, (
        "if the exit id were used as a position directly this would be 6, so the assertion "
        "above would not distinguish the two implementations"
    )
