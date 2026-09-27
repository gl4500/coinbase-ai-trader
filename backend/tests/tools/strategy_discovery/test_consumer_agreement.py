"""Both consumers, one published fixture, one answer.

Before this integration the miner measured occupancy on a wall clock while the portfolio replay
measured it on a different wall clock, and nothing compared them. Each is now individually
tested against the endpoints, but that leaves the failure this effort is actually about
unguarded: the two paths DRIFTING APART again. A change to either basis would keep both
modules internally consistent and their own suites green.

This is not a tautology even though both paths end in `eligibility_boundaries`. The miner side
travels the full production route -- parquet on disk, sidecar discovery, loader validation,
retention, device placement -- while the replay side is computed directly from the validated
records. If anyone reintroduces a clock, a sort, a different retention rule or a different
filter on either side, the two stop matching here.
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
_HOLE = 11


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
    columns[f"label_h{_HORIZON}"] = [
        float("nan") if (row == _HOLE or row + _HORIZON >= _ROWS) else 0.04 for row in range(_ROWS)
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


def _replay_eligibility(frame, tmp_path):
    """The boundaries the replay would use, from the validated records directly."""
    import json

    from tools.strategy_discovery.endpoint_consumers import (
        eligibility_boundaries,
        load_validated_endpoints,
    )

    sidecar = json.loads((tmp_path / f"{_PID}.endpoints.json").read_text(encoding="utf-8"))
    validated = load_validated_endpoints(
        tmp_path / "endpoints" / _PID, frame=frame, sidecar=sidecar, product_id=_PID
    )
    retained = frame[np.isfinite(frame[f"label_h{_HORIZON}"].to_numpy(dtype="float64"))]
    return eligibility_boundaries(validated, retained["source_row_id"], horizon=_HORIZON)


def test_the_miner_and_the_replay_derive_the_same_eligibility(tmp_path, monkeypatch):
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
