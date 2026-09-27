"""Tests for tools.strategy_discovery.build_phase2 (Phase 2 orchestrator)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.build_phase2 import (
    build_phase2_for_pid,
    build_phase2_for_universe,
)
from tools.strategy_discovery.endpoint_consumers import (  # noqa: E402
    MissingEndpoint,
    load_validated_endpoints,
)
from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER  # noqa: E402
from tools.strategy_discovery.labels import (  # noqa: E402
    _DEFAULT_HORIZONS,
    _DEFAULT_MAX_HOLD_BARS,
)

_HOUR_S = 3_600
_DAY_S = 86_400


def _write_history_parquet(path: Path, n_hours: int = 400, start_day_s: int = 1_000 * _DAY_S):
    rng = np.random.default_rng(11)
    close = 100.0 + rng.normal(0.0, 0.5, size=n_hours).cumsum()
    df = pd.DataFrame(
        {
            "start": start_day_s + np.arange(n_hours, dtype="int64") * _HOUR_S,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n_hours, 1_000.0),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def _write_marketcap_parquet(path: Path, n_days: int = 20, start_day_s: int = 1_000 * _DAY_S):
    df = pd.DataFrame(
        {
            "start": start_day_s + np.arange(n_days, dtype="int64") * _DAY_S,
            "market_cap": np.full(n_days, 100_000.0),
            "volume_24h": np.full(n_days, 5_000.0),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def _write_supply_snapshot(path: Path, pid: str = "FOO-USD"):
    schema = pa.schema(
        [
            pa.field("pid", pa.string()),
            pa.field("circulating", pa.float64()),
            pa.field("total", pa.float64()),
            pa.field("max_supply", pa.float64()),
            pa.field("ingest_ts", pa.int64()),
            pa.field("schema_version", pa.int32()),
        ]
    )
    tbl = pa.table(
        {
            "pid": [pid],
            "circulating": [1_000_000.0],
            "total": [2_000_000.0],
            "max_supply": [None],
            "ingest_ts": [1_700_000_000],
            "schema_version": [1],
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="snappy")


def test_build_phase2_for_pid_writes_parquet(tmp_path: Path):
    pid = "FOO-USD"
    history_dir = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    output_dir = tmp_path / "phase2"
    _write_history_parquet(history_dir / f"{pid}.parquet", n_hours=400)
    _write_marketcap_parquet(marketcap_dir / f"{pid}.parquet", n_days=20)
    _write_supply_snapshot(supply_path, pid=pid)

    result = build_phase2_for_pid(pid, history_dir, marketcap_dir, supply_path, output_dir)

    assert result.error is None, f"unexpected error: {result.error}"
    assert result.rows_written > 0
    assert (output_dir / f"{pid}.parquet").exists()

    out = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()
    # Must have all 13 features + 5 labels + identifiers + schema_version
    for col in (
        "ts",
        "pid",
        "market_cap",
        "fdv",
        "fdv_over_mc",
        "circ_over_total",
        "vol_24h",
        "vol_over_mc",
        "price_over_ema20",
        "price_over_ema50",
        "price_over_ema200",
        "ret_1h_sign",
        "ret_24h_sign",
        "ret_7d_sign",
        "atr14_pct",
        "label_h1",
        "label_h4",
        "label_h24",
        "label_h72",
        "label_h168",
        "schema_version",
    ):
        assert col in out.columns, f"missing column {col}"
    assert (out["pid"] == pid).all()
    assert (out["schema_version"] == 1).all()


def test_build_phase2_for_universe_iterates_all_pids(tmp_path: Path):
    pids = ["FOO-USD", "BAR-USD", "BAZ-USD"]
    history_dir = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    output_dir = tmp_path / "phase2"

    for p in pids:
        _write_history_parquet(history_dir / f"{p}.parquet", n_hours=400)
        _write_marketcap_parquet(marketcap_dir / f"{p}.parquet", n_days=20)
    # Universe JSON uses Phase 1 cohort layout: {cohort: [pids]}
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "large": ["FOO-USD"],
                "mid": ["BAR-USD"],
                "high_fdv_ratio": ["BAZ-USD"],
                "low_turnover": [],
            }
        ),
        encoding="utf-8",
    )

    # Single supply snapshot parquet for all three pids
    schema = pa.schema(
        [
            pa.field("pid", pa.string()),
            pa.field("circulating", pa.float64()),
            pa.field("total", pa.float64()),
            pa.field("max_supply", pa.float64()),
            pa.field("ingest_ts", pa.int64()),
            pa.field("schema_version", pa.int32()),
        ]
    )
    tbl = pa.table(
        {
            "pid": pids,
            "circulating": [1_000_000.0] * 3,
            "total": [2_000_000.0] * 3,
            "max_supply": [None] * 3,
            "ingest_ts": [1_700_000_000] * 3,
            "schema_version": [1] * 3,
        },
        schema=schema,
    )
    supply_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, supply_path, compression="snappy")

    results = build_phase2_for_universe(
        universe_path,
        history_dir=history_dir,
        marketcap_dir=marketcap_dir,
        supply_path=supply_path,
        output_dir=output_dir,
    )
    assert len(results) == 3
    assert {r.pid for r in results} == set(pids)
    assert all(r.error is None for r in results)
    for p in pids:
        assert (output_dir / f"{p}.parquet").exists()


def test_build_result_reports_drop_counts(tmp_path: Path):
    # Build marketcap parquet with a NaN volume_24h on day D+5 — should drop
    # 24 hourly rows from the output and report it in BuildResult.
    pid = "FOO-USD"
    history_dir = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    output_dir = tmp_path / "phase2"

    _write_history_parquet(history_dir / f"{pid}.parquet", n_hours=400)
    _write_supply_snapshot(supply_path, pid=pid)

    # Marketcap with a single NaN-volume day.
    # Warmup cut is at hour 200 = day 1008.33 from epoch.
    # NaN at day index 9 (day 1009): after T+1 shift it covers day 1010,
    # which lands inside the post-warmup hourly range (1008.33–1016.62).
    n_days = 20
    start_day_s = 1_000 * _DAY_S
    vols = np.full(n_days, 5_000.0)
    vols[9] = np.nan
    df = pd.DataFrame(
        {
            "start": start_day_s + np.arange(n_days, dtype="int64") * _DAY_S,
            "market_cap": np.full(n_days, 100_000.0),
            "volume_24h": vols,
        }
    )
    marketcap_path = marketcap_dir / f"{pid}.parquet"
    marketcap_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False), marketcap_path, compression="snappy"
    )

    result = build_phase2_for_pid(pid, history_dir, marketcap_dir, supply_path, output_dir)
    assert result.error is None
    assert result.rows_dropped_missing_volume > 0
    # nan_label_counts should be a dict over all 5 horizons
    assert set(result.nan_label_counts.keys()) == {
        "label_h1",
        "label_h4",
        "label_h24",
        "label_h72",
        "label_h168",
    }


# ── the producer publishes endpoints alongside the labels it already wrote ────
#
# Contract §9.2. The sidecar is what a consumer reads the manifest digest and the declared
# exit config from, so it verifies the dataset against values it did not compute. The
# schema is fixed here and consumed by the adapter.


def _build_with_endpoints(tmp_path: Path, pid: str = "FOO-USD"):
    history_dir = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    output_dir = tmp_path / "phase2"
    _write_history_parquet(history_dir / f"{pid}.parquet", n_hours=400)
    _write_marketcap_parquet(marketcap_dir / f"{pid}.parquet", n_days=20)
    _write_supply_snapshot(supply_path, pid=pid)
    result = build_phase2_for_pid(pid, history_dir, marketcap_dir, supply_path, output_dir)
    assert result.error is None, f"unexpected error: {result.error}"
    return result, output_dir, pid


def test_the_parquet_carries_source_row_id_as_exact_ordinals(tmp_path: Path):
    """`dropna(...).reset_index(drop=True)` downstream destroys row identity. This column
    is the only bridge from a filtered frame back to the rows the endpoints reference."""
    _, output_dir, pid = _build_with_endpoints(tmp_path)
    frame = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()
    assert frame["source_row_id"].tolist() == list(range(len(frame)))


def test_publishing_endpoints_does_not_move_a_single_label(tmp_path: Path):
    """The producer swaps one function for another. The only acceptable outcome is labels
    that are identical bit-for-bit, compared by float.hex() rather than by tolerance -- a
    relative tolerance would hide a real move."""
    from tools.strategy_discovery.labels import simulate_dynamic_exit_labels

    _, output_dir, pid = _build_with_endpoints(tmp_path)
    frame = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()

    # recompute from the published frame's own inputs, independently of the endpoint path
    recomputed = simulate_dynamic_exit_labels(
        frame[["ts", "open", "high", "low", "close", "atr14_pct"]].copy(),
        horizons=list(_DEFAULT_HORIZONS),
    )
    for horizon in _DEFAULT_HORIZONS:
        column = f"label_h{horizon}"
        published = [None if pd.isna(v) else float(v).hex() for v in frame[column]]
        expected = [None if pd.isna(v) else float(v).hex() for v in recomputed[column]]
        assert published == expected, f"{column} moved"


def test_the_sidecar_declares_everything_a_consumer_needs(tmp_path: Path):
    result, output_dir, pid = _build_with_endpoints(tmp_path)
    sidecar = json.loads((output_dir / f"{pid}.endpoints.json").read_text(encoding="utf-8"))

    assert sidecar["sidecar_version"] == 1
    assert sidecar["manifest_digest"] == result.endpoint_manifest_digest
    assert sidecar["manifest_digest"].startswith("sha256:")
    assert sidecar["product_id"] == pid
    assert sorted(sidecar["horizons"]) == sorted(_DEFAULT_HORIZONS)
    assert sidecar["bar_duration_ms"] == 3_600_000
    assert sidecar["feature_recipe"] == "atr14_pct_wilder_v1"

    # The CAP, not a horizon. A consumer rebuilding this from record.horizon would reject
    # every valid short-horizon record.
    assert sidecar["exit_config"]["max_hold_bars"] == _DEFAULT_MAX_HOLD_BARS
    # Worth stating because it nearly hid the bug: the default cap 168 EQUALS the longest
    # default horizon, so a consumer that wrongly rebuilt the cap from record.horizon would
    # still validate horizon-168 records. Only a shorter horizon exposes it -- which is why
    # the adapter regression uses horizon 1.
    assert _DEFAULT_MAX_HOLD_BARS in _DEFAULT_HORIZONS
    assert any(h != _DEFAULT_MAX_HOLD_BARS for h in _DEFAULT_HORIZONS)
    assert set(sidecar["exit_config"]) == {
        "stop_loss_pct",
        "atr_trail_floor",
        "max_hold_bars",
        "round_trip_fee",
    }
    assert (output_dir / "endpoints" / pid / "endpoints.jsonl").exists()


def test_per_record_digests_are_stored_at_publication(tmp_path: Path):
    """A digest recomputed from the record under test attests nothing. These are captured
    while the record is being written, so a later validation compares against a value it
    did not derive from the thing it is checking."""
    from tools.strategy_discovery.endpoint_records import endpoint_digest

    _, output_dir, pid = _build_with_endpoints(tmp_path)
    sidecar = json.loads((output_dir / f"{pid}.endpoints.json").read_text(encoding="utf-8"))
    frame = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()

    validated = load_validated_endpoints(
        output_dir / "endpoints" / pid, frame=frame, sidecar=sidecar, product_id=pid
    )
    assert validated.records, "the fixture must produce endpoints"
    for record in validated.records:
        key = f"{record.horizon}:{record.entry_row_id}"
        assert sidecar["record_digests"][key] == endpoint_digest(record)


def test_the_producers_own_output_passes_the_consumer_adapter(tmp_path: Path):
    """The loop closed. Everything the adapter checks -- recomputed data_id, complete
    coverage across every declared horizon, the configured cap, per-record semantics --
    is satisfied by what the producer actually writes, with no fixture in between.

    If the producer and the adapter ever disagree about the sidecar schema or the identity
    recompute, this is the test that fails.
    """
    _, output_dir, pid = _build_with_endpoints(tmp_path)
    sidecar = json.loads((output_dir / f"{pid}.endpoints.json").read_text(encoding="utf-8"))
    frame = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()

    validated = load_validated_endpoints(
        output_dir / "endpoints" / pid, frame=frame, sidecar=sidecar, product_id=pid
    )
    assert validated.semantic_validation_performed is True
    assert validated.data_id == sidecar["data_id"]
    # the version-wide blocker survives publication and validation
    for record in validated.records:
        assert CAUSALITY_BLOCKER in record.blockers


def test_a_stale_sidecar_from_an_earlier_run_cannot_validate_a_new_frame(tmp_path: Path):
    """Publication is per-file, so an interrupted rebuild can leave a NEW parquet beside an
    OLD sidecar and dataset. Nothing silently validates: the adapter recomputes `data_id`
    from the frame, and the retained sidecar describes different content.

    This is containment by the identity recompute, not by atomicity, and the distinction is
    worth a test rather than a claim.
    """
    _, output_dir, pid = _build_with_endpoints(tmp_path)
    stale_sidecar = json.loads((output_dir / f"{pid}.endpoints.json").read_text(encoding="utf-8"))

    # a different frame lands in place, as an interrupted regeneration would leave it
    frame = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()
    frame.loc[0, "close"] = float(frame.loc[0, "close"]) + 1.0

    with pytest.raises(MissingEndpoint, match="data_id"):
        load_validated_endpoints(
            output_dir / "endpoints" / pid,
            frame=frame,
            sidecar=stale_sidecar,
            product_id=pid,
        )


def test_a_frame_with_no_endpoints_leaves_no_stale_sidecar_behind(tmp_path: Path):
    """A rebuild that produces no endpoints must not leave an earlier run's sidecar in
    place, where it would describe a dataset that no longer corresponds to the parquet.

    `write_dataset` already refuses to publish an empty set, so nothing-survived cannot look
    like nothing-was-attempted; this closes the other half, where the previous run's
    description survives its own data.
    """
    result, output_dir, pid = _build_with_endpoints(tmp_path)
    sidecar_path = output_dir / f"{pid}.endpoints.json"
    assert sidecar_path.exists() and result.endpoint_manifest_digest is not None

    from tools.strategy_discovery.build_phase2 import _publish_endpoints

    assert _publish_endpoints(output_dir, pid, []) is None
    assert not sidecar_path.exists(), "an earlier run's sidecar outlived its data"
