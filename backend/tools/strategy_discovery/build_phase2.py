"""Phase 2 orchestrator — universe → per-token feature+label parquet.

Loads inputs (curated universe JSON, per-pid hourly OHLCV parquet, per-pid
daily CoinPaprika marketcap parquet, single supply-snapshot parquet),
applies features → tokenomic stamping → labels, and writes one parquet
per pid to backend/data/phase2/.

Run:
    cd backend && python -m tools.strategy_discovery.build_phase2 \\
        --universe ../docs/superpowers/specs/2026-05-23-universe-50.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.endpoint_dataset import write_dataset  # noqa: E402
from tools.strategy_discovery.endpoint_records import endpoint_digest  # noqa: E402
from tools.strategy_discovery.features import (  # noqa: E402
    add_trend_features,
    first_valid_index,
)
from tools.strategy_discovery.labels import (  # noqa: E402
    _DEFAULT_ATR_TRAIL_FLOOR,
    _DEFAULT_HORIZONS,
    _DEFAULT_MAX_HOLD_BARS,
    _DEFAULT_ROUND_TRIP_FEE,
    _DEFAULT_STOP_LOSS_PCT,
    COST_VERSION,
    LABEL_VERSION,
    simulate_labels_with_endpoints,
)
from tools.strategy_discovery.tokenomic_stamp import (  # noqa: E402
    SupplySnapshot,
    stamp_tokenomic,
)

_SCHEMA_VERSION = 1
_DEFAULT_HISTORY_DIR = Path(BACKEND) / "data" / "history"
_DEFAULT_MARKETCAP_DIR = Path(BACKEND) / "data" / "marketcap"
_DEFAULT_SUPPLY_PATH = Path(BACKEND) / "data" / "supply" / "snapshot.parquet"
_DEFAULT_OUTPUT_DIR = Path(BACKEND) / "data" / "phase2"


# The sidecar schema a consumer reads; bump together with any field change.
_SIDECAR_VERSION = 1
# Names the recipe that produced atr14_pct, so a consumer can recompute the frame identity.
_FEATURE_RECIPE = "atr14_pct_wilder_v1"


@dataclass
class BuildResult:
    pid: str
    rows_written: int = 0
    rows_dropped_missing_volume: int = 0
    nan_label_counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    # The digest a consumer must retain to verify the published endpoints against
    # something it did not compute. None when the frame produced no endpoints at all.
    endpoint_manifest_digest: Optional[str] = None
    endpoint_dispositions: Dict[str, int] = field(default_factory=dict)


def _load_supply_snapshot(supply_path: Path, pid: str) -> Optional[SupplySnapshot]:
    """Read one pid's row from the supply snapshot parquet. Returns None if absent."""
    if not supply_path.exists():
        return None
    tbl = pq.read_table(supply_path).to_pandas()
    row = tbl[tbl["pid"] == pid]
    if row.empty:
        return None
    r = row.iloc[0]
    max_supply = None if pd.isna(r["max_supply"]) else float(r["max_supply"])
    return SupplySnapshot(
        pid=pid,
        circulating=float(r["circulating"]),
        total=float(r["total"]),
        max_supply=max_supply,
    )


def _load_history_parquet(path: Path) -> pd.DataFrame:
    """Read history parquet, rename 'start' (epoch s) → 'ts' (epoch ms)."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={"start": "ts"})
    df["ts"] = df["ts"].astype("int64") * 1_000
    return df.sort_values("ts").reset_index(drop=True)


def _load_marketcap_parquet(path: Path) -> pd.DataFrame:
    """Read marketcap parquet, rename 'start' (epoch s) → 'ts' (epoch ms)."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={"start": "ts"})
    df["ts"] = df["ts"].astype("int64") * 1_000
    return df.sort_values("ts").reset_index(drop=True)


def build_phase2_for_pid(
    pid: str,
    history_dir: Path,
    marketcap_dir: Path,
    supply_path: Path,
    output_dir: Path,
) -> BuildResult:
    """End-to-end Phase 2 build for one pid. Writes output_dir/{pid}.parquet."""
    history_path = Path(history_dir) / f"{pid}.parquet"
    marketcap_path = Path(marketcap_dir) / f"{pid}.parquet"
    if not history_path.exists():
        return BuildResult(pid=pid, error=f"missing history: {history_path}")
    if not marketcap_path.exists():
        return BuildResult(pid=pid, error=f"missing marketcap: {marketcap_path}")
    supply = _load_supply_snapshot(Path(supply_path), pid)
    if supply is None:
        return BuildResult(pid=pid, error=f"missing supply: {pid}")

    df_hourly = _load_history_parquet(history_path)
    if len(df_hourly) < 200:
        return BuildResult(pid=pid, error=f"history too short ({len(df_hourly)} < 200 bars)")
    df_daily = _load_marketcap_parquet(marketcap_path)

    # features → drop warmup → stamp → labels
    df_feat = add_trend_features(df_hourly)
    cut = first_valid_index(df_feat, min_warmup=200)
    df_feat = df_feat.iloc[cut:].reset_index(drop=True)
    rows_pre_drop = len(df_feat)
    df_stamped = stamp_tokenomic(df_feat, df_daily, supply, drop_on_missing_volume=True)
    rows_dropped = rows_pre_drop - len(df_stamped)
    # Publishes the exit the simulation already chose, instead of discarding it and
    # leaving three consumers to re-derive it from different clocks. Label values are
    # unchanged: both paths share one _SimResult.
    df_labeled, endpoints, dispositions = simulate_labels_with_endpoints(
        df_stamped,
        horizons=list(_DEFAULT_HORIZONS),
        product_id=pid,
        with_dispositions=True,
    )

    df_labeled["pid"] = pid
    df_labeled["schema_version"] = _SCHEMA_VERSION
    nan_counts = {
        f"label_h{h}": int(df_labeled[f"label_h{h}"].isna().sum()) for h in _DEFAULT_HORIZONS
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{pid}.parquet"
    df_labeled.to_parquet(out_path, compression="snappy", index=False)

    endpoint_digest_value = _publish_endpoints(Path(output_dir), pid, endpoints, _DEFAULT_HORIZONS)
    return BuildResult(
        pid=pid,
        rows_written=len(df_labeled),
        rows_dropped_missing_volume=rows_dropped,
        nan_label_counts=nan_counts,
        endpoint_manifest_digest=endpoint_digest_value,
        endpoint_dispositions=dict(dispositions),
    )


def _publish_endpoints(
    output_dir: Path, pid: str, endpoints: List, declared_horizons: Sequence[int]
) -> Optional[str]:
    """Write the endpoint dataset and the sidecar a consumer verifies it against.

    Returns the manifest digest, or None when the frame produced no endpoints -- which is
    normal for a frame too short to carry any label, and must not be confused with a
    failure. `write_dataset` refuses to publish an empty set precisely so that
    nothing-survived cannot look like nothing-was-attempted.

    The SIDECAR is the point. A consumer reads the manifest digest and the declared exit
    config from here, so it checks the dataset against values it did not compute. Per-record
    digests are captured now, at publication: one recomputed later from the record under
    test would attest nothing. Contract section 9.2 also states the limit -- this is an
    anchor, not a root of trust, since replacing frame, dataset and sidecar together is
    coherent and undetectable.
    """
    sidecar_path = output_dir / f"{pid}.endpoints.json"
    if not endpoints:
        # Remove any earlier run's sidecar rather than leaving it to describe a dataset
        # that no longer corresponds to the parquet beside it. `write_dataset` already
        # refuses to publish an empty set; this closes the other half.
        sidecar_path.unlink(missing_ok=True)
        return None

    endpoint_dir = output_dir / "endpoints" / pid
    data_id = endpoints[0].data_id
    manifest_digest = write_dataset(endpoint_dir, endpoints=endpoints, data_id=data_id)

    sidecar = {
        "sidecar_version": _SIDECAR_VERSION,
        "manifest_digest": manifest_digest,
        "data_id": data_id,
        "product_id": pid,
        # The REQUESTED horizons, never the surviving ones. A frame too short for a long
        # horizon emits no endpoints for it, and a survivor-derived set would drop that
        # horizon from the sidecar entirely -- so a consumer would build expectations only
        # for horizons that happened to survive, and complete coverage would pass because
        # the missing horizon was never expected. That is coverage derived from the thing
        # under test, the same defect the dataset loader already refuses. Executed on a
        # 30-row frame: declared [1, 4, 24, 72, 168], surviving [1, 4, 24].
        "horizons": sorted(int(h) for h in declared_horizons),
        "bar_duration_ms": int(endpoints[0].bar_duration_ms),
        "feature_recipe": _FEATURE_RECIPE,
        "label_version": LABEL_VERSION,
        "cost_version": COST_VERSION,
        # The CONFIGURED cap, which is not a horizon. A consumer that rebuilt it from a
        # record's own horizon would reject every valid short-horizon record.
        "exit_config": {
            "stop_loss_pct": _DEFAULT_STOP_LOSS_PCT,
            "atr_trail_floor": _DEFAULT_ATR_TRAIL_FLOOR,
            "max_hold_bars": _DEFAULT_MAX_HOLD_BARS,
            "round_trip_fee": _DEFAULT_ROUND_TRIP_FEE,
        },
        "record_digests": {f"{e.horizon}:{e.entry_row_id}": endpoint_digest(e) for e in endpoints},
    }
    # Same-directory temp then os.replace: a half-written sidecar would be a parse error
    # at best, and a plausible-looking partial document at worst.
    temporary = sidecar_path.with_name(sidecar_path.name + ".partial")
    try:
        # The WRITE is inside the cleanup scope too, not just the rename: a failure
        # part-way through writing would otherwise leave a .partial behind that the
        # cleanup claim did not actually cover.
        temporary.write_text(json.dumps(sidecar, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, sidecar_path)
    except OSError:
        # What this guarantees is that the previous sidecar is INTACT -- not that it is
        # still correct. If the dataset beside it was already replaced, the pair is
        # mismatched, and it is the consumer's data_id recompute that must reject it.
        temporary.unlink(missing_ok=True)
        raise
    return manifest_digest


def _pids_from_universe_json(universe_path: Path) -> List[str]:
    """Flatten {cohort: [pids]} into a deduplicated sorted pid list."""
    with open(universe_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen: set = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def build_phase2_for_universe(
    universe_path: Path,
    history_dir: Optional[Path] = None,
    marketcap_dir: Optional[Path] = None,
    supply_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[BuildResult]:
    """Iterate every pid in the universe JSON; collect per-pid BuildResults."""
    history_dir = Path(history_dir) if history_dir else _DEFAULT_HISTORY_DIR
    marketcap_dir = Path(marketcap_dir) if marketcap_dir else _DEFAULT_MARKETCAP_DIR
    supply_path = Path(supply_path) if supply_path else _DEFAULT_SUPPLY_PATH
    output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    pids = _pids_from_universe_json(Path(universe_path))
    results: List[BuildResult] = []
    for pid in pids:
        results.append(
            build_phase2_for_pid(
                pid,
                history_dir,
                marketcap_dir,
                supply_path,
                output_dir,
            )
        )
    return results


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint — build Phase 2 parquet for an entire universe."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 2 features+labels for a universe.")
    parser.add_argument(
        "--universe",
        default=os.path.join(
            BACKEND, "..", "docs", "superpowers", "specs", "2026-05-23-universe-50.json"
        ),
    )
    parser.add_argument("--history-dir", default=None)
    parser.add_argument("--marketcap-dir", default=None)
    parser.add_argument("--supply", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    results = build_phase2_for_universe(
        Path(args.universe),
        history_dir=Path(args.history_dir) if args.history_dir else None,
        marketcap_dir=Path(args.marketcap_dir) if args.marketcap_dir else None,
        supply_path=Path(args.supply) if args.supply else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    n_ok = sum(1 for r in results if r.error is None)
    n_err = len(results) - n_ok
    print(f"  ok:    {n_ok}", flush=True)
    print(f"  error: {n_err}", flush=True)
    for r in results:
        if r.error:
            print(f"    [ERR] {r.pid}: {r.error}", flush=True)
        else:
            print(
                f"    {r.pid}: {r.rows_written:,} rows "
                f"(dropped {r.rows_dropped_missing_volume:,} missing-vol)",
                flush=True,
            )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
