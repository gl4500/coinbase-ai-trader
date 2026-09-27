"""The ATR diagnostic restricted to entries whose OWN forward window is contiguous hourly.

Every Phase 2 frame on disk fails the full-frame contiguous-hourly guard, so
`atr_causality_report.run_probe` returns `no_data` and the ATR question stays unmeasured. But only
~10% of labelled entries actually have a window spanning a gap. This module measures the variants
on the sub-population where a row-count horizon genuinely equals the nominal duration.

Separate module rather than more surface on the report layer: this is a POPULATION SELECTION
concern, and mixing it into the full-frame scanner would blur which population a number came
from -- the exact confusion that voided the first set of results.

The selection rule is declared once in `SELECTION_RULE`, reported verbatim in the payload, and
never depends on any outcome. What it cannot do is stated just as plainly in `CAVEATS`: removing
gap-crossing windows does not repair the ATR itself, because the precomputed `atr14_pct` at any
row is a recursive ewm(alpha=1/14) over the whole preceding series, so earlier gaps never fully
decay out of it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from tools.strategy_discovery.atr_causality_probe import (
    ALL_VARIANTS,
    LEGACY,
    VariantSpec,
    simulate_variant,
)

_BAR_MS = 3_600_000
_REQUIRED_COLUMNS = ("ts", "open", "high", "low", "close", "atr14_pct")

DEFAULT_CONFIG: Mapping[str, float] = {
    "stop_loss_pct": 0.08,
    "atr_trail_floor": 0.06,
    "max_hold_bars": 168,
    "round_trip_fee": 0.012,
}

SELECTION_RULE = (
    "For each product, consider source-entry POSITIONS 0..max_entries-1 in original frame order. "
    "Retain an entry if and only if (a) its stored label for this horizon is finite, (b) its full "
    "forward window [entry, entry+horizon] lies inside the frame, and (c) every step inside that "
    "window is exactly one hour. Original source positions and timestamps are preserved, and the "
    "PRECOMPUTED atr14_pct column is used as-is with no feature recomputation. Every eligible "
    "window is scanned; selection never depends on an outcome, a variant result or a PnL."
)

CAVEATS = (
    "Selection removes entries whose own window crosses a gap. It does NOT repair the ATR. "
    "features._wilder_atr14 is ewm(alpha=1/14, adjust=False), i.e. RECURSIVE smoothing with "
    "infinite memory and geometric decay -- not a finite 14-bar window (Codex 995a6ac8). So every "
    "earlier bar contributes to atr14_pct at any row, and no number of clean predecessor bars "
    "removes the influence of an earlier gap. A retained entry can still carry an ATR shaped by "
    "holes outside its own window, and this diagnostic does not claim otherwise.",
    "The retained population is NOT representative of the full universe. Per-product gap exposure "
    "ranges from about 0.3% to 84%, so any pooled figure is weighted towards the products with "
    "the cleanest clocks.",
    "legacy_matches_stored is the provenance anchor. Below retained_entries it means the stored "
    "labels came from a different config or code, and every delta is then relative to a "
    "recomputation rather than to the published artifact.",
    "Not a profitability measurement, and not a re-measurement of any archived verdict.",
)


@dataclass(frozen=True)
class VariantDelta:
    variant: str
    records: int = 0
    pnl_changed: int = 0
    result_changed: int = 0
    bars_held_changed: int = 0
    gapped_fills: int = 0
    sign_flips: int = 0
    kind_migration: Dict[str, int] = field(default_factory=dict)
    pnl_delta_min: Optional[float] = None
    pnl_delta_max: Optional[float] = None
    pnl_delta_abs_mean_over_changed: Optional[float] = None


@dataclass(frozen=True)
class ContiguousScan:
    product_id: str
    horizon: int
    rows: int
    positions_considered: int
    retained_entries: int
    excluded_label_not_finite: int
    excluded_window_incomplete: int
    excluded_window_has_gap: int
    legacy_matches_stored: int
    retained_first_ts: Optional[int]
    retained_last_ts: Optional[int]
    variants: List[VariantDelta] = field(default_factory=list)
    skipped_reason: Optional[str] = None


def _finite(value) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _skip(product_id, horizon, rows, reason) -> ContiguousScan:
    return ContiguousScan(
        product_id=product_id,
        horizon=int(horizon),
        rows=rows,
        positions_considered=0,
        retained_entries=0,
        excluded_label_not_finite=0,
        excluded_window_incomplete=0,
        excluded_window_has_gap=0,
        legacy_matches_stored=0,
        retained_first_ts=None,
        retained_last_ts=None,
        skipped_reason=reason,
    )


def _frame_defect(frame) -> Optional[str]:
    """Defects that make a frame unusable even for window selection.

    A GAPPED clock is explicitly allowed here -- that is the whole point. A reversed or duplicated
    clock is not: that is a broken frame rather than a sparse one, and no window selection makes
    it meaningful.
    """
    import numpy as np
    from pandas.api.types import is_integer_dtype

    if not is_integer_dtype(frame["ts"].dtype) or frame["ts"].isna().any():
        return "ts must be non-null integer milliseconds"
    ts = frame["ts"].to_numpy(dtype="int64")
    if len(ts) > 1 and (np.diff(ts) <= 0).any():
        return (
            "ts must be strictly increasing; a reversed or duplicated clock is a broken frame, "
            "not a gapped one"
        )
    for name in ("open", "high", "low", "close"):
        values = frame[name].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values <= 0.0).any():
            return f"{name} must be finite and positive"
    return None


def scan_contiguous_windows(
    frame,
    *,
    product_id: str,
    horizon: int,
    config: Mapping[str, float] = DEFAULT_CONFIG,
    variants: Sequence[VariantSpec] = ALL_VARIANTS,
    max_entries: Optional[int] = 1500,
) -> ContiguousScan:
    """Variants over entries whose own forward window is contiguous hourly."""
    import numpy as np

    missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        return _skip(product_id, horizon, len(frame), f"missing columns: {sorted(missing)}")
    label_col = f"label_h{int(horizon)}"
    if label_col not in frame.columns:
        return _skip(product_id, horizon, len(frame), f"no {label_col} column")
    defect = _frame_defect(frame)
    if defect is not None:
        return _skip(product_id, horizon, len(frame), defect)

    ts = frame["ts"].to_numpy(dtype="int64")
    steps = np.diff(ts) if len(ts) > 1 else np.zeros(0, dtype="int64")
    bad = (steps != _BAR_MS).astype("int64")
    prefix = np.concatenate([[0], np.cumsum(bad)])
    stored = frame[label_col].to_numpy(dtype="float64")
    shared = dict(
        opens=frame["open"].to_numpy(dtype="float64"),
        closes=frame["close"].to_numpy(dtype="float64"),
        highs=frame["high"].to_numpy(dtype="float64"),
        lows=frame["low"].to_numpy(dtype="float64"),
        atr_pcts=frame["atr14_pct"].to_numpy(dtype="float64"),
        config=config,
    )

    rows = len(frame)
    considered = rows if max_entries is None else min(rows, int(max_entries))
    buckets: Dict[str, dict] = {
        spec.name: {
            "records": 0,
            "pnl_changed": 0,
            "result_changed": 0,
            "bars": 0,
            "gapped": 0,
            "flips": 0,
            "migration": {},
            "deltas": [],
        }
        for spec in variants
    }
    retained = matched = 0
    no_label = incomplete = has_gap = 0
    first_ts = last_ts = None

    for entry in range(considered):
        if not _finite(float(stored[entry])):
            no_label += 1
            continue
        end = entry + int(horizon)
        if end >= rows:
            incomplete += 1
            continue
        if prefix[end] != prefix[entry]:
            has_gap += 1
            continue

        base = simulate_variant(entry_idx=entry, horizon=horizon, spec=LEGACY, **shared)
        if not _finite(base.pnl):
            incomplete += 1
            continue

        retained += 1
        if first_ts is None:
            first_ts = int(ts[entry])
        last_ts = int(ts[entry])
        if base.pnl == float(stored[entry]):
            matched += 1

        for spec in variants:
            other = (
                base
                if spec.name == LEGACY.name
                else simulate_variant(entry_idx=entry, horizon=horizon, spec=spec, **shared)
            )
            bucket = buckets[spec.name]
            bucket["records"] += 1
            if other.gapped:
                bucket["gapped"] += 1
            if not _finite(other.pnl):
                bucket["pnl_changed"] += 1
                bucket["result_changed"] += 1
                continue
            changed = False
            if other.pnl != base.pnl:
                bucket["pnl_changed"] += 1
                bucket["deltas"].append(other.pnl - base.pnl)
                if (other.pnl > 0) != (base.pnl > 0):
                    bucket["flips"] += 1
                changed = True
            if other.bars_held != base.bars_held:
                bucket["bars"] += 1
                changed = True
            if other.exit_kind != base.exit_kind:
                key = f"{base.exit_kind}->{other.exit_kind}"
                bucket["migration"][key] = bucket["migration"].get(key, 0) + 1
                changed = True
            if changed:
                bucket["result_changed"] += 1

    return ContiguousScan(
        product_id=product_id,
        horizon=int(horizon),
        rows=rows,
        positions_considered=considered,
        retained_entries=retained,
        excluded_label_not_finite=no_label,
        excluded_window_incomplete=incomplete,
        excluded_window_has_gap=has_gap,
        legacy_matches_stored=matched,
        retained_first_ts=first_ts,
        retained_last_ts=last_ts,
        variants=[_summarise(spec.name, buckets[spec.name]) for spec in variants],
    )


def _summarise(name: str, bucket: dict) -> VariantDelta:
    deltas = bucket["deltas"]
    return VariantDelta(
        variant=name,
        records=bucket["records"],
        pnl_changed=bucket["pnl_changed"],
        result_changed=bucket["result_changed"],
        bars_held_changed=bucket["bars"],
        gapped_fills=bucket["gapped"],
        sign_flips=bucket["flips"],
        kind_migration=dict(sorted(bucket["migration"].items())),
        pnl_delta_min=min(deltas) if deltas else None,
        pnl_delta_max=max(deltas) if deltas else None,
        pnl_delta_abs_mean_over_changed=(
            (sum(abs(d) for d in deltas) / len(deltas)) if deltas else None
        ),
    )


def run_contiguous_probe(
    frames_dir,
    output_dir,
    *,
    horizons: Sequence[int] = (24,),
    config: Mapping[str, float] = DEFAULT_CONFIG,
    limit: Optional[int] = 8,
    max_entries: Optional[int] = 1500,
) -> dict:
    """Scan read-only, write one JSON to an isolated directory, report no-data explicitly."""
    import pyarrow.parquet as pq

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    resolved_frames = frames_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_frames or resolved_frames in resolved_output.parents:
        raise ValueError("output_dir must not be the frames directory or inside it")

    before = _fingerprint(frames_dir) if frames_dir.exists() else {}
    candidates = sorted(frames_dir.glob("*.parquet")) if frames_dir.exists() else []
    if limit is not None:
        candidates = candidates[: int(limit)]

    scans: List[ContiguousScan] = []
    for path in candidates:
        frame = pq.read_table(path).to_pandas()
        for horizon in horizons:
            scans.append(
                scan_contiguous_windows(
                    frame,
                    product_id=path.stem,
                    horizon=int(horizon),
                    config=config,
                    max_entries=max_entries,
                )
            )

    with_evidence = [s for s in scans if s.skipped_reason is None and s.retained_entries > 0]
    payload = {
        "schema_version": 1,
        "status": "ok" if with_evidence else "no_data",
        "frames_dir": str(resolved_frames),
        "horizons": [int(h) for h in horizons],
        "config": dict(config),
        "max_entries_per_frame": max_entries,
        "selection_rule": SELECTION_RULE,
        "frames_found": len(candidates),
        "frames_with_evidence": len(with_evidence),
        "retained_entries_total": sum(s.retained_entries for s in scans),
        "legacy_matches_stored_total": sum(s.legacy_matches_stored for s in scans),
        "scans": [asdict(s) for s in scans],
        "caveats": list(CAVEATS),
    }
    if not with_evidence:
        payload["no_data_reason"] = (
            f"no scannable frames under {frames_dir}"
            if not candidates
            else "no entry in any candidate frame satisfied the selection rule"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "contiguous_window_report.json"
    temporary = destination.with_name(destination.name + ".partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise

    if frames_dir.exists() and _fingerprint(frames_dir) != before:
        raise RuntimeError("the frames directory changed during the run; the probe is read-only")
    payload["output_path"] = str(destination)
    return payload


def _fingerprint(frames_dir: Path) -> Dict[str, tuple]:
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(frames_dir.glob("*.parquet"))
    }
