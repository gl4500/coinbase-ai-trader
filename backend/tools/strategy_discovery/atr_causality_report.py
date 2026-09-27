"""Run the ATR-causality variants over Phase 2 frames and write an isolated report.

Read-only on its inputs and write-only to a directory that must not be the input directory. It
regenerates nothing, fetches nothing, and does not touch `phase2/`, any published endpoint
directory, any model artifact or the live database.

Each variant is summarised SEPARATELY against legacy, because a single combined diff cannot
attribute an effect to a cause. The per-variant `changed_vs_legacy` for `legacy` itself must be
zero; if it is not, the probe is measuring its own noise and the report says so.

**What this is not:** it is not a measurement of profitability, not a re-measurement of any
archived verdict, and not an approved label policy. It reports how many records each proposed
change would move and in which direction. Every caveat in
`docs/specs/2026-09-27-atr-causality-repair-proposal.md` applies, in particular §1 and §4.1.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from tools.strategy_discovery.atr_causality_probe import (
    ALL_VARIANTS,
    LEGACY,
    VariantSpec,
    bracket_orderings,
    simulate_variant,
)

_REQUIRED_COLUMNS = ("ts", "open", "high", "low", "close", "atr14_pct")

DEFAULT_CONFIG: Mapping[str, float] = {
    "stop_loss_pct": 0.08,
    "atr_trail_floor": 0.06,
    "max_hold_bars": 168,
    "round_trip_fee": 0.012,
}


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    records: int = 0
    changed_vs_legacy: int = 0
    kind_migration: Dict[str, int] = field(default_factory=dict)
    sign_flips: int = 0
    bars_held_changed: int = 0
    gapped_fills: int = 0
    pnl_delta_min: Optional[float] = None
    pnl_delta_max: Optional[float] = None
    pnl_delta_abs_mean: Optional[float] = None


@dataclass(frozen=True)
class FrameReport:
    product_id: str
    horizon: int
    rows: int
    comparable_records: int
    variants: List[VariantSummary] = field(default_factory=list)
    ordering_ambiguous: int = 0
    order_insensitive: int = 0
    skipped_reason: Optional[str] = None


def scan_frame(
    frame,
    *,
    product_id: str,
    horizon: int,
    config: Mapping[str, float] = DEFAULT_CONFIG,
    variants: Sequence[VariantSpec] = ALL_VARIANTS,
    max_entries: Optional[int] = None,
) -> FrameReport:
    """Every variant over one frame and horizon, each summarised against legacy."""
    missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
    label_col = f"label_h{int(horizon)}"
    if missing:
        return FrameReport(
            product_id,
            int(horizon),
            len(frame),
            0,
            skipped_reason=f"missing columns: {sorted(missing)}",
        )
    if label_col not in frame.columns:
        return FrameReport(
            product_id,
            int(horizon),
            len(frame),
            0,
            skipped_reason=f"no {label_col} column",
        )

    opens = frame["open"].to_numpy(dtype="float64")
    closes = frame["close"].to_numpy(dtype="float64")
    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    atrs = frame["atr14_pct"].to_numpy(dtype="float64")
    shared = dict(opens=opens, closes=closes, highs=highs, lows=lows, atr_pcts=atrs, config=config)

    rows = len(frame)
    entries = range(rows if max_entries is None else min(rows, int(max_entries)))

    per_variant: Dict[str, dict] = {
        spec.name: {
            "records": 0,
            "changed": 0,
            "migration": {},
            "flips": 0,
            "bars": 0,
            "gapped": 0,
            "deltas": [],
        }
        for spec in variants
    }
    ambiguous = 0
    insensitive = 0
    comparable = 0

    for entry_idx in entries:
        base = simulate_variant(entry_idx=entry_idx, horizon=horizon, spec=LEGACY, **shared)
        if not _finite(base.pnl):
            continue
        comparable += 1
        for spec in variants:
            other = (
                base
                if spec.name == LEGACY.name
                else simulate_variant(entry_idx=entry_idx, horizon=horizon, spec=spec, **shared)
            )
            bucket = per_variant[spec.name]
            bucket["records"] += 1
            if other.gapped:
                bucket["gapped"] += 1
            if not _finite(other.pnl):
                # A variant that cannot produce a label where legacy could is itself a finding.
                bucket["changed"] += 1
                bucket["migration"][f"{base.exit_kind}->unavailable"] = (
                    bucket["migration"].get(f"{base.exit_kind}->unavailable", 0) + 1
                )
                continue
            if other.pnl != base.pnl:
                bucket["changed"] += 1
                bucket["deltas"].append(other.pnl - base.pnl)
                if (other.pnl > 0) != (base.pnl > 0):
                    bucket["flips"] += 1
            if other.bars_held != base.bars_held:
                bucket["bars"] += 1
            if other.exit_kind != base.exit_kind:
                key = f"{base.exit_kind}->{other.exit_kind}"
                bucket["migration"][key] = bucket["migration"].get(key, 0) + 1

        report = bracket_orderings(entry_idx=entry_idx, horizon=horizon, atr_lag_bars=1, **shared)
        if not report.agree:
            ambiguous += 1
        else:
            insensitive += 1

    summaries = [_summarise(spec.name, per_variant[spec.name]) for spec in variants]
    return FrameReport(
        product_id, int(horizon), rows, comparable, summaries, ambiguous, insensitive
    )


def _finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _summarise(name: str, bucket: dict) -> VariantSummary:
    deltas = bucket["deltas"]
    return VariantSummary(
        variant=name,
        records=bucket["records"],
        changed_vs_legacy=bucket["changed"],
        kind_migration=dict(sorted(bucket["migration"].items())),
        sign_flips=bucket["flips"],
        bars_held_changed=bucket["bars"],
        gapped_fills=bucket["gapped"],
        pnl_delta_min=min(deltas) if deltas else None,
        pnl_delta_max=max(deltas) if deltas else None,
        pnl_delta_abs_mean=(sum(abs(d) for d in deltas) / len(deltas)) if deltas else None,
    )


def _input_fingerprint(frames_dir: Path) -> Dict[str, Tuple[int, int]]:
    """Size and mtime of every input file, so a test can prove the run touched nothing."""
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(frames_dir.glob("*.parquet"))
    }


def run_probe(
    frames_dir,
    output_dir,
    *,
    horizons: Sequence[int] = (24,),
    config: Mapping[str, float] = DEFAULT_CONFIG,
    products: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    max_entries: Optional[int] = 2000,
) -> dict:
    """Scan frames read-only and write one JSON report. Returns the summary payload.

    `status` is `"ok"` when at least one frame was scanned and `"no_data"` otherwise, with a
    reason -- an empty result is reported explicitly rather than looking like a clean run with
    nothing to say.
    """
    import pyarrow.parquet as pq

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    if frames_dir.resolve() == output_dir.resolve():
        raise ValueError(
            "output_dir must not be the frames directory; the probe writes nothing next to its "
            "read-only inputs"
        )

    before = _input_fingerprint(frames_dir) if frames_dir.exists() else {}
    candidates = sorted(frames_dir.glob("*.parquet")) if frames_dir.exists() else []
    if products is not None:
        wanted = set(products)
        candidates = [path for path in candidates if path.stem in wanted]
    if limit is not None:
        candidates = candidates[: int(limit)]

    reports: List[FrameReport] = []
    for path in candidates:
        frame = pq.read_table(path).to_pandas()
        for horizon in horizons:
            reports.append(
                scan_frame(
                    frame,
                    product_id=path.stem,
                    horizon=int(horizon),
                    config=config,
                    max_entries=max_entries,
                )
            )

    scanned = [r for r in reports if r.skipped_reason is None]
    payload = {
        "schema_version": 1,
        "status": "ok" if scanned else "no_data",
        "frames_dir": str(frames_dir),
        "horizons": [int(h) for h in horizons],
        "config": dict(config),
        "max_entries_per_frame": max_entries,
        "frames_found": len(candidates),
        "frames_scanned": len(scanned),
        "frames_skipped": [
            {"product_id": r.product_id, "horizon": r.horizon, "reason": r.skipped_reason}
            for r in reports
            if r.skipped_reason is not None
        ],
        "reports": [asdict(r) for r in reports],
        "caveats": [
            "Not a profitability measurement and not a re-measurement of any archived verdict.",
            "min(B1,B2) is a proposed diagnostic, not an approved label policy, and bounds only "
            "the two enumerated orderings -- not all intrabar paths.",
            "Under lag 0 every level is computed retrospectively from the bar's own OHLC, so "
            "lag-0 gap and opening-event figures are counterfactual.",
            "low_before_high is a delayed-update policy: a peak raised by a bar's high takes "
            "effect from the next bar, and the high-to-close descent is not modelled.",
        ],
    }
    if not scanned:
        payload["no_data_reason"] = (
            f"no scannable Phase 2 frames under {frames_dir}"
            if not candidates
            else "every candidate frame was skipped; see frames_skipped"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "atr_causality_report.json"
    temporary = destination.with_name(destination.name + ".partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise

    if frames_dir.exists():
        after = _input_fingerprint(frames_dir)
        if after != before:
            raise RuntimeError(
                "the frames directory changed during the run; the probe must be read-only"
            )
    payload["output_path"] = str(destination)
    return payload
