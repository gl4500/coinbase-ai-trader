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
    """One variant against legacy.

    `pnl_changed` and `result_changed` are separate on purpose (Codex c6f81f65): a variant can
    move the exit kind or the holding period while landing on the same PnL, and counting only PnL
    understates the change. `result_changed` is the union of pnl, exit kind and bars held.
    """

    variant: str
    records: int = 0
    pnl_changed: int = 0
    result_changed: int = 0
    kind_migration: Dict[str, int] = field(default_factory=dict)
    sign_flips: int = 0
    bars_held_changed: int = 0
    gapped_fills: int = 0
    pnl_delta_min: Optional[float] = None
    pnl_delta_max: Optional[float] = None
    # Averaged over the records that CHANGED, not the population -- a conditional denominator,
    # named so nobody reads it as a population mean.
    pnl_delta_abs_mean_over_changed: Optional[float] = None


@dataclass(frozen=True)
class FrameReport:
    """One product and horizon.

    `comparable_records` counts rows the PUBLISHED frame labelled (finite stored label) whose
    legacy recomputation is also finite -- the population is taken from the artifact, not from
    whatever the recomputation happened to produce.

    `legacy_matches_stored_label` is the anchor for that population: when it equals
    `comparable_records`, the probe's baseline IS the published label bit for bit. Any shortfall
    means the stored labels were produced under a different config or code, and every delta in
    this report is then relative to a recomputation rather than to the artifact.

    `enumerated_policy_agreement` counts records where the two enumerated orderings agree. It is
    NOT evidence of path independence -- a third intrabar path could still differ, as §4.2 of the
    proposal states. An earlier version called this `order_insensitive`, which contradicted that
    very section (Codex c6f81f65).
    """

    product_id: str
    horizon: int
    rows: int
    comparable_records: int
    variants: List[VariantSummary] = field(default_factory=list)
    ordering_ambiguous: int = 0
    enumerated_policy_agreement: int = 0
    stored_labels_finite: int = 0
    legacy_matches_stored_label: int = 0
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

    invalid = _frame_defect(frame)
    if invalid is not None:
        return FrameReport(product_id, int(horizon), len(frame), 0, skipped_reason=invalid)

    stored = frame[label_col].to_numpy(dtype="float64")
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
            "pnl_changed": 0,
            "result_changed": 0,
            "migration": {},
            "flips": 0,
            "bars": 0,
            "gapped": 0,
            "deltas": [],
        }
        for spec in variants
    }
    ambiguous = 0
    agreeing = 0
    comparable = 0

    stored_finite = 0
    matches_stored = 0
    for entry_idx in entries:
        # The PUBLISHED artifact decides who is a candidate. A row the producer left unlabelled is
        # not part of the population, however the recomputation would have scored it.
        if not _finite(float(stored[entry_idx])):
            continue
        stored_finite += 1
        base = simulate_variant(entry_idx=entry_idx, horizon=horizon, spec=LEGACY, **shared)
        if not _finite(base.pnl):
            continue
        comparable += 1
        if base.pnl == float(stored[entry_idx]):
            matches_stored += 1
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
                bucket["pnl_changed"] += 1
                bucket["result_changed"] += 1
                bucket["migration"][f"{base.exit_kind}->unavailable"] = (
                    bucket["migration"].get(f"{base.exit_kind}->unavailable", 0) + 1
                )
                continue
            if other.pnl != base.pnl:
                bucket["pnl_changed"] += 1
                bucket["deltas"].append(other.pnl - base.pnl)
                if (other.pnl > 0) != (base.pnl > 0):
                    bucket["flips"] += 1
            if other.bars_held != base.bars_held:
                bucket["bars"] += 1
            if other.exit_kind != base.exit_kind:
                key = f"{base.exit_kind}->{other.exit_kind}"
                bucket["migration"][key] = bucket["migration"].get(key, 0) + 1
            if (
                other.pnl != base.pnl
                or other.bars_held != base.bars_held
                or other.exit_kind != base.exit_kind
            ):
                bucket["result_changed"] += 1

        report = bracket_orderings(entry_idx=entry_idx, horizon=horizon, atr_lag_bars=1, **shared)
        if not report.agree:
            ambiguous += 1
        else:
            agreeing += 1

    summaries = [_summarise(spec.name, per_variant[spec.name]) for spec in variants]
    return FrameReport(
        product_id,
        int(horizon),
        rows,
        comparable,
        summaries,
        ambiguous,
        agreeing,
        stored_finite,
        matches_stored,
    )


def _frame_defect(frame) -> Optional[str]:
    """Why this frame cannot carry a row-count label, or None.

    `ts` was previously required and never checked (Codex c6f81f65). The label semantics are
    row-count based, so a gapped, reversed or duplicated clock silently turns row offsets into
    false hourly horizons -- the exact defect class the endpoint contract exists to remove.
    """
    import numpy as np
    from pandas.api.types import is_integer_dtype

    if not is_integer_dtype(frame["ts"].dtype) or frame["ts"].isna().any():
        return "ts must be non-null integer milliseconds"
    ts = frame["ts"].to_numpy(dtype="int64")
    if len(ts) > 1:
        steps = np.diff(ts)
        if (steps != _BAR_MS).any():
            return (
                "ts must be unique, ascending and contiguous hourly; a gapped or reordered clock "
                "makes row-count horizons false"
            )
    for name in ("open", "high", "low", "close"):
        values = frame[name].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values <= 0.0).any():
            return f"{name} must be finite and positive"
    if (frame["high"].to_numpy(dtype="float64") < frame["low"].to_numpy(dtype="float64")).any():
        return "high must not be below low"
    return None


_BAR_MS = 3_600_000


def _finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _summarise(name: str, bucket: dict) -> VariantSummary:
    deltas = bucket["deltas"]
    return VariantSummary(
        variant=name,
        records=bucket["records"],
        pnl_changed=bucket["pnl_changed"],
        result_changed=bucket["result_changed"],
        kind_migration=dict(sorted(bucket["migration"].items())),
        sign_flips=bucket["flips"],
        bars_held_changed=bucket["bars"],
        gapped_fills=bucket["gapped"],
        pnl_delta_min=min(deltas) if deltas else None,
        pnl_delta_max=max(deltas) if deltas else None,
        pnl_delta_abs_mean_over_changed=(
            (sum(abs(d) for d in deltas) / len(deltas)) if deltas else None
        ),
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
    resolved_frames = frames_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_frames or resolved_frames in resolved_output.parents:
        raise ValueError(
            "output_dir must not be the frames directory or inside it; the probe writes nothing "
            "under its read-only inputs (Codex c6f81f65: an equality-only guard permitted a "
            "subdirectory, contradicting the promise)"
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

    # "scanned" requires actual comparable records. A frame that is too short, entirely
    # unlabelled or wholly invalid produced zero evidence, and reporting that as a successful
    # scan would make an empty result look like a finding of no effect (Codex c6f81f65).
    scanned = [r for r in reports if r.skipped_reason is None and r.comparable_records > 0]
    empty = [r for r in reports if r.skipped_reason is None and r.comparable_records == 0]
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
        "frames_without_evidence": [
            {"product_id": r.product_id, "horizon": r.horizon, "rows": r.rows} for r in empty
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
            "enumerated_policy_agreement is agreement between TWO orderings, not proof of path "
            "independence; a third intrabar path could still differ.",
            "pnl_delta_abs_mean_over_changed has a conditional denominator: it averages the "
            "records that changed, not the population.",
            "Read-only was checked by file size and mtime, which detects modification by this "
            "run but is not a content-integrity proof of the inputs themselves.",
            "legacy_matches_stored_label is the population anchor: below comparable_records, the "
            "stored labels came from a different config or code and every delta here is relative "
            "to a recomputation rather than to the published artifact.",
        ],
    }
    if not scanned:
        if not candidates:
            payload["no_data_reason"] = f"no scannable Phase 2 frames under {frames_dir}"
        elif empty:
            payload["no_data_reason"] = (
                "candidate frames were readable but produced no comparable records; see "
                "frames_without_evidence"
            )
        else:
            payload["no_data_reason"] = "every candidate frame was skipped; see frames_skipped"

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
