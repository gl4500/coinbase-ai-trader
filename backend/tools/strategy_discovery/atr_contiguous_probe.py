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
    "legacy_matches_stored counts entries where the recomputation REPRODUCES THE STORED VALUE "
    "bitwise (float.hex, so signed zero cannot pass as a match). That is value identity, and "
    "does not by itself establish which code or config produced the stored label. Below "
    "retained_entries the baseline is NOT "
    "the published label and every delta is relative to a recomputation. A shortfall does not by "
    "itself identify the cause: a different config or code version would do it, but so would "
    "float storage or round-tripping, or any intervening transformation of the frame "
    "(Codex dd462ea4).",
    "Not a profitability measurement, and not a re-measurement of any archived verdict.",
    # Carried over from atr_causality_report so this file is interpretable on its own.
    "low_before_high is a DELAYED-UPDATE policy, not a literal low-first path: a peak raised by a "
    "bar's high takes effect from the next bar, and the high-to-close descent is not modelled.",
    "Under lag 0 the opening levels are computed retrospectively from the bar's own completed "
    "OHLC, so lag-0 gap and opening-event figures are counterfactual.",
    "pnl_delta_abs_mean_over_changed has a CONDITIONAL denominator: it averages the records that "
    "changed, not the population.",
    "The two enumerated orderings are not exhaustive bounds over intrabar paths. A bar may visit "
    "its low, recover, set its high and fall back, touching a level neither ordering triggers.",
    "Read-only was verified by file size and mtime, which detects modification BY THIS RUN but is "
    "not a content-integrity proof of the inputs; per-file sha256 is recorded separately.",
    "source_provenance.reproducible_from_commit is False whenever the tree carried tracked "
    "modifications at run time. In that case the recorded commit does NOT reproduce the behaviour "
    "that produced this artifact, and the run must be repeated from a clean tree before the "
    "numbers are quoted as reproducible.",
    "source_provenance is source-tree evidence ONLY, in both directions. A True flag says the "
    "recorded commit describes the tracked sources that ran; it says nothing about the "
    "interpreter, the installed package versions, or any untracked importable file, each of "
    "which can change a rerun that checks out exactly this commit.",
)


def _retained_ranges(positions) -> list:
    """Run-length encode retained positions as [start, end] inclusive pairs.

    First and last timestamps cannot reproduce WHICH entries were excluded (Codex dd462ea4), and a
    flat list of 11k integers per product is unwieldy. Ranges are exact and compact.
    """
    ranges = []
    for position in positions:
        if ranges and position == ranges[-1][1] + 1:
            ranges[-1][1] = position
        else:
            ranges.append([position, position])
    return [[int(a), int(b)] for a, b in ranges]


def _file_digest(path) -> str:
    """sha256 of one input file, so a reader can confirm they hold the same bytes."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git_provenance() -> Dict[str, object]:
    """The commit this ran from AND whether the tree was modified at the time.

    A bare commit SHA is worse than none when the executed code was uncommitted: it invites a
    reader to check out that SHA and get different behaviour (Codex 35985ceb -- the first artifact
    recorded a9e07b1 while the hex matching, coherence guards and these very fields were still
    working-tree changes later committed as e94408b). `tree_dirty` makes that condition visible in
    the artifact instead of inferable only by someone who already knows.
    """
    import subprocess

    def _run(args):
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(Path(__file__).resolve().parent),
            )
        except (OSError, subprocess.SubprocessError):
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    commit = head.stdout.strip() if head is not None and head.returncode == 0 else None

    # Tracked modifications only: an untracked file cannot change what `git checkout <sha>` yields,
    # but it is counted separately so the reader can judge.
    diff = _run(["git", "diff", "--quiet", "HEAD"])
    dirty = None if diff is None else (diff.returncode != 0)

    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"])
    count = (
        len([line for line in untracked.stdout.splitlines() if line.strip()])
        if untracked is not None and untracked.returncode == 0
        else None
    )
    return {
        "commit": commit,
        "tree_dirty": dirty,
        "untracked_files": count,
        "reproducible_from_commit": bool(commit) and dirty is False,
    }


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
    # Exact, reproducible record of WHICH positions were retained, run-length encoded.
    retained_position_ranges: List[List[int]] = field(default_factory=list)
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

    # The RELATIONS, not just the magnitudes (Codex dd462ea4). A bar whose low exceeds its high,
    # or whose open or close sits outside [low, high], describes no traversable path -- yet the
    # simulation would still produce an exit from it, so the diagnostic must refuse the frame
    # rather than report a number derived from an impossible bar.
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    if (low > high).any():
        return "low must not exceed high"
    for name in ("open", "close"):
        values = frame[name].to_numpy(dtype="float64")
        if (values > high).any() or (values < low).any():
            return f"{name} must lie within [low, high]"
    return None


def _checked_positive_int(value, field: str) -> int:
    """A strictly positive integer, without coercion.

    `int(value)` accepted 24.9 and True, and a negative `max_entries` produced an empty range that
    would read as a clean run with no evidence rather than as bad input.
    """
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{field} must be an integer without coercion, got {value!r}")
    if int(value) <= 0:
        raise ValueError(f"{field} must be strictly positive, got {value!r}")
    return int(value)


def _checked_config(config) -> Mapping[str, float]:
    """Every threshold present, finite and in range. A missing key must not default silently."""
    required = ("stop_loss_pct", "atr_trail_floor", "max_hold_bars", "round_trip_fee")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config is missing {sorted(missing)}")
    for key in ("stop_loss_pct", "atr_trail_floor"):
        value = float(config[key])
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"config[{key!r}] must be a finite fraction in (0, 1), got {value!r}")
    fee = float(config["round_trip_fee"])
    if not math.isfinite(fee) or fee < 0.0:
        raise ValueError(f"config['round_trip_fee'] must be finite and non-negative, got {fee!r}")
    _checked_positive_int(config["max_hold_bars"], "config['max_hold_bars']")
    return config


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

    horizon = _checked_positive_int(horizon, "horizon")
    config = _checked_config(config)
    if max_entries is not None:
        max_entries = _checked_positive_int(max_entries, "max_entries")

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
    retained_positions = []

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
        retained_positions.append(entry)
        if first_ts is None:
            first_ts = int(ts[entry])
        last_ts = int(ts[entry])
        # float.hex(), not ==: 0.0 == -0.0 is True while the bits differ, so equality alone cannot
        # support a bitwise claim (Codex 15e8c1c7). And even a bitwise match establishes only that
        # the VALUES agree -- never by itself which code or config produced them.
        if base.pnl.hex() == float(stored[entry]).hex():
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
        retained_position_ranges=_retained_ranges(retained_positions),
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

    config = _checked_config(config)
    horizons = tuple(_checked_positive_int(h, "horizon") for h in horizons)
    if not horizons:
        raise ValueError("at least one horizon is required")
    if limit is not None:
        limit = _checked_positive_int(limit, "limit")
    if max_entries is not None:
        max_entries = _checked_positive_int(max_entries, "max_entries")

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
    inputs = {}
    for path in candidates:
        inputs[path.name] = _file_digest(path)
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
        "input_digests": inputs,
        "source_provenance": _git_provenance(),
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
