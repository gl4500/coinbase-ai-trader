"""Phase 4 orchestrator + CLI.

Iterates caps ∈ {3, 4, 5}, dispatches knapsack search per cap, writes:
  - backend/data/phase4/scorecard.md
  - backend/data/phase4/deployment_n{N}.json  (one per cap)
  - backend/data/phase4/portfolio_telemetry_n{N}.parquet  (one per cap)

Only module in Phase 4 that touches the filesystem.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.knapsack_search import beam_search_knapsack  # noqa: E402
from tools.strategy_discovery.profile_loader import (  # noqa: E402
    load_all_profiles,
    load_pid_features,
)
from tools.strategy_discovery.scorecard import (  # noqa: E402
    DEPLOYMENT_BLOCKERS,
    CapScorecard,
    evaluate_cap_gates,
    render_scorecard,
)

_DEFAULT_PHASE3_DIR = Path(BACKEND) / "data" / "phase3"
_DEFAULT_PHASE2_DIR = Path(BACKEND) / "data" / "phase2"
_DEFAULT_OUTPUT_DIR = Path(BACKEND) / "data" / "phase4"
_DEFAULT_CAPS = (3, 4, 5)


def _write_deployment_json(
    card: CapScorecard,
    output_path: Path,
) -> None:
    payload = {
        "schema_version": 2,
        "evaluation_scope": "research_selection",
        "deployment_eligible": False,
        "deployment_blockers": list(DEPLOYMENT_BLOCKERS),
        "cap": int(card.cap),
        "selected_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "k_subsets_evaluated": int(card.k_evaluated),
        "portfolio_metrics": {
            "cumulative_profit_raw": float(card.metrics.cumulative_profit_raw),
            "cumulative_profit_deflated": float(card.metrics.cumulative_profit_deflated),
            "deflation_pp": float(card.inflation),
            "max_dd": float(card.metrics.max_dd),
            "sortino": float(card.metrics.sortino),
            "trade_count": int(card.metrics.trade_count),
            "pct_slots_full": float(card.metrics.pct_slots_full),
            "mean_concurrent": float(card.metrics.mean_concurrent),
        },
        # The universe travels with the numbers: a reader can see what was left out rather
        # than having to notice that a profit figure got smaller.
        "universe": dict(card.universe),
        "gates": {
            **card.gates,
            "overall": "pass" if card.overall_pass else "fail",
            "scope": "research_only",
        },
        "profiles": [
            {
                "pid": p.pid,
                "horizon": int(p.horizon),
                "leaf_id": int(p.leaf_id),
                "profile_id": p.profile_id,
                "rounded_display_summary": p.rule_path,
                "machine_rule": p.machine_rule,
                "rule_digest": p.rule_digest,
                "group_search_metrics": {
                    "scope": "qualifying_leaves_not_representative_policy",
                    "avg_win": float(p.avg_win),
                    "avg_loss": float(p.avg_loss),
                    "max_dd": float(p.max_dd),
                    "trade_count": int(p.trade_count),
                    "sortino": float(p.sortino),
                    "cumulative_profit_deflated": float(p.cumulative_profit_deflated),
                },
            }
            for p in card.selected_profiles
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_telemetry_parquet(
    telemetry,
    output_path: Path,
) -> None:
    if not telemetry:
        return
    rows = [
        {
            "ts": int(t.ts),
            "equity": float(t.equity),
            "n_open": int(t.n_open),
            "fired_profile_id": t.fired_profile_id,
            "closed_profile_id": t.closed_profile_id,
            "realized_pnl": None if t.realized_pnl is None else float(t.realized_pnl),
            "schema_version": 1,
        }
        for t in telemetry
    ]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False), output_path, compression="snappy"
    )


_BAR_DURATION_MS = 3_600_000


class EndpointArtifactError(RuntimeError):
    """A published endpoint artifact is corrupt or does not describe its frame.

    Deliberately NOT an exclusion. Dropping such a product would shrink the optimisation
    universe while still producing a scorecard that looks successful, which is the silent
    thinning §9.3 forbids.
    """


def _load_endpoints(pid_features, phase2_dir: Path):
    """Validated endpoints per product, plus the products with no publication at all.

    The two cases are treated differently on purpose (§9.3):

      * NO publication -- no sidecar on disk -- is an exclusion with a named reason. One
        unpublished product should not abort a sweep over many, and the reason is returned so
        the caller can persist it.
      * A publication that FAILS validation raises `EndpointArtifactError` naming the product.
        A mismatched `data_id`, a bad checksum, incomplete coverage or a record that fails
        semantic validation all mean the artifact and the frame are not a matched pair, and no
        amount of per-product exclusion repairs that.
    """
    from tools.strategy_discovery.endpoint_consumers import load_validated_endpoints

    loaded: Dict[str, object] = {}
    unpublished: Dict[str, str] = {}
    for pid, frame in pid_features.items():
        sidecar_path = Path(phase2_dir) / f"{pid}.endpoints.json"
        if not sidecar_path.exists():
            unpublished[pid] = "endpoint_sidecar_missing"
            continue
        try:
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, ValueError) as exc:
            raise EndpointArtifactError(
                f"{pid}: endpoint sidecar is present but unreadable: {exc}"
            ) from exc
        try:
            loaded[pid] = load_validated_endpoints(
                Path(phase2_dir) / "endpoints" / pid,
                frame=frame,
                sidecar=sidecar,
                product_id=pid,
            )
        except (ValueError, KeyError, OSError) as exc:
            raise EndpointArtifactError(
                f"{pid}: published endpoints failed validation against its own Phase 2 frame "
                f"({exc}); excluding the product would shrink the optimisation universe while "
                f"still producing a scorecard that looks successful"
            ) from exc
    return loaded, unpublished


def build_phase4(
    *,
    phase3_dir: Path = _DEFAULT_PHASE3_DIR,
    phase2_dir: Path = _DEFAULT_PHASE2_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    caps=_DEFAULT_CAPS,
    horizons: List[int] = None,
    beam_width: int = 20,
    pool_size: int = 100,
    bootstrap_iter: int = 1000,
    seed: int = 42,
) -> Dict[int, CapScorecard]:
    """Sweep caps; per cap: knapsack search -> score -> write artifacts."""
    if horizons is None:
        horizons = [1, 4, 24, 72, 168]
    phase3_dir = Path(phase3_dir)
    phase2_dir = Path(phase2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=horizons)
    requested_pids = sorted({p.pid for p in profiles})
    loaded_frames = {pid: load_pid_features(pid, phase2_dir=phase2_dir) for pid in requested_pids}
    # Frames dropped for missing or empty features were previously discarded by a bare
    # comprehension, leaving no trace. They are classified here so the report can name them.
    exclusions = {
        pid: "phase2_features_missing_or_empty"
        for pid, frame in loaded_frames.items()
        if frame.empty
    }
    pid_features = {pid: frame for pid, frame in loaded_frames.items() if not frame.empty}

    # Endpoints come from the SAME directory the producer published them to, and are validated
    # against the very frame loaded above -- so the replay cannot be handed one product's
    # records with another's frame. A product without a usable sidecar is excluded with a named
    # reason rather than replayed on horizon arithmetic.
    endpoints_by_pid, unpublished_products = _load_endpoints(pid_features, phase2_dir)
    exclusions.update(unpublished_products)
    if unpublished_products:
        pid_features = {
            pid: frame for pid, frame in pid_features.items() if pid in endpoints_by_pid
        }
        profiles = [p for p in profiles if p.pid in endpoints_by_pid]
    universe = {
        "requested_products": requested_pids,
        "evaluated_products": sorted(pid_features),
        "excluded_products": dict(sorted(exclusions.items())),
        "requested_product_count": len(requested_pids),
        "evaluated_product_count": len(pid_features),
    }

    cards: Dict[int, CapScorecard] = {}
    for cap in caps:
        result = beam_search_knapsack(
            all_qualifying=profiles,
            cap=int(cap),
            pid_features=pid_features,
            endpoints_by_pid=endpoints_by_pid,
            bar_duration_ms=_BAR_DURATION_MS,
            beam_width=int(beam_width),
            pool_size=int(pool_size),
            bootstrap_iter=int(bootstrap_iter),
            seed=int(seed),
        )
        gates, overall = evaluate_cap_gates(result.best_metrics)
        card = CapScorecard(
            universe=universe,
            cap=int(cap),
            metrics=result.best_metrics,
            k_evaluated=result.k_evaluated,
            inflation=result.inflation,
            gates=gates,
            overall_pass=overall,
            selected_profiles=result.best_subset,
        )
        cards[int(cap)] = card
        _write_deployment_json(card, output_dir / f"deployment_n{int(cap)}.json")
        _write_telemetry_parquet(
            result.best_telemetry, output_dir / f"portfolio_telemetry_n{int(cap)}.parquet"
        )
    # Render scorecard
    md = render_scorecard(list(cards.values()))
    (output_dir / "scorecard.md").write_text(md, encoding="utf-8")
    return cards


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4 -- research scorecard; deployment blocked."
    )
    parser.add_argument("--phase3-dir", default=str(_DEFAULT_PHASE3_DIR))
    parser.add_argument("--phase2-dir", default=str(_DEFAULT_PHASE2_DIR))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--caps", default="3,4,5")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    caps = [int(c.strip()) for c in args.caps.split(",") if c.strip()]
    cards = build_phase4(
        phase3_dir=Path(args.phase3_dir),
        phase2_dir=Path(args.phase2_dir),
        output_dir=Path(args.output_dir),
        caps=caps,
        beam_width=args.beam_width,
        pool_size=args.pool_size,
        seed=args.seed,
    )
    n_passing = sum(1 for c in cards.values() if c.overall_pass)
    print(f"  scorecard written to {args.output_dir}/scorecard.md", flush=True)
    print(f"  {n_passing} of {len(cards)} caps passed", flush=True)
    return 0 if n_passing > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
