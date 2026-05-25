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
from dataclasses import asdict
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
    LoadedProfile,
    load_all_profiles,
    load_pid_features,
)
from tools.strategy_discovery.scorecard import (  # noqa: E402
    CapScorecard,
    evaluate_cap_gates,
    render_scorecard,
)

_DEFAULT_PHASE3_DIR = Path(BACKEND) / "data" / "phase3"
_DEFAULT_PHASE2_DIR = Path(BACKEND) / "data" / "phase2"
_DEFAULT_OUTPUT_DIR = Path(BACKEND) / "data" / "phase4"
_DEFAULT_CAPS       = (3, 4, 5)


def _write_deployment_json(
    card: CapScorecard,
    output_path: Path,
) -> None:
    payload = {
        "cap": int(card.cap),
        "selected_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "k_subsets_evaluated": int(card.k_evaluated),
        "portfolio_metrics": {
            "cumulative_profit_raw":      float(card.metrics.cumulative_profit_raw),
            "cumulative_profit_deflated": float(card.metrics.cumulative_profit_deflated),
            "deflation_pp":               float(card.inflation),
            "max_dd":                     float(card.metrics.max_dd),
            "sortino":                    float(card.metrics.sortino),
            "trade_count":                int(card.metrics.trade_count),
            "pct_slots_full":             float(card.metrics.pct_slots_full),
            "mean_concurrent":            float(card.metrics.mean_concurrent),
        },
        "gates": {**card.gates, "overall": "pass" if card.overall_pass else "fail"},
        "profiles": [
            {
                "pid": p.pid,
                "horizon": int(p.horizon),
                "leaf_id": int(p.leaf_id),
                "rule_path": p.rule_path,
                "expected_avg_win": float(p.avg_win),
                "expected_avg_loss": float(p.avg_loss),
                "expected_max_dd": float(p.max_dd),
                "expected_trade_count": int(p.trade_count),
                "expected_sortino": float(p.sortino),
                "phase3_cumulative_profit_deflated": float(p.cumulative_profit_deflated),
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
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path,
                   compression="snappy")


def build_phase4(
    *,
    phase3_dir: Path = _DEFAULT_PHASE3_DIR,
    phase2_dir: Path = _DEFAULT_PHASE2_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    caps = _DEFAULT_CAPS,
    horizons: List[int] = [1, 4, 24, 72, 168],
    beam_width: int = 20,
    pool_size: int = 100,
    bootstrap_iter: int = 1000,
    seed: int = 42,
) -> Dict[int, CapScorecard]:
    """Sweep caps; per cap: knapsack search -> score -> write artifacts."""
    phase3_dir = Path(phase3_dir)
    phase2_dir = Path(phase2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=horizons)
    pid_features = {pid: load_pid_features(pid, phase2_dir=phase2_dir)
                    for pid in {p.pid for p in profiles}}
    pid_features = {k: v for k, v in pid_features.items() if not v.empty}
    cards: Dict[int, CapScorecard] = {}
    for cap in caps:
        result = beam_search_knapsack(
            all_qualifying=profiles, cap=int(cap), pid_features=pid_features,
            beam_width=int(beam_width), pool_size=int(pool_size),
            bootstrap_iter=int(bootstrap_iter), seed=int(seed),
        )
        gates, overall = evaluate_cap_gates(result.best_metrics)
        card = CapScorecard(
            cap=int(cap), metrics=result.best_metrics,
            k_evaluated=result.k_evaluated, inflation=result.inflation,
            gates=gates, overall_pass=overall,
            selected_profiles=result.best_subset,
        )
        cards[int(cap)] = card
        _write_deployment_json(card, output_dir / f"deployment_n{int(cap)}.json")
        _write_telemetry_parquet(result.best_telemetry,
                                  output_dir / f"portfolio_telemetry_n{int(cap)}.parquet")
    # Render scorecard
    md = render_scorecard(list(cards.values()))
    (output_dir / "scorecard.md").write_text(md, encoding="utf-8")
    return cards


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 -- scorecard + deployment selection.")
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
        phase3_dir=Path(args.phase3_dir), phase2_dir=Path(args.phase2_dir),
        output_dir=Path(args.output_dir), caps=caps,
        beam_width=args.beam_width, pool_size=args.pool_size, seed=args.seed,
    )
    n_passing = sum(1 for c in cards.values() if c.overall_pass)
    print(f"  scorecard written to {args.output_dir}/scorecard.md", flush=True)
    print(f"  {n_passing} of {len(cards)} caps passed", flush=True)
    return 0 if n_passing > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
