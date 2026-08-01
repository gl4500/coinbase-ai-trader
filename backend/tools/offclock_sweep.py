"""CLI: sweep the 16 off-the-clock XGB configs and scorecard each.

For each (substrate, label_variant, horizon) it builds samples, runs 5-fold
purged-WF OOF prediction, scores the result through compute_scorecard, and
writes a results table. See 2026-05-21-offclock-xgb-track-design.md.

Operator-run and offline: it trains 16 x 5 boosters. Requires
data/history/dollar/ to be populated (SP1's backfill + build steps).
"""

from __future__ import annotations

import argparse
import os
import traceback

SUBSTRATES = ("dollar", "time")
LABEL_VARIANTS = ("direction", "triple_barrier")
HORIZONS = (4, 24, 72, 168)

_DEFAULT_CACHE = "cnn_dataset_cache.pt"
_DEFAULT_SAMPLE_STEP = 24
_DEFAULT_OUT = os.path.join(
    "..", "docs", "superpowers", "specs", "2026-05-21-offclock-sweep-results.md"
)


def _config_grid() -> list[tuple[str, str, int]]:
    """All 16 (substrate, label_variant, horizon) configs."""
    return [(s, lv, k) for s in SUBSTRATES for lv in LABEL_VARIANTS for k in HORIZONS]


def _gates_passed(row: dict) -> int:
    """Count of the 4 hard gates a config row passes."""
    return sum((row["precision"], row["expected_return"], row["paper_sharpe"], row["ece"]))


def _render_results_doc(rows: list[dict]) -> str:
    """Render the sweep results as a markdown doc: per-config table + the
    dollar-minus-time delta per (label_variant, horizon) cell."""
    lines = [
        "# Off-the-Clock Sweep Results",
        "",
        "Spec: `2026-05-21-offclock-xgb-track-design.md`. Each row is one XGB",
        "config: 5-fold purged-WF OOF, scored by the deployment scorecard.",
        "",
    ]
    if not rows:
        lines.append("_Sweep produced no configs (no samples / all failed)._")
        return "\n".join(lines)

    lines += [
        "## Per-config scorecard",
        "",
        "| substrate | label | horizon | n | AUC | precision | E[r] | Sharpe | ECE | rec_tau |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['substrate']} | {r['label_variant']} | {r['horizon']} | "
            f"{r['n']} | {r['auc']:.4f} | {r['precision']} | "
            f"{r['expected_return']} | {r['paper_sharpe']} | {r['ece']} | "
            f"{r['recommended_tau']} |"
        )

    n_pass = sum(1 for r in rows if _gates_passed(r) == 4)
    lines += ["", f"**{n_pass} of {len(rows)} configs pass all 4 hard gates.**", ""]

    # Dollar - time delta per (label_variant, horizon) cell — the clean A/B:
    # label and horizon are held fixed, so the delta isolates bar structure.
    by_key = {(r["substrate"], r["label_variant"], r["horizon"]): r for r in rows}
    lines += [
        "## Dollar - time delta",
        "",
        "Each row holds label + horizon fixed, so the delta isolates the",
        "bar-structure effect. Positive = dollar bars beat time bars.",
        "",
        "| label | horizon | dAUC | dGates |",
        "|---|---|---|---|",
    ]
    for lv in LABEL_VARIANTS:
        for k in HORIZONS:
            d = by_key.get(("dollar", lv, k))
            t = by_key.get(("time", lv, k))
            if d is None or t is None:
                continue
            d_auc = d["auc"] - t["auc"]
            d_gates = _gates_passed(d) - _gates_passed(t)
            lines.append(f"| {lv} | {k} | {d_auc:+.4f} | {d_gates:+d} |")
    return "\n".join(lines)


def _run_one(substrate: str, label_variant: str, k: int, pids: list[str], sample_step: int) -> dict:
    """Run + scorecard one config. Returns a results row."""
    from sklearn.metrics import roc_auc_score

    from tools._scorecard._offclock_harness import run_config
    from tools.scorecard import compute_scorecard

    data = run_config(substrate, label_variant, k, pids, sample_step)
    report = compute_scorecard(
        data["scores"],
        data["labels"],
        data["returns"],
        data["fold_ids"],
        data["fold_spans_days"],
    )
    try:
        auc = float(roc_auc_score(data["labels"], data["scores"]))
    except Exception:
        auc = float("nan")
    return {
        "substrate": substrate,
        "label_variant": label_variant,
        "horizon": k,
        "n": data["n"],
        "auc": auc,
        "precision": report.gates_passed["precision"],
        "expected_return": report.gates_passed["expected_return"],
        "paper_sharpe": report.gates_passed["paper_sharpe"],
        "ece": report.gates_passed["ece"],
        "recommended_tau": report.recommended_operating_tau,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep 16 off-the-clock XGB configs through the scorecard"
    )
    parser.add_argument(
        "--cache", default=_DEFAULT_CACHE, help="cache for the survivorship-aware top-20 ranking"
    )
    parser.add_argument(
        "--pids", default=None, help="comma-separated product ids (overrides --cache)"
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=_DEFAULT_SAMPLE_STEP,
        help="roll one sample every N bars (default: 24)",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT, help="results doc path (default: the SP2 results spec)"
    )
    args = parser.parse_args()

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    else:
        from tools._scorecard._cv_harness import top_n_pids_from_cache

        pids = list(top_n_pids_from_cache(args.cache))

    grid = _config_grid()
    print(f"offclock_sweep: {len(grid)} configs, {len(pids)} products", flush=True)
    rows: list[dict] = []
    for i, (substrate, label_variant, k) in enumerate(grid, 1):
        tag = f"{substrate}/{label_variant}/h{k}"
        print(f"[{i}/{len(grid)}] {tag} ...", flush=True)
        try:
            rows.append(_run_one(substrate, label_variant, k, pids, args.sample_step))
        except Exception as e:
            print(f"    SKIPPED {tag}: {e}", flush=True)
            traceback.print_exc()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(_render_results_doc(rows))
    print(f"wrote {args.out} ({len(rows)} configs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
