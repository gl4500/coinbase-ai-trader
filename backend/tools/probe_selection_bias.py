"""Selection-bias meta-analysis of the XGB feature-search probe history.

Deflates the recorded probe results for the size of the search: given N trials
and a per-trial noise scale, computes the best result a true-null search would
still produce, and a Deflated-Sharpe-style probability that the observed edge
exceeds that floor. See 2026-05-22-probe-selection-bias-design.md.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_NORM = NormalDist()  # standard normal
_EULER_GAMMA = 0.5772156649015329


def iid_auc_se(n_pos: int, n_neg: int) -> float:
    """SE of an AUC under H0 (true AUC = 0.5), treating samples as iid.

    The Mann-Whitney null variance: (n_pos + n_neg + 1) / (12 * n_pos * n_neg).
    Optimistic for overlapping financial labels — reported only for contrast.

    Raises:
        ValueError: if either count is not positive.
    """
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError(f"counts must be positive, got n_pos={n_pos} n_neg={n_neg}")
    return math.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))


def fold_level_se(fold_aucs: list[float]) -> float:
    """Empirical SE from a list of per-fold AUCs — the sample standard deviation.

    The honest noise unit for non-iid overlapping labels.

    Raises:
        ValueError: if fewer than 2 folds are given.
    """
    n = len(fold_aucs)
    if n < 2:
        raise ValueError(f"need >= 2 folds for an empirical SE, got {n}")
    mean = sum(fold_aucs) / n
    var = sum((a - mean) ** 2 for a in fold_aucs) / (n - 1)
    return math.sqrt(var)


def expected_max_under_null(n_trials: int, se: float, center: float) -> float:
    """Best result a true-null search of N trials would still produce.

    `center + se * E[max of N standard normals]`, using Bailey & Lopez de
    Prado's expected-max approximation. `center` is the null value — 0.5 for
    an AUC, 0.0 for a lift Delta.

    Raises:
        ValueError: if n_trials < 2.
    """
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2, got {n_trials}")
    e_max_z = (
        (1.0 - _EULER_GAMMA) * _NORM.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER_GAMMA * _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )
    return center + se * e_max_z


def deflated_probability(observed: float, n_trials: int, se: float,
                         center: float) -> float:
    """Probability the observed result exceeds the best-of-N-trials noise floor.

    A Deflated-Sharpe-style statistic: `Phi((observed - expected_max) / se)`.
    ~1.0 when the observed result is far above the noise floor; ~0.5 when it
    sits exactly on it; ~0.0 when it is below.
    """
    floor = expected_max_under_null(n_trials, se, center)
    return _NORM.cdf((observed - floor) / se)


# Trial table — transcribed from xgb_probe_results_log.md. Each entry is one
# single-add channel-candidate probe run. The marketcap probe is excluded
# (null-coverage — CoinGecko 401/429, never a real signal trial). sma50_d1's
# Delta is the leak-corrected value (the +0.0852 was 100% lookahead leak).
TRIALS: list[dict] = [
    {"name": "RSI-rank (survivorship)", "delta": +0.0124, "passed": True},
    {"name": "RSI-rank (legacy)",       "delta": +0.0208, "passed": True},
    {"name": "log10-vol-rank",          "delta": -0.0010, "passed": False},
    {"name": "MFI-rank",                "delta": +0.0031, "passed": False},
    {"name": "BTC-dominance",           "delta": +0.0077, "passed": False},
    {"name": "OKX long/short",          "delta": +0.0014, "passed": False},
    {"name": "sma50_1h",                "delta": +0.0007, "passed": False},
    {"name": "sma200_1h",               "delta": +0.0007, "passed": False},
    {"name": "sma50_d1",                "delta": -0.0002, "passed": False},
    {"name": "sma200_d1",               "delta": +0.0004, "passed": False},
    {"name": "golden_cross_d1",         "delta": +0.0006, "passed": False},
    {"name": "btc_ret_lag_1",           "delta": -0.0021, "passed": False},
    {"name": "btc_ret_lag_4",           "delta": -0.0100, "passed": False},
    {"name": "btc_ret_lag_12",          "delta": -0.0038, "passed": False},
    {"name": "btc_beta_60",             "delta": -0.0003, "passed": False},
    {"name": "btc_beta_residual_60",    "delta": -0.0003, "passed": False},
    {"name": "btc_residual_ch9",        "delta": -0.0000, "passed": False},
]

# Documented constants (from xgb_feature_optimization_findings.md / the
# scorecard baseline / xgb_probe_results_log.md).
N_SAMPLES = 167933          # pooled top-20, 28-channel cache
POS_RATE = 0.488            # label balance
V3_BEST_AUC = 0.5284        # best documented config (top-40, tuned)
V3_OOF_AUC = 0.512          # scorecard 5-fold purged-WF OOF AUC
BEST_LIFT_DELTA = 0.0124    # RSI-rank survivorship-aware — the only confirmed PASS
BASELINE_FOLD_AUCS = [0.516, 0.507, 0.527, 0.523, 0.529]  # logged purged-WF folds
N_TIERS = (17, 100, 200)    # channel candidates / full search / conservative


def analyze() -> list[dict]:
    """Run both deflation tracks across the N tiers and both noise scales.

    Track A — v3's base AUC edge (center 0.5), for the best documented AUC and
    the scorecard OOF AUC. Track B — the best channel-add lift (center 0.0).
    Returns one row dict per (track, observed value, N tier, noise scale).
    """
    n_pos = round(N_SAMPLES * POS_RATE)
    n_neg = N_SAMPLES - n_pos
    noise_scales = (
        ("fold", fold_level_se(BASELINE_FOLD_AUCS)),
        ("iid", iid_auc_se(n_pos, n_neg)),
    )

    rows: list[dict] = []
    track_a = (("v3 best documented AUC", V3_BEST_AUC),
               ("v3 scorecard OOF AUC", V3_OOF_AUC))
    for label, observed in track_a:
        for n in N_TIERS:
            for noise_label, se in noise_scales:
                rows.append({
                    "track": "A: base AUC edge",
                    "observed_label": label, "observed": observed,
                    "n_trials": n, "noise": noise_label, "se": se,
                    "center": 0.5,
                    "expected_max": expected_max_under_null(n, se, 0.5),
                    "deflated_prob": deflated_probability(observed, n, se, 0.5),
                })
    for n in N_TIERS:
        for noise_label, se in noise_scales:
            rows.append({
                "track": "B: best channel lift",
                "observed_label": "RSI-rank Delta", "observed": BEST_LIFT_DELTA,
                "n_trials": n, "noise": noise_label, "se": se, "center": 0.0,
                "expected_max": expected_max_under_null(n, se, 0.0),
                "deflated_prob": deflated_probability(BEST_LIFT_DELTA, n, se, 0.0),
            })
    return rows


def verdict_line(headline_prob: float) -> str:
    """One-line verdict from the headline deflated probability.

    headline = Track A, best documented AUC, fold-level noise, N=100.
    """
    if headline_prob >= 0.95:
        return (f"VERDICT: real edge — deflated probability {headline_prob:.2f} "
                "clears the 0.95 bar; feature work on this cache is justified.")
    if headline_prob <= 0.50:
        return (f"VERDICT: indistinguishable from noise — deflated probability "
                f"{headline_prob:.2f} sits at/below the best-of-N noise floor; "
                "further single-add feature work on this cache is not justified.")
    return (f"VERDICT: marginal — deflated probability {headline_prob:.2f} is "
            "above the noise floor but short of the 0.95 bar; the edge is weak "
            "and selection-fragile.")


def _headline_prob(rows: list[dict]) -> float:
    """Track A, best documented AUC, fold noise, N=100 — the headline cell."""
    for r in rows:
        if (r["track"].startswith("A") and r["n_trials"] == 100
                and r["noise"] == "fold" and "best" in r["observed_label"]):
            return r["deflated_prob"]
    raise RuntimeError("headline cell not found in analysis rows")


def render_report(rows: list[dict]) -> str:
    """Render the full meta-analysis as a markdown doc."""
    lines = [
        "# Probe Selection-Bias Meta-Analysis",
        "",
        "Spec: `2026-05-22-probe-selection-bias-design.md`. Deflates the recorded",
        "XGB feature-search results for the size of the search.",
        "",
        "## Verdict",
        "",
        verdict_line(_headline_prob(rows)),
        "",
        "## Trial table",
        "",
        f"{len(TRIALS)} single-add channel-candidate probes (marketcap excluded —",
        "null-coverage). Source: `xgb_probe_results_log.md`.",
        "",
        "| probe | Delta AUC | passed |",
        "|---|---|---|",
    ]
    for t in TRIALS:
        lines.append(f"| {t['name']} | {t['delta']:+.4f} | {t['passed']} |")

    lines += [
        "",
        "## Deflation analysis",
        "",
        "Noise scales: `fold` = empirical purged-WF per-fold SE (honest for",
        "overlapping labels); `iid` = Mann-Whitney null SE (contrast — optimistic).",
        "",
        "| track | observed | N | noise | SE | exp_max_under_null | deflated_prob |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['track']} | {r['observed_label']} {r['observed']:.4f} | "
            f"{r['n_trials']} | {r['noise']} | {r['se']:.5f} | "
            f"{r['expected_max']:.5f} | {r['deflated_prob']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe selection-bias meta-analysis (roadmap task #16)"
    )
    parser.add_argument(
        "--out",
        default="../docs/superpowers/specs/2026-05-22-probe-selection-bias-results.md",
        help="results doc path (default: the SP-#16 results spec)",
    )
    args = parser.parse_args()
    doc = render_report(analyze())
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(doc, flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
