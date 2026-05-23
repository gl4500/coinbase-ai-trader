# Probe Selection-Bias Meta-Analysis

Spec: `2026-05-22-probe-selection-bias-design.md`. Deflates the recorded
XGB feature-search results for the size of the search.

## Verdict

VERDICT: marginal — deflated probability 0.74 is above the noise floor but short of the 0.95 bar; the edge is weak and selection-fragile.

## Trial table

17 single-add channel-candidate probes (marketcap excluded —
null-coverage). Source: `xgb_probe_results_log.md`.

| probe | Delta AUC | passed |
|---|---|---|
| RSI-rank (survivorship) | +0.0124 | True |
| RSI-rank (legacy) | +0.0208 | True |
| log10-vol-rank | -0.0010 | False |
| MFI-rank | +0.0031 | False |
| BTC-dominance | +0.0077 | False |
| OKX long/short | +0.0014 | False |
| sma50_1h | +0.0007 | False |
| sma200_1h | +0.0007 | False |
| sma50_d1 | -0.0002 | False |
| sma200_d1 | +0.0004 | False |
| golden_cross_d1 | +0.0006 | False |
| btc_ret_lag_1 | -0.0021 | False |
| btc_ret_lag_4 | -0.0100 | False |
| btc_ret_lag_12 | -0.0038 | False |
| btc_beta_60 | -0.0003 | False |
| btc_beta_residual_60 | -0.0003 | False |
| btc_residual_ch9 | -0.0000 | False |

## Deflation analysis

Noise scales: `fold` = empirical purged-WF per-fold SE (honest for
overlapping labels); `iid` = Mann-Whitney null SE (contrast — optimistic).

| track | observed | N | noise | SE | exp_max_under_null | deflated_prob |
|---|---|---|---|---|---|---|
| A: base AUC edge | v3 best documented AUC 0.5284 | 17 | fold | 0.00899 | 0.51643 | 0.908 |
| A: base AUC edge | v3 best documented AUC 0.5284 | 17 | iid | 0.00141 | 0.50258 | 1.000 |
| A: base AUC edge | v3 best documented AUC 0.5284 | 100 | fold | 0.00899 | 0.52275 | 0.735 |
| A: base AUC edge | v3 best documented AUC 0.5284 | 100 | iid | 0.00141 | 0.50357 | 1.000 |
| A: base AUC edge | v3 best documented AUC 0.5284 | 200 | fold | 0.00899 | 0.52486 | 0.653 |
| A: base AUC edge | v3 best documented AUC 0.5284 | 200 | iid | 0.00141 | 0.50390 | 1.000 |
| A: base AUC edge | v3 scorecard OOF AUC 0.5120 | 17 | fold | 0.00899 | 0.51643 | 0.311 |
| A: base AUC edge | v3 scorecard OOF AUC 0.5120 | 17 | iid | 0.00141 | 0.50258 | 1.000 |
| A: base AUC edge | v3 scorecard OOF AUC 0.5120 | 100 | fold | 0.00899 | 0.52275 | 0.116 |
| A: base AUC edge | v3 scorecard OOF AUC 0.5120 | 100 | iid | 0.00141 | 0.50357 | 1.000 |
| A: base AUC edge | v3 scorecard OOF AUC 0.5120 | 200 | fold | 0.00899 | 0.52486 | 0.076 |
| A: base AUC edge | v3 scorecard OOF AUC 0.5120 | 200 | iid | 0.00141 | 0.50390 | 1.000 |
| B: best channel lift | RSI-rank Delta 0.0124 | 17 | fold | 0.00899 | 0.01643 | 0.327 |
| B: best channel lift | RSI-rank Delta 0.0124 | 17 | iid | 0.00141 | 0.00258 | 1.000 |
| B: best channel lift | RSI-rank Delta 0.0124 | 100 | fold | 0.00899 | 0.02275 | 0.125 |
| B: best channel lift | RSI-rank Delta 0.0124 | 100 | iid | 0.00141 | 0.00357 | 1.000 |
| B: best channel lift | RSI-rank Delta 0.0124 | 200 | fold | 0.00899 | 0.02486 | 0.083 |
| B: best channel lift | RSI-rank Delta 0.0124 | 200 | iid | 0.00141 | 0.00390 | 1.000 |