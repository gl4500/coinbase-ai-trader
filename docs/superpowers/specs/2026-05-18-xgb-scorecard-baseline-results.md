# XGB Scorecard Baseline Results — v3 driver

**Date:** 2026-05-19
**Spec:** `2026-05-18-xgb-deployment-scorecard-design.md`
**Plan:** `2026-05-18-xgb-deployment-scorecard.md` (Task 7/8/9, corrected 2026-05-19)
**Track:** v3 driver only. v4 / v4.5 pending Tasks 7b/7c.

## Run configuration

- **Harness:** `tools/scorecard.py --track v3 --gate-tier retail`
- **Samples:** built from per-pid OHLCV parquets the same way `train_xgb.train_xgb_v3` does — tiered slices (micro 60 / meso 168 / macro 336), v3 features (350), label `close[t+4] > close[t]`.
- **Pids:** survivorship-aware top-20 (cache used only for the ranking).
- **`sample_step`:** 24 (matches `train_xgb_v3`; one sample per 24 bars).
- **Total samples:** 7,386. **Positive rate:** 0.458.
- **CV:** 5-fold purged walk-forward, 4h embargo; per-fold fresh booster (no in-sample leakage from the deployed `xgb_model.json`).
- **Fees:** retail 0.6% / mid 0.25% / pro 0.05% per side (round-trip 2×).

## v3 driver report

```
OOF mean AUC: 0.5120
pos_rate:     0.4575
ECE:          0.0468   (gate <0.05 => PASS)
Recommended operating tau: nan   (no tau qualifies)

   tau    prec   n_fired     E[r]_R     E[r]_M     E[r]_P     Sh_R     Sh_M     Sh_P
  0.50   0.477      2033   -0.01227   -0.00527   -0.00127  -26.587  -11.398   -2.718
  0.55   0.481       941   -0.01223   -0.00523   -0.00123  -18.439   -7.947   -1.951
  0.60   0.490       404   -0.01218   -0.00518   -0.00118  -12.346   -5.169   -1.067
  0.65   0.472       142   -0.01067   -0.00367    0.00033   -6.543   -2.176    0.320
  0.70   0.425        40   -0.01279   -0.00579   -0.00179   -4.520   -2.169   -0.826
  0.75   0.500         8   -0.01088   -0.00388    0.00012   -5.943   -1.094    1.678
  0.80   0.000         1   -0.02746   -0.02046   -0.01646      nan      nan      nan
  0.85     nan         0        nan        nan        nan      nan      nan      nan
  0.90     nan         0        nan        nan        nan      nan      nan      nan
  0.95     nan         0        nan        nan        nan      nan      nan      nan
```

## Hard-gate summary

| Metric | Gate | Result |
|---|---|---|
| Precision-at-gate | `P_τ ≥ pos_rate + 0.03` at a τ with ≥100 fires | **FAIL** — best precision among ≥100-fire taus is 0.490 (τ=0.60) vs the 0.488 bar; never a clear beat, and it is not net-positive |
| Expected return | `E[r_τ] > 0` after retail fees | **FAIL** — every τ is negative at the retail tier |
| Paper-Sharpe (annualized) | `> 0` | **FAIL** — deeply negative at retail across all τ |
| ECE | `< 0.05` | **PASS** — 0.0468 |

**v3 passes 1 of 4 hard gates.**

## Interpretation

1. **v3 is not deployable as a fee-paying retail strategy.** The model's directional edge is real but tiny — precision peaks at ~0.49 against a 0.458 base rate, an edge of ~3 percentage points. A 1.2% retail round-trip fee is far larger than any per-signal edge, so expected return is negative at every threshold. E[r] only crosses zero at the `pro` tier (0.05%/side), and only marginally (τ 0.65/0.75).

2. **The AUC reframe was correct.** The project began because v3's AUC (~0.528, a top-40 feature-selected figure) sat under a 0.55 gate. The deployment scorecard makes the situation sharper and more honest: the full-350-feature purged-WF OOF AUC is **0.512**, and even where the model ranks correctly the economics do not survive fees. Chasing AUC toward 0.55 would not have produced a profitable strategy at retail fees.

3. **Calibration is the one healthy signal.** ECE 0.047 means the probabilities are roughly trustworthy — when v3 says 0.6 it is near 0.6. The problem is not miscalibration; it is that the true edge is too small to overcome costs.

4. **Gate-retirement recommendation: retire the 0.55 AUC gate, replace it with the scorecard.** AUC never measured the thing that matters. A model should promote only if it clears all four hard gates on the retail tier. By that standard v3 does not qualify for fee-paying deployment today — which matches its observed paper-trading performance. Keep v3 running only as a baseline/shadow until a variant clears the scorecard.

## Caveats

- `sample_step=24` yields 7,386 samples (~1,500/fold); high-τ rows fire too rarely (≤8 at τ≥0.75) for stable estimates. Re-run with a smaller `--sample-step` for denser high-threshold statistics.
- AUC 0.512 < the documented 0.528: that 0.528 was a *top-40 feature-selected* peak (`xgb_feature_optimization_findings.md`); 0.512 is the honest full-feature purged-WF OOF and is the more representative number.
- Per-fold Sharpe magnitudes are large because annualization multiplies by `√(signals/year)`; read the sign and cross-tier ordering, not the absolute value.

## Follow-up

- Tasks 7b / 7c — v4 and v4.5 scorecard tracks (separate OHLCV-parquet harnesses; v4.5 adds the 3-class × 3-horizon × 3-rule grid).
- SELL-side scorecard — deferred per spec O5.
- Amend the design spec's "Val fold convention" — the ~167k figure is the CNN cache count, not v3's parquet-derived sample count.
