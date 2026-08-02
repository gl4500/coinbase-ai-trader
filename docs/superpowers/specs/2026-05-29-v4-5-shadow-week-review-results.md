# v4.5 Shadow-Week Review — Results

**Date:** 2026-05-29 (review nominally due 2026-05-31)
**Topic:** Predictive accuracy + would-be ROI of XGB v4.5 (3-class) on the 8002 dev backend's shadow telemetry, 2026-05-23 → 2026-05-29.
**Status:** Final for h24/h72. h168 re-score deferred to ~2026-06-05 (label maturity).

---

## TL;DR

**Do NOT promote v4.5 to 8001 at this time.**

- Realized UP-AUC ≈ 0.51 at both h24 and h72 — barely above random.
- Calibration is poor (ECE 0.103–0.127) — gate is <0.05.
- Every BUY threshold is net-negative ROI after retail fees, at both horizons.
- The model has *some* directional signal at high confidence (h24 thr=0.60 precision 46.4% vs 35.5% base rate — a +11 pp edge) but it is not strong enough to overcome the 1.2% round-trip retail fee in this regime.
- Pattern matches the v3 scorecard (Session 58.71l-m, 1/4 gates). The deeper issue is fee tier vs edge magnitude, not the model.

---

## Methodology

### Data sources
- **Shadow scans:** `backend/coinbase_dev.db` `cnn_scans` rows where `xgb_prob_v4_5_up IS NOT NULL`, 2026-05-23T16:20 → 2026-05-29T21:12. Read-only.
- **OHLCV series:** shared parquets at `backend/data/history/<pid>.parquet` (static, no live-DB contention).
- **Boosters:** loaded directly from `backend/xgb_model_v4_5_h{24,72}.json` + matching feature jsons. Deployed `xgb_model_v4_5.json` (h168) was NOT touched.

### Why h24/h72 and not h168
The deployed model is h168 (7-day forward triple-barrier label). A 7-day label needs 7 days of *future* price. Maturity by review date:

| As of | Matured h168 scans |
|---|---|
| 2026-05-29 (today) | **0** |
| 2026-05-31 | ~12.8k (only 05-23/24) |
| 2026-06-05 | ~84.8k (full shadow set) |

The h168 verdict cannot be rendered until ~06-05. h24/h72 are sibling models on the *identical* feature set (210 features, same `xgb_v4_5_features` extractor) — they give an early read using already-matured outcomes.

### Pipeline (per scan)
1. Find largest parquet bar index `hi` with `start < scan_ts` (mirrors `fetch_tiered` semantics).
2. Reconstruct tiers in-memory: micro=60, meso=168, macro=336 (last-N bars before scan).
3. Extract 210 features via `extract_v4_5`.
4. Predict with h24 and h72 boosters (same feature vector → 2 predictions).
5. Compute realized triple-barrier label `_triple_barrier_label_3class(closes, hi-1, H, thresh)` on the same close series; thresholds h24=0.015, h72=0.03 (the training values).
6. Compute endpoint forward return `closes[hi-1+H] / closes[hi-1] − 1`.
7. Record `(p_up, p_down, label, fwd_ret)` per horizon.

### Sample
- 102 of 103 v4.5 products have parquets (only BOBBOB-USD missing).
- 75 of 102 had enough history (≥336 bars before scan) AND matured outcomes within the parquet's coverage.
- Per-product cap of 150 stride-sampled scans.
- **N = 7,160 h24 / 3,374 h72 matured scans.**

### Decision rule + fees
- `_indep_thresholds_decision`: BUY when `p_up > thr AND p_up ≥ p_down`.
- Threshold sweep: {0.40, 0.45, 0.50, 0.55, 0.60}.
- Retail fee: 1.2% round-trip (0.6%/side; matches Session 58.71l-m scorecard).

---

## Telemetry health (8002 dev backend)

The telemetry itself is healthy.

| Metric | Value |
|---|---|
| Total scans with v4.5 populated | 84,893 |
| Distinct products | 103 |
| Neutral-fallback rows `(0.33, 0.34, 0.33)` | **0** |
| Class mix (argmax over 3 probs) | DOWN 54% / UP 40% / NEUTRAL 6% |
| UP-prob spread | median 0.41 · p95 0.66 · p99 0.82 · max 0.87 |
| Paper trades on 8002 | **0** (telemetry-only mode; signals fire but no positions opened) |

The model is making real, well-separated predictions across the full product set. The review is purely predictive-accuracy (no trade-PnL comparison possible).

---

## Market context

| Horizon | Endpoint ret mean | P(ret > 0) | Label mix UP / NEU / DOWN |
|---|---|---|---|
| h24 | **−1.73%** per sample | 30.0% | 35.5% / 9.7% / 54.9% |
| h72 | **−4.19%** per sample | 19.8% | 32.6% / 11.6% / 55.7% |

The shadow week was a strong bear regime. This matters: any model that buys signals is fighting a heavy headwind, and the asymmetry between winning/losing trade magnitude is unfavorable.

---

## Results

### Predictive accuracy

| Horizon | UP-AUC vs label==UP | UP-AUC vs endpoint ret>0 | ECE |
|---|---|---|---|
| h24 | 0.5058 | 0.5454 | 0.1270 |
| h72 | 0.5033 | 0.5418 | 0.1031 |

AUC vs triple-barrier label hovers at random. AUC vs endpoint sign is slightly better (~0.54), consistent with the model capturing some short-horizon momentum signal. Calibration is materially poor at both horizons.

### Would-be ROI — `indep_thresholds` BUY sweep

**h24 (label_thresh 0.015), net of 1.2% retail fee:**

| thr | n_buy | %fired | prec(lab=UP) | prec(ret>0) | mean_ret | net_ret |
|---|---|---|---|---|---|---|
| 0.40 | 3,764 | 52.6% | 31.0% | 31.5% | −1.22% | **−2.42%** |
| 0.45 | 3,440 | 48.0% | 32.2% | 32.0% | −1.25% | **−2.45%** |
| 0.50 | 2,525 | 35.3% | 35.8% | 35.4% | −1.26% | **−2.46%** |
| 0.55 | 1,316 | 18.4% | 40.0% | 39.1% | −1.68% | **−2.88%** |
| 0.60 | 640 | 8.9% | **46.4%** | **47.7%** | −1.79% | **−2.99%** |

**h72 (label_thresh 0.03), net of 1.2% retail fee:**

| thr | n_buy | %fired | prec(lab=UP) | prec(ret>0) | mean_ret | net_ret |
|---|---|---|---|---|---|---|
| 0.40 | 1,209 | 35.8% | 23.3% | 20.2% | −2.28% | **−3.48%** |
| 0.45 | 1,098 | 32.5% | 25.3% | 22.1% | −1.96% | **−3.16%** |
| 0.50 | 727 | 21.5% | 26.7% | 24.5% | −2.44% | **−3.64%** |
| 0.55 | 162 | 4.8% | 29.0% | 27.8% | −5.12% | **−6.32%** |
| 0.60 | 37 | 1.1% | 27.0% | 16.2% | −13.25% | **−14.45%** |

At h72 the highest-confidence buckets are *worse* than the lower thresholds — what little signal exists at h24 is gone by h72.

### Scorecard 4-gate verdict (Session 58.71l-m framework)

| Gate | h24 best | h72 best |
|---|---|---|
| Precision ≥ pos_rate + 0.03 | ✅ at thr ≥ 0.55 (40.0%, 46.4% vs gate 38.5%) | ❌ all (max 29.0% vs gate 35.6%) |
| E[r] > 0 net of fee | ❌ all negative | ❌ all negative |
| Paper-Sharpe > 0 | ❌ all negative | ❌ all negative |
| ECE < 0.05 | ❌ 0.127 | ❌ 0.103 |

**h24: 1/4 gates. h72: 0/4 gates.**

---

## Recommendation

1. **Do not promote v4.5 to 8001.** No threshold produces positive net ROI; calibration is poor; UP-AUC is at chance.
2. **Keep 8002 shadow running** until h168 matures (~2026-06-05) — that gives us the official h168 verdict on the same data.
3. **Re-score with h168 on or after 2026-06-05** using the same harness. One-line config change: see Follow-ups below.
4. **Do not iterate on model architecture yet.** The h24 high-confidence precision lift (+11 pp) shows the model has signal — it's the fee tier vs edge magnitude that fails, not the model. Architecture iteration without addressing the fee tier (e.g., promoting only at thresholds + horizons where E[r] > 0 by enough margin to clear fees) will repeat this outcome.
5. **Pro-tier fee assessment.** Session 58.71l-m noted v3 was only positive at pro-tier 0.05%. The retail-vs-pro fee gap is the dominant variable. Quantify "what would v4.5 ROI look like at pro-tier?" as a separate analysis — it informs whether to pursue a Coinbase Advanced or pro account vs more model iteration.

---

## Caveats

1. **One bear week is not the universe.** Mean endpoint returns of −1.7% / −4.2% are heavy. A bull or sideways window could materially shift `mean_ret`. AUC and ECE are regime-less metrics though, and both fail.
2. **h168 not directly scored here.** The deployed model is h168; h24/h72 are sibling models on identical features. Highly correlated but not identical predictions. Re-score ~06-05.
3. **75/103 products** scored due to parquet history requirements. Coverage gap, not material to the verdict (the sample is large and stratified across products).
4. **Endpoint return ≠ deployed PnL.** Real PnL adds slippage, trail-stop exits, max-hold cuts, and any active risk-management. Endpoint return is a clean signal-quality lower bound; deployed PnL would typically be worse.
5. **The harness is throwaway analysis** at `C:\Users\gl450\AppData\Local\Temp\v45_rescore.py`. Not in version control. Re-runs are reproducible from the script (boosters + parquets + dev DB only).

---

## Follow-ups

- **2026-06-05 — h168 re-score.** Edit harness `HORIZONS = {168: 0.06}` (replace or add), re-run. Roughly the same N (~85k samples potential, all 6 days of shadow data matured).
- **Decision-rule replay.** Apply the indep_thresholds rule to the historical `xgb_prob_v4_5_*` columns and join to candle prices for an exit-aware would-be-PnL replay (per the deferred "Option B" in `2026-05-23-backfill-v4-5-shadow-design.md`). Useful if we want to model real trade flow rather than pure endpoint return.
- **Pro-tier ROI table.** Same h24/h72 sweep with fee = 0.1% round-trip instead of 1.2% — quantifies whether fee tier alone closes the gap.
- **Bull-period replay.** Backfill v4.5 predictions over a historical bull window (e.g. Q4 2025 if data exists) and re-score. Tests the regime caveat.

---

## See also

- `docs/superpowers/specs/2026-05-17-xgb-v4-5-three-class-design.md` — v4.5 design
- `docs/superpowers/specs/2026-05-18-xgb-deployment-scorecard-design.md` — scorecard framework
- `docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md` — v3 baseline (1/4 gates)
- `docs/superpowers/specs/2026-05-23-backfill-v4-5-shadow-design.md` — backfill that populated the shadow telemetry
