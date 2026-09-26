# v4.5 h168 Re-score — Results

**Date:** 2026-06-08 (was due 2026-06-05)
**Topic:** Post-maturity h168 verdict on the deployed v4.5 booster.
**Status:** Final. v4.5 fully closed as a deployment candidate.

---

## TL;DR

**Do NOT promote v4.5 at h168 — 0/4 scorecard gates pass at retail.**

- All three v4.5 horizons (h24, h72, h168) now confirmed deployment-blocked across two independent matured-sample windows.
- UP-AUC vs endpoint-return = **0.4329** — *below random*. In the strong-bear regime that dominated the matured h168 window, the model's BUY signals are anti-correlated with positive 7-day returns.
- The longer-horizon hypothesis ("less fee drag wins") is empirically refuted. h168's lower fee drag (~580 pp/year savings vs h24) does NOT compensate for the larger per-trade loss.
- Next-direction work (information-driven bar sampling) is being handled on a parallel CLI workstream and is NOT in scope for this doc.

---

## Methodology

### Data sources
- **Shadow scans:** `backend/coinbase_dev.db` `cnn_scans` rows where `xgb_prob_v4_5_up IS NOT NULL`, 2026-05-23 → 2026-06-08. Read-only.
- **OHLCV series:** shared parquets at `backend/data/history/<pid>.parquet`.
- **Booster:** loaded directly from `backend/xgb_model_v4_5_h168.json` + matching feature json (210 features, 600 trees).
- The deployed `xgb_model_v4_5.json` (h168 singleton) was NOT touched.

### Harness
Adapted from Session 58.71t's `C:/Users/gl450/AppData/Local/Temp/v45_rescore.py` with `HORIZONS = {168: 0.06}`. Throwaway at `C:/tmp/h168_rescore.py`. Same pipeline:

1. Find largest parquet bar index `hi` with `start < scan_ts` (mirrors `fetch_tiered` semantics).
2. Reconstruct tiers in-memory: micro=60, meso=168, macro=336 (last-N bars before scan).
3. Extract 210 features via `extract_v4_5`.
4. Predict with h168 booster.
5. Compute realized triple-barrier label `_triple_barrier_label_3class(closes, hi-1, 168, 0.06)`.
6. Compute endpoint forward return `closes[hi-1+168] / closes[hi-1] − 1`.
7. Record `(p_up, p_down, label, fwd_ret)`.

### Sample
- 218,962 total shadow scans surveyed across 133 pids with v4.5 telemetry.
- 78 of 133 pids had enough parquet history (≥336 bars before scan) AND ≥1 matured h168 outcome.
- Per-pid cap of 150 stride-sampled scans.
- **N = 5,774 matured h168 scans.**
- Matured fraction: 5,774 / 218,962 ≈ **2.6%**, consistent with the 7-day maturity gate.

### Decision rule + fees
- `_indep_thresholds_decision`: BUY when `p_up > thr AND p_up ≥ p_down`.
- Threshold sweep: {0.40, 0.45, 0.50, 0.55, 0.60}.
- Retail fee: 1.2% round-trip (0.6%/side; matches Session 58.71l-m scorecard).
- Pro fee: 0.1% round-trip (comparison).

---

## Market context (h168 matured window)

| Metric | Value |
|---|---|
| Endpoint ret mean | **−9.21%** |
| P(endpoint ret > 0) | 19.6% |
| Label mix UP / NEU / DOWN | **26.6% / 7.0% / 66.5%** |
| Regime | **strong bear** |

Two-thirds of all matured 7-day windows ended DOWN by ≥6% (the triple-barrier threshold). The realized UP-rate (26.6%) is well below 50% — the asymmetric market context the model was deployed into.

---

## Threshold sweep at h168

```
  thr   n_buy %fired prec(UP) prec(ret>0) mean_ret  net_retail  net_pro
 0.40    1957  33.9%   23.4%       14.5%   −8.83%     −10.03%   −8.93%
 0.45    1676  29.0%   25.8%       15.6%   −8.45%      −9.65%   −8.55%
 0.50    1124  19.5%   28.2%       16.1%   −8.14%      −9.34%   −8.24%
 0.55     534   9.2%   27.7%       15.4%   −7.68%      −8.88%   −7.78%
 0.60     229   4.0%   17.5%       12.7%   −8.61%      −9.81%   −8.71%
```

Every threshold is net-negative ROI at both retail AND pro fee tiers. There is no calibration point at which the model is profitable.

Note the **precision DROP at thr=0.60** (17.5%) versus thr=0.55 (27.7%). The most-confident predictions are actually the WORST. This is the signature of a model that has overfit to features that invert in the deployment regime — the more confident the BUY, the worse the outcome.

---

## Scorecard at thr=0.60 (retail tier)

| Gate | Value | Threshold | Pass |
|---|---|---|---|
| Precision (lift over pos_rate) | 17.5% | ≥ 29.6% (pos_rate 26.6% + 0.03) | **FAIL** |
| E[r] net of 1.2% RT fee | −9.81% | > 0 | **FAIL** |
| Per-trade Sharpe | −0.477 | > 0 | **FAIL** |
| ECE (decile bins) | 0.166 | < 0.05 | **FAIL** |
| **Total** | | | **0 / 4** |

---

## Cross-horizon comparison

Combining Session 58.71t (h24/h72) and this run (h168):

| Metric | h24 (58.71t) | h72 (58.71t) | **h168 (this)** |
|---|---|---|---|
| N matured | 7,160 | 3,374 | 5,774 |
| Pids used | 75 | 102 | 78 |
| Sample mean endpoint ret | −1.73% | −4.19% | **−9.21%** |
| Sample P(endpoint > 0) | ~40% | ~35% | **19.6%** |
| Scorecard gates passed | 1 / 4 | 0 / 4 | **0 / 4** |
| Precision @ thr=0.60 | 46.4% (PASS) | FAIL | **17.5%** |
| ECE | 0.127 (FAIL) | FAIL | **0.166** (FAIL) |
| Net E[r] @ thr=0.60 (retail) | −2.99% | FAIL | **−9.81%** |
| UP-AUC vs label==UP | ~0.515 | ~0.51 | **0.5388** |
| UP-AUC vs endpoint ret > 0 | ~0.51 | ~0.51 | **0.4329** (negative edge) |

The "UP-AUC vs endpoint" row is the cleanest cross-horizon comparator (independent of label-threshold choice). h168 is the only horizon **below 0.5** — actively misranking samples relative to forward return.

---

## Annualization (cross-horizon, retail fees, per slot)

| Horizon | Trades/year | Fee drag/year | Per-trade net E[r] | Annualized net E[r] |
|---|---|---|---|---|
| h24 | ~365 | 438% | −2.99% | **−1,090%** |
| h72 | ~122 | 146% | (worse than h24) | (worse) |
| h168 | ~52 | 62% | −9.81% | **−510%** |

h168's annualized loss is "only" half of h24's — entirely because it trades 7× less often. But this is not a victory. The **per-trade economics** were the original h168 thesis: ~6× lower fee drag, hoping the per-trade edge would survive. It did not. The model lost more per trade than h24 did, by enough to halve but not flip the annualized comparison.

---

## Interpretation: why h168 is the worst horizon

Three compounding effects:

1. **Regime asymmetry.** The matured h168 window is dominated by a deep bear (mean −9.2%). v4.5 was trained on data that included bull regimes; deployed into bear, the "BUY high p_up" signals invert. h24's matured window was milder (−1.7%); h168's is severe (−9.2%); the model's miscalibration scales with the regime gap.

2. **Triple-barrier asymmetry.** label_thresh=0.06 means UP requires a +6% move in 7 days. In a strong bear, hitting +6% is far rarer than hitting −6% (pos_rate 26.6%). The model's predictions don't compensate — they're calibrated for a more symmetric prior.

3. **Negative-edge structure.** UP-AUC vs endpoint = 0.4329 means *the model's confidence is anti-informative*. The booster has found 600 trees of features that, in this regime, point exactly the wrong way. This is not "no signal" — it's "negative signal." Adjusting the threshold won't save it.

---

## Verdict

- **v4.5 closed across all 3 horizons.** Same plateau v3 hit in Session 58.71l-m's scorecard.
- The cause is structural (no signal in the current sample distribution + regime sensitivity), not transformational (no feature/architecture tweak will save it).
- Pattern matches the v3 baseline scorecard (1 of 4 gates) and is consistent with the v3 feature-pruning research (PR #16 Session 58.71v) that confirmed the v3 model's "edge" reduces to overfitting noise at sub-5-feature isolation.

---

## Files referenced

- Harness: `C:/tmp/h168_rescore.py` (throwaway; mirrors Session 58.71t's `C:/Users/gl450/AppData/Local/Temp/v45_rescore.py`)
- Run log: `C:/tmp/h168_rescore.log`
- Result JSON: `C:/tmp/h168_rescore.json`
- Prior shadow-week review: `docs/superpowers/specs/2026-05-29-v4-5-shadow-week-review-results.md`

---

## Closes

- Task **#79** (h168 re-score on/after 2026-06-05) — DONE 2026-06-08 with 0/4 gates.

---

## See also

- `docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md` — parallel CLI's bar-structure design (strategy-discovery Phase 2 → 3 → 4 on dollar bars). Tests whether information-driven sampling changes the deflation-corrected edge for a *different* pipeline (strategy-discovery, not XGB).
- `docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md` — its implementation plan.
- `docs/superpowers/specs/2026-05-29-v4-5-shadow-week-review-results.md` — Session 58.71t h24/h72 review that deferred h168 to "after 2026-06-05" maturity.
- The two 2026-06 workstreams (this doc + the dollar-bar discovery work) are complementary chapters of the same operator-level question — *"does bar structure matter?"* — at the XGB v4.5 layer and the strategy-discovery layer respectively. They do not overlap on files, branches, or tests.

---

## Future consolidation (NOT in scope for this doc)

When the parallel CLI's dollar-bar Phase 4 results land, a single combined "2026-06 bar-structure + XGB closure" results doc may be appropriate. Tracked here as a deferred follow-up; not undertaken in this work.
