# Dollar-Bar Strategy-Discovery — Design

**Date:** 2026-05-29
**Status:** approved (brainstorm), pending implementation plan
**Author:** operator + Claude (session 9950fd63)

## Goal

Re-run the strategy-discovery Phase 2→3→4 pipeline on **information-driven dollar bars** instead of fixed 1h time bars, per the operator's 2026-05-20 priority to move away from time-bar sampling. The just-completed 1h run produced a clean **ABORT** verdict (raw profit positive but selection-bias deflation flipped every portfolio cap negative; see `backend/data/phase4/scorecard.md`). The hypothesis under test: information-driven bars give more IID-like, less heteroskedastic returns, so the deflation-corrected edge is more honestly estimated — and may survive where the 1h run did not.

## Decisions (locked during brainstorm)

1. **Bar source = 1h-aggregated dollar bars.** Aggregate the existing 1h OHLCV (no 1m backfill — there is no 1m data on disk, and a true-dollar-bar build would require a large minute-candle pull for all 50 tokens). Bar edges snap to 1h boundaries; this is a coarser but zero-new-data test of the sampling hypothesis.
2. **Accumulation metric = dollar-volume** (`volume × (H+L+C)/3`), AFML-standard, more stationary across price regimes than raw volume.
3. **Matched bar count.** Threshold = `total_dollar_value / n_1h_rows`, so the dollar-bar count ≈ the 1h-bar count. This holds sample size fixed, isolating the *sampling clock* as the only variable changed vs the failed 1h run.
4. **Horizons = bar-count (activity-anchored).** A hold of h bars = h dollar-bars ahead. Wall-clock duration of a hold now varies with activity. Concurrency, embargo, and labels all operate in bar-index space.
5. **Universe = reuse universe-50** (`docs/superpowers/specs/2026-05-23-universe-50.json`). Its Phase 1 inputs (daily marketcap, supply snapshot) already exist and are reused unchanged.
6. **Separate output namespaces** (`data/phase2_dbar`, `data/phase3_dbar`, `data/phase4_dbar`) so the 1h baseline is preserved for A/B comparison.
7. **Identical Q0 gates + deflation** in Phase 4 (no gate changes) for an apples-to-apples comparison against the 1h abort.

## Architecture / file map

| File | Change |
|---|---|
| `tools/strategy_discovery/info_bars.py` | **NEW** — aggregate a 1h OHLCV frame into dollar bars |
| `tools/strategy_discovery/build_info_bars.py` | **NEW** — CLI: map `aggregate_dollar_bars` over the universe → `data/history/dollar_1h/` |
| `tools/strategy_discovery/profit_split.py` | **MODIFY** — `build_next_eligible` from ms-based → bar-index |
| `tools/strategy_discovery/mine_profiles.py` | **MODIFY** — `build_next_eligible` call site: pass `n` not `ts_ms` |
| `tools/strategy_discovery/_diag_mining_hang.py` | **MODIFY** — same call-site update (untracked diagnostic) |
| `build_phase2.py`, `mine_universe.py`, `build_phase4.py` | **NO code change** — already accept `--history-dir` / `--phase2-dir` / `--phase3-dir` / `--output-dir` |
| `features.py`, `labels.py`, `tokenomic_stamp.py`, `purged_wf.py` | **NO change** — already bar-index based; feature/label column names kept (see Convention below) |

## Component 1 — `info_bars.py`

```python
def aggregate_dollar_bars(df_1h: pd.DataFrame) -> pd.DataFrame
```
Walks time-ordered 1h rows accumulating `dollar_value = volume × (high+low+close)/3` until cumulative value crosses `threshold = total_dollar_value / n_1h_rows`. A 1h row is atomic (never split). The trailing sub-threshold residual is dropped. Emitted columns (history schema so the existing Phase 2 loader reads it unchanged):

- `start` — the **first** merged 1h row's `start` (epoch **seconds**, = bar-open time), matching both the history-parquet and existing `build_dollar_bars` conventions, so `_load_history_parquet` reads it as `ts` unchanged.
- `end` — the **closing** merged row's `start` (epoch seconds), retained for traceability (mirrors `build_dollar_bars` schema).
- `open` — first merged row's open; `high`/`low` — max/min over merged rows; `close` — last merged row's close.
- `volume` — sum; `dollar_value` — sum; `n_1h` — number of 1h rows merged.

Causality note: `ts` is used only by `tokenomic_stamp` (date → T+1 daily-marketcap join, which is look-ahead-safe regardless of open-vs-close choice) and is no longer used by `build_next_eligible` (now bar-index). Features and labels are purely positional. So bar-open `start` introduces no look-ahead.

This reuses the exact accumulation contract of `tools/build_dollar_bars.dollar_bars_from_candles` (a tested pattern), fed 1h rows in place of 1m candles.

## Component 2 — `profit_split.build_next_eligible` refactor

Before (fixed-width, ms-based):
```python
_MS_PER_BAR = 3_600_000
def build_next_eligible(ts_ms, horizon_bars):
    target = ts_ms + horizon_bars * _MS_PER_BAR
    return torch.searchsorted(ts_ms, target, right=False).clamp_max(n)
```
After (bar-index):
```python
def build_next_eligible(n_rows: int, horizon_bars: int) -> torch.Tensor:
    idx = torch.arange(n_rows)
    return (idx + horizon_bars).clamp_max(n_rows)
```
`_MS_PER_BAR` is deleted. **Downstream concurrency math is unchanged** — `walk_and_sum` already treats `next_eligible` as row indices; only its construction changes. Call sites pass `n_rows` instead of the `ts_ms` tensor.

## Data flow (Phase 1 reused, never re-run)

```
data/history/{pid}.parquet           (1h OHLCV, exists)
  → build_info_bars                  → data/history/dollar_1h/{pid}.parquet
  → build_phase2 --history-dir data/history/dollar_1h --output-dir data/phase2_dbar
  → mine_universe --phase2-dir data/phase2_dbar --output-dir data/phase3_dbar --device cuda --seed 42
  → build_phase4 --phase3-dir data/phase3_dbar --phase2-dir data/phase2_dbar --output-dir data/phase4_dbar --seed 42
```
Mining horizons stay {24, 72, 168} (now bar counts). Phase 4 caps {3,4,5} and gates unchanged.

## Conventions and caveats

- **Feature-name convention.** `features.py` columns `ret_24h_sign` / `ret_7d_sign` are kept verbatim (no rename) but on dollar bars mean "24 bars back" / "168 bars back". This matches the same loose convention already adopted for horizons ("h=72" = 72 bars). Avoids rippling a rename into `mine_profiles`'s hardcoded `FEATS` list and the feature test suite for cosmetic gain only.
- **Max-hold fidelity caveat.** `labels._DEFAULT_MAX_HOLD_BARS = 168` mirrors the live 7-day wall-clock max-hold (`cnn_agent._CNN_MAX_HOLD_SECS`). On dollar bars, 168 bars ≠ 7 days. For this activity-anchored research run the cap stays in bars. **Before any live deployment of a dollar-bar rule, the bar-count cap must be reconciled with the live wall-clock exit policy.** Documented here, not addressed in this work.

## Error handling

- Degenerate threshold (zero total dollar-volume for a pid) → skip pid, log, non-fatal.
- A pid yielding < 200 info-bars → caught by `build_phase2`'s existing `< 200 bars` guard.
- Per-pid failures captured in `build_phase2.BuildResult.error`; the batch continues.

## Testing (TDD)

- `tests/tools/strategy_discovery/test_info_bars.py` (**new**): OHLC integrity (open=first / close=last / high=max / low=min over merged rows), volume + dollar_value sums, monotonic non-decreasing `start`, residual drop at series end, matched-count threshold (~`n_1h` bars emitted), degenerate-volume guard.
- `test_profit_split.py` / `test_profit_tree.py` (**modify**): rewrite the 5 `build_next_eligible` cases for bar-index semantics (`next_eligible[i] == min(i + h, n)`); write-failing-first against the old ms impl, then refactor to green.
- Integration smoke: `build_info_bars` → `build_phase2` on 1–2 pids → assert output columns present and `label_h*` finite for non-warmup rows.
- Full suite green before commit.

## Out of scope (YAGNI)

- Volume bars (dollar only).
- Wall-clock horizons (bar-count chosen).
- 1-minute backfill / true-resolution dollar bars.
- Phase 4 gate changes (identical gates for clean A/B).
- Live-deployment exit-policy reconciliation (caveat documented above).

## Success criteria

A `data/phase4_dbar/scorecard.md` produced end-to-end on dollar bars, directly comparable to the 1h `scorecard.md`. The research question — does the deflation-corrected edge survive on information-driven bars — is answered either way (deploy verdict or another documented abort). A second abort is still a successful, informative run.
