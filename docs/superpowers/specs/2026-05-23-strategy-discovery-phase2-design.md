# Strategy Discovery Rebuild — Phase 2 Design Spec

**Date:** 2026-05-23
**Author:** Claude Opus 4.7 (post Phase 1 cutover)
**Status:** Approved (operator 2026-05-23)
**Predecessor brainstorm:** `2026-05-23-strategy-discovery-rebuild-brainstorm.md`
**Phase 1 plan:** `2026-05-23-strategy-discovery-phase1-data-foundation.md` (complete — commits `a37c9c7..95dd4d6`)

---

## Goal

Turn the curated 50-token universe (Phase 1 output) into a per-token **feature matrix + multi-horizon dynamic-exit PnL labels**, ready for Phase 3 mining (custom-criterion decision tree with profit-based split).

Phase 2 owns: trend feature compute, daily tokenomic stamping onto hourly grid, per-horizon label simulation. Phase 2 does NOT own: mining algorithm, scorecard evaluation, deployment artifacts — those live in Phase 3/4.

---

## Inputs (from Phase 1)

| Source | Path | Content |
|---|---|---|
| Curated universe | `docs/superpowers/specs/2026-05-23-universe-50.{md,json}` | 50 pid list with cohort tags (large/mid/high-fdv-mc/low-turnover) |
| 1h OHLCV | `backend/data/history/{pid}.parquet` | hourly bars per token |
| Daily tokenomic | `backend/data/marketcap/{pid}.parquet` | CoinPaprika daily: `(ts_ms, market_cap, volume_24h)` |
| Supply snapshot | `backend/data/supply/snapshot.parquet` | `(pid, circulating, total, max_supply, ingest_ts)` |

---

## Output

**Per-token parquet** at `backend/data/phase2/{pid}.parquet`. One file per pid in the universe (~50 files).

### Row schema (1 row per hourly candidate)

| Family | Columns | Type | Source |
|---|---|---|---|
| Identifiers | `ts`, `pid` | int64 (ms), str | row position |
| **Tokenomic (6)** | `market_cap`, `fdv`, `fdv_over_mc`, `circ_over_total`, `vol_24h`, `vol_over_mc` | float64 | T+1-stamped from CoinPaprika daily + supply snapshot |
| **Trend (7)** | `price_over_ema20`, `price_over_ema50`, `price_over_ema200`, `ret_1h_sign`, `ret_24h_sign`, `ret_7d_sign`, `atr14_pct` | float64 (signs ∈ {-1, 0, +1}) | derived from 1h OHLCV |
| **Labels (5)** | `label_h1`, `label_h4`, `label_h24`, `label_h72`, `label_h168` | float64 (net PnL fraction) | simulated dynamic-exit at each horizon |
| Provenance | `schema_version` | int | constant `1` |

**Cadence:** one row per (token, hour). Warm-up = 200 bars per token (driven by EMA200). First valid candidate at hour ≥ 200 of token history.

---

## Causality Contract

- **Features at row `t`** use only data with timestamp `< t`:
  - Trend features computed from OHLCV bars `[t-200, t-1]` (closed bars only — OHLCV row at `t` is the *forming* bar).
  - Tokenomic features stamped from CoinPaprika daily row for UTC-day `D-1` (T+1 stamping; see below).
- **Labels at row `t`** use future OHLCV bars `[t+1, t+horizon]`.
- No feature ever reads a future bar; no label ever reads a past bar.

### T+1 stamping rule

CoinPaprika returns one row per UTC day timestamped `D 00:00:00 UTC`, but the aggregate (market_cap, volume_24h) is computed from end-of-day data. Per existing `services/coinpaprika_marketcap.py:24-25` convention ("Strict causality: same EOD-UTC stamping as CoinGecko, so the same 1-day lag applies at align time"), the row for UTC-day `D` is only **knowable** starting `D+1 00:00 UTC`.

**Stamping:** candidate at hour `t` (UTC) reads the CoinPaprika row dated `floor((t - 24h) / 1day) × 1day`. Equivalently: snapshot dated UTC-day `D` applies to candidate hours in UTC-day `D+1` and later.

---

## Feature Definitions

### Tokenomic (6)

| # | Column | Formula | Lookup source |
|---|---|---|---|
| 1 | `market_cap` | `cp_row.market_cap` | CoinPaprika daily (T+1) |
| 2 | `fdv` | `price_t × supply.total` | Live close × supply snapshot (T+1 fallback for supply if missing) |
| 3 | `fdv_over_mc` | `fdv / max(market_cap, 1e-12)` | derived |
| 4 | `circ_over_total` | `supply.circulating / max(supply.total, 1e-12)` | supply snapshot |
| 5 | `vol_24h` | `cp_row.volume_24h` | CoinPaprika daily (T+1) |
| 6 | `vol_over_mc` | `vol_24h / max(market_cap, 1e-12)` | derived |

**Supply snapshot semantics:** supply snapshot is a single point-in-time reading per pid (Phase 1 T3). For Phase 2 we treat it as constant across the backfill window — accepting the simplification that supply changes (unlocks, burns) within the window are not modeled in Phase 2. Phase 2.5 backlog: periodic supply re-snapshots for historical accuracy.

### Trend (7) — all from 1h OHLCV closes

| # | Column | Formula |
|---|---|---|
| 7 | `price_over_ema20` | `close_t / EMA(close, span=20).at(t)` |
| 8 | `price_over_ema50` | `close_t / EMA(close, span=50).at(t)` |
| 9 | `price_over_ema200` | `close_t / EMA(close, span=200).at(t)` |
| 10 | `ret_1h_sign` | `numpy.sign(close_t - close_{t-1})` ∈ {-1, 0, +1} |
| 11 | `ret_24h_sign` | `numpy.sign(close_t - close_{t-24})` |
| 12 | `ret_7d_sign` | `numpy.sign(close_t - close_{t-168})` |
| 13 | `atr14_pct` | `Wilder_ATR(high, low, close, period=14).at(t) / close_t` |

EMA uses pandas `ewm(span=N, adjust=False).mean()` — the recursive form `EMA_t = (2/(N+1)) × price_t + (1 - 2/(N+1)) × EMA_{t-1}`.

Wilder ATR: `TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)`, then `ATR_t = (13 × ATR_{t-1} + TR_t) / 14` (Wilder's smoothing).

---

## Label Definitions — Dynamic-Exit Simulation

For each candidate `(pid, t)` at each horizon `h ∈ {1, 4, 24, 72, 168}`:

```
entry_price = close_t
peak        = close_t
horizon_cap = min(h, 168)             # max-hold cap matches deployed system
for s in 1..horizon_cap:
    bar = ohlcv[t+s]
    # 1. Stop-loss check first (priority)
    if bar.low / entry_price - 1 <= -0.08:
        exit_price = entry_price * 0.92    # SL trigger price; conservative
        break

    # 2. Trail-stop check
    peak = max(peak, bar.high)
    atr_pct = max(atr14_pct.at(t+s), 0.06)  # ATR floor matches _CNN_ATR_TRAIL_MIN
    if bar.low / peak - 1 <= -atr_pct:
        exit_price = peak * (1 - atr_pct)    # trail trigger price
        break
else:
    # 3. Horizon reached without trigger
    exit_price = ohlcv[t + horizon_cap].close

label_h{h} = (exit_price / entry_price - 1) - 0.012   # retail round-trip fee
```

### Numeric constants (mirror live system)

| Constant | Value | Source |
|---|---|---|
| Stop-loss threshold | 0.08 (8%) | `_CNN_STOP_LOSS_PCT`, CLAUDE.md invariant #3 |
| Trail-stop floor | 0.06 (6%) | `_CNN_ATR_TRAIL_MIN`, exit_watcher fallback |
| Max-hold cap (bars) | 168 (7 days) | `_CNN_MAX_HOLD_SECS / 3600`, CLAUDE.md invariant #4 |
| ATR window | 14 bars (1h) | Wilder convention; matches `atr14_pct` feature |
| Round-trip fee | 0.012 (1.2% net) | Q0-locked retail tier (0.6%/side) |

### Priority & edge cases

- **Stop-loss precedes trail-stop**: matches `agents/exit_watcher.on_price_tick` and `_check_risk_exits` (cnn_agent.py:1671-1674).
- **Both triggered in same bar**: stop-loss wins.
- **Horizon `h > 168`**: dropped from Phase 2 (operator decision); see backlog task #52.
- **Insufficient forward bars** (`t + horizon_cap` out of history): `label_h{h} = NaN`. Reported in build summary.
- **Conservative SL trigger price**: when `bar.low / entry - 1 ≤ -0.08`, the actual fill price is unknown without intra-bar tick data; we assume fill at the SL boundary (`entry × 0.92`) rather than `bar.low` (which could be a wick). This matches live behavior where WS fires at the first tick crossing the threshold.

---

## Missing-Data Policy

| Field family | Missing data behavior |
|---|---|
| MC, FDV, supply ratios (slow-moving) | Forward-fill last known snapshot indefinitely (no time cap) |
| 24h_volume, vol/MC (activity) | Drop the candidate row entirely; track per-pid in build summary |
| 1h OHLCV gap (`< 200`-bar warm-up) | Skip the token entirely from the universe (log warning) |
| Forward OHLCV gap during label simulation | `NaN` label for the affected horizon(s) only; other horizons unaffected |

Rationale: slow-moving features carry meaningful information even when stale by a few days (supply doesn't change overnight). Activity features (volume) are time-sensitive — stale volume mis-signals the "hot float" state, so candidates must be dropped rather than silently lying.

---

## Module Structure

```
backend/tools/strategy_discovery/
  features.py            (NEW)  Trend features from 1h OHLCV
  tokenomic_stamp.py     (NEW)  T+1 daily-snapshot stamping
  labels.py              (NEW)  Per-horizon dynamic-exit simulation
  build_phase2.py        (NEW)  Orchestrator: universe → per-token compute → parquet
```

### `features.py` — Public API

```python
def add_trend_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame
    """Adds 7 trend columns to df. Requires columns: ts, open, high, low, close.
    Returns df with 7 added columns. First 199 rows have NaN/0 in EMA200-dependent cols."""

def first_valid_index(df: pd.DataFrame, min_warmup: int = 200) -> int
    """Returns the row index where all trend features are valid (post-warmup)."""
```

### `tokenomic_stamp.py` — Public API

```python
def stamp_tokenomic(
    df_hourly: pd.DataFrame,
    df_daily_marketcap: pd.DataFrame,
    supply_row: SupplySnapshot,
    drop_on_missing_volume: bool = True,
) -> pd.DataFrame
    """Adds 6 tokenomic columns to df_hourly via T+1 stamping. Returns df with
    dropped rows where vol_24h is missing (per missing-data policy)."""
```

### `labels.py` — Public API

```python
def simulate_dynamic_exit_labels(
    df: pd.DataFrame,
    horizons: List[int] = [1, 4, 24, 72, 168],
    stop_loss_pct: float = 0.08,
    atr_trail_floor: float = 0.06,
    max_hold_bars: int = 168,
    round_trip_fee: float = 0.012,
) -> pd.DataFrame
    """Adds label_h{h} column per horizon. Requires columns: ts, open, high, low,
    close, atr14_pct. Returns df with len(horizons) added columns."""
```

### `build_phase2.py` — Public API

```python
def build_phase2_for_pid(
    pid: str,
    history_dir: Path,
    marketcap_dir: Path,
    supply_path: Path,
    output_dir: Path,
) -> BuildResult
    """End-to-end build for one pid. Loads inputs, computes features+labels,
    writes output_dir/{pid}.parquet. Returns BuildResult with row counts,
    drop counts, error if any."""

def build_phase2_for_universe(
    universe_path: Path,
    history_dir: Path = Path("backend/data/history"),
    marketcap_dir: Path = Path("backend/data/marketcap"),
    supply_path: Path = Path("backend/data/supply/snapshot.parquet"),
    output_dir: Path = Path("backend/data/phase2"),
) -> List[BuildResult]
    """Iterate universe pids; build each; collect summary. CLI-callable via main()."""

def main(argv: Optional[List[str]] = None) -> int
    """CLI entrypoint."""
```

Module boundary rationale (per `feedback_loose_coupling`): `features.py` knows nothing about tokenomics or labels; `tokenomic_stamp.py` knows nothing about trend features or labels; `labels.py` only knows about OHLCV + ATR. The orchestrator composes them via clean interfaces.

---

## Testing

| Test file | Test | What it pins |
|---|---|---|
| `tests/tools/strategy_discovery/test_features.py` | `test_ema20_matches_pandas_ewm` | EMA formula correctness |
| | `test_ema_warmup_returns_finite_after_n_bars` | Warm-up boundary (200 bars) |
| | `test_atr14_wilder_smoothing_formula` | Wilder ATR formula correctness |
| | `test_ret_sign_at_zero_returns_zero` | Edge case: identical adjacent closes |
| | `test_first_valid_index_at_200` | Warm-up exposed correctly |
| `tests/tools/strategy_discovery/test_tokenomic_stamp.py` | `test_t_plus_1_boundary_uses_yesterday_snapshot` | T+1 causality |
| | `test_forward_fill_supplies_carry_indefinitely` | Slow-feature ffill |
| | `test_missing_volume_drops_candidate_row` | Volume-missing drop policy |
| | `test_fdv_derived_from_price_and_total_supply` | FDV formula |
| `tests/tools/strategy_discovery/test_labels.py` | `test_stop_loss_fires_at_8pct_drawdown` | SL trigger |
| | `test_trail_stop_fires_at_atr_floor` | Trail trigger with 6% floor |
| | `test_max_hold_cap_at_168_for_h168` | Max-hold cap |
| | `test_horizon_reached_without_trigger_uses_close` | Default exit at horizon |
| | `test_stop_loss_priority_over_trail` | SL beats trail in same bar |
| | `test_fee_subtracted_from_label` | 1.2% net |
| | `test_insufficient_forward_bars_returns_nan` | NaN edge case |
| `tests/tools/strategy_discovery/test_build_phase2.py` | `test_build_phase2_for_pid_writes_parquet` | Per-token write |
| | `test_build_phase2_for_universe_iterates_all_pids` | Orchestrator loop |
| | `test_build_result_reports_drop_counts` | Summary fidelity |

**Total new test surface:** 17 tests, mock-only (no live API, no real DB writes — per CLAUDE.md test conventions).

---

## Operator Integration (post-implementation)

After Phase 2 code lands and tests pass:

1. Operator runs `python -m tools.strategy_discovery.build_phase2 --universe docs/superpowers/specs/2026-05-23-universe-50.json`
2. Expected runtime: ~50 tokens × ~2 sec/token (vectorized) = ~2 min total
3. Verify summary: row counts per pid, drop counts per (pid, reason), NaN-label counts per horizon
4. Commit the `data/phase2/{pid}.parquet` artifacts (~50 files, ~10-20 MB total)
5. Phase 2 complete → kick off Phase 3 brainstorm (mining algorithm)

---

## What Phase 2 is NOT

- Not the mining algorithm (Phase 3 — custom-criterion decision tree with profit-based split)
- Not the scorecard / Q0-gate evaluator (Phase 4)
- Not a backfill mechanism for CoinPaprika data (that's Phase 1 Task 6 — orchestrator handles its own backfill)
- Not feature-engineering exploration (the 13 features are locked in the brainstorm; new features require an explicit brainstorm round)

---

## Backlog (deferred)

| ID | Item | Trigger to revisit |
|---|---|---|
| #52 | Add 336h horizon | After max-hold redesign (#26) raises the deployed cap |
| #18 | Sample uniqueness weighting | Phase 3 (mining time, not labeling) |
| #20 | CUSUM event-driven candidate filter | Future upgrade after Phase 4 baseline lands |
| — | Periodic supply re-snapshot for historical accuracy | If Phase 4 results show supply-staleness materially affects profile metrics |

---

## See also

- `2026-05-23-strategy-discovery-rebuild-brainstorm.md` — Phase 1–4 spec (Q0–Q6 decisions)
- `2026-05-23-strategy-discovery-phase1-data-foundation.md` — Phase 1 plan + completed task log
- `2026-05-23-ws-exit-checker-design.md` — defines the deployed dynamic-exit logic that Phase 2 labels mirror
- `coinbase_trader_architecture.md` (memory) — current backend architecture + strategy_discovery package overview
