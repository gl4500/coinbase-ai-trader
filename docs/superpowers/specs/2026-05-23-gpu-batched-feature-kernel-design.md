# GPU-Batched XGB v4.5 Feature Kernel — Design

**Date:** 2026-05-23
**Status:** Draft (delegated approval via `/loop refactor until complete`)
**Author:** Claude (Opus 4.7), invoked by gl4500

---

## Problem

`backend/tools/train_xgb_v4_5.py` extracts features via a Python for-loop calling
`extract_v4_5(tier_slices)` once per sample bar (~7570 bars/PID × 222 PIDs). Empirical
rate: ~6 min/PID = **~22 hr per horizon** of wall time on this RTX 2060 box, even with
the trainer pinned to a full CPU core. Horizon-sweep workflow needs three trainings
(h24/h72/h168) → ~3 days of compute. Unworkable for routine retraining.

Bottleneck is **Python loop overhead and per-bar numpy launches**, not the XGBoost fit
itself (the fit is ~10-30 min CPU, ~1-3 min GPU). GPU only helps if we
**batch all sample bars per PID into one vectorized tensor op**.

## Goal

Cut feature-extraction wall time per horizon from ~22 hr to ≤30 min by replacing
the per-bar Python loop with batched PyTorch tensor ops that run on either CPU or GPU,
selected by CLI flag.

**Success criteria:**
1. `--device cuda` extracts features for one 222-PID sweep in ≤30 min on RTX 2060.
2. `--device cpu` (default) preserves bit-exact behavior of today's trainer — the
   existing per-bar code path is **untouched** when `--device cpu` is selected.
3. Numerical parity between batched-CPU and per-bar-CPU paths: `max|batch - per_bar| < 1e-4`
   on a real BTC-USD slice. (Tighter tolerance than CPU-vs-GPU because both run on numpy.)
4. Numerical parity between batched-GPU and batched-CPU paths: `max|gpu - cpu| < 1e-4`.
   This is the only path that crosses devices; expected drift comes from
   reduction-order in floating-point sums.

## Non-Goals

- Porting `agents/xgb_signal.py` inference to GPU. Inference is single-bar per tick;
  Python overhead is negligible and GPU launch latency would be slower. **Inference
  stays on numpy via the existing `extract_v4_5` per-bar API.**
- Replacing the current per-bar code path or its tests. CPU default stays bit-exact.
- Multi-PID GPU batching (would complicate VRAM management for marginal gain — single-PID
  tensors are bounded at ~200MB even at the macro tier).
- GPU-accelerated v4 (2-class) trainer. v4.5 is the only horizon-sweep workflow.

---

## Architecture

### Module layout

```
backend/tools/
├── xgb_v4_5_features.py          (UNCHANGED — per-bar extract_v4_5, used at inference + CPU training)
├── xgb_v4_5_features_batch.py    (NEW — batched all-bars-per-PID, used at GPU training)
└── train_xgb_v4_5.py             (MODIFIED — adds --device flag + dispatches batched path)
```

### Dispatch in `train_xgb_v4_5.py`

```python
# CLI: --device {cpu,cuda} default=cpu
if args.device == "cpu":
    X, y, ts = _build_samples_for_pid(candles, ...)   # existing per-bar path
else:
    X, y, ts = batch_build_samples_for_pid(           # new batched path
        candles, ..., device="cuda",
    )
# Returns identical (np.float64 (N,210), int8 (N,), int64 (N,)) shape contract.
```

The batched module exports a single function with the same return contract as the
existing per-bar `_build_samples_for_pid`, so the trainer's downstream code (pooling,
split, xgb.train) is unchanged.

### Why "batch" not "gpu" in the module name

The new module's distinguishing feature is *algorithmic shape* (one-shot all bars
per PID via `tensor.unfold` strided windows), not the device. The same vectorized
algorithm runs on CPU when `device="cpu"` is passed — useful as a numerical reference
that controls one variable when debugging GPU drift. `device` is a runtime arg, not
a module-level decision.

---

## Public API of `xgb_v4_5_features_batch.py`

### Constants (re-exported from `xgb_v4_5_features.py` to avoid duplication)

```python
from .xgb_v4_5_features import (
    N_CHANNELS_V45, N_TIERS_V45, N_STATS_V45, N_FEATURES_V45,
    TIER_WINDOWS_V45, BB_PERIOD, BB_MULT,
)
```

### Top-level function

```python
def batch_build_samples_for_pid(
    candles: List[Dict[str, float]],
    *,
    forward_hours: int,
    label_thresh: float,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched-tensor equivalent of train_xgb_v4_5._build_samples_for_pid.

    For one PID, vectorize feature extraction across all valid sample bars.

    Args:
        candles: chronologically-sorted OHLCV list (same format as per-bar path).
        forward_hours: triple-barrier horizon (e.g. 72).
        label_thresh: triple-barrier threshold (e.g. 0.03).
        device: "cpu" or "cuda". On "cuda", tensors are created on the GPU and
            results are moved back to host as numpy arrays.

    Returns:
        (X, y, ts) — same shape and dtype as _build_samples_for_pid:
            X:  (N, 210) float64
            y:  (N,)     int8 (0=DOWN, 1=NEUTRAL, 2=UP)
            ts: (N,)     int64 epoch seconds

    Empty/insufficient input -> three empty arrays.
    """
```

### Lower-level helpers (private, but exposed for unit testing)

```python
def _build_channel_matrix(
    candles_array: torch.Tensor,    # (n_candles, 5) OHLCV
    device: str,
) -> torch.Tensor:                  # (n_candles, 7)
    """Stack OHLCV + computed (bb_position, bb_width) into 7-channel tensor."""

def _rolling_bb(
    closes: torch.Tensor,           # (n,) float64
    period: int = BB_PERIOD,
    mult: float = BB_MULT,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (bb_position, bb_width), each (n,)
    """Rolling Bollinger band position + width with [0, 1] clamp on position."""

def _strided_windows(
    channels: torch.Tensor,         # (n_candles, 7)
    *,
    window: int,                    # tier_window + bb_prefix
    n_samples: int,                 # number of sample bars
    sample_offset: int,             # first valid sample bar index
) -> torch.Tensor:                  # (n_samples, window, 7) view (no copy)
    """tensor.unfold-based strided windows for one tier across all sample bars."""

def _batch_stats(
    windows: torch.Tensor,          # (n_samples, window, 7)
) -> torch.Tensor:                  # (n_samples, 7, 10) stats
    """Compute all 10 stats in one batched call. Order matches _STAT_NAMES_V45."""

def _batch_triple_barrier(
    closes: torch.Tensor,           # (n_candles,)
    sample_indices: torch.Tensor,   # (n_samples,) int64
    forward_hours: int,
    label_thresh: float,
) -> torch.Tensor:                  # (n_samples,) int8 — DOWN=0/NEUTRAL=1/UP=2
    """Vectorized triple-barrier labeling. Returns -1 for samples where the
    forward window extends past end of series; caller must filter."""
```

---

## Tensor layout & algorithm

### Step 1 — Load PID candles into channel matrix (one-time per PID)

```python
candles_arr = torch.tensor(           # (n_candles, 5) float64
    [[c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in candles],
    dtype=torch.float64, device=device,
)
closes = candles_arr[:, 3]
bb_pos, bb_wid = _rolling_bb(closes)              # each (n_candles,)
channels = torch.cat([candles_arr, bb_pos[:, None], bb_wid[:, None]], dim=1)
# channels: (n_candles, 7) float64
```

### Step 2 — For each tier, build (n_samples, window, 7) strided view

```python
tier_window = TIER_WINDOWS_V45["micro"]       # 60
window_len = tier_window + BB_PERIOD          # 80 (matches per-bar tier_slices length)

# Valid sample bars: i in [macro+bb_prefix .. n_candles] s.t. triple-barrier label exists
sample_offset = TIER_WINDOWS_V45["macro"] + BB_PERIOD    # 356
n_samples = (n_candles - sample_offset) - forward_hours  # subtract right-edge labels

# Strided unfold: channels[i-window_len:i] for i in [sample_offset .. sample_offset+n_samples)
unfold = channels.unfold(0, window_len, 1)         # (n_candles - window_len + 1, 7, window_len)
unfold = unfold.permute(0, 2, 1)                   # (n_useable, window_len, 7)
# Extract just the windows for our sample bars:
windows = unfold[sample_offset - window_len + 1 : sample_offset - window_len + 1 + n_samples]
# windows: (n_samples, window_len, 7)  — NO MEMORY COPY (strided view)
```

### Step 3 — Compute 10 stats across window dim (vectorized)

```python
last     = windows[:, -1, :]                              # (n_samples, 7)
mean     = windows.mean(dim=1)                            # (n_samples, 7)
std      = windows.std(dim=1, correction=0)               # ddof=0 to match numpy default
slope    = _vectorized_ols_slope(windows)                 # (n_samples, 7)
mn       = windows.min(dim=1).values                      # (n_samples, 7)
mx       = windows.max(dim=1).values                      # (n_samples, 7)
pct_rank = _vectorized_pct_rank(windows, last)            # (n_samples, 7)
dlt5     = windows[:, -1, :] - windows[:, -1 - 5, :]      # (n_samples, 7)
dlt10    = windows[:, -1, :] - windows[:, -1 - 10, :]
dlt30    = windows[:, -1, :] - windows[:, -1 - 30, :]

tier_stats = torch.stack([last, mean, std, slope, mn, mx, pct_rank,
                          dlt5, dlt10, dlt30], dim=-1)   # (n_samples, 7, 10)
```

### Step 4 — Concat tiers + flatten to feature matrix

```python
# After computing per-tier stats for micro/meso/macro:
all_stats = torch.stack([micro, meso, macro], dim=2)     # (n_samples, 7, 3, 10)
X = all_stats.reshape(n_samples, N_FEATURES_V45)         # (n_samples, 210)
# Layout: channel-major -> tier-major -> stat-major, matching feature_names_v4_5().
```

### Step 5 — Triple-barrier labels (vectorized)

```python
# For each sample bar i, scan closes[i+1 .. i+forward_hours] looking for first
# UP/DOWN barrier crossing. Use a cumulative-max / cumulative-min sliding approach
# that's pure tensor — no Python loop over samples.
y = _batch_triple_barrier(closes, sample_indices, forward_hours, label_thresh)
# y: (n_samples,) int8

# Drop rows where triple-barrier returned -1 (no label possible at the right edge):
valid = y >= 0
X, y, ts = X[valid], y[valid], ts[valid]
```

### Step 6 — Move to CPU + return numpy

```python
return X.cpu().numpy().astype(np.float64), \
       y.cpu().numpy().astype(np.int8), \
       ts.cpu().numpy().astype(np.int64)
```

---

## Trainer integration

Edit `train_xgb_v4_5.py`:

1. **Add `--device` argparse:**
   ```python
   p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                  help="Feature extraction backend. cpu (default) uses the per-bar "
                       "numpy loop. cuda uses the batched PyTorch kernel.")
   ```

2. **Dispatch in main loop:** when iterating PIDs, branch on `args.device`:
   ```python
   if args.device == "cpu":
       X, y, ts = _build_samples_for_pid(candles, ...)
   else:
       from tools.xgb_v4_5_features_batch import batch_build_samples_for_pid
       X, y, ts = batch_build_samples_for_pid(candles, ..., device="cuda")
   ```

3. **xgb.train device:** when `--device cuda`, also pass `device="cuda"` to XGB params:
   ```python
   if args.device == "cuda":
       params["device"] = "cuda"  # XGBoost 2.x API
   ```

4. **Determinism:** if `--device cuda`, call once at startup:
   ```python
   torch.use_deterministic_algorithms(True, warn_only=True)
   ```
   Stable reduction order. `warn_only=True` lets non-critical ops fall back to
   non-deterministic if no deterministic kernel exists (won't fail-hard on rare ops).

---

## Testing strategy

### Unit tests (`tests/test_xgb_v4_5_features_batch.py`)

One test per stat × parity vs per-bar reference on a synthetic input:

1. `test_batch_last_matches_per_bar` — synthetic 100-bar OHLCV, check `last` slot for one tier matches `extract_v4_5` output.
2. `test_batch_mean_matches_per_bar` — same, for `mean`.
3. `test_batch_std_matches_per_bar` — same, for `std`. (Catches ddof=0 vs ddof=1 trap.)
4. `test_batch_slope_matches_per_bar` — same, for `slope`. (Catches OLS reduction-order drift.)
5. `test_batch_min_max_matches_per_bar` — same, for `min` and `max`.
6. `test_batch_pct_rank_matches_per_bar` — same, for `pct_rank`. (Catches comparison-tie behavior.)
7. `test_batch_delta_matches_per_bar` — `dlt5/10/30`.
8. `test_batch_bb_position_matches_per_bar` — Bollinger position + clamp.
9. `test_batch_bb_width_matches_per_bar` — Bollinger width.

All run on CPU device (no CUDA required). Tolerance: `1e-4` max abs diff.

### Integration test (`tests/test_xgb_v4_5_features_batch.py`)

10. `test_full_pipeline_parity_cpu` — runs `batch_build_samples_for_pid(device="cpu")`
    against the current per-bar `_build_samples_for_pid` on a slice of real BTC-USD
    parquet (or synthetic 1000-bar series if parquet unavailable). Asserts
    `max|batched - per_bar| < 1e-4` on the (N, 210) feature matrix and exact equality
    on labels + timestamps.

### CUDA-conditional test

11. `test_full_pipeline_parity_cuda` — gated on `@pytest.mark.skipif(not torch.cuda.is_available())`.
    Same as #10 but with `device="cuda"`. Same tolerance. Skipped on CI/dev boxes
    without a GPU; runs on operator's RTX 2060.

### Triple-barrier vectorization test

12. `test_batch_triple_barrier_matches_per_bar` — synthetic closes with known
    UP/DOWN/NEUTRAL labels at hand-picked indices; assert vectorized labels match
    the existing `_triple_barrier_label_3class` exactly. Integer equality (no drift).

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ddof=0 vs ddof=1 std drift | High | `correction=0` arg in `torch.std`; covered by `test_batch_std_matches_per_bar`. |
| Slope OLS reduction order drift | Medium | Tolerance test (1e-4) accepts ~1e-5 drift; `test_batch_slope_matches_per_bar` confirms it's within bound. |
| pct_rank tie-handling differs | Low | Test #6 uses synthetic with explicit ties; assert `(below + 0.5*equal)/n` formula matches. |
| GPU OOM on macro-tier window | Low | Per-PID VRAM bounded at ~200MB (8000 bars × 356 window × 7 ch × 8 bytes); RTX 2060 has 6GB. |
| `tensor.unfold` returns non-contiguous view; downstream ops fail | Medium | Add `.contiguous()` if needed; covered by integration test. |
| Trainer's `tier_slices` length includes BB prefix but treats whole slice as stats input — comment in trainer claims prefix is excluded | High (behavioral) | Per the **actual code** in `extract_v4_5`, stats are computed over the full slice (prefix + tier_window). Batched path **must replicate this** (i.e., window length = `tier_window + BB_PERIOD`). |
| Numerical parity test #10 fails on a stat we didn't anticipate | Medium | Per-stat tests #1-9 localize the failure; tolerance budget can be relaxed to 1e-3 if drift is bounded but larger than expected. |
| Model trained on GPU features behaves materially differently on shadow week | Medium | Confine GPU re-train to a single horizon initially; compare horizon_compare report against the prior CPU-trained baseline. Don't promote until shadow-week parity holds. |

---

## Out-of-scope (deferred)

- Multiprocess feature extraction on CPU (option 4 from the brainstorm). Skipped:
  this design already covers the same speedup ceiling via GPU, and we get the
  flexibility of CPU fallback via `device="cpu"` to the same batched code path.
- Per-PID parallelism on GPU (multiple PIDs in flight simultaneously). Skipped:
  ~30 min wall time is already acceptable; complexity isn't justified.
- Inference-side GPU. Skipped: per-tick single-bar inference doesn't benefit.

---

## Files-only constraint (until 8001 pause)

Per `feedback_no_pytest_during_trading.md`: live trading on 8001 is currently active.
The implementation phase will write code and tests but **will not run pytest, will
not commit, and will not push** until the operator pauses 8001 at a natural break.

Implementation order:
1. Write tests (red phase — they'll be untested-against-running-impl until pause).
2. Write `xgb_v4_5_features_batch.py`.
3. Plumb `--device` flag through `train_xgb_v4_5.py`.
4. **Halt here. Wait for 8001 pause.**
5. Operator pauses → run full pytest, verify green, commit atomically, push.
6. Operator runs sweep: `python -m tools.train_xgb_v4_5 --pids ... --device cuda --forward-hours 72 --label-thresh 0.03`.
7. After all 3 horizons retrained on GPU: run `v4_5_horizon_compare`, inspect HTML report,
   verify AUC drift < 0.005 vs prior CPU run.
8. If parity holds: GPU path is the new default for routine retraining.

---

## See also

- [[xgb_feature_optimization_findings]] — prior v4 feature work, decides 210-col layout.
- [[xgb_post_scorecard_roadmap]] — broader v4.5 roadmap. GPU port enables routine retraining for the bar-structure probes (path 1 of 8).
- [[feedback_no_pytest_during_trading]] — pytest/commit deferred until pause window.
- [[feedback_tdd_workflow]] — tests before code.
- [[polymarket_app_python_interpreter]] — `.venv/Scripts/python.exe` is the interpreter.
- [[coinbase_trader_schema]] — feature names + DB landmarks for v4.5 telemetry.
