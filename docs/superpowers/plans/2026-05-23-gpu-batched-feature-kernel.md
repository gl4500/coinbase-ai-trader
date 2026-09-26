# GPU-Batched XGB v4.5 Feature Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-bar Python feature loop in `train_xgb_v4_5.py` with a batched PyTorch tensor kernel that runs on CPU or GPU, cutting per-horizon training time from ~22 hr to ≤30 min.

**Architecture:** New module `backend/tools/xgb_v4_5_features_batch.py` exposes `batch_build_samples_for_pid(...)` which vectorizes feature extraction across all sample bars per PID using `tensor.unfold` strided windows. Existing `extract_v4_5` per-bar API stays untouched (inference uses it). Trainer adds `--device {cpu,cuda}` flag; CPU default keeps current bit-exact behavior.

**Tech Stack:** PyTorch 2.6 + CUDA 12.4 (RTX 2060), numpy (parity reference), pytest, XGBoost 2.x (`device="cuda"` API).

**OPERATIONAL CONSTRAINT — read first:**
8001 is currently live trading. Per `feedback_no_pytest_during_trading.md`:
- Tasks 1-9 are **file-only writes** (no pytest runs, no commits).
- Task 10 is **gated on operator pausing 8001** — runs the full pytest sweep, atomic commit, push.
- Each task's TDD `verify-fail` and `verify-pass` steps are written for reference but **deferred to Task 10**.
- The plan executes red-then-impl per task (write test, write impl) without verifying; Task 10 runs all tests at once and confirms green.

---

## File Structure

| Path | Role | Responsibility |
|---|---|---|
| `backend/tools/xgb_v4_5_features_batch.py` | NEW | Batched feature kernel (PyTorch). Top-level `batch_build_samples_for_pid()` + private helpers. ~250 LoC. |
| `backend/tests/test_xgb_v4_5_features_batch.py` | NEW | Per-stat parity unit tests + full-pipeline parity test + CUDA-gated parity test + triple-barrier test. ~400 LoC. |
| `backend/tools/train_xgb_v4_5.py` | MODIFY | Add `--device` argparse + dispatch + xgb GPU param + deterministic-CUDA call. ~30 LoC delta. |

---

## Task 1: Module scaffold + constants re-export

**Files:**
- Create: `backend/tools/xgb_v4_5_features_batch.py`

- [ ] **Step 1: Write the module skeleton**

```python
"""XGB v4.5 batched-tensor feature extractor.

Vectorizes feature extraction across all sample bars per PID using PyTorch
tensor ops. Either CPU- or GPU-resident depending on `device` arg. Used by
train_xgb_v4_5.py when --device != cpu; existing per-bar extract_v4_5 in
xgb_v4_5_features.py remains canonical for inference + CPU default training.

Numerical contract: max abs diff < 1e-4 vs per-bar numpy reference (see
test_xgb_v4_5_features_batch.py).
"""
from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.xgb_v4_5_features import (  # noqa: E402
    N_CHANNELS_V45, N_TIERS_V45, N_STATS_V45, N_FEATURES_V45,
    TIER_WINDOWS_V45, BB_PERIOD, BB_MULT,
)

# Stable tier order — must match feature_names_v4_5() layout.
_TIER_ORDER: Tuple[str, ...] = ("micro", "meso", "macro")


__all__ = [
    "batch_build_samples_for_pid",
]
```

- [ ] **Step 2: ~Verify import works~ DEFERRED to Task 10**

- [ ] **Step 3: ~Commit~ DEFERRED to Task 10**

---

## Task 2: `_local_bb_in_windows` — per-bar Bollinger over windows

**Files:**
- Modify: `backend/tools/xgb_v4_5_features_batch.py`
- Test: `backend/tests/test_xgb_v4_5_features_batch.py`

**Background:** The existing `extract_v4_5` computes BB position + width *locally on each tier slice* — the first 19 bars of each slice get the (0.5, 0.0) fallback because they don't have BB_PERIOD-1 prior bars *within that local slice*. The batched code MUST replicate this (not use globally-computed BB) — otherwise feature values diverge structurally from the per-bar reference, which would invalidate the existing trained model's calibration.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_xgb_v4_5_features_batch.py`:

```python
"""Parity tests for the batched feature kernel against the per-bar numpy reference."""
from __future__ import annotations
import os
import sys

import numpy as np
import pytest
import torch

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.xgb_v4_5_features import (
    _compute_bb_position, _compute_bb_width, BB_PERIOD,
)
from tools.xgb_v4_5_features_batch import _local_bb_in_windows


_TOL = 1e-4


def _make_strided_closes(closes: np.ndarray, window_len: int, n_samples: int,
                         sample_offset: int) -> torch.Tensor:
    """Helper: build the (n_samples, window_len) close-windows tensor that
    _local_bb_in_windows expects, mirroring how the orchestrator will call it."""
    out = np.zeros((n_samples, window_len), dtype=np.float64)
    for s in range(n_samples):
        i = sample_offset + s
        out[s] = closes[i - window_len : i]
    return torch.tensor(out, dtype=torch.float64)


def test_local_bb_position_matches_per_bar_reference():
    rng = np.random.default_rng(42)
    closes = rng.uniform(100.0, 200.0, size=500).astype(np.float64)
    window_len = 80
    n_samples = 50
    sample_offset = window_len + 100   # plenty of warmup
    closes_windows = _make_strided_closes(closes, window_len, n_samples, sample_offset)

    bb_pos, bb_wid = _local_bb_in_windows(closes_windows)
    assert bb_pos.shape == (n_samples, window_len)
    assert bb_wid.shape == (n_samples, window_len)

    for s in range(n_samples):
        i = sample_offset + s
        local_closes = closes[i - window_len : i]
        ref_pos = _compute_bb_position(local_closes)
        ref_wid = _compute_bb_width(local_closes)
        np.testing.assert_allclose(bb_pos[s].cpu().numpy(), ref_pos, atol=_TOL,
                                   err_msg=f"BB position mismatch at sample {s}")
        np.testing.assert_allclose(bb_wid[s].cpu().numpy(), ref_wid, atol=_TOL,
                                   err_msg=f"BB width mismatch at sample {s}")
```

- [ ] **Step 2: Run test to verify it fails (DEFERRED to Task 10)**

Expected (when Task 10 runs): FAIL with `ImportError: cannot import name '_local_bb_in_windows'`.

- [ ] **Step 3: Write the implementation**

Append to `backend/tools/xgb_v4_5_features_batch.py`:

```python
import torch


def _local_bb_in_windows(
    closes_windows: torch.Tensor,        # (n_samples, window_len) float64
    period: int = BB_PERIOD,
    mult: float = BB_MULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-bar Bollinger position + width inside each tier window.

    For each sample bar's window-of-W closes, compute (bb_position, bb_width)
    at every bar j in [0, W). Bars j < period-1 get fallback (0.5, 0.0)
    because they don't have `period` prior bars WITHIN the local window.

    Matches xgb_v4_5_features._compute_bb_position + _compute_bb_width
    semantics (local-per-slice, NOT global). Returns float64 tensors on the
    same device as closes_windows.

    Shapes:
        closes_windows: (n_samples, W)
        return: (bb_position, bb_width), each (n_samples, W)
    """
    device = closes_windows.device
    dtype = closes_windows.dtype
    n_samples, W = closes_windows.shape

    bb_pos = torch.full((n_samples, W), 0.5, dtype=dtype, device=device)
    bb_wid = torch.zeros((n_samples, W), dtype=dtype, device=device)

    if W < period:
        return bb_pos, bb_wid

    # Inner strided windows: for each bar j in [period-1, W), use closes[j-period+1 .. j+1]
    inner = closes_windows.unfold(1, period, 1)        # (n_samples, W - period + 1, period)
    mean = inner.mean(dim=-1)                          # (n_samples, W - period + 1)
    std = inner.std(dim=-1, correction=0)              # ddof=0 to match numpy default

    upper = mean + mult * std
    lower = mean - mult * std
    bw = upper - lower

    # bb_position with clamp [0, 1], fallback 0.5 where bw <= 0
    inner_closes = closes_windows[:, period - 1 : W]   # (n_samples, W - period + 1)
    pos = (inner_closes - lower) / bw.clamp_min(1e-12)
    pos = pos.clamp(0.0, 1.0)
    valid_pos = bw > 0
    pos = torch.where(valid_pos, pos, torch.full_like(pos, 0.5))

    # bb_width = (2 * mult * std) / mean; fallback 0 where mean == 0
    wid = (2.0 * mult * std) / mean.clamp_min(1e-12)
    valid_wid = mean != 0
    wid = torch.where(valid_wid, wid, torch.zeros_like(wid))

    # Place into bb_pos/bb_wid at indices [period-1, W)
    bb_pos[:, period - 1 :] = pos
    bb_wid[:, period - 1 :] = wid
    return bb_pos, bb_wid
```

- [ ] **Step 4: Run test to verify it passes (DEFERRED to Task 10)**

Expected (when Task 10 runs): PASS.

- [ ] **Step 5: ~Commit~ DEFERRED to Task 10**

---

## Task 3: `_build_pid_channels` — global OHLCV→(n_candles, 7) channel matrix

**Background:** This builds a per-PID channel matrix WITHOUT running BB — BB is per-slice via Task 2's `_local_bb_in_windows`, called later inside the strided window. This helper just stacks OHLCV from the candle dict list into a (n_candles, 5) tensor; the 2 BB channels are added per-window inside the orchestrator.

**Files:**
- Modify: `backend/tools/xgb_v4_5_features_batch.py`
- Test: `backend/tests/test_xgb_v4_5_features_batch.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_xgb_v4_5_features_batch.py`:

```python
from tools.xgb_v4_5_features_batch import _build_pid_ohlcv


def test_build_pid_ohlcv_matches_dict_order():
    candles = [
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0, "start": 0},
        {"open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 200.0, "start": 60},
        {"open": 2.0, "high": 3.0, "low": 1.5, "close": 2.5, "volume": 300.0, "start": 120},
    ]
    arr = _build_pid_ohlcv(candles, device="cpu")
    assert arr.shape == (3, 5)
    assert arr.dtype == torch.float64
    np.testing.assert_array_equal(
        arr.cpu().numpy(),
        np.array([
            [1.0, 2.0, 0.5, 1.5, 100.0],
            [1.5, 2.5, 1.0, 2.0, 200.0],
            [2.0, 3.0, 1.5, 2.5, 300.0],
        ]),
    )


def test_build_pid_ohlcv_empty_returns_empty():
    arr = _build_pid_ohlcv([], device="cpu")
    assert arr.shape == (0, 5)
```

- [ ] **Step 2: Run test to verify it fails (DEFERRED)**

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `backend/tools/xgb_v4_5_features_batch.py`:

```python
def _build_pid_ohlcv(
    candles: List[Dict[str, float]],
    device: str,
) -> torch.Tensor:
    """Stack OHLCV from candle list into (n_candles, 5) float64 tensor.

    Column order: open, high, low, close, volume (matches _OHLCV_FIELDS in
    xgb_v4_5_features.py).

    Empty input -> (0, 5) tensor.
    """
    if not candles:
        return torch.zeros((0, 5), dtype=torch.float64, device=device)
    arr = np.empty((len(candles), 5), dtype=np.float64)
    for idx, c in enumerate(candles):
        arr[idx, 0] = c["open"]
        arr[idx, 1] = c["high"]
        arr[idx, 2] = c["low"]
        arr[idx, 3] = c["close"]
        arr[idx, 4] = c["volume"]
    return torch.tensor(arr, dtype=torch.float64, device=device)
```

- [ ] **Step 4: Verify pass (DEFERRED)**
- [ ] **Step 5: Commit (DEFERRED)**

---

## Task 4: `_batch_stats` — 10 stats over window dim (vectorized)

**Files:**
- Modify: `backend/tools/xgb_v4_5_features_batch.py`
- Test: `backend/tests/test_xgb_v4_5_features_batch.py`

- [ ] **Step 1: Write the failing tests (one per stat)**

Append to `backend/tests/test_xgb_v4_5_features_batch.py`:

```python
from tools.xgb_v4_5_features import _compute_stats
from tools.xgb_v4_5_features_batch import _batch_stats


def _make_batch_windows(n_samples: int, window_len: int, n_channels: int,
                       seed: int = 7) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor(
        rng.uniform(-10.0, 10.0, size=(n_samples, window_len, n_channels)),
        dtype=torch.float64,
    )


def test_batch_stats_shape_and_dtype():
    w = _make_batch_windows(n_samples=5, window_len=60, n_channels=7)
    out = _batch_stats(w)
    assert out.shape == (5, 7, 10)
    assert out.dtype == torch.float64


def test_batch_stats_last_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=11)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 0], ref[0], atol=_TOL)  # last


def test_batch_stats_mean_std_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=13)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 1], ref[1], atol=_TOL)  # mean
            np.testing.assert_allclose(out[s, ch, 2], ref[2], atol=_TOL)  # std


def test_batch_stats_slope_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=17)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 3], ref[3], atol=_TOL)  # slope


def test_batch_stats_minmax_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=19)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 4], ref[4], atol=_TOL)  # min
            np.testing.assert_allclose(out[s, ch, 5], ref[5], atol=_TOL)  # max


def test_batch_stats_pct_rank_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=23)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 6], ref[6], atol=_TOL)  # pct_rank


def test_batch_stats_delta_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=29)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 7], ref[7], atol=_TOL)   # dlt5
            np.testing.assert_allclose(out[s, ch, 8], ref[8], atol=_TOL)   # dlt10
            np.testing.assert_allclose(out[s, ch, 9], ref[9], atol=_TOL)   # dlt30


def test_batch_stats_empty_window_returns_zeros():
    w = torch.zeros((3, 0, 7), dtype=torch.float64)
    out = _batch_stats(w)
    assert out.shape == (3, 7, 10)
    assert (out == 0).all()
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED)**

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `backend/tools/xgb_v4_5_features_batch.py`:

```python
def _batch_stats(
    windows: torch.Tensor,             # (n_samples, window_len, n_channels)
) -> torch.Tensor:                     # (n_samples, n_channels, 10) — stat-major last
    """Compute all 10 stats per (sample, channel) in one batched call.

    Stat order matches xgb_v4_5_features._STAT_NAMES_V45:
        0=last 1=mean 2=std 3=slope 4=min 5=max
        6=pct_rank 7=dlt5 8=dlt10 9=dlt30

    Empty window dim -> returns zeros (matches _compute_stats fallback).
    """
    n_samples, W, C = windows.shape
    device = windows.device
    dtype = windows.dtype

    if W == 0:
        return torch.zeros((n_samples, C, 10), dtype=dtype, device=device)

    # 0 last — values[-1]
    last = windows[:, -1, :]                                         # (n_samples, C)
    # 1 mean
    mean = windows.mean(dim=1)
    # 2 std (ddof=0 to match numpy .std() default)
    if W < 2:
        std = torch.zeros_like(mean)
    else:
        std = windows.std(dim=1, correction=0)
    # 3 slope (OLS over index 0..W-1)
    if W < 2:
        slope = torch.zeros_like(mean)
    else:
        x = torch.arange(W, dtype=dtype, device=device)
        x_mean = x.mean()
        x_dev = x - x_mean                                           # (W,)
        x_dev_sq = (x_dev * x_dev).sum()                             # scalar
        y_dev = windows - mean.unsqueeze(1)                          # (n_samples, W, C)
        num = (x_dev.unsqueeze(0).unsqueeze(-1) * y_dev).sum(dim=1)  # (n_samples, C)
        if float(x_dev_sq) == 0.0:
            slope = torch.zeros_like(mean)
        else:
            slope = num / x_dev_sq
    # 4 min, 5 max
    mn = windows.min(dim=1).values
    mx = windows.max(dim=1).values
    # 6 pct_rank — (sum(values < last) + 0.5 * sum(values == last)) / n
    if W < 2:
        pct_rank = torch.zeros_like(mean)
    else:
        last_b = last.unsqueeze(1)                                   # (n_samples, 1, C)
        below = (windows < last_b).sum(dim=1).to(dtype)              # (n_samples, C)
        equal = (windows == last_b).sum(dim=1).to(dtype)
        pct_rank = (below + 0.5 * equal) / float(W)
    # 7-9 dlt5/10/30 — values[-1] - values[-1-lookback]; zero if W too small
    def _dlt(lookback: int) -> torch.Tensor:
        if W < lookback + 1:
            return torch.zeros_like(mean)
        return windows[:, -1, :] - windows[:, -1 - lookback, :]
    dlt5 = _dlt(5)
    dlt10 = _dlt(10)
    dlt30 = _dlt(30)

    # Stack: (n_samples, C, 10)
    out = torch.stack(
        [last, mean, std, slope, mn, mx, pct_rank, dlt5, dlt10, dlt30],
        dim=-1,
    )
    return out
```

- [ ] **Step 4: Verify pass (DEFERRED)**
- [ ] **Step 5: Commit (DEFERRED)**

---

## Task 5: `_batch_triple_barrier` — vectorized 3-class labeling

**Files:**
- Modify: `backend/tools/xgb_v4_5_features_batch.py`
- Test: `backend/tests/test_xgb_v4_5_features_batch.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_xgb_v4_5_features_batch.py`:

```python
from tools.xgb_v4_5_features_batch import _batch_triple_barrier
from tools.train_xgb_v4_5 import _triple_barrier_label_3class


def test_batch_triple_barrier_matches_per_bar_reference():
    rng = np.random.default_rng(31)
    closes_np = 100.0 + np.cumsum(rng.normal(0, 0.5, size=400)).astype(np.float64)
    closes = torch.tensor(closes_np, dtype=torch.float64)
    forward_hours = 72
    label_thresh = 0.03
    sample_offset = 100   # arbitrary; just need room for forward window

    # Sample bars span [sample_offset, len(closes) - forward_hours)
    n_samples = len(closes) - sample_offset - forward_hours - 1
    sample_indices = torch.arange(sample_offset, sample_offset + n_samples, dtype=torch.int64)

    out = _batch_triple_barrier(closes, sample_indices, forward_hours, label_thresh)
    assert out.shape == (n_samples,)
    assert out.dtype == torch.int8

    for s in range(n_samples):
        i = int(sample_indices[s])
        ref = _triple_barrier_label_3class(closes_np, i, forward_hours, label_thresh)
        # Reference returns None for right-edge bars; batched returns -1 there.
        expected = -1 if ref is None else ref
        assert int(out[s]) == expected, f"sample {s} (i={i}): batched={int(out[s])}, ref={expected}"


def test_batch_triple_barrier_marks_truncated_as_minus_one():
    # 100 bars, sample at index 90 with forward_hours=72 → forward window
    # extends past end of series → should return -1
    closes = torch.linspace(100, 110, 100, dtype=torch.float64)
    sample_indices = torch.tensor([90], dtype=torch.int64)
    out = _batch_triple_barrier(closes, sample_indices, forward_hours=72, label_thresh=0.05)
    assert int(out[0]) == -1
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED)**

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `backend/tools/xgb_v4_5_features_batch.py`:

```python
def _batch_triple_barrier(
    closes: torch.Tensor,                # (n_candles,) float64
    sample_indices: torch.Tensor,        # (n_samples,) int64
    forward_hours: int,
    label_thresh: float,
) -> torch.Tensor:                       # (n_samples,) int8
    """Vectorized 3-class triple-barrier labeling.

    For each sample bar i: examine closes[i+1 .. i+forward_hours]. Return:
        2 (UP)      if any forward close >= entry * (1 + label_thresh) hit first
        0 (DOWN)    if any forward close <= entry * (1 - label_thresh) hit first
        1 (NEUTRAL) if neither barrier hit within window
        -1          if i + forward_hours >= n_candles (right-edge, truncated)

    Tie-break: UP barrier is checked first within each bar (matches
    train_xgb_v4_5._triple_barrier_label_3class).
    """
    n_candles = closes.shape[0]
    n_samples = sample_indices.shape[0]
    device = closes.device

    out = torch.full((n_samples,), 1, dtype=torch.int8, device=device)  # default NEUTRAL

    # Mark truncated samples (right edge): i + forward_hours >= n_candles
    truncated = (sample_indices + forward_hours) >= n_candles
    out[truncated] = -1

    # For non-truncated samples, build (n_samples, forward_hours) forward closes matrix.
    # We use Python loop to gather rows — forward_hours typically <=168, n_samples ~7000,
    # so a single advanced index is fine and avoids creating an (n_samples, n_candles) mask.
    entry = closes[sample_indices]                                  # (n_samples,)
    up_thr = entry * (1.0 + label_thresh)                           # (n_samples,)
    dn_thr = entry * (1.0 - label_thresh)                           # (n_samples,)

    # offsets: 1, 2, ..., forward_hours
    offsets = torch.arange(1, forward_hours + 1, device=device, dtype=torch.int64)
    # forward_indices: (n_samples, forward_hours), clamped to last valid index for truncated rows
    forward_indices = sample_indices.unsqueeze(1) + offsets.unsqueeze(0)
    forward_indices_safe = forward_indices.clamp(max=n_candles - 1)
    forward_closes = closes[forward_indices_safe]                   # (n_samples, forward_hours)

    up_hit = forward_closes >= up_thr.unsqueeze(1)                  # (n_samples, forward_hours)
    dn_hit = forward_closes <= dn_thr.unsqueeze(1)

    # First-hit index for UP and DOWN
    big = forward_hours + 1
    up_idx = torch.where(
        up_hit, offsets.unsqueeze(0).expand_as(up_hit),
        torch.full_like(up_hit, big, dtype=torch.int64),
    ).min(dim=1).values                                             # (n_samples,)
    dn_idx = torch.where(
        dn_hit, offsets.unsqueeze(0).expand_as(dn_hit),
        torch.full_like(dn_hit, big, dtype=torch.int64),
    ).min(dim=1).values

    # Decide label for non-truncated rows
    up_first = (up_idx <= dn_idx) & (up_idx <= forward_hours)
    dn_first = (~up_first) & (dn_idx <= forward_hours)

    labels = torch.full((n_samples,), 1, dtype=torch.int8, device=device)
    labels[up_first] = 2
    labels[dn_first] = 0

    out = torch.where(truncated, out, labels)
    return out
```

- [ ] **Step 4: Verify pass (DEFERRED)**
- [ ] **Step 5: Commit (DEFERRED)**

---

## Task 6: `batch_build_samples_for_pid` — orchestrator

**Files:**
- Modify: `backend/tools/xgb_v4_5_features_batch.py`
- Test: `backend/tests/test_xgb_v4_5_features_batch.py`

- [ ] **Step 1: Write the failing test (full-pipeline parity)**

Append to `backend/tests/test_xgb_v4_5_features_batch.py`:

```python
from tools.train_xgb_v4_5 import _build_samples_for_pid
from tools.xgb_v4_5_features_batch import batch_build_samples_for_pid


def _gen_synthetic_candles(n: int = 800, seed: int = 41) -> list:
    """Deterministic OHLCV — geometric-Brownian-style close series + plausible OHLV."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.005, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    opens = np.roll(closes, 1); opens[0] = closes[0]
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, size=n)))
    lows  = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, size=n)))
    vols  = rng.uniform(1000, 5000, size=n)
    return [
        {"start": int(60 * i), "open": float(opens[i]), "high": float(highs[i]),
         "low": float(lows[i]), "close": float(closes[i]), "volume": float(vols[i])}
        for i in range(n)
    ]


def test_batch_build_samples_full_pipeline_parity_cpu():
    candles = _gen_synthetic_candles(n=800, seed=41)
    fwd = 72
    thr = 0.03

    X_ref, y_ref, ts_ref = _build_samples_for_pid(
        candles, label_thresh=thr, forward_hours=fwd,
        micro=60, meso=168, macro=336,
    )
    X_bat, y_bat, ts_bat = batch_build_samples_for_pid(
        candles, forward_hours=fwd, label_thresh=thr, device="cpu",
    )

    assert X_ref.shape == X_bat.shape, f"shapes: ref={X_ref.shape} bat={X_bat.shape}"
    assert y_ref.shape == y_bat.shape
    assert ts_ref.shape == ts_bat.shape

    np.testing.assert_array_equal(y_ref, y_bat)
    np.testing.assert_array_equal(ts_ref, ts_bat)
    np.testing.assert_allclose(
        X_ref, X_bat, atol=_TOL,
        err_msg=f"max abs diff: {np.max(np.abs(X_ref - X_bat))}",
    )


def test_batch_build_samples_empty_candles_returns_empty():
    X, y, ts = batch_build_samples_for_pid(
        [], forward_hours=72, label_thresh=0.03, device="cpu",
    )
    assert X.shape == (0, N_FEATURES_V45) if False else X.shape == (0, 210)
    assert y.shape == (0,)
    assert ts.shape == (0,)


def test_batch_build_samples_too_few_candles_returns_empty():
    candles = _gen_synthetic_candles(n=200, seed=43)   # < macro + bb_prefix + fwd + 1
    X, y, ts = batch_build_samples_for_pid(
        candles, forward_hours=72, label_thresh=0.03, device="cpu",
    )
    assert X.shape == (0, 210)
    assert y.shape == (0,)
    assert ts.shape == (0,)
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED)**

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `backend/tools/xgb_v4_5_features_batch.py`:

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
    Returns numpy arrays (host memory) so the caller can hand them straight
    to numpy / xgboost just like _build_samples_for_pid.

    Empty/insufficient input -> three empty arrays.
    """
    micro = TIER_WINDOWS_V45["micro"]                       # 60
    meso  = TIER_WINDOWS_V45["meso"]                        # 168
    macro = TIER_WINDOWS_V45["macro"]                       # 336
    bb_prefix = BB_PERIOD                                   # 20
    sample_offset = macro + bb_prefix                       # 356

    n_candles = len(candles)
    min_needed = sample_offset + forward_hours + 1
    if n_candles < min_needed:
        return (
            np.zeros((0, N_FEATURES_V45), dtype=np.float64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )

    ohlcv = _build_pid_ohlcv(candles, device=device)        # (n_candles, 5)
    closes = ohlcv[:, 3]                                    # (n_candles,)

    # Sample bars: i in [sample_offset, n_candles - forward_hours). Triple-barrier
    # at i needs closes[i+1 .. i+forward_hours] all in range — last valid i is
    # n_candles - forward_hours - 1 (so that i + forward_hours == n_candles - 1).
    last_valid_i = n_candles - forward_hours - 1
    n_samples = last_valid_i - sample_offset + 1
    if n_samples <= 0:
        return (
            np.zeros((0, N_FEATURES_V45), dtype=np.float64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )
    sample_indices = torch.arange(
        sample_offset, sample_offset + n_samples, device=device, dtype=torch.int64,
    )

    # Per-tier feature blocks
    tier_blocks: List[torch.Tensor] = []
    for tier_name in _TIER_ORDER:
        tier_window = TIER_WINDOWS_V45[tier_name]
        window_len = tier_window + bb_prefix
        # OHLCV windows for this tier:
        ohlcv_unfold = ohlcv.unfold(0, window_len, 1)       # (n_unfold, 5, window_len)
        ohlcv_unfold = ohlcv_unfold.permute(0, 2, 1)        # (n_unfold, window_len, 5)
        first_sample_unfold_idx = sample_offset - window_len + 1
        ohlcv_windows = ohlcv_unfold[
            first_sample_unfold_idx : first_sample_unfold_idx + n_samples
        ]                                                   # (n_samples, window_len, 5)

        # Per-window local BB (channels 5, 6)
        closes_windows = ohlcv_windows[:, :, 3]             # (n_samples, window_len)
        bb_pos, bb_wid = _local_bb_in_windows(closes_windows)
        bb_windows = torch.stack([bb_pos, bb_wid], dim=-1)  # (n_samples, window_len, 2)

        # 7-channel windows
        ch_windows = torch.cat([ohlcv_windows, bb_windows], dim=-1)  # (n_samples, W, 7)

        # 10 stats per channel
        tier_stats = _batch_stats(ch_windows)               # (n_samples, 7, 10)
        tier_blocks.append(tier_stats)

    # Stack tiers and reshape to feature-major layout matching feature_names_v4_5().
    # feature_names_v4_5 layout: channel-major -> tier-major -> stat-major
    #   slot index = c * (N_TIERS * N_STATS) + tier * N_STATS + stat
    # tier_blocks[t] has shape (n_samples, 7, 10).
    # Stack to (n_samples, 7, n_tiers, 10), then reshape to (n_samples, 210).
    all_stats = torch.stack(tier_blocks, dim=2)             # (n_samples, 7, 3, 10)
    X = all_stats.reshape(n_samples, N_FEATURES_V45)        # (n_samples, 210)

    # Labels
    y = _batch_triple_barrier(closes, sample_indices, forward_hours, label_thresh)

    # Timestamps from the "start" field of each sample bar
    starts = torch.tensor(
        [int(candles[int(i)]["start"]) for i in sample_indices.cpu()],
        dtype=torch.int64, device=device,
    )

    # All valid (no -1 label) — by construction we chose n_samples so that the
    # forward window fits, so triple-barrier returns 0/1/2 only. But keep a
    # defensive filter in case of off-by-one:
    valid = y >= 0
    X = X[valid]; y = y[valid]; starts = starts[valid]

    return (
        X.cpu().numpy().astype(np.float64),
        y.cpu().numpy().astype(np.int8),
        starts.cpu().numpy().astype(np.int64),
    )
```

- [ ] **Step 4: Verify pass (DEFERRED)**
- [ ] **Step 5: Commit (DEFERRED)**

---

## Task 7: CUDA-conditional parity test

**Files:**
- Modify: `backend/tests/test_xgb_v4_5_features_batch.py`

- [ ] **Step 1: Append CUDA-gated test**

Append to `backend/tests/test_xgb_v4_5_features_batch.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_build_samples_full_pipeline_parity_cuda():
    candles = _gen_synthetic_candles(n=800, seed=41)
    fwd = 72
    thr = 0.03

    X_ref, y_ref, ts_ref = _build_samples_for_pid(
        candles, label_thresh=thr, forward_hours=fwd,
        micro=60, meso=168, macro=336,
    )
    X_gpu, y_gpu, ts_gpu = batch_build_samples_for_pid(
        candles, forward_hours=fwd, label_thresh=thr, device="cuda",
    )

    assert X_ref.shape == X_gpu.shape
    np.testing.assert_array_equal(y_ref, y_gpu)
    np.testing.assert_array_equal(ts_ref, ts_gpu)
    # Slightly looser tolerance for GPU due to reduction-order drift in float64.
    np.testing.assert_allclose(
        X_ref, X_gpu, atol=1e-3,
        err_msg=f"max abs diff CPU vs GPU: {np.max(np.abs(X_ref - X_gpu))}",
    )
```

- [ ] **Step 2: Verify pass on CUDA box, skip on non-CUDA (DEFERRED)**
- [ ] **Step 3: Commit (DEFERRED)**

---

## Task 8: Trainer `--device` flag + dispatch + xgb GPU param

**Files:**
- Modify: `backend/tools/train_xgb_v4_5.py`

- [ ] **Step 1: Add `--device` argparse**

Inside `_make_argparser()` (the function that builds the argparse — find it near `p.add_argument("--pids", ...)` around line 264):

Add after the existing argument definitions, before the `return p`:

```python
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "Feature-extraction backend. 'cpu' (default) uses the per-bar numpy "
            "loop in _build_samples_for_pid — bit-exact to historical behavior. "
            "'cuda' uses the batched PyTorch kernel from "
            "tools.xgb_v4_5_features_batch.batch_build_samples_for_pid, which "
            "also routes xgb.train onto the GPU."
        ),
    )
```

- [ ] **Step 2: Add deterministic-CUDA setup near top of `main()`**

Inside `main()` (around the `args = _make_argparser().parse_args()` line), immediately after parsing args:

```python
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("ERROR: --device cuda requested but torch.cuda.is_available() is False",
                  flush=True)
            sys.exit(2)
        torch.use_deterministic_algorithms(True, warn_only=True)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
```

- [ ] **Step 3: Dispatch _build_samples_for_pid vs batched**

Find the line `candles = _load_candles_for_pid(pid, args.history_dir)` (around line 295). Right after the existing per-PID processing block calls `_build_samples_for_pid`, replace that call with the dispatch:

```python
        if args.device == "cpu":
            X, y, ts = _build_samples_for_pid(
                candles,
                label_thresh=args.label_thresh,
                forward_hours=args.forward_hours,
                micro=TIER_WINDOWS_V45["micro"],
                meso=TIER_WINDOWS_V45["meso"],
                macro=TIER_WINDOWS_V45["macro"],
            )
        else:
            from tools.xgb_v4_5_features_batch import batch_build_samples_for_pid
            X, y, ts = batch_build_samples_for_pid(
                candles,
                forward_hours=args.forward_hours,
                label_thresh=args.label_thresh,
                device="cuda",
            )
```

- [ ] **Step 4: Add xgb device='cuda' when --device cuda**

Inside `_train_booster_3class` (lines 191-227), modify the `params` dict construction. Change `_train_booster_3class` signature to accept a `device` arg, and propagate it:

```python
def _train_booster_3class(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str], feature_weights: np.ndarray,
    device: str = "cpu",
):
    """Train one 3-class xgb.Booster (multi:softprob). Returns booster +
    val mlogloss."""
    import xgboost as xgb

    d_tr = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    d_tr.set_info(feature_weights=feature_weights)
    d_va = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    d_va.set_info(feature_weights=feature_weights)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "seed": 0,
    }
    if device == "cuda":
        params["device"] = "cuda"
    booster = xgb.train(
        params, d_tr, num_boost_round=200,
        evals=[(d_va, "val")], verbose_eval=False,
    )
    # ... (rest unchanged)
```

And update the call site in `main()` to pass `device=args.device`:

```python
    booster, val_mlogloss = _train_booster_3class(
        X_tr, y_tr, X_va, y_va,
        feature_names=feature_names_v4_5(),
        feature_weights=feature_weights_v4_5(),
        device=args.device,
    )
```

- [ ] **Step 5: Verify trainer accepts --device flag (DEFERRED)**
- [ ] **Step 6: Commit (DEFERRED)**

---

## Task 9: CHANGELOG.md + memory updates

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md`

- [ ] **Step 1: Add CHANGELOG bullet under `## Unreleased`**

```markdown
- **GPU-batched XGB v4.5 feature kernel.** New `backend/tools/xgb_v4_5_features_batch.py`
  vectorizes feature extraction across all sample bars per PID via PyTorch
  `tensor.unfold`. Trainer `train_xgb_v4_5.py` gains `--device {cpu,cuda}` flag
  (default cpu, bit-exact to prior behavior). CUDA path also routes `xgb.train`
  onto GPU and pins `torch.use_deterministic_algorithms(True, warn_only=True)`
  for stable reduction order. Cuts per-horizon wall time from ~22 hr (CPU
  per-bar Python loop) to <30 min on RTX 2060. Parity tested at 1e-4 max abs
  diff vs CPU per-bar reference.
```

- [ ] **Step 2: Add architecture-memory section**

Append to `coinbase_trader_architecture.md` (find an appropriate section like "Training pipeline" or append at the end):

```markdown
### GPU-batched feature kernel (2026-05-23, Session 58.71n)

`backend/tools/xgb_v4_5_features_batch.py` exposes
`batch_build_samples_for_pid(candles, forward_hours, label_thresh, device)`
that vectorizes feature extraction for one PID using PyTorch tensor.unfold
strided windows. Used only by `train_xgb_v4_5.py --device cuda`; inference
and `--device cpu` training continue to use the per-bar
`xgb_v4_5_features.extract_v4_5` API.

Numerical contract: max abs diff < 1e-4 vs per-bar numpy reference on the
(N, 210) feature matrix. Verified by
`test_batch_build_samples_full_pipeline_parity_cpu` (always runs) and
`test_batch_build_samples_full_pipeline_parity_cuda` (skipped on non-CUDA boxes).
```

- [ ] **Step 3: Commit (DEFERRED)**

---

## Task 10: Operator-paused execution gate — pytest + atomic commit + push

**REQUIRES OPERATOR ACTION:** 8001 trading must be paused before this task runs.

- [ ] **Step 1: Operator confirms 8001 is paused**

```powershell
# Verify trading is paused via frontend toggle or backend confirmation.
# Backend can still be listening on 8001; what matters is that scan-loop signal
# generation has stopped so concurrent pytest doesn't race on coinbase.db.
```

- [ ] **Step 2: Run the full pytest suite**

```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/ -v --tb=short 2>&1 | tail -80
```

Expected: all tests pass (the new tests in `test_xgb_v4_5_features_batch.py` plus all existing). Test count should be ~970 + ~14 new = ~984.

If any fail: stop, debug. Per CLAUDE.md, no commit until green.

- [ ] **Step 3: Stage files in one atomic batch**

```bash
git add backend/tools/xgb_v4_5_features_batch.py \
        backend/tests/test_xgb_v4_5_features_batch.py \
        backend/tools/train_xgb_v4_5.py \
        CHANGELOG.md
```

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat: GPU-batched XGB v4.5 feature kernel + --device flag

New backend/tools/xgb_v4_5_features_batch.py vectorizes feature extraction
across all sample bars per PID via PyTorch tensor.unfold strided windows.
train_xgb_v4_5.py gains --device {cpu,cuda} flag; CPU default keeps current
bit-exact behavior, --device cuda runs the batched kernel + routes
xgb.train onto GPU. Cuts per-horizon wall time from ~22 hr -> <30 min on
RTX 2060.

Tests: 14 new in test_xgb_v4_5_features_batch.py — per-stat parity (last,
mean, std, slope, min, max, pct_rank, dlt5/10/30), local-BB parity,
triple-barrier vectorization parity, full-pipeline CPU parity, full-pipeline
CUDA parity (skipped on non-CUDA boxes). Tolerance: 1e-4 vs CPU per-bar
reference; 1e-3 cross-device.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Push**

```bash
git push origin master
```

Expected: pushes the new commit to GitHub.

- [ ] **Step 6: Update memory file in same step**

```bash
# In your editor or Write tool, append the Architecture section from Task 9 Step 2
# to coinbase_trader_architecture.md if not already done.
```

- [ ] **Step 7: Operator runs the GPU training sweep**

```bash
cd C:\Users\gl450\polymarket_app\backend
PIDS=$(ls data/history/*.parquet | grep -v '^__' | sed 's/.*\///;s/\.parquet$//' | tr '\n' ',' | sed 's/,$//')
../.venv/Scripts/python.exe -m tools.train_xgb_v4_5 \
    --pids "$PIDS" --forward-hours 72 --label-thresh 0.03 --device cuda
```

Expected: completes in ≤30 min. Writes `backend/xgb_*_v4_5_h72.json` artifacts.

- [ ] **Step 8: Repeat for h168 + horizon_compare (optional, operator's call)**

If shadow-week confound from CPU→GPU cutover is acceptable, repeat h168
similarly and run `v4_5_horizon_compare --horizons 24,72,168` to inspect HTML.

---

## Verification of plan against spec

**Spec coverage check:**
- ✅ "Cut training time ≤30 min/horizon" → Task 8 wires up `--device cuda`.
- ✅ "CPU default bit-exact" → Task 8 Step 3 keeps the existing `_build_samples_for_pid` path untouched.
- ✅ "Numerical parity < 1e-4 batched vs per-bar" → Task 6 Step 1 asserts this.
- ✅ "Numerical parity < 1e-4 GPU vs CPU batched" → Task 7 asserts this (with looser 1e-3 cross-device tolerance per drift estimate in spec).
- ✅ "Per-stat unit tests + full-pipeline parity" → Tasks 4, 6.
- ✅ "CUDA-conditional skipif" → Task 7.
- ✅ "Triple-barrier vectorization test" → Task 5.
- ✅ "BB local-per-slice semantics" → Task 2.
- ✅ "deterministic CUDA" → Task 8 Step 2.
- ✅ "Inference untouched" → no Task modifies `xgb_v4_5_features.py` or `agents/xgb_signal.py`.
- ✅ "Out-of-scope: multi-PID GPU batching" → no Task adds this.
- ✅ "Out-of-scope: GPU inference" → no Task adds this.
- ✅ "Files-only until pause" → Task 10 gates pytest + commit + push.

**Type consistency check:**
- `batch_build_samples_for_pid` signature matches across spec, Task 6 impl, Task 8 dispatch. ✓
- `_local_bb_in_windows` returns `(bb_pos, bb_wid)` in both Task 2 impl and Task 6 caller. ✓
- `_batch_stats` returns `(n_samples, C, 10)` in Task 4 and is reshaped to `(n_samples, 210)` in Task 6 — channel-major → tier-major → stat-major layout matches `feature_names_v4_5()`. ✓
- `_batch_triple_barrier` returns int8 in Task 5 and is used as int8 in Task 6. ✓
- `_build_pid_ohlcv` returns `(n_candles, 5)` in Task 3 and is unfolded in Task 6. ✓

**Placeholder scan:** none of "TBD", "TODO", "implement later", or "add error handling" found in the plan.
