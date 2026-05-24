"""Parity tests for the batched feature kernel against the per-bar numpy reference.

Numerical contract:
    - Same-device (CPU batched vs CPU per-bar): max abs diff < 1e-4
    - Cross-device (GPU batched vs CPU per-bar): max abs diff < 1e-3 (looser
      because floating-point reduction order on GPU differs from numpy)
    - Labels + timestamps: exact equality (integer dtype, no drift)
"""
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
    _compute_stats, _compute_bb_position, _compute_bb_width,
    BB_PERIOD, N_FEATURES_V45,
)
from tools.xgb_v4_5_features_batch import (
    _build_pid_ohlcv,
    _local_bb_in_windows,
    _batch_stats,
    _batch_triple_barrier,
    batch_build_samples_for_pid,
)
from tools.train_xgb_v4_5 import (
    _build_samples_for_pid, _triple_barrier_label_3class,
)


_TOL_SAMEDEVICE = 1e-4
_TOL_CROSSDEVICE = 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# _build_pid_ohlcv
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# _local_bb_in_windows  — per-slice BB parity
# ═══════════════════════════════════════════════════════════════════════════

def _make_strided_closes(closes: np.ndarray, window_len: int, n_samples: int,
                         sample_offset: int) -> torch.Tensor:
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
    sample_offset = window_len + 100   # ample warmup so no wrap issues
    closes_windows = _make_strided_closes(closes, window_len, n_samples, sample_offset)

    bb_pos, bb_wid = _local_bb_in_windows(closes_windows)
    assert bb_pos.shape == (n_samples, window_len)
    assert bb_wid.shape == (n_samples, window_len)

    for s in range(n_samples):
        i = sample_offset + s
        local_closes = closes[i - window_len : i]
        ref_pos = _compute_bb_position(local_closes)
        ref_wid = _compute_bb_width(local_closes)
        np.testing.assert_allclose(bb_pos[s].cpu().numpy(), ref_pos, atol=_TOL_SAMEDEVICE,
                                   err_msg=f"BB position mismatch at sample {s}")
        np.testing.assert_allclose(bb_wid[s].cpu().numpy(), ref_wid, atol=_TOL_SAMEDEVICE,
                                   err_msg=f"BB width mismatch at sample {s}")


def test_local_bb_fallback_when_window_shorter_than_period():
    # Window length < BB_PERIOD -> entire output should be fallback values
    closes_windows = torch.zeros((3, BB_PERIOD - 1), dtype=torch.float64)
    bb_pos, bb_wid = _local_bb_in_windows(closes_windows)
    np.testing.assert_array_equal(bb_pos.cpu().numpy(),
                                  np.full((3, BB_PERIOD - 1), 0.5))
    np.testing.assert_array_equal(bb_wid.cpu().numpy(),
                                  np.zeros((3, BB_PERIOD - 1)))


# ═══════════════════════════════════════════════════════════════════════════
# _batch_stats — one test per stat
# ═══════════════════════════════════════════════════════════════════════════

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
            np.testing.assert_allclose(out[s, ch, 0], ref[0], atol=_TOL_SAMEDEVICE)


def test_batch_stats_mean_std_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=13)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 1], ref[1], atol=_TOL_SAMEDEVICE)
            np.testing.assert_allclose(out[s, ch, 2], ref[2], atol=_TOL_SAMEDEVICE)


def test_batch_stats_slope_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=17)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 3], ref[3], atol=_TOL_SAMEDEVICE)


def test_batch_stats_minmax_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=19)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 4], ref[4], atol=_TOL_SAMEDEVICE)
            np.testing.assert_allclose(out[s, ch, 5], ref[5], atol=_TOL_SAMEDEVICE)


def test_batch_stats_pct_rank_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=23)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 6], ref[6], atol=_TOL_SAMEDEVICE)


def test_batch_stats_delta_matches_per_bar():
    w = _make_batch_windows(n_samples=4, window_len=80, n_channels=7, seed=29)
    out = _batch_stats(w).cpu().numpy()
    for s in range(4):
        for ch in range(7):
            ref = _compute_stats(w[s, :, ch].cpu().numpy())
            np.testing.assert_allclose(out[s, ch, 7], ref[7], atol=_TOL_SAMEDEVICE)
            np.testing.assert_allclose(out[s, ch, 8], ref[8], atol=_TOL_SAMEDEVICE)
            np.testing.assert_allclose(out[s, ch, 9], ref[9], atol=_TOL_SAMEDEVICE)


def test_batch_stats_empty_window_returns_zeros():
    w = torch.zeros((3, 0, 7), dtype=torch.float64)
    out = _batch_stats(w)
    assert out.shape == (3, 7, 10)
    assert (out == 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# _batch_triple_barrier
# ═══════════════════════════════════════════════════════════════════════════

def test_batch_triple_barrier_matches_per_bar_reference():
    rng = np.random.default_rng(31)
    closes_np = 100.0 + np.cumsum(rng.normal(0, 0.5, size=400)).astype(np.float64)
    closes = torch.tensor(closes_np, dtype=torch.float64)
    forward_hours = 72
    label_thresh = 0.03
    sample_offset = 100

    # Sample bars span [sample_offset, len(closes) - forward_hours)
    last_valid_i = len(closes_np) - forward_hours - 1
    n_samples = last_valid_i - sample_offset + 1
    sample_indices = torch.arange(sample_offset, sample_offset + n_samples, dtype=torch.int64)

    out = _batch_triple_barrier(closes, sample_indices, forward_hours, label_thresh)
    assert out.shape == (n_samples,)
    assert out.dtype == torch.int8

    for s in range(n_samples):
        i = int(sample_indices[s])
        ref = _triple_barrier_label_3class(closes_np, i, forward_hours, label_thresh)
        # All chosen indices are within bounds, so ref should never be None.
        assert ref is not None
        assert int(out[s]) == ref, f"sample {s} (i={i}): batched={int(out[s])}, ref={ref}"


def test_batch_triple_barrier_marks_truncated_as_minus_one():
    # 100 bars, sample at index 90 with forward_hours=72 → forward window
    # extends past end of series → should return -1
    closes = torch.linspace(100, 110, 100, dtype=torch.float64)
    sample_indices = torch.tensor([90], dtype=torch.int64)
    out = _batch_triple_barrier(closes, sample_indices, forward_hours=72, label_thresh=0.05)
    assert int(out[0]) == -1


# ═══════════════════════════════════════════════════════════════════════════
# Full-pipeline parity — CPU batched vs CPU per-bar
# ═══════════════════════════════════════════════════════════════════════════

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
    max_diff = float(np.max(np.abs(X_ref - X_bat)))
    np.testing.assert_allclose(
        X_ref, X_bat, atol=_TOL_SAMEDEVICE,
        err_msg=f"max abs diff: {max_diff}",
    )


def test_batch_build_samples_empty_candles_returns_empty():
    X, y, ts = batch_build_samples_for_pid(
        [], forward_hours=72, label_thresh=0.03, device="cpu",
    )
    assert X.shape == (0, N_FEATURES_V45)
    assert y.shape == (0,)
    assert ts.shape == (0,)


def test_batch_build_samples_too_few_candles_returns_empty():
    candles = _gen_synthetic_candles(n=200, seed=43)   # < macro + bb_prefix + fwd + 1
    X, y, ts = batch_build_samples_for_pid(
        candles, forward_hours=72, label_thresh=0.03, device="cpu",
    )
    assert X.shape == (0, N_FEATURES_V45)
    assert y.shape == (0,)
    assert ts.shape == (0,)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-device parity — GPU batched vs CPU per-bar (skipped if no CUDA)
# ═══════════════════════════════════════════════════════════════════════════

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
    max_diff = float(np.max(np.abs(X_ref - X_gpu)))
    np.testing.assert_allclose(
        X_ref, X_gpu, atol=_TOL_CROSSDEVICE,
        err_msg=f"max abs diff CPU-ref vs CUDA-batched: {max_diff}",
    )
