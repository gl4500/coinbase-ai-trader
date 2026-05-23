"""Tests for `tools/fit_xgb_calibration.py` (#187).

The signal_outcomes-fit calibrator shipped in #180 plateaus at a single
output value (~0.4346) across ~95% of the booster's [0.4, 0.6] range,
because the source sample is too small (~300 resolved CNN BUYs) and biased
toward the high-prob tail. Live result: XGB output collapses to a constant
and AUC drops from 0.6 (raw) to 0.05 (calibrated).

Fix: add a `cache` source mode that fits isotonic on the booster's
predictions over the full ~83k chronological val split of the per-product
dataset cache, with the cache's own forward-4h ±0.3% labels.

These tests cover the new path; the legacy db-shadow path is unchanged.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_BACKEND = os.path.abspath(os.path.join(_HERE, ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _make_synthetic_pairs(n: int = 80_000, seed: int = 0):
    """Mimic the booster's actual val-set output: tight [0.4, 0.6] range with
    a real monotone link between raw_prob and the binary outcome."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.4, 0.6, size=n)
    # P(win|raw) is monotone increasing across the booster's narrow range
    p_win = 0.3 + 1.5 * (raw - 0.4)  # 0.30 at raw=0.40 -> 0.60 at raw=0.60
    wins = (rng.random(n) < p_win).astype(np.float64)
    return raw.astype(np.float64), wins


class TestCacheSourceCalibratorFit:
    """Targets the new `source='cache'` mode that #187b adds."""

    def test_cache_source_function_exists(self):
        """The new fit entry-point must exist with a `source` kwarg."""
        from tools.fit_xgb_calibration import fit_calibration

        # Sentinel call signature; actual execution patched in later tests.
        # If the kwarg is missing this raises TypeError → RED.
        import inspect
        sig = inspect.signature(fit_calibration)
        assert "source" in sig.parameters, (
            "fit_calibration must accept `source` so callers can pick "
            "between the legacy signal_outcomes path and the new cache path"
        )

    def test_cache_source_writes_non_collapsed_calibrator(self, tmp_path, monkeypatch):
        """With ~80k predictive (raw, win) pairs, isotonic must NOT collapse."""
        import tools.fit_xgb_calibration as mod

        raw, wins = _make_synthetic_pairs()
        monkeypatch.setattr(
            mod, "_load_cache_pairs",
            lambda cache_path, model_path, features_path: (raw, wins),
        )

        out_path = str(tmp_path / "calib.pkl")
        mod.fit_calibration(
            source="cache",
            cache_path="ignored",
            model_path="ignored",
            features_path="ignored",
            out_path=out_path,
        )

        with open(out_path, "rb") as f:
            _loaded = pickle.load(f); iso = _loaded["calibrator"] if isinstance(_loaded, dict) else _loaded
        grid = np.linspace(0.4, 0.6, 11)
        cal = iso.transform(grid)
        n_unique = len(np.unique(np.round(cal, 4)))
        assert n_unique >= 5, (
            f"calibrator collapsed: only {n_unique} unique outputs across "
            f"the booster's [0.4, 0.6] range — this is the regression #187 fixes"
        )

    def test_cache_source_calibrator_is_monotone(self, tmp_path, monkeypatch):
        """Isotonic must be monotone non-decreasing — preserves ranking."""
        import tools.fit_xgb_calibration as mod

        raw, wins = _make_synthetic_pairs()
        monkeypatch.setattr(
            mod, "_load_cache_pairs",
            lambda cache_path, model_path, features_path: (raw, wins),
        )

        out_path = str(tmp_path / "calib.pkl")
        mod.fit_calibration(
            source="cache",
            cache_path="ignored",
            model_path="ignored",
            features_path="ignored",
            out_path=out_path,
        )

        with open(out_path, "rb") as f:
            _loaded = pickle.load(f); iso = _loaded["calibrator"] if isinstance(_loaded, dict) else _loaded
        grid = np.linspace(0.4, 0.6, 50)
        cal = iso.transform(grid)
        diffs = np.diff(cal)
        # Allow tiny numerical noise but no true regressions
        assert np.all(diffs >= -1e-9), (
            f"calibrator violates monotonicity: {diffs.min():+.6f} min step"
        )

    def test_cache_source_preserves_auc_within_tolerance(self, tmp_path, monkeypatch):
        """Calibrated AUC must be within 0.02 of raw AUC (ranking preserved)."""
        import tools.fit_xgb_calibration as mod

        raw, wins = _make_synthetic_pairs()
        monkeypatch.setattr(
            mod, "_load_cache_pairs",
            lambda cache_path, model_path, features_path: (raw, wins),
        )

        out_path = str(tmp_path / "calib.pkl")
        mod.fit_calibration(
            source="cache",
            cache_path="ignored",
            model_path="ignored",
            features_path="ignored",
            out_path=out_path,
        )

        with open(out_path, "rb") as f:
            _loaded = pickle.load(f); iso = _loaded["calibrator"] if isinstance(_loaded, dict) else _loaded
        cal = iso.transform(raw)

        def _auc(p, y):
            pos = p[y == 1]
            neg = p[y == 0]
            order = np.argsort(np.concatenate([pos, neg]))
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1)
            u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
            return float(u / (len(pos) * len(neg)))

        raw_auc = _auc(raw, wins)
        cal_auc = _auc(cal, wins)
        assert abs(raw_auc - cal_auc) <= 0.02, (
            f"calibration broke ranking: raw_auc={raw_auc:.4f} "
            f"cal_auc={cal_auc:.4f}"
        )

    def test_cache_source_default_when_no_args(self):
        """The legacy positional-arg call still routes through the
        signal_outcomes path so #180 callers don't break."""
        from tools.fit_xgb_calibration import fit_calibration
        import inspect
        sig = inspect.signature(fit_calibration)
        # Backward compat: default source must keep the legacy "shadow" path
        default = sig.parameters["source"].default
        assert default in ("shadow", "signal_outcomes"), (
            f"backward compat: default source kwarg must keep legacy path, "
            f"got {default!r}"
        )


class TestFeatureSetDetection:
    """Regression: at N_CHANNELS=28 the v1 set is 280 features, which is
    > 270. The legacy `len > 270 -> v2` heuristic flips a v1 booster to
    the v2 extraction path and produces a 290-vs-280 column mismatch when
    DMatrix-binding feature_names. Detect by name prefix instead."""

    def test_v1_at_28_channels_detected_as_v1(self):
        from tools.fit_xgb_calibration import _detect_feature_set
        v1_names = [f"ch{c}_{s}" for c in range(28)
                    for s in ("last", "mean", "std", "slope", "min", "max",
                              "pct_rank", "delta_5", "delta_10", "delta_30")]
        assert len(v1_names) == 280
        assert _detect_feature_set(v1_names) == "v1"

    def test_v2_at_28_channels_detected_as_v2(self):
        from tools.fit_xgb_calibration import _detect_feature_set
        v1_names = [f"ch{c}_{s}" for c in range(28)
                    for s in ("last", "mean", "std", "slope", "min", "max",
                              "pct_rank", "delta_5", "delta_10", "delta_30")]
        v2_addons = ["xt_vol_regime_ratio", "xt_vol_of_vol", "xt_ret_full"]
        assert _detect_feature_set(v1_names + v2_addons) == "v2"


# ── v3 calibrator dict-pickle shape (added 2026-05-16, #311f) ─────────────


class TestV3CalibrationPickle:
    def test_pickle_writes_dict_with_feature_set(self, tmp_path):
        import pickle
        from sklearn.isotonic import IsotonicRegression
        import numpy as np
        from tools.fit_xgb_calibration import _save_calibrator

        out = tmp_path / "xgb_calibration.pkl"
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.5, 0.9]),
        )
        _save_calibrator(iso, str(out), feature_set="v3")
        loaded = pickle.load(open(out, "rb"))
        assert isinstance(loaded, dict)
        assert loaded["feature_set"] == "v3"
        assert "calibrator" in loaded

    def test_detect_feature_set_recognises_v3(self):
        from tools.fit_xgb_calibration import _detect_feature_set
        v3_names = ["ch0_last", "ch15_m060_mean", "ch15_m168_slope",
                    "ch20_m336_pct_rank"]
        assert _detect_feature_set(v3_names) == "v3"
