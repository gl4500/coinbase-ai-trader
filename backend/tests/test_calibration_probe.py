"""TDD tests for tools/calibration_probe.py — Phase 4 HARD GATE.

The probe runs purged walk-forward CV, collects out-of-fold predictions,
buckets them against realized labels, and decides whether the model passes:

  - Walk-forward mean AUC >= 0.55, AND
  - Calibration is monotonic (mean actual rises across buckets)

If either fails, do not proceed to integration. This is the check the CNN
training path never had.
"""

import os
import sys

import numpy as np

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

N_CHANNELS = 28
SEQ_LEN = 60


def _learnable_dataset(n_samples: int = 240, seed: int = 0):
    """ch 0 mean drift > 0 -> label 1. Same shape as Phase 3 fixture."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n_samples, N_CHANNELS, SEQ_LEN)).astype(np.float32)
    drift = rng.choice([-0.5, 0.5], size=n_samples)
    samples[:, 0, :] += drift[:, None]
    labels = (drift > 0).astype(np.float32)
    timestamps = np.arange(1_700_000_000, 1_700_000_000 + n_samples * 3600, 3600, dtype=np.int64)
    return samples, labels, timestamps


def _noise_dataset(n_samples: int = 600, seed: int = 1):
    """No signal: labels are independent of features.

    Larger sample size so out-of-fold AUC concentrates near 0.5 (LLN);
    a 240-sample noise dataset can hit AUC > 0.55 by chance.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n_samples, N_CHANNELS, SEQ_LEN)).astype(np.float32)
    labels = rng.integers(0, 2, size=n_samples).astype(np.float32)
    timestamps = np.arange(1_700_000_000, 1_700_000_000 + n_samples * 3600, 3600, dtype=np.int64)
    return samples, labels, timestamps


# ── Signature ───────────────────────────────────────────────────────────


class TestSignature:
    def test_function_exists(self):
        from tools.calibration_probe import calibration_probe

        assert callable(calibration_probe)


# ── Output structure ────────────────────────────────────────────────────


class TestOutput:
    def test_returns_expected_keys(self):
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _learnable_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=20,
            seed=0,
        )
        for k in (
            "fold_aucs",
            "mean_auc",
            "calibration_table",
            "is_monotonic",
            "gate_passed",
            "gate_reason",
        ):
            assert k in result, f"missing key: {k}"

    def test_fold_aucs_length_matches_n_folds(self):
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _learnable_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=20,
            seed=0,
        )
        assert len(result["fold_aucs"]) == 3

    def test_calibration_table_has_n_buckets(self):
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _learnable_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=30,
            seed=0,
        )
        assert len(result["calibration_table"]) == 5
        for row in result["calibration_table"]:
            for k in ("bucket", "n", "mean_pred", "mean_actual"):
                assert k in row, f"missing calibration row key: {k}"


# ── Hard gate semantics ─────────────────────────────────────────────────


class TestGate:
    def test_learnable_signal_passes_gate(self):
        """With injected signal, walk-forward AUC should clear 0.55."""
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _learnable_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 4, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=80,
            seed=0,
        )
        assert result["mean_auc"] >= 0.55, (
            f"Expected learnable signal to clear 0.55 AUC; got {result['mean_auc']:.3f}"
        )
        assert result["gate_passed"] is True, (
            f"Gate should pass for learnable signal; reason: {result['gate_reason']}"
        )

    def test_pure_noise_fails_gate(self):
        """Random labels -> AUC ~0.5, gate must fail."""
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _noise_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=20,
            seed=0,
        )
        assert result["gate_passed"] is False, (
            f"Gate must fail on pure noise; mean_auc={result['mean_auc']:.3f}"
        )

    def test_gate_reason_is_string(self):
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _noise_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=20,
            seed=0,
        )
        assert isinstance(result["gate_reason"], str)
        assert len(result["gate_reason"]) > 0


# ── Calibration monotonicity ────────────────────────────────────────────


class TestCalibration:
    def test_buckets_are_chronological_in_pred_space(self):
        """Calibration table buckets must be ordered by predicted prob ascending."""
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _learnable_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 4, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=80,
            seed=0,
        )
        preds = [row["mean_pred"] for row in result["calibration_table"]]
        assert preds == sorted(preds), (
            f"Buckets must be ordered by mean_pred ascending; got {preds}"
        )

    def test_monotonic_calibration_for_learnable_signal(self):
        """With real signal, mean_actual should generally rise with mean_pred."""
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _learnable_dataset()
        result = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 4, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=80,
            seed=0,
        )
        assert result["is_monotonic"] is True, (
            f"Learnable signal must produce monotonic calibration; "
            f"table: {result['calibration_table']}"
        )


# ── feature_set passthrough (v2 comparison) ─────────────────────────────


class TestFeatureSetPassthrough:
    def test_v2_produces_more_feature_names_than_v1(self):
        """calibration_probe(feature_set='v2') must thread to extract_features."""
        from tools.calibration_probe import calibration_probe

        samples, labels, ts = _learnable_dataset(n_samples=180)
        r1 = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=10,
            seed=0,
            feature_set="v1",
        )
        r2 = calibration_probe(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            n_buckets=5,
            params={"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            n_estimators=10,
            seed=0,
            feature_set="v2",
        )
        assert len(r2["feature_names"]) == len(r1["feature_names"]) + 10
        assert r1["feature_names"][:280] == r2["feature_names"][:280]
        assert r2["feature_names"][280:] == [
            "xt_vol_regime_ratio",
            "xt_vol_of_vol",
            "xt_ret_full",
            "xt_ret_skew",
            "xt_ret_kurt",
            "xt_rsi_below_30",
            "xt_rsi_above_70",
            "xt_rsi_cross_up_3",
            "xt_macd_sign_change_3",
            "xt_funding_x_trend",
        ]
