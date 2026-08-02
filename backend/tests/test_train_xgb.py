"""TDD tests for tools/train_xgb.py — XGBoost training with purged
walk-forward CV and grid search.

Phase 3 of the CNN -> XGBoost transition. Uses small synthetic data with
a learnable signal (y = ch0_mean > 0) so tests run in <5 seconds.
"""

import json
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

N_CHANNELS = 28
SEQ_LEN = 60
N_SAMPLES = 240


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Learnable signal: label = 1 iff mean of channel 0 over the window > 0."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((N_SAMPLES, N_CHANNELS, SEQ_LEN)).astype(np.float32)
    # Inject signal in channel 0 — half the samples have positive drift
    drift = rng.choice([-0.5, 0.5], size=N_SAMPLES)
    samples[:, 0, :] += drift[:, None]
    labels = (drift > 0).astype(np.float32)
    timestamps = np.arange(1_700_000_000, 1_700_000_000 + N_SAMPLES * 3600, 3600, dtype=np.int64)
    return samples, labels, timestamps


# ── Signature ───────────────────────────────────────────────────────────


class TestSignature:
    def test_train_xgb_exists(self):
        from tools.train_xgb import train_xgb

        assert callable(train_xgb)


# ── Output structure ────────────────────────────────────────────────────


class TestOutput:
    def test_returns_dict_with_expected_keys(self, synthetic_dataset, tmp_path):
        from tools.train_xgb import train_xgb

        samples, labels, ts = synthetic_dataset
        result = train_xgb(
            samples,
            labels,
            ts,
            n_folds=3,
            embargo_hours=4,
            grid=[{"max_depth": 3, "min_child_weight": 1, "subsample": 1.0}],
            out_dir=tmp_path,
            n_estimators=20,
            seed=0,
        )
        for k in (
            "best_params",
            "fold_aucs",
            "mean_auc",
            "feature_names",
            "model_path",
            "features_path",
        ):
            assert k in result, f"missing key: {k}"

    def test_fold_aucs_length_matches_n_folds(self, synthetic_dataset, tmp_path):
        from tools.train_xgb import train_xgb

        samples, labels, ts = synthetic_dataset
        result = train_xgb(
            samples,
            labels,
            ts,
            n_folds=3,
            grid=[{"max_depth": 3, "min_child_weight": 1, "subsample": 1.0}],
            out_dir=tmp_path,
            n_estimators=20,
            seed=0,
        )
        assert len(result["fold_aucs"]) == 3
        assert all(0.0 <= a <= 1.0 for a in result["fold_aucs"])


# ── Persisted artifacts ─────────────────────────────────────────────────


class TestArtifacts:
    def test_model_file_saved_and_loadable(self, synthetic_dataset, tmp_path):
        import xgboost as xgb

        from tools.train_xgb import train_xgb

        samples, labels, ts = synthetic_dataset
        result = train_xgb(
            samples,
            labels,
            ts,
            n_folds=3,
            grid=[{"max_depth": 3, "min_child_weight": 1, "subsample": 1.0}],
            out_dir=tmp_path,
            n_estimators=20,
            seed=0,
        )
        model_path = result["model_path"]
        assert os.path.exists(model_path)
        booster = xgb.Booster()
        booster.load_model(model_path)
        # Booster should accept inference on a sample-shaped input
        from tools.xgb_features import extract_features

        features, _ = extract_features(samples[:2])
        dmat = xgb.DMatrix(features, feature_names=result["feature_names"])
        preds = booster.predict(dmat)
        assert preds.shape == (2,)
        assert ((preds >= 0) & (preds <= 1)).all()

    def test_features_json_saved_with_names(self, synthetic_dataset, tmp_path):
        from tools.train_xgb import train_xgb

        samples, labels, ts = synthetic_dataset
        result = train_xgb(
            samples,
            labels,
            ts,
            n_folds=3,
            grid=[{"max_depth": 3, "min_child_weight": 1, "subsample": 1.0}],
            out_dir=tmp_path,
            n_estimators=20,
            seed=0,
        )
        with open(result["features_path"]) as f:
            schema = json.load(f)
        assert "feature_names" in schema
        assert isinstance(schema["feature_names"], list)
        assert len(schema["feature_names"]) == 280  # 28 channels x 10 stats
        assert schema["feature_names"] == result["feature_names"]


# ── Grid search ─────────────────────────────────────────────────────────


class TestGridSearch:
    def test_best_params_come_from_grid(self, synthetic_dataset, tmp_path):
        from tools.train_xgb import train_xgb

        samples, labels, ts = synthetic_dataset
        grid = [
            {"max_depth": 3, "min_child_weight": 1, "subsample": 1.0},
            {"max_depth": 4, "min_child_weight": 5, "subsample": 0.7},
        ]
        result = train_xgb(
            samples,
            labels,
            ts,
            n_folds=3,
            grid=grid,
            out_dir=tmp_path,
            n_estimators=20,
            seed=0,
        )
        assert result["best_params"] in grid

    def test_signal_is_learnable_mean_auc_above_random(self, synthetic_dataset, tmp_path):
        """Sanity: with injected signal in ch 0, a tiny model should beat 0.5."""
        from tools.train_xgb import train_xgb

        samples, labels, ts = synthetic_dataset
        result = train_xgb(
            samples,
            labels,
            ts,
            n_folds=3,
            grid=[{"max_depth": 4, "min_child_weight": 1, "subsample": 1.0}],
            out_dir=tmp_path,
            n_estimators=50,
            seed=0,
        )
        assert result["mean_auc"] > 0.6, f"Signal not learned: mean_auc={result['mean_auc']:.3f}"
