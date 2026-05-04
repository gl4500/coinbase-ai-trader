"""XGBoost inference signal — Phase 5 of CNN→XGBoost transition (#135).

Provides `xgb_prob(channels) -> float`: the same contract as `_cnn_prob` so
`MODEL_BACKEND=xgb` can swap probability sources without touching the
buy gate (`model_prob > cnn_buy_threshold`).

Lazy-load: the Booster + feature_names are read from disk on first call
and cached for subsequent calls. If the artifacts don't exist the call
returns 0.5 (neutral) and never raises — that's how Phase 5 ships safely
with default `MODEL_BACKEND=cnn`: the XGB code path is wired in but
silently inert until a model file is dropped on disk.

Default artifact paths sit beside the CNN checkpoint at backend/, so the
sibling files are discoverable by the same Operator runbook.

XGB-Step2 (#180): A pickled IsotonicRegression at xgb_calibration.pkl is
loaded alongside the booster (when present) and applied to raw output
before clipping. Fixes the live U-shape calibration discovered in the
shadow window without retraining the booster.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from typing import List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BACKEND_DIR, "xgb_model.json")
_FEATURES_PATH = os.path.join(_BACKEND_DIR, "xgb_features.json")
_CALIBRATION_PATH = os.path.join(_BACKEND_DIR, "xgb_calibration.pkl")

_NEUTRAL = 0.5
_FEATURE_SET_DEFAULT = "v1"

_lock = threading.Lock()
_booster = None
_feature_names: List[str] = []
_feature_set: str = _FEATURE_SET_DEFAULT
_calibration: Optional[object] = None
_load_attempted: bool = False
_load_succeeded: bool = False


ChannelsLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _try_load() -> bool:
    """Load Booster + feature_names + optional calibrator from disk once. Idempotent."""
    global _booster, _feature_names, _feature_set, _calibration, _load_attempted, _load_succeeded
    with _lock:
        if _load_attempted:
            return _load_succeeded
        _load_attempted = True
        if not (os.path.exists(_MODEL_PATH) and os.path.exists(_FEATURES_PATH)):
            logger.info(
                "xgb_signal: artifacts missing (model=%s features=%s) — fallback to %.2f",
                _MODEL_PATH, _FEATURES_PATH, _NEUTRAL,
            )
            return False
        try:
            import xgboost as xgb
            with open(_FEATURES_PATH, "r") as f:
                meta = json.load(f)
            names = list(meta.get("feature_names", []))
            if not names:
                logger.warning("xgb_signal: features.json has empty feature_names")
                return False
            booster = xgb.Booster()
            booster.load_model(_MODEL_PATH)
            _booster = booster
            _feature_names = names
            _feature_set = "v2" if len(names) > 270 else "v1"
            _load_succeeded = True
            logger.info(
                "xgb_signal: loaded booster (%d features, set=%s)",
                len(names), _feature_set,
            )
            if os.path.exists(_CALIBRATION_PATH):
                try:
                    with open(_CALIBRATION_PATH, "rb") as f:
                        _calibration = pickle.load(f)
                    logger.info(
                        "xgb_signal: loaded isotonic calibrator from %s",
                        _CALIBRATION_PATH,
                    )
                except Exception as exc:
                    logger.exception(
                        "xgb_signal: failed to load calibrator (raw passthrough): %s",
                        exc,
                    )
                    _calibration = None
            else:
                logger.info(
                    "xgb_signal: no calibrator at %s — raw passthrough",
                    _CALIBRATION_PATH,
                )
            return True
        except Exception as exc:
            logger.exception("xgb_signal: failed to load artifacts: %s", exc)
            return False


def xgb_prob(channels: ChannelsLike) -> float:
    """XGBoost probability of upward target, in [0.01, 0.99].

    Returns 0.5 if the model artifacts are missing or fail to load — the
    caller (`_cnn_prob` branch on `config.model_backend`) should treat
    that as "no signal" rather than error.
    """
    if not _try_load():
        return _NEUTRAL
    try:
        import xgboost as xgb
        from tools.xgb_features import extract_features

        arr = np.asarray(channels, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        features, _ = extract_features(arr, feature_set=_feature_set)
        dmat = xgb.DMatrix(features, feature_names=_feature_names)
        raw = float(_booster.predict(dmat)[0])
        if _calibration is not None:
            raw = float(_calibration.transform(np.asarray([raw]))[0])
        return float(np.clip(raw, 0.01, 0.99))
    except Exception as exc:
        logger.exception("xgb_signal.xgb_prob failed, returning neutral: %s", exc)
        return _NEUTRAL
