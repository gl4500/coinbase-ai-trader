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
from typing import List, Optional, Sequence, Tuple, Union

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

# ── v4 shadow state (Step B.1) ────────────────────────────────────────────
_MODEL_PATH_V4    = os.path.join(_BACKEND_DIR, "xgb_model_v4.json")
_FEATURES_PATH_V4 = os.path.join(_BACKEND_DIR, "xgb_features_v4.json")
_CALIB_PATH_V4    = os.path.join(_BACKEND_DIR, "xgb_calibration_v4.pkl")

_booster_v4 = None
_feature_names_v4: List[str] = []
_calibration_v4: Optional[object] = None
_load_attempted_v4: bool = False
_load_succeeded_v4: bool = False


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
            if any(("_m060_" in str(n)) or ("_m168_" in str(n)) or ("_m336_" in str(n)) for n in names):
                _feature_set = "v3"
            elif any(str(n).startswith("xt_") for n in names):
                _feature_set = "v2"
            else:
                _feature_set = "v1"
            _load_succeeded = True
            logger.info(
                "xgb_signal: loaded booster (%d features, set=%s)",
                len(names), _feature_set,
            )
            if os.path.exists(_CALIBRATION_PATH):
                try:
                    with open(_CALIBRATION_PATH, "rb") as f:
                        obj = pickle.load(f)
                    if not (isinstance(obj, dict) and "calibrator" in obj):
                        logger.warning(
                            "xgb_signal: calibrator pkl is not the canonical "
                            '{"calibrator","feature_set"} dict shape — '
                            "skipping calibration. Legacy bare-isotonic "
                            "format dropped #311-refactor-b."
                        )
                        _calibration = None
                    else:
                        cal_set = obj.get("feature_set")
                        if cal_set is not None and cal_set != _feature_set:
                            logger.warning(
                                "xgb_signal: calibrator feature_set=%s differs from "
                                "booster feature_set=%s — skipping calibration",
                                cal_set, _feature_set,
                            )
                            _calibration = None
                        else:
                            _calibration = obj["calibrator"]
                            logger.info(
                                "xgb_signal: loaded isotonic calibrator (feature_set=%s)",
                                cal_set,
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


def _try_load_v4() -> bool:
    """Load v4 booster + feature_names + calibrator from disk once. Idempotent.

    Returns True iff all artifacts loaded successfully. Failures log + return
    False; never raise. Mirrors _try_load but for v4 paths.
    """
    global _booster_v4, _feature_names_v4, _calibration_v4
    global _load_attempted_v4, _load_succeeded_v4
    with _lock:
        if _load_attempted_v4:
            return _load_succeeded_v4
        _load_attempted_v4 = True
        if not (os.path.exists(_MODEL_PATH_V4) and os.path.exists(_FEATURES_PATH_V4)):
            logger.info(
                "xgb_signal: v4 artifacts missing (model=%s features=%s) — shadow disabled",
                _MODEL_PATH_V4, _FEATURES_PATH_V4,
            )
            return False
        try:
            import xgboost as xgb
            with open(_FEATURES_PATH_V4, "r") as f:
                meta = json.load(f)
            names = list(meta.get("feature_names", []))
            if not names:
                logger.warning("xgb_signal: v4 features.json has empty feature_names")
                return False
            booster = xgb.Booster()
            booster.load_model(_MODEL_PATH_V4)
            _booster_v4 = booster
            _feature_names_v4 = names
            if os.path.exists(_CALIB_PATH_V4):
                try:
                    with open(_CALIB_PATH_V4, "rb") as f:
                        obj = pickle.load(f)
                    if isinstance(obj, dict) and "calibrator" in obj:
                        cal_set = obj.get("feature_set")
                        if cal_set is not None and cal_set != "v4":
                            logger.warning(
                                "xgb_signal: v4 calibrator feature_set=%s != 'v4' — skipping",
                                cal_set,
                            )
                            _calibration_v4 = None
                        else:
                            _calibration_v4 = obj["calibrator"]
                            logger.info("xgb_signal: loaded v4 isotonic calibrator")
                    else:
                        logger.warning(
                            "xgb_signal: v4 calibrator pkl is not dict-shape — skipping"
                        )
                        _calibration_v4 = None
                except Exception as exc:
                    logger.exception("xgb_signal: v4 calibrator load failed: %s", exc)
                    _calibration_v4 = None
            else:
                logger.info("xgb_signal: no v4 calibrator at %s — raw passthrough",
                            _CALIB_PATH_V4)
            _load_succeeded_v4 = True
            logger.info("xgb_signal: loaded v4 booster (%d features)", len(names))
            return True
        except Exception as exc:
            logger.exception("xgb_signal: v4 load failed: %s", exc)
            return False


def xgb_prob_v4(channels, pid: Optional[str] = None) -> float:
    """v4 booster probability in [0.01, 0.99]. Neutral 0.5 if artifacts missing or pid None.

    Mirrors xgb_prob() but always uses feature_set='v4' and requires pid for
    tiered_history fetch. Used by xgb_prob_shadow; not called directly by
    cnn_agent during the shadow week.
    """
    if not _try_load_v4():
        return _NEUTRAL
    if pid is None:
        logger.warning("xgb_signal: v4 requires pid, got None — returning neutral")
        return _NEUTRAL
    try:
        import xgboost as xgb
        from services.tiered_history import fetch_tiered
        from tools.xgb_v4_features import extract_v4

        tiers = fetch_tiered(pid, source="live")
        features, _ = extract_v4(tiers)
        dmat = xgb.DMatrix(features, feature_names=_feature_names_v4)
        raw = float(_booster_v4.predict(dmat)[0])
        if _calibration_v4 is not None:
            raw = float(_calibration_v4.transform(np.asarray([raw]))[0])
        return float(np.clip(raw, 0.01, 0.99))
    except Exception as exc:
        logger.exception("xgb_signal.xgb_prob_v4 failed, returning neutral: %s", exc)
        return _NEUTRAL


def xgb_prob_shadow(
    channels, pid: Optional[str] = None,
) -> Tuple[float, Optional[float]]:
    """Return (v3_prob, v4_prob) with v4 fully isolated from v3.

    v3 path runs normally (its own exception handling). v4 wrapped in
    try/except: any failure -> v4=None + log, NEVER affects v3. This is the
    only function cnn_agent should call during the shadow week.
    """
    prob_v3 = xgb_prob(channels, pid=pid)
    try:
        prob_v4 = xgb_prob_v4(channels, pid=pid)
    except Exception as exc:
        logger.exception("xgb_signal.xgb_prob_shadow: v4 path raised (isolated): %s", exc)
        prob_v4 = None
    return prob_v3, prob_v4


def force_reload() -> bool:
    """Drop cached booster + calibrator + feature names and re-read from disk.

    Use after refitting `xgb_calibration.pkl` (#187) or swapping the booster
    artifacts so the running backend picks up the new files without a process
    restart. Returns the new load_succeeded so callers can branch on whether
    artifacts are present.
    """
    global _booster, _feature_names, _feature_set, _calibration
    global _load_attempted, _load_succeeded
    with _lock:
        _booster = None
        _feature_names = []
        _feature_set = _FEATURE_SET_DEFAULT
        _calibration = None
        _load_attempted = False
        _load_succeeded = False
    return _try_load()


def xgb_prob(channels: ChannelsLike, pid: Optional[str] = None) -> float:
    """XGBoost probability of upward target, in [0.01, 0.99].

    Returns 0.5 if the model artifacts are missing or fail to load — the
    caller (`_cnn_prob` branch on `config.model_backend`) should treat
    that as "no signal" rather than error.

    pid: required when feature_set == "v3" so tiered_history can be fetched.
         Ignored for v1/v2. When None under v3, returns neutral with warning.
    """
    if not _try_load():
        return _NEUTRAL
    if _feature_set == "v3":
        if pid is None:
            logger.warning(
                "xgb_signal: v3 booster requires pid, got None — returning neutral",
            )
            return _NEUTRAL
        try:
            import xgboost as xgb
            from services.tiered_history import fetch_tiered
            from tools.xgb_features import extract_features as _extract

            tiers = fetch_tiered(pid, source="live")
            features, _ = _extract(tiers, feature_set="v3")
            dmat = xgb.DMatrix(features, feature_names=_feature_names)
            raw = float(_booster.predict(dmat)[0])
            if _calibration is not None:
                raw = float(_calibration.transform(np.asarray([raw]))[0])
            return float(np.clip(raw, 0.01, 0.99))
        except Exception as exc:
            logger.exception(
                "xgb_signal.xgb_prob v3 path failed, returning neutral: %s", exc,
            )
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
