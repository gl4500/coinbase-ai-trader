"""
HMM Regime Detector — 3-state Gaussian HMM on hourly log-returns + vol.

States (labeled by post-fit characterisation):
  0 = TRENDING     — persistent directional move, momentum strategies work
  1 = RANGING      — mean-reverting, oscillators work, low vol
  2 = CHAOTIC      — high vol, unpredictable, reduce size or sit out

Replaces the binary ADX ≥ 25 gate in cnn_agent with probabilistic regime
probabilities, enabling finer CNN/LLM blend ratios and Kelly scaling.

Minimum 60 candles required; falls back to ADX-based heuristic if not fitted.
"""
import logging
import math
import pickle
import os
from typing import List, Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)

_N_STATES   = 3
_MIN_BARS   = 60
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hmm_model.pkl")

# Regime label names indexed by state id (assigned after fitting by vol ranking)
REGIME_NAMES = {0: "TRENDING", 1: "RANGING", 2: "CHAOTIC"}


def _build_obs(closes: List[float], window: int = 10) -> Optional[np.ndarray]:
    """
    Build observation matrix: [log_return, rolling_vol].
    Rolling vol = std of last `window` log returns.
    """
    if len(closes) < window + 2:
        return None
    log_ret = [math.log(closes[i] / closes[i - 1])
               for i in range(1, len(closes))
               if closes[i - 1] > 0 and closes[i] > 0]
    if len(log_ret) < window + 1:
        return None
    rows = []
    for i in range(window, len(log_ret)):
        win = log_ret[i - window: i]
        mean = sum(win) / window
        vol  = math.sqrt(sum((r - mean) ** 2 for r in win) / window)
        rows.append([log_ret[i], vol])
    return np.array(rows, dtype=np.float64)


def _label_states(model) -> Dict[int, str]:
    """
    Assign semantic labels by ranking state means on the volatility dimension.
    Lowest vol → RANGING, highest vol → CHAOTIC, middle → TRENDING.
    """
    vols = model.means_[:, 1]   # volatility is column 1
    order = np.argsort(vols)    # ascending
    return {
        int(order[0]): "RANGING",
        int(order[1]): "TRENDING",
        int(order[2]): "CHAOTIC",
    }


class HMMRegimeDetector:
    def __init__(self):
        self._model = None
        self._state_labels: Dict[int, str] = {}
        self._load()

    def _load(self):
        if not os.path.exists(_MODEL_PATH):
            return
        try:
            with open(_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self._model       = data["model"]
            self._state_labels = data["labels"]
            logger.info("HMMRegime loaded from %s", _MODEL_PATH)
        except Exception as e:
            logger.warning("HMMRegime load failed: %s", e)

    def _save(self):
        try:
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            with open(_MODEL_PATH, "wb") as f:
                pickle.dump({"model": self._model, "labels": self._state_labels}, f)
        except Exception as e:
            logger.warning("HMMRegime save failed: %s", e)

    def fit(self, closes: List[float]) -> bool:
        """
        Fit HMM on the given close series.  Tries progressively smaller windows
        on failure so a bad recent data patch doesn't permanently break the detector.
        Returns True if a valid model was fitted and saved.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed — pip install hmmlearn")
            return False

        # Try full history first, then fall back to shorter windows
        for window in (len(closes), 500, 200):
            sub = closes[-window:] if len(closes) > window else closes
            obs = _build_obs(sub)
            if obs is None or len(obs) < _MIN_BARS:
                continue
            try:
                candidate = GaussianHMM(
                    n_components=_N_STATES,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                )
                candidate.fit(obs)

                # ── Validate: states must be sufficiently distinct ────────────
                vols = candidate.means_[:, 1]
                if max(vols) - min(vols) < 1e-6:
                    logger.warning(
                        "HMM fit produced degenerate states (all similar vol, window=%d) "
                        "— trying smaller window", window
                    )
                    continue

                new_labels = _label_states(candidate)

                # Warn if labels flipped vs previous fit (blend weights will change)
                if self._state_labels and new_labels != self._state_labels:
                    logger.warning(
                        "HMM state label mapping changed after refit: %s → %s "
                        "(CNN/LLM blend weights updated accordingly)",
                        self._state_labels, new_labels,
                    )

                self._model        = candidate
                self._state_labels = new_labels
                self._save()
                logger.info(
                    "HMMRegime fitted on %d bars (window=%d) | labels=%s",
                    len(obs), window, self._state_labels,
                )
                return True

            except Exception as e:
                logger.warning("HMM fit failed (window=%d): %s", window, e)

        # All windows failed — keep existing model (don't clear it)
        logger.warning(
            "HMM fit failed for all window sizes — keeping %s",
            "existing model" if self._model is not None else "unfitted state (ADX fallback active)",
        )
        return False

    def predict(self, closes: List[float]) -> Tuple[str, float, int]:
        """
        Predict current regime from recent closes.
        Returns (regime_name, confidence, raw_state_id).
        Falls back to ADX-heuristic string if model not fitted.
        """
        if self._model is None:
            return "UNKNOWN", 0.5, -1

        obs = _build_obs(closes)
        if obs is None or len(obs) < 2:
            return "UNKNOWN", 0.5, -1

        try:
            log_prob, state_seq = self._model.decode(obs, algorithm="viterbi")
            current_state = int(state_seq[-1])
            # Get posterior probability for the current bar
            posteriors = self._model.predict_proba(obs)
            confidence = float(posteriors[-1, current_state])
            regime     = self._state_labels.get(current_state, "UNKNOWN")
            return regime, confidence, current_state
        except Exception as e:
            logger.debug("HMM predict failed: %s", e)
            return "UNKNOWN", 0.5, -1

    def is_ready(self) -> bool:
        return self._model is not None


# Module-level singleton — shared across all CNN scans
_detector: Optional[HMMRegimeDetector] = None


def get_detector() -> HMMRegimeDetector:
    global _detector
    if _detector is None:
        _detector = HMMRegimeDetector()
    return _detector


def regime_blend(regime: str, confidence: float) -> Tuple[float, float]:
    """
    Returns (cnn_weight, llm_weight) blend based on HMM regime.
    TRENDING:  CNN 75% / LLM 25%  (momentum signal reliable)
    RANGING:   CNN 55% / LLM 45%  (LLM context useful but CNN keeps majority)
    CHAOTIC:   CNN 40% / LLM 60%  (model less reliable in chaos)
    UNKNOWN:   CNN 60% / LLM 40%  (neutral fallback, favour model)
    Confidence scales blend toward 50/50 when low.
    """
    base = {
        "TRENDING": (0.75, 0.25),
        "RANGING":  (0.55, 0.45),
        "CHAOTIC":  (0.40, 0.60),
        "UNKNOWN":  (0.60, 0.40),
    }.get(regime, (0.60, 0.40))

    # Pull toward 0.5/0.5 proportionally when confidence is low
    cnn = base[0] * confidence + 0.5 * (1 - confidence)
    llm = 1.0 - cnn
    return round(cnn, 3), round(llm, 3)
