"""
LightGBM entry filter for CNN BUY decisions.

Trained on closed CNN trades (features from cnn_scans at entry time).
Gates future BUY signals: only execute when predicted win probability
>= LGBM_GATE_THRESHOLD.

Falls back to allow_buy=True when fewer than MIN_SAMPLES trades exist
so the CNN can still trade while accumulating training data.
"""
import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_SAMPLES        = 50    # Minimum closed trades needed before model activates
MIN_WINS           = 20    # Minimum wins required — avoids "block everything" when all-loss history
# Predicted win prob must exceed this to allow BUY.
# Overridable via env LGBM_GATE_THRESHOLD (e.g. 0.35 to unblock CNN while
# binary-label filter mis-scores asymmetric-pnl picks; see Task #39 root cause).
LGBM_GATE_THRESHOLD = float(os.getenv("LGBM_GATE_THRESHOLD", "0.52"))

_FEATURES = [
    "cnn_prob", "rsi", "adx", "strength", "macd",
    "mfi", "stoch_k", "hour_of_day", "day_of_week", "usd_open",
]

# ── Model ─────────────────────────────────────────────────────────────────────

class LGBMFilter:
    """
    Lightweight LightGBM binary classifier: predict P(win) for a CNN BUY.

    Usage:
        f = LGBMFilter()
        f.load("lgbm_filter.pkl")          # no-op if missing
        if f.allow_buy(features):
            book.buy(...)
        ...
        metrics = f.train(closed_trade_rows)  # retrain periodically
        f.save("lgbm_filter.pkl")
    """

    def __init__(self) -> None:
        self._model = None
        self._n_samples: int = 0
        self._n_wins:    int = 0
        # Seam for unit testing — override to inject fixed predictions
        self._model_predict = None

    # ── Public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True once trained on enough rows with meaningful win representation."""
        return (self._model is not None
                and self._n_samples >= MIN_SAMPLES
                and self._n_wins >= MIN_WINS)

    def train(self, rows: List[Dict]) -> Optional[Dict]:
        """
        Train on a list of closed-trade dicts.  Each dict must contain all
        _FEATURES keys plus a 'pnl' key (positive = win).

        Returns a metrics dict on success, None if too few samples.
        """
        if len(rows) < MIN_SAMPLES:
            _log.info("LGBMFilter: only %d rows — need %d to train", len(rows), MIN_SAMPLES)
            return None

        X, y = self._build_xy(rows)
        w = self._sample_weights(rows)

        # Need both classes to train a binary classifier
        if len(set(y.tolist())) < 2:
            _log.info(
                "LGBMFilter: only one class in %d rows (all %s) — need wins AND losses to train",
                len(rows), "wins" if y[0] == 1 else "losses",
            )
            return None

        try:
            import lightgbm as lgb
        except ImportError:
            _log.error("LGBMFilter: lightgbm not installed — pip install lightgbm")
            return None

        # Time-ordered split: last 20% as validation (no shuffle — respects time order)
        split = max(1, int(len(X) * 0.8))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        w_tr, w_val = w[:split], w[split:]
        # Disable validation if training split lacks both classes OR val has
        # labels not seen in training (causes LightGBM "unseen labels" error)
        if len(set(y_tr.tolist())) < 2 or not set(y_val.tolist()).issubset(set(y_tr.tolist())):
            X_val, y_val, w_val = X_tr[:0], y_tr[:0], w_tr[:0]  # empty — disables eval_set

        params = {
            "objective":       "binary",
            "metric":          "auc",
            "n_estimators":    100,
            "max_depth":       4,
            "min_child_samples": max(5, len(X_tr) // 20),
            "learning_rate":   0.05,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "reg_lambda":      2.0,
            "verbose":         -1,
        }

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)] if len(X_val) >= 5 else None,
                eval_sample_weight=[w_val] if len(X_val) >= 5 else None,
                callbacks=[lgb.early_stopping(10, verbose=False),
                           lgb.log_evaluation(-1)] if len(X_val) >= 5 else [lgb.log_evaluation(-1)])

        self._model     = clf
        self._n_samples = len(rows)
        self._n_wins    = int(y.sum())

        # Compute AUC on validation set (or full set if too small)
        eval_X, eval_y = (X_val, y_val) if len(X_val) >= 5 else (X, y)
        probs = clf.predict_proba(eval_X)[:, 1]
        auc = self._roc_auc(eval_y, probs)

        n_wins = self._n_wins
        metrics = {
            "n_samples": len(rows),
            "n_wins":    n_wins,
            "n_losses":  len(rows) - n_wins,
            "win_rate":  round(n_wins / len(rows) * 100, 1),
            "auc":       round(auc, 3),
        }
        _log.info("LGBMFilter trained: %s", metrics)
        return metrics

    def predict(self, features: Dict) -> float:
        """Return P(win) for a single feature dict. Returns 0.5 if not ready."""
        if not self.is_ready():
            return 0.5
        X = self._features_to_array(features)
        if self._model_predict is not None:
            return float(self._model_predict(X)[0])
        prob = self._model.predict_proba(X)[:, 1][0]
        return float(prob)

    def allow_buy(self, features: Dict) -> bool:
        """
        True if the model is not yet ready (pass-through) OR predicted
        win probability >= LGBM_GATE_THRESHOLD.
        """
        if not self.is_ready():
            _log.debug(
                "LGBMFilter pass-through (n=%d wins=%d — need %d/%d)",
                self._n_samples, self._n_wins, MIN_SAMPLES, MIN_WINS,
            )
            return True
        return self.predict(features) >= LGBM_GATE_THRESHOLD

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "n_samples": self._n_samples, "n_wins": self._n_wins}, f)
        _log.info("LGBMFilter saved → %s", path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            _log.info("LGBMFilter: no saved model at %s", path)
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._model     = data["model"]
            self._n_samples = data["n_samples"]
            self._n_wins    = data.get("n_wins", 0)
            _log.info("LGBMFilter loaded from %s (n=%d)", path, self._n_samples)
        except Exception as e:
            _log.warning("LGBMFilter load failed: %s", e)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_xy(self, rows: List[Dict]):
        X = np.array([[self._safe(r, k) for k in _FEATURES] for r in rows],
                     dtype=np.float32)
        y = np.array([1 if (r.get("pnl") or 0) > 0 else 0 for r in rows],
                     dtype=np.int32)
        return X, y

    @staticmethod
    def _sample_weights(rows: List[Dict]) -> np.ndarray:
        """
        Weight each training sample by |pnl| so large winners/losers dominate
        learning and near-zero noise trades contribute minimally. A tiny floor
        prevents LightGBM from silently dropping 0-weight rows.
        """
        w = np.array([abs(float(r.get("pnl") or 0.0)) for r in rows], dtype=np.float32)
        w = np.maximum(w, 1e-3)
        return w

    def _features_to_array(self, features: Dict):
        return np.array([[self._safe(features, k) for k in _FEATURES]],
                        dtype=np.float32)

    @staticmethod
    def _safe(d: Dict, key: str) -> float:
        v = d.get(key)
        return float(v) if v is not None and not (isinstance(v, float) and (v != v)) else 0.0

    @staticmethod
    def _roc_auc(y_true, y_score) -> float:
        """Simple trapezoid AUC — avoids sklearn dependency."""
        pairs = sorted(zip(y_score, y_true), reverse=True)
        tp = fp = 0
        tp_prev = fp_prev = 0
        auc = 0.0
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        for _, label in pairs:
            if label:
                tp += 1
            else:
                fp += 1
                auc += (tp + tp_prev) / 2 / n_pos
                tp_prev = tp
            fp_prev = fp
        return auc / n_neg if n_neg else 0.5
