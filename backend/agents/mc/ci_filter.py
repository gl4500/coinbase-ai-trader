"""Entry confidence-interval filter (#311-mc-ci).

Algorithm: take the cumulative-prediction trajectory across the v3 booster's
trees (cheap; ~7s per scan for 51 products), compute its stdev as a proxy
for ensemble uncertainty, and require the lower bound (point - K*stdev) to
exceed cnn_buy_threshold before allowing BUY.

K is configurable via MC_CI_K (default 1.0). Skips gracefully for non-v3
boosters, missing pid, or any predict failure — decision stays the caller's.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from agents.mc.base import BuyFilter

logger = logging.getLogger(__name__)


class CIFilter(BuyFilter):
    name = "ci"

    def __init__(self) -> None:
        try:
            self._K = float(os.getenv("MC_CI_K", "1.0"))
        except (TypeError, ValueError):
            self._K = 1.0

    def evaluate(
        self,
        side: str,
        model_prob: float,
        pid: str,
        channels: List[List[float]],
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        # Lazy imports — survives test monkeypatching of xgb_signal state.
        from agents import xgb_signal as xs

        if not getattr(xs, "_load_succeeded", False) or xs._booster is None:
            return side, {"ci": {"skipped": "booster-unavailable"}}
        if getattr(xs, "_feature_set", "v1") != "v3":
            return side, {"ci": {"skipped": "non-v3-booster"}}
        if pid is None:
            return side, {"ci": {"skipped": "pid-none"}}

        try:
            import xgboost as xgb

            import config as cfg
            from services.tiered_history import fetch_tiered
            from tools.xgb_features import extract_features

            tiers = fetch_tiered(pid, source="live")
            features, _ = extract_features(tiers, feature_set="v3")
            dmat = xgb.DMatrix(features, feature_names=xs._feature_names)
            n = xs._booster.num_boosted_rounds()
            trajectory = [
                float(xs._booster.predict(dmat, iteration_range=(0, k + 1))[0]) for k in range(n)
            ]
            point = trajectory[-1]
            stdev = float(np.std(trajectory))
            lower = max(0.0, point - self._K * stdev)
            threshold = float(cfg.config.cnn_buy_threshold)
            decision = "keep" if lower > threshold else "block"
            new_side = side if decision == "keep" else "HOLD"
            tele = {
                "ci": {
                    "stdev": round(stdev, 6),
                    "lower": round(lower, 6),
                    "K": self._K,
                    "decision": decision,
                }
            }
            return new_side, tele
        except Exception as exc:
            logger.warning("CIFilter predict failed: %s", exc)
            return side, {"ci": {"skipped": "predict-error", "error": str(exc)}}


# Self-register with the registry on import.
try:
    from agents.mc.registry import _FILTER_CLASSES

    _FILTER_CLASSES["ci"] = CIFilter
except Exception:
    pass
