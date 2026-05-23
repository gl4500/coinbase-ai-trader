"""ABC for Monte Carlo decision filters (#311-mc).

Each filter wraps one decision point in cnn_agent. Filters live under
agents/mc/<name>_filter.py, expose a class with .name and .evaluate(...),
and are listed by name in the MC_FILTERS env var (comma-separated).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BuyFilter(ABC):
    """Filter invoked at the BUY gate. May change the side and adds telemetry.

    Contract:
        evaluate(side, model_prob, pid, channels, context) -> (new_side, telemetry)
        - `side` is "BUY" when this is invoked (registry skips non-BUY).
        - return value's first slot is the post-filter side ("BUY" or "HOLD").
          Filters should never up-grade HOLD to BUY in MVP scope.
        - second slot is a dict keyed by self.name with arbitrary serializable
          telemetry; gets merged into the chain-level telemetry dict.
    """

    name: str = ""

    @abstractmethod
    def evaluate(
        self,
        side: str,
        model_prob: float,
        pid: str,
        channels: List[List[float]],
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError
