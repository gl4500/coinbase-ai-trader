"""Large-move check for a spot market. Pure: no I/O. A heads-up to review, not a trade signal."""

from __future__ import annotations

import math
from typing import Optional


def move_alerts(
    product_id: str,
    now_px: Optional[float],
    px_1h: Optional[float],
    px_24h: Optional[float],
    th_1h: float = 0.05,
    th_24h: float = 0.10,
) -> list:
    out = []
    for kind, ref, th in (("move_1h", px_1h, th_1h), ("move_24h", px_24h, th_24h)):
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (now_px, ref)
        ):
            continue
        change = now_px / ref - 1
        if abs(change) >= th:
            out.append({"kind": kind, "product_id": product_id, "change": change, "threshold": th})
    return out
