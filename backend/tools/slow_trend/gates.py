"""Preregistered gates and verdict order. KILL abandons THIS candidate; it is not a general
falsification of trend following."""

from tools.slow_trend import prereg as P


def passes(trend: dict, bh: dict, initial: float, ratio: float) -> dict:
    g1 = trend["terminal_value"] > initial
    g2 = (
        trend["net_return"] >= bh["net_return"]
        or trend["max_drawdown"] <= ratio * bh["max_drawdown"]
    )
    return {"G1": bool(g1), "G2": bool(g2), "pass": bool(g1 and g2)}


def affected(trend: dict, bh: dict) -> bool:
    """A gating comparison is affected when a trend OR buy-and-hold sleeve ended holding units
    it could not sell: its terminal cash is a lower bound, not an economic value."""
    return bool(any(trend["unliquidatable"]) or any(bh["unliquidatable"]))


def verdict(data_ok: bool, results, block_round_trips: int) -> dict:
    if not data_ok:
        return {"verdict": "INCONCLUSIVE", "reason": "data"}
    periods = ("dev", "block")
    p0 = [results[p]["P0"] for p in periods]
    if any(not r["pass"] and not r["affected"] for r in p0):
        return {"verdict": "KILL", "reason": "primary_failed"}
    if any(r["affected"] for r in p0):
        return {"verdict": "INCONCLUSIVE", "reason": "terminal_size"}
    sens = [results[p][s] for p in periods for s in P.GATING_SENSITIVITIES]
    if any(r["affected"] for r in sens):
        return {"verdict": "INCONCLUSIVE", "reason": "terminal_size"}
    if not all(r["pass"] for r in sens):
        return {"verdict": "INCONCLUSIVE", "reason": "fragile"}
    if block_round_trips < P.MIN_BLOCK_ROUND_TRIPS:
        return {"verdict": "INCONCLUSIVE", "reason": "insufficient_transitions"}
    return {"verdict": "PASS_TO_FORWARD", "reason": "all_gates"}
