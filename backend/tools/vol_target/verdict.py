"""Vol-target verdict: slow_trend's preregistered order, with the binary round-trip activity
gate replaced by per-sleeve weekly decision coverage (a partial reduction is not a round trip).
Re-stated rather than passing a decision count into slow_trend's `block_round_trips` argument."""

from tools.vol_target import prereg as P


def verdict(data_ok: bool, results, block_valid_decisions: dict) -> dict:
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
    if any(block_valid_decisions.get(p, 0) < P.MIN_BLOCK_VALID_DECISIONS for p in P.PRODUCTS):
        return {"verdict": "INCONCLUSIVE", "reason": "insufficient_coverage"}
    return {"verdict": "PASS_TO_FORWARD", "reason": "all_gates"}
