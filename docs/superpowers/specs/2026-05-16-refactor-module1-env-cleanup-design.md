# Refactor Sweep — Module 1: Dead Env-Var Cleanup

**Date:** 2026-05-16
**Status:** APPROVED 2026-05-16 (operator: "approved, continue" → "go" → "yes")
**Scope:** `backend/` + repo-root `.env` only
**Branch:** continue on `feat/gpu-coord-mirror`
**Sweep position:** Module 1 of N (module-by-module strategy, .env/config first)
**Predecessors:** XGB v3 cutover (#311-cut), MC CIFilter (#311-mc-sync)
**Successor:** Module 2 — likely `xgb_signal.py` legacy bare-isotonic path (operator picks order at next brainstorm)

---

## 1. Problem

The repo accumulated 60+ sessions of work. Some env vars are defined in `.env` and `config.py` but consumed by **nothing in `backend/`** — they were scaffolded for a future tuning path that never got wired. They pollute `.env`, mislead operators about what's tunable, and create a false impression that flipping them changes behavior.

Investigation (grep `config\.cnn_trending|cnn_ranging|cnn_train_every` across `backend/`):

| Env var | Config field | Live consumer |
|---|---|---|
| `CNN_TRENDING_CNN_W` | `config.cnn_trending_cnn_w` | **None.** `regime_blend()` in `services/hmm_regime.py:198` uses hardcoded constants. |
| `CNN_TRENDING_LLM_W` | `config.cnn_trending_llm_w` | **None.** Same. |
| `CNN_RANGING_CNN_W` | `config.cnn_ranging_cnn_w` | **None.** Same. |
| `CNN_RANGING_LLM_W` | `config.cnn_ranging_llm_w` | **None.** Same. |
| `CNN_TRAIN_EVERY_N_SCANS` | `config.cnn_train_every_n_scans` | `main.py:465` (passed to `run_loop`); auto-train gated off under `MODEL_BACKEND=xgb` per #300, but knob is still meaningful when CNN backend is active |

## 2. Goal

Delete dead-on-arrival env vars and config fields. Tag the legitimately-gated `CNN_TRAIN_EVERY_N_SCANS` so operators know when it matters. Establish a policy comment that future env vars MUST trace to a live consumer.

This module establishes the cleanup precedent for the module-by-module refactor sweep (#311-refactor). Future modules tackle larger surface area (xgb_signal legacy paths, cnn_agent dead branches, deprecated CNN_ARCH variants, probe scripts, etc.) — each gets its own spec + plan.

## 3. Non-goals

- No changes to `regime_blend()` weights (they're hardcoded and that's fine — this module is about *config*, not the regime function itself).
- No changes to `main.py`, `cnn_agent.py`, `xgb_signal.py`, or any other module.
- No schema/DB changes.
- No test infrastructure changes.
- No deletion of `CNN_TRAIN_EVERY_N_SCANS` — operator chose to keep + tag it.

## 4. Approach

Single atomic commit on `feat/gpu-coord-mirror`. Pure deletion of dead code + one new regression test that locks in the policy + docstring/comment additions for the kept knob.

### 4.1 Files touched

| Path | Action | Diff size |
|---|---|---|
| `backend/config.py:60-63` | DELETE 4 fields | ~6 lines removed |
| `backend/config.py` (module docstring) | EDIT — add policy line | ~2 lines added |
| `.env` | DELETE 4 keys; ADD policy comment block at top; ADD tag comment above `CNN_TRAIN_EVERY_N_SCANS` | ~5 lines deleted, ~4 added |
| `backend/services/hmm_regime.py:198-217` | EDIT — `regime_blend()` docstring now explicitly notes weights are hardcoded (was misleading) | docstring-only |
| `backend/tests/test_config.py` | NEW | ~15 LOC, 4 assertions in 1 test |
| `backend/CHANGELOG.md` | APPEND — Session 58.71a entry | new entry |
| `polymarket_app/CLAUDE.md` | EDIT — add invariant #15 | +3 lines |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | EDIT — add Refactor Sweep entry | +5 lines (outside repo, no commit) |

### 4.2 Code changes

**`backend/config.py:60-65`** — before:

```python
    cnn_trending_cnn_w:     float = field(default_factory=lambda: float(os.getenv("CNN_TRENDING_CNN_W",     "0.75")))
    cnn_trending_llm_w:     float = field(default_factory=lambda: float(os.getenv("CNN_TRENDING_LLM_W",     "0.25")))
    cnn_ranging_cnn_w:      float = field(default_factory=lambda: float(os.getenv("CNN_RANGING_CNN_W",      "0.40")))
    cnn_ranging_llm_w:      float = field(default_factory=lambda: float(os.getenv("CNN_RANGING_LLM_W",      "0.60")))

    cnn_train_every_n_scans: int  = field(default_factory=lambda: int(os.getenv("CNN_TRAIN_EVERY_N_SCANS",  "8")))
```

After:

```python
    # NOTE: Active only under MODEL_BACKEND=cnn. Auto-train is gated off
    # under MODEL_BACKEND=xgb per #300; flipping this knob has no effect
    # while xgb backend is live.
    cnn_train_every_n_scans: int  = field(default_factory=lambda: int(os.getenv("CNN_TRAIN_EVERY_N_SCANS",  "8")))
```

**`backend/config.py` module docstring** — add at the top of the file (after existing docstring if present, or as a new one):

```python
"""Backend config. All env vars defined here MUST trace to a live consumer
in backend/. Dead entries are deleted on sight per the refactor sweep policy
(#311-refactor)."""
```

**`.env`** — remove these four lines:

```
CNN_TRENDING_CNN_W=0.75
CNN_TRENDING_LLM_W=0.25
CNN_RANGING_CNN_W=0.40
CNN_RANGING_LLM_W=0.60
```

Add the tag comment above the kept knob (locate `CNN_TRAIN_EVERY_N_SCANS=` line):

```
# Active only under MODEL_BACKEND=cnn (auto-train gated off otherwise, #300)
CNN_TRAIN_EVERY_N_SCANS=4
```

Add policy comment block at the very top of `.env`:

```
# All env vars in this file MUST trace to a live consumer in backend/.
# Dead entries are deleted on sight per the refactor sweep policy (#311-refactor).
```

**`backend/services/hmm_regime.py:198-217`** — `regime_blend()` docstring update. Replace:

```python
def regime_blend(regime: str, confidence: float) -> Tuple[float, float]:
    """
    Returns (cnn_weight, llm_weight) blend based on HMM regime.
    TRENDING:  CNN 75% / LLM 25%  (momentum signal reliable)
    RANGING:   CNN 55% / LLM 45%  (LLM context useful but CNN keeps majority)
    CHAOTIC:   CNN 40% / LLM 60%  (model less reliable in chaos)
    UNKNOWN:   CNN 60% / LLM 40%  (neutral fallback, favour model)
    Confidence scales blend toward 50/50 when low.
    """
```

With:

```python
def regime_blend(regime: str, confidence: float) -> Tuple[float, float]:
    """
    Returns (cnn_weight, llm_weight) blend based on HMM regime.
    Weights are HARDCODED below; they are NOT config-driven despite the
    historical CNN_*_CNN_W / CNN_*_LLM_W env vars (deleted #311-refactor-a).

    TRENDING:  CNN 75% / LLM 25%  (momentum signal reliable)
    RANGING:   CNN 55% / LLM 45%  (LLM context useful but CNN keeps majority)
    CHAOTIC:   CNN 40% / LLM 60%  (model less reliable in chaos)
    UNKNOWN:   CNN 60% / LLM 40%  (neutral fallback, favour model)
    Confidence scales blend toward 50/50 when low.
    """
```

### 4.3 New test

`backend/tests/test_config.py` (NEW):

```python
"""Policy test for refactor sweep (#311-refactor-a).

Locks in: any env var defined in config.py MUST trace to a live consumer.
The four CNN_*_CNN_W / CNN_*_LLM_W env vars were dead-on-arrival (never
read anywhere in backend/) and were deleted. If anyone re-adds them
without a live consumer, this test fails."""
import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestDeadBlendFieldsStayDeleted:
    def test_no_dead_llm_blend_fields(self):
        from config import config
        dead = (
            "cnn_trending_cnn_w",
            "cnn_trending_llm_w",
            "cnn_ranging_cnn_w",
            "cnn_ranging_llm_w",
        )
        for name in dead:
            assert not hasattr(config, name), (
                f"config.{name} was deleted #311-refactor-a — re-adding it requires "
                f"a live consumer in backend/ first."
            )
```

## 5. Architecture

No architectural change. This module is pure code deletion + one policy test + docstring fixes.

The `regime_blend()` function continues to use its hardcoded weights exactly as today. `main.py` continues to read `config.cnn_train_every_n_scans` exactly as today. `cnn_agent.generate_signal` is untouched.

## 6. Data flow

**Before:**

```
.env CNN_TRENDING_CNN_W=0.75
    → os.getenv read at module load
    → config.cnn_trending_cnn_w = 0.75
    → (read by nothing)

.env CNN_TRAIN_EVERY_N_SCANS=4
    → config.cnn_train_every_n_scans = 4
    → main.py:465 passes to cnn_agent.run_loop
    → run_loop._maybe_auto_train (gated off if MODEL_BACKEND=xgb per #300)
```

**After:**

```
(env var deleted)
    → (field deleted)
    → N/A

.env CNN_TRAIN_EVERY_N_SCANS=4  (unchanged, now with tag comment)
    → config.cnn_train_every_n_scans = 4   (unchanged)
    → main.py:465 → cnn_agent.run_loop      (unchanged)
```

Zero live-behavior change. The deleted fields were never on a hot path.

## 7. Error handling

| Condition | Behavior |
|---|---|
| User leaves stale env var in their local `.env` after pulling this commit | `os.getenv()` is no longer called for them — the env var is silently ignored. No error. Operator just has unused lines in their `.env`. |
| User pre-imports `config.cnn_trending_cnn_w` somewhere we missed | `AttributeError` at import time. Caught at test run by the new `test_no_dead_llm_blend_fields`. |
| Future PR re-adds `cnn_trending_cnn_w` without a live consumer | `test_no_dead_llm_blend_fields` fails in pre-commit hook. |

## 8. Tests

| File | Status | Tests | Coverage |
|---|---|---:|---|
| `backend/tests/test_config.py` | NEW | 1 (4 assertions) | dead-field policy lock |
| **Total new** | | **1** | |

Plus: the existing 1100+ test suite MUST stay green (pre-commit hook enforces).

What we don't test:
- `.env` file contents (not a Python testable surface; the policy comment is documentation, not behavior).
- `regime_blend()` weights (already covered by existing tests in `test_hmm_regime.py`).

## 9. Rollout

### Phase 0 — Atomic commit
Single commit on `feat/gpu-coord-mirror`:

```
git add backend/config.py backend/services/hmm_regime.py \
        backend/tests/test_config.py .env CHANGELOG.md CLAUDE.md
git commit -m "refactor(#311-refactor-a): delete dead LLM-blend env vars..."
```

Pre-commit hook runs full suite (~5 min). On green, commit lands.

### Phase 1 — Push
`git push` to origin (per standing feedback rule).

### Phase 2 — Verification (operator)
No backend restart needed — these fields weren't consumed, so process state isn't affected.

```bash
.venv/Scripts/python.exe -c "from backend.config import config; print(hasattr(config, 'cnn_trending_cnn_w'))"
# Expected: False
```

### Rollback
`git revert <commit>` if anything unexpected surfaces. Pure code deletion → zero state side-effects.

## 10. Memory + CLAUDE.md sync (per CLAUDE.md rule)

Bundled into the same commit as the code changes (small scope, no separate docs commit needed).

- `CHANGELOG.md` — Session 58.71a entry (in same commit)
- `polymarket_app/CLAUDE.md` — new invariant #15: *Env vars MUST trace to a live consumer. Dead entries are deleted on sight per refactor policy (#311-refactor).*
- `memory/coinbase_trader_architecture.md` (outside repo) — new "Refactor Sweep" section with Module 1 logged + the dead-env-var policy.

## 11. Open questions

None — operator approved every clarifying question on 2026-05-16.

## 12. References

- HTML overview of accumulated cruft: discussed inline in Task #19 description
- v3 cutover commit: `9a5e084` (`fix(#311j) + ops(#311-cut)`)
- MC chain wire-up: `f349125` (`feat(#311-mc-wire)`)
- CNN auto-train backend gate: `#300` (`fix: gate CNN auto-train behind MODEL_BACKEND=='cnn'`, commit `954b8b8`)
- CLAUDE.md sync rule: `polymarket_app/CLAUDE.md` "CLAUDE.md ↔ Memory Sync Rule" section
