# Remove CNN Driver + Add XGB v4.5 Driver Path — Design Spec

**Date:** 2026-05-23
**Topic:** CNN paper-trading deprecation cleanup + v4.5 driver path
**Status:** Design approved, plan pending

---

## Goal

Remove the CNN model as a viable trade driver. Add XGB v4.5 as a driver path selectable via `MODEL_BACKEND=xgb_v45`. Make v4.5 shadow telemetry log on every scan regardless of which driver is active.

## Why

- Operator declared CNN deprecated on 2026-05-18 (`feedback_xgb_focus_not_cnn.md`). The driver branch in `_cnn_prob` and the `cnn_model_glu1.pt` checkpoint load remained as dead-but-loadable code.
- The dev backend on port 8002 currently runs `MODEL_BACKEND=cnn` solely because that is the only value that activates v4.5 shadow logging (gate at `cnn_agent.py:1899` reads `if model_backend == "xgb": skip v4.5`). This forces CNN to drive paper trades on 8002 — directly contradicting the "XGB only" directive.
- With CNN as the 8002 driver, v4.5 paper-ROI is unmeasurable. Per `feedback_roi_first_priority.md`, ROI is the operator's #1 metric. Promoting v4.5 to driver on 8002 enables direct ROI measurement against `coinbase_dev.db.trades`.

## Scope

**IN:**
- Delete CNN driver branch + checkpoint loading + Glu1 class + `_linear` fallback
- Add `MODEL_BACKEND=xgb_v45` value with v4.5-driven decision via `indep_thresholds`
- Make v4.5 shadow logging always-on regardless of `MODEL_BACKEND`
- Test suite updates (delete stale CNN tests, add new driver/shadow/migration tests)
- Memory + CHANGELOG + architecture doc updates
- Dev backend relaunch on 8002 with `MODEL_BACKEND=xgb_v45` (operator action, post-deploy)

**OUT (deferred to backlog task #7, gated on shadow-week success):**
- Class rename `CoinbaseCNNAgent` → XGB-aware name
- DB column rename `cnn_scans` / `cnn_w` / `llm_w`
- Frontend file rename `CNNDashboard.tsx`
- On-disk artifact cleanup of `cnn_model_glu1.pt` (host-side, operator's call)

## Architecture

### Config (`backend/config.py`)

```python
# Valid values: "xgb" (v3 driver) | "xgb_v45" (v4.5 driver). Default flipped
# from legacy "cnn" to "xgb" on 2026-05-23. Legacy "cnn" raises at startup.
model_backend: str = field(
    default_factory=lambda: _validate_backend(
        os.getenv("MODEL_BACKEND", "xgb").lower()
    )
)
```

New helper at module scope:

```python
_VALID_BACKENDS = {"xgb", "xgb_v45"}

def _validate_backend(value: str) -> str:
    if value == "cnn":
        raise ValueError(
            "MODEL_BACKEND=cnn is deprecated as of 2026-05-23. "
            "Use MODEL_BACKEND=xgb (default, v3 driver) or "
            "MODEL_BACKEND=xgb_v45 (v4.5 driver). See "
            "docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md"
        )
    if value not in _VALID_BACKENDS:
        raise ValueError(
            f"MODEL_BACKEND={value!r} invalid. Valid: {sorted(_VALID_BACKENDS)}"
        )
    return value
```

New thresholds:

```python
# v4.5 indep_thresholds decision rule (matches tools/v4_5_horizon_compare.py
# winning rule). BUY when p_up > thresh_up AND p_up >= p_down. SELL when
# p_down > thresh_down AND p_down > p_up. Else HOLD.
xgb_v45_thresh_up:   float = field(default_factory=lambda: float(os.getenv("XGB_V45_THRESH_UP",   "0.50")))
xgb_v45_thresh_down: float = field(default_factory=lambda: float(os.getenv("XGB_V45_THRESH_DOWN", "0.50")))
```

### Driver routing (`backend/agents/cnn_agent.py`)

**Delete** the CNN branch in `_cnn_prob` (lines ~1619-1620):

```python
# DELETE these two lines:
if _TORCH and self.model:
    return self.model.predict(self.fb.to_tensor(channels))
```

**Delete** the `_linear` fallback (lines ~1624-1638). It existed only as the CNN-side last-resort and is dead under all valid `MODEL_BACKEND` values after this change.

After the changes, `_cnn_prob` becomes:

```python
def _cnn_prob(self, channels, pid: Optional[str] = None) -> float:
    channels = _mask_training_constant_channels(channels)
    if config.model_backend in ("xgb", "xgb_v45"):
        from agents import xgb_signal
        return xgb_signal.xgb_prob(channels, pid=pid)
    # Unreachable under _validate_backend, but defensive:
    raise RuntimeError(f"unsupported model_backend={config.model_backend!r}")
```

Note: under `xgb_v45`, `_cnn_prob` still returns the **v3 prob** (single float). This preserves invariant #13 (v3 inference plumbing) and keeps `_cnn_prob` callers backward-compatible. v4.5 driver decision happens in `generate_signal`, not in `_cnn_prob`.

### Shadow gate flip (`cnn_agent.generate_signal` line ~1899)

Replace the current branch:

```python
# BEFORE:
if config.model_backend == "xgb":
    xgb_shadow = cnn_prob
    xgb_shadow_v45 = None
else:
    xgb_shadow, xgb_shadow_v45 = _xgb.xgb_prob_shadow_v4_5(...)
```

With unconditional shadow logging:

```python
# AFTER:
try:
    from agents import xgb_signal as _xgb
    xgb_shadow, xgb_shadow_v45 = _xgb.xgb_prob_shadow_v4_5(
        _mask_training_constant_channels(channels),
        pid=pid,
    )
except Exception:
    xgb_shadow      = None
    xgb_shadow_v45  = None
```

The `xgb_prob_shadow_v4_5` function already returns `(v3_prob, (p_down, p_neutral, p_up) or None)` with isolated v4.5 try/except (invariant #16, #17). Reusing it here is safe.

### v4.5 driver decision (`cnn_agent.generate_signal`)

Add a branch in the side-decision section (currently at line ~1964-1973). Existing 2-class gate stays for `xgb`. New 3-class gate for `xgb_v45`:

```python
if config.model_backend == "xgb_v45":
    if xgb_shadow_v45 is None:
        # v4.5 inference failed — HOLD to avoid trading on stale/garbage signal.
        side, strength = "HOLD", 0.0
    else:
        p_down, p_neutral, p_up = xgb_shadow_v45
        side, strength = _indep_thresholds_decision(
            p_down, p_neutral, p_up,
            thresh_up   = config.xgb_v45_thresh_up,
            thresh_down = config.xgb_v45_thresh_down,
        )
else:
    # Existing xgb (v3) 2-class gate — unchanged.
    if model_prob > config.cnn_buy_threshold:
        side     = "BUY"
        strength = round((model_prob - 0.5) * 2, 3)
    elif model_prob < config.cnn_sell_threshold:
        side     = "SELL"
        strength = round((0.5 - model_prob) * 2, 3)
    else:
        side     = "HOLD"
        strength = 0.0
```

New helper at module scope (in `cnn_agent.py`, near other private helpers):

```python
def _indep_thresholds_decision(
    p_down: float, p_neutral: float, p_up: float,
    thresh_up: float, thresh_down: float,
) -> Tuple[str, float]:
    """v4.5 indep_thresholds rule. Mirrors tools/v4_5_horizon_compare.py:138.

    BUY  when p_up   > thresh_up   AND p_up   >= p_down (tie -> BUY).
    SELL when p_down > thresh_down AND p_down >  p_up  (strict).
    Else HOLD.

    Returns (side, strength) where strength is the winning class prob.
    """
    if p_up > thresh_up and p_up >= p_down:
        return "BUY", round(p_up, 3)
    if p_down > thresh_down and p_down > p_up:
        return "SELL", round(p_down, 3)
    return "HOLD", 0.0
```

Asymmetric tie-handling (`>=` on BUY, `>` on SELL) mirrors horizon_compare exactly — preserves the rule's measured AUC behavior.

### Removals (full inventory)

`backend/agents/cnn_agent.py`:
- DELETE the `if _TORCH and self.model:` branch in `_cnn_prob`
- DELETE `_linear` method
- KEEP `SignalCNNGlu1` class — still referenced by `_build_cnn` and indirectly by training code (`self.model`-typed in __init__, `save_model`, `_load`). Scheduled for full removal in backlog task #7 alongside the training-infrastructure cleanup.
- KEEP `_build_cnn()` factory — same reason.
- KEEP `_TORCH` import flag — still gates `if _TORCH:` blocks in training-related code paths (`__init__`, `_load`, `save_model`, `train_on_history`). Removing it requires the broader training-infra rip-out in task #7.
- KEEP `self.model` + `self.fb` field init in `__init__` — `self.fb` (FeatureBuilder) is shared infrastructure for XGB inference (`channels = self.fb.build(...)` in `generate_signal`). `self.model` is unused at inference under XGB but kept for training-code back-compat.

**Scope note (revised during T6 execution):** The original spec listed Glu1/_build_cnn/_TORCH/checkpoint-load deletions as in-scope. During implementation it became clear that training code (`train_on_history`, `_load`, `save_model`) still references all of these. Deleting them breaks training-infrastructure entry points that are dead-but-callable. Per CLAUDE.md "Don't add features, refactoring, or abstractions beyond the task scope" + [[feedback_xgb_focus_not_cnn]] which scheduled this for backlog #7, the broader removal is deferred. This commit scopes to: CNN trade-driver path only.

`backend/tests/test_cnn_agent.py`:
- DELETE any `TestSignalCNNGlu1` class (forward pass / param count / shape tests)
- DELETE any test that sets `MODEL_BACKEND=cnn` as a driver — i.e. tests that assert CNN drives `generate_signal`. Tests that set `MODEL_BACKEND=cnn` to assert post-deprecation **ValueError** are NEW (see below).
- KEEP tests that exercise `_cnn_prob` under `MODEL_BACKEND=xgb` (still the v3 path).

`backend/config.py`:
- Flip `MODEL_BACKEND` default from `"cnn"` to `"xgb"`.
- Add `_validate_backend` + `_VALID_BACKENDS`.
- Add `xgb_v45_thresh_up`, `xgb_v45_thresh_down` fields.

`.env`:
- If `MODEL_BACKEND=cnn` exists, change to `MODEL_BACKEND=xgb` (or delete to use default). Operator does this on the live 8001 host.

## Test Plan (TDD, RED → GREEN → REFACTOR)

Each test added RED first, then implementation, then verify GREEN, then commit. Tests grouped by file:

### `tests/test_config.py`

1. **`test_model_backend_cnn_raises_value_error`** — instantiate `Config` with `MODEL_BACKEND=cnn` patched into env; assert `ValueError` with "deprecated" in message.
2. **`test_model_backend_xgb_v45_accepted`** — instantiate with `MODEL_BACKEND=xgb_v45`; assert `config.model_backend == "xgb_v45"`.
3. **`test_model_backend_unknown_raises`** — `MODEL_BACKEND=lstm` → `ValueError`.
4. **`test_xgb_v45_threshold_defaults`** — `xgb_v45_thresh_up == 0.50`, `xgb_v45_thresh_down == 0.50`.

### `tests/test_xgb_v45_decision.py` (NEW FILE)

5. **`test_indep_strong_up`** — `(0.10, 0.10, 0.80)` → `("BUY", 0.800)`.
6. **`test_indep_strong_down`** — `(0.80, 0.10, 0.10)` → `("SELL", 0.800)`.
7. **`test_indep_neutral_dominant`** — `(0.25, 0.50, 0.25)` → `("HOLD", 0.0)`.
8. **`test_indep_both_below_threshold`** — `(0.49, 0.02, 0.49)` → `("HOLD", 0.0)` (neither class strictly above 0.50).
9. **`test_indep_p_up_at_exact_threshold_holds`** — `(0.49, 0.01, 0.50)` → `("HOLD", 0.0)`. Verifies strict-greater rule (`p_up > 0.50`, not `>=`).
10. **`test_indep_buy_wins_tie_when_both_above_threshold`** — `(0.51, 0.00, 0.51)` → `("BUY", 0.510)`. Verifies asymmetric tie-handling: BUY rule's `p_up >= p_down` accepts tie, SELL rule's `p_down > p_up` does not.
11. **`test_indep_buy_when_p_up_marginally_exceeds`** — `(0.49, 0.00, 0.51)` → `("BUY", 0.510)`.
12. **`test_indep_sell_when_p_down_marginally_exceeds`** — `(0.51, 0.00, 0.49)` → `("SELL", 0.510)`.

### `tests/test_cnn_agent.py`

13. **`test_generate_signal_xgb_v45_driver_path`** — set `MODEL_BACKEND=xgb_v45`, mock `xgb_prob_shadow_v4_5` to return `(0.6_v3, (0.1, 0.1, 0.8))`. Assert `generate_signal` produces `side="BUY"`. Assert `save_cnn_scan` called with `xgb_prob_v4_5_up=0.8` (after renormalization).
14. **`test_generate_signal_xgb_v45_holds_on_v45_failure`** — set `MODEL_BACKEND=xgb_v45`, mock `xgb_prob_shadow_v4_5` to return `(0.6_v3, None)`. Assert `side="HOLD"`. v3 prob still persisted.
15. **`test_generate_signal_xgb_logs_v45_shadow`** — set `MODEL_BACKEND=xgb` (v3 driver), mock `xgb_prob_shadow_v4_5` to return `(0.6_v3, (0.1, 0.1, 0.8))`. Assert v3 drives decision, AND all 3 v4.5 probs are persisted to `save_cnn_scan` call (always-on shadow).
16. **`test_generate_signal_xgb_handles_v45_shadow_failure`** — mock `xgb_prob_shadow_v4_5` raising; assert `save_cnn_scan` called with `xgb_prob_v4_5_*=None` (all three NULL atomically per invariant #17). v3 driver decision unaffected.

### Deletions

17. Search `tests/test_cnn_agent.py` for any `MODEL_BACKEND=cnn` patches that expect CNN to drive — delete them (they will fail under the new ValueError anyway). Search for `SignalCNNGlu1` references in tests — delete.

## Commit shape

Single atomic commit per find-list-fix discipline. The changes are tightly coupled: removing the CNN driver branch in `_cnn_prob` without simultaneously updating the shadow gate would leave v4.5 unlogged when `MODEL_BACKEND=xgb`. So the commit covers:

- `backend/config.py`
- `backend/agents/cnn_agent.py`
- `backend/tests/test_config.py`
- `backend/tests/test_cnn_agent.py`
- `backend/tests/test_xgb_v45_decision.py` (new file)
- `CHANGELOG.md`
- `docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md` (this file)

Memory file updates (`feedback_xgb_focus_not_cnn.md`, `coinbase_trader_architecture.md`, `coinbase_trader_thresholds.md`) land in the same commit OR in an immediate follow-up commit per the CLAUDE.md ↔ Memory Sync Rule.

## Deployment sequence

1. Land the commit (8001 paused for the pytest window).
2. Operator restarts 8001 with `MODEL_BACKEND=xgb` (or just default — same effect).
3. Verify 8001 logs show v4.5 shadow probs being persisted (`SELECT xgb_prob_v4_5_up FROM cnn_scans ORDER BY id DESC LIMIT 5;`).
4. Kill the current 8002 process (PID 17492).
5. Relaunch 8002 with `PORT=8002 MODEL_BACKEND=xgb_v45 DATABASE_URL=coinbase_dev.db`.
6. Verify 8002 is now executing v4.5-driven paper trades (signals in `coinbase_dev.db.signals` table should match v4.5 BUY/SELL when `p_up > 0.50` or `p_down > 0.50`).

## Rollback

If post-deploy issues:

1. `git revert <commit-sha>` — single revert undoes everything.
2. Restart 8001 + relaunch 8002 with old envs.
3. Old `cnn_model_glu1.pt` still on disk → CNN driver path returns to working state under reverted code.

## Invariants preserved

- **#13** (v3 inference REQUIRES `pid` kwarg through `_cnn_prob → xgb_signal.xgb_prob(channels, pid=pid)`) — `_cnn_prob` still routes through `xgb_signal.xgb_prob` for both `xgb` and `xgb_v45` (the latter's v4.5 path happens elsewhere).
- **#16** (Shadow telemetry isolation) — `xgb_prob_shadow_v4_5` retained as the only shadow entry point. Always-on logging just calls it more often.
- **#17** (v4.5 3-class telemetry contract — atomic write or all-NULL) — preserved by reusing `xgb_prob_shadow_v4_5`'s existing isolation.

## Risks

- **Live 8001 disruption window** — pytest + commit + restart. Estimated 30-60 min of paused live trading. Operator-coordinated.
- **Tests touching `_linear`** — grep first; if `_linear` has test coverage, those tests get deleted alongside the method.
- **`_TORCH` import side-effects** — `import torch` is heavy. Removing it speeds up backend startup but may surface latent imports elsewhere that assumed torch was loaded. Test plan covers basic import smoke (`python -c "import agents.cnn_agent"`).
- **8002 dev backend during transition** — 8002 will be inoperative between the commit landing and the env relaunch. Acceptable since shadow-week telemetry on 8002 has been accumulating for hours and a 5-min relaunch gap is noise.
