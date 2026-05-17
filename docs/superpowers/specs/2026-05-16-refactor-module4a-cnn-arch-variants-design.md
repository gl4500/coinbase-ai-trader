# Refactor Sweep — Module 4a: CNN_ARCH Dead Variants Cleanup

**Date:** 2026-05-16
**Status:** APPROVED 2026-05-16 (operator: "approved")
**Scope:** `backend/agents/cnn_agent.py` + `backend/tests/` + `.env` + host-side `retired/`
**Branch:** continue on `feat/gpu-coord-mirror`
**Sweep position:** Module 4a of N (first sub-module of cnn_agent.py cleanup)
**Predecessors:** Module 3 (#311-refactor-c/d, TechAgent removal, commits `d5991bd` + `d7d01a9`)
**Successor:** Module 4b — CNN-backend-only inference branches (Hurst/LGBM/regime/Ollama) under `MODEL_BACKEND=xgb`

---

## 1. Problem

`backend/agents/cnn_agent.py` is 3073 lines — the largest file in `backend/`. Among the cleanup clusters, the smallest and lowest-risk is the CNN_ARCH dead variants:

- `.env` sets `CNN_ARCH=glu1`. Per Session 58.x notes, glu1 is the active arch.
- `cnn_agent.py` defines **three** architecture classes: `SignalCNN` (glu2, the original ~280k-param variant), `SignalCNNGlu1` (~12k params, active), `SignalCNNGluM` (~55k params, mid-size).
- `_ARCH_REGISTRY = {"glu2": SignalCNN, "glu1": SignalCNNGlu1, "glum": SignalCNNGluM}` maps env values to classes.
- `_active_arch()` reads `os.environ.get("CNN_ARCH", "glu2")` at runtime — defaults to glu2 even though the actual env is glu1.
- Per-arch path helpers `_model_path_for(arch)` and `_best_loss_path_for(arch)` route to one of three filesystem locations.

Under `MODEL_BACKEND=xgb` (live) the entire CNN path is bypassed — even glu1 is dead today. But the operator scoped this module as "delete dead variants, keep glu1 in case of future flip-back." That preserves the option to revert to `MODEL_BACKEND=cnn` without re-importing deleted classes.

## 2. Goal

Eliminate the multi-arch registry surface. After this module:
- One CNN class (`SignalCNNGlu1`) survives in code.
- One on-disk checkpoint (`cnn_model_glu1.pt`) is the only path the loader looks at.
- `.env` has no `CNN_ARCH` knob.
- `_active_arch` / `_ARCH_REGISTRY` / `_model_path_for` / `_best_loss_path_for` functions are gone or trivialized.
- Glu2 weights and best-loss artifacts move to `backend/retired/` for graveyard preservation.
- Policy test locks in: re-introducing the env-var lookup fails the suite.

## 3. Non-goals

- No changes to `SignalCNNGlu1` (the surviving class).
- No changes to the XGB path / `xgb_signal.py` / MC chain.
- No changes to gated-off Hurst/LGBM/Ollama code (that's Module 4b).
- No removal of training infrastructure (auto-train, train_worker — that's Module 4c).
- No class rename (`CoinbaseCNNAgent` → `XGBAgent` — that's Module 4d if ever).

## 4. Approach

Single atomic commit on `feat/gpu-coord-mirror`. Pure deletion of glu2 + glum classes + multi-arch registry + per-arch path helpers. Hardcode the remaining glu1 paths. Delete env-var lookup. Move glu2 checkpoint to `backend/retired/` (host-side, gitignored). Extend Module 1's policy test.

### 4.1 Files touched

| Path | Action | Diff scope |
|---|---|---|
| `backend/agents/cnn_agent.py` lines 1141-1184 | DELETE `class SignalCNN` (glu2) | ~44 lines deleted |
| `backend/agents/cnn_agent.py` lines 1218-1252 | DELETE `class SignalCNNGluM` | ~34 lines deleted |
| `backend/agents/cnn_agent.py` lines 1253-1267 | DELETE `_ARCH_REGISTRY` + `_active_arch()` + `_build_cnn()` lookup; simplify `_build_cnn` to instantiate `SignalCNNGlu1` directly | ~15 lines deleted, ~5 added |
| `backend/agents/cnn_agent.py` lines 1096-1117 | SIMPLIFY `_model_path_for()` + `_best_loss_path_for()` — drop the `arch` parameter; return the glu1-suffixed paths directly | ~20 lines simplified to ~4 |
| `backend/agents/cnn_agent.py` lines 1693-1800 (multiple sites) | EDIT — replace `self._arch = _active_arch()` with hardcoded "glu1"; replace `_model_path_for(self._arch)` with `_model_path_for()`; remove `self._arch` field entirely (consumers updated to drop the parameter) | ~10 sites |
| `.env` | DELETE the `CNN_ARCH=glu1` line + its `# CNN model arch` comment if present | -1 to -2 lines |
| `backend/tests/test_cnn_agent.py` | DELETE `TestSignalCNNGluM` (entire class), `TestActiveArch` tests asserting glu2 default, `TestBuildCnn::test_build_cnn_glu2_returns_signal_cnn`, `test_build_cnn_glum_returns_glum_class`, `TestModelPath` glu2/glum tests, `TestBestLossPath` glu2/glum tests, `TestSignalCNNGlu1::test_fewer_params_than_glu2` | ~150 LOC across ~10 tests |
| `backend/tests/test_config.py` | EXTEND — add `test_no_cnn_arch_env_var` policy test | +12 LOC, +1 test |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71f entry | new |
| `polymarket_app/CLAUDE.md` | No edit needed (no invariant changes) | 0 |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND — Refactor Sweep Module 4a entry | +3 lines (outside repo) |

### 4.2 Host-side artifact retirement (NOT in commit)

```bash
mkdir -p backend/retired
mv backend/cnn_model.pt           backend/retired/cnn_model_glu2.pt
mv backend/cnn_best_loss.txt      backend/retired/cnn_best_loss_glu2.txt
```

`backend/retired/` is gitignored by the existing `backend/*.pt` and `backend/*.txt` patterns (they're already excluded from git tracking). The directory is operator-side preservation only; remote sees nothing.

If glu2 needs to come back, the operator can `mv backend/retired/cnn_model_glu2.pt backend/cnn_model.pt` after reverting the code changes.

### 4.3 Code change sketches

**`_active_arch()` deletion**, lines 1258-1267:
```python
# BEFORE
def _active_arch() -> str:
    """Read CNN_ARCH env at call-time so flips take effect without reimport."""
    return os.environ.get("CNN_ARCH", "glu2").strip().lower() or "glu2"

# AFTER (deleted entirely; no replacement function)
```

**`_ARCH_REGISTRY` deletion**, line 1253:
```python
# BEFORE
_ARCH_REGISTRY = {
    "glu2": SignalCNN,
    "glu1": SignalCNNGlu1,
    "glum": SignalCNNGluM,
}

def _build_cnn(arch: str) -> nn.Module:
    cls = _ARCH_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(f"Unknown CNN_ARCH '{arch}' — known: {sorted(_ARCH_REGISTRY)}")
    return cls()

# AFTER (registry gone; _build_cnn trivialized)
def _build_cnn() -> nn.Module:
    """Returns the glu1 SignalCNN. Multi-arch registry deleted #311-refactor-e."""
    return SignalCNNGlu1()
```

**Path helpers `_model_path_for` + `_best_loss_path_for`**, lines 1096-1117:
```python
# BEFORE — both took an arch parameter and constructed suffix
def _model_path_for(arch: str) -> str:
    if arch == "glu2":
        return MODEL_PATH                      # legacy unsuffixed
    return os.path.join(_BACKEND_DIR, f"cnn_model_{arch}.pt")

def _best_loss_path_for(arch: str) -> str:
    if arch == "glu2":
        return _BEST_LOSS_PATH                  # legacy unsuffixed
    return os.path.join(_BACKEND_DIR, f"cnn_best_loss_{arch}.txt")

# AFTER — no arch parameter; glu1 hardcoded
def _model_path_for() -> str:
    """Path to the active CNN checkpoint (glu1; multi-arch deleted #311-refactor-e)."""
    return os.path.join(_BACKEND_DIR, "cnn_model_glu1.pt")

def _best_loss_path_for() -> str:
    """Path to the active best-loss baseline (glu1)."""
    return os.path.join(_BACKEND_DIR, "cnn_best_loss_glu1.txt")
```

**`CoinbaseCNNAgent.__init__` arch field**, lines 1693-1697:
```python
# BEFORE
# Active arch is read at construction time so flipping CNN_ARCH only
# takes effect after agent re-instantiation (not env mutation at runtime).
self._arch = _active_arch()
if _TORCH:
    self.model = _build_cnn(self._arch).to(_DEVICE)

# AFTER
# Multi-arch registry deleted #311-refactor-e — glu1 is the only CNN arch.
if _TORCH:
    self.model = _build_cnn().to(_DEVICE)
```

All `_model_path_for(self._arch)` / `_best_loss_path_for(self._arch)` call sites become `_model_path_for()` / `_best_loss_path_for()`. `self._arch` field removed entirely.

### 4.4 New policy test

`backend/tests/test_config.py` (extend the existing file from Module 1):

```python
class TestNoCnnArchEnvVar:
    def test_cnn_arch_env_var_lookup_removed_from_cnn_agent(self):
        """Locks in #311-refactor-e: CNN_ARCH env var lookup was deleted
        when the multi-arch registry was removed. Only glu1 survives.
        Re-introducing the lookup requires reverting the dead-variant
        cleanup, not just adding a config field."""
        import os
        BACKEND = os.path.join(os.path.dirname(__file__), "..")
        src = open(
            os.path.join(BACKEND, "agents", "cnn_agent.py"),
            encoding="utf-8",
        ).read()
        for needle in (
            'os.environ.get("CNN_ARCH"',
            "os.environ.get('CNN_ARCH'",
            'os.getenv("CNN_ARCH"',
            "os.getenv('CNN_ARCH'",
        ):
            assert needle not in src, (
                f"cnn_agent.py contains '{needle}' — CNN_ARCH env-var "
                f"lookup was deleted #311-refactor-e. Multi-arch registry "
                f"committed to single-arch (glu1)."
            )
```

## 5. Architecture

No architectural change. Pure code deletion in one module + one .env line + ~150 LOC of test removal + 1 new test + host-side mkdir/mv.

## 6. Data flow

**Before:** `CNN_ARCH=glu1` env var → `_active_arch()` returns "glu1" → `_build_cnn("glu1")` → `SignalCNNGlu1()` → `_load(_model_path_for("glu1"))` → loads from `backend/cnn_model_glu1.pt`.

**After:** `_build_cnn()` → `SignalCNNGlu1()` → `_load(_model_path_for())` → loads from `backend/cnn_model_glu1.pt`.

Same on-disk artifact; same class instance; same predictions. The plumbing is just shorter.

## 7. Error handling

| Condition | Behavior |
|---|---|
| User has `CNN_ARCH=glu1` still in their local `.env` after pulling | Silently ignored — `os.environ` is never read for that key. No error. |
| User has `CNN_ARCH=glu2` in local `.env` | Same — silently ignored. Operator may be confused that flipping back to glu2 doesn't take effect. Documented in the CHANGELOG entry. |
| Future PR re-introduces `os.environ.get("CNN_ARCH"` | `test_cnn_arch_env_var_lookup_removed_from_cnn_agent` fires in pre-commit. |
| Operator wants to restore glu2 | Documented in CHANGELOG: `mv backend/retired/cnn_model_glu2.pt backend/cnn_model.pt` + `git revert` the commit. |

## 8. Tests

| File | Action | Net |
|---|---|---|
| `tests/test_cnn_agent.py` — `TestSignalCNNGluM` (entire class) | DELETE | -1 class, ~5 tests |
| `tests/test_cnn_agent.py` — `TestActiveArch` | DELETE | ~3 tests |
| `tests/test_cnn_agent.py` — `TestBuildCnn` glu2 + glum tests | DELETE | -2 tests |
| `tests/test_cnn_agent.py` — `TestModelPath` glu2 + glum tests | DELETE | -2 tests |
| `tests/test_cnn_agent.py` — `TestBestLossPath` glu2 + glum tests | DELETE | -2 tests |
| `tests/test_cnn_agent.py` — `TestSignalCNNGlu1::test_fewer_params_than_glu2` | DELETE | -1 test |
| `tests/test_config.py` — `TestNoCnnArchEnvVar` | NEW | +1 test |
| **Total** | | **net ~-15 tests, ~-150 LOC** |

The existing 1100+ test suite MUST stay green (pre-commit hook).

## 9. Rollout

### Phase 0 — Atomic commit
```bash
cd C:\Users\gl450\polymarket_app
git add backend/agents/cnn_agent.py backend/tests/test_cnn_agent.py \
        backend/tests/test_config.py CHANGELOG.md
git commit -m "refactor(#311-refactor-e): delete dead CNN_ARCH variants (glu2, glum)"
```

Pre-commit hook runs full suite (~5 min). On green, commit lands.

### Phase 1 — Push
```bash
git push
```

### Phase 2 — Host-side artifact retirement (operator)
```bash
mkdir -p backend/retired
mv backend/cnn_model.pt           backend/retired/cnn_model_glu2.pt
mv backend/cnn_best_loss.txt      backend/retired/cnn_best_loss_glu2.txt
```

### Phase 3 — Update local .env (operator)
```bash
# Remove the line:  CNN_ARCH=glu1
# from the local .env file
```

(.env is gitignored — host-only change.)

### Phase 4 — Verification
No backend restart needed. The deleted classes weren't loaded by the live XGB path (we're on `MODEL_BACKEND=xgb`). If you flip back to `MODEL_BACKEND=cnn`, the agent boots with glu1; any `CNN_ARCH=glu2` in `.env` is silently ignored.

### Rollback
1. `git revert <commit>` — restores classes + registry + helpers + tests.
2. `mv backend/retired/cnn_model_glu2.pt backend/cnn_model.pt` (host-side).
3. Re-add `CNN_ARCH=glu2` to `.env` if you actually want glu2 active.

## 10. Memory + CLAUDE.md sync (per CLAUDE.md rule)

Bundled into the same commit:
- `CHANGELOG.md` — Session 58.71f entry
- `polymarket_app/CLAUDE.md` — no invariant change (the existing notes about CNN training are still accurate; just `arch` flexibility is gone)
- `memory/coinbase_trader_architecture.md` (outside repo) — Refactor Sweep Module 4a entry

## 11. Open questions

None — operator approved every clarifying question on 2026-05-16.

## 12. References

- Module 1: `97dc8c9` (dead env vars — established the policy test pattern this module extends)
- Module 2: `cff73a0` (bare-isotonic — same single-commit pattern)
- Module 3: `d5991bd` + `d7d01a9` (TechAgent — bigger module that needed phases; this 4a is back to single-commit shape)
- Session 45 (2026-04-26): glu1 variant introduced + dual-arch capability
- Session 48 (2026-04-27): glum variant added as mid-size sibling
- CLAUDE.md invariant #15: env vars must trace to live consumer
