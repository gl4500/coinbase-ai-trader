# Refactor Sweep — Module 2: Bare-Isotonic Calibrator Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the bare-isotonic calibrator load path in `xgb_signal._try_load`; lock the dict-shape `{"calibrator","feature_set"}` as the only supported format via a new policy test.

**Architecture:** Single atomic commit on `feat/gpu-coord-mirror`. ~20 lines deleted from `_try_load` + 3 existing bare-isotonic tests retired + 1 new policy test added. Zero live-behavior change (current state: bare-isotonic on disk + v3 booster → raw passthrough; after: same → raw passthrough with different warning message).

**Tech Stack:** Python 3.11, pytest. No new dependencies.

**Spec source:** `docs/superpowers/specs/2026-05-16-refactor-module2-bare-isotonic-removal-design.md`
**Branch:** `feat/gpu-coord-mirror` (continue)

---

## File map

| Path | Action | Diff scope |
|---|---|---|
| `backend/agents/xgb_signal.py` (lines 90-128) | EDIT — collapse the dict-vs-bare branch into single dict-shape check | ~20 lines deleted, ~5 added |
| `backend/tests/test_xgb_signal.py` | EDIT — delete 3 tests, append 1 new test under `TestCalibration` | net ~-100 lines |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71c entry | new entry |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND — Refactor Sweep Module 2 entry | +3 lines (outside repo, no commit) |

---

## Task 1: Module 2 bare-isotonic removal — single atomic commit

**Files:**
- Modify: `backend/agents/xgb_signal.py` lines 90-128 (the `if os.path.exists(_CALIBRATION_PATH):` block)
- Modify: `backend/tests/test_xgb_signal.py` lines 180-340 (TestCalibration + TestForceReload classes)
- Modify: `polymarket_app/CHANGELOG.md`
- Modify: `~/.claude/projects/.../memory/coinbase_trader_architecture.md` (outside repo, not committed)

### Step 1.1 — Write the failing test FIRST (TDD red)

Append this test to `backend/tests/test_xgb_signal.py` inside `class TestCalibration` (currently lines 180-279). After deletion of the two bare-isotonic tests (Step 1.3), this becomes the only test in that class besides `test_no_calibration_pkl_falls_back_to_raw`.

```python
    def test_bare_isotonic_pkl_skipped_with_warning(
        self, tmp_path, monkeypatch, fresh_xgb_module, caplog
    ):
        """Locks in #311-refactor-b: bare-isotonic pickle format is no longer
        supported. A bare pickle on disk is treated as 'unknown shape' —
        skipped with a warning. Raw passthrough remains the failure mode."""
        import logging
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        _train_tiny_xgb(str(tmp_path), feature_set="v1")
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.5, 0.9])
        )
        with open(tmp_path / "xgb_calibration.pkl", "wb") as f:
            pickle.dump(iso, f)  # bare isotonic — no longer supported

        monkeypatch.setattr(
            fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json")
        )
        monkeypatch.setattr(
            fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json")
        )
        monkeypatch.setattr(
            fresh_xgb_module, "_CALIBRATION_PATH",
            str(tmp_path / "xgb_calibration.pkl"),
        )

        with caplog.at_level(logging.WARNING):
            fresh_xgb_module._try_load()
        assert fresh_xgb_module._calibration is None
        assert any(
            "bare-isotonic" in r.message.lower()
            or "not the canonical" in r.message.lower()
            for r in caplog.records
        )
```

Note: the new test relies on the `_train_tiny_xgb` helper and `pickle` (both already imported at the top of `test_xgb_signal.py`). No new imports needed at module level.

- [ ] **Step 1.1** — Append the test to `TestCalibration` (insert after `test_calibration_clipped_to_safe_range` which will be deleted in Step 1.3 — but adding it first means after Step 1.3 it's correctly positioned).

### Step 1.2 — Run the new test; expect 1 FAIL (current bare-isotonic path returns the calibrator instead of None)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_xgb_signal.py::TestCalibration::test_bare_isotonic_pkl_skipped_with_warning -v
```

Expected: 1 FAIL. The current code path at `xgb_signal.py:111-115` (the `if _feature_set == "v1"` branch) will set `_calibration = obj` (the bare isotonic) — so the `assert fresh_xgb_module._calibration is None` fails.

- [ ] **Step 1.2** — Run and observe red.

### Step 1.3 — Delete 3 bare-isotonic tests from `test_xgb_signal.py`

Open `backend/tests/test_xgb_signal.py`. Find and delete these three test methods entirely (including their decorators if any):

1. **`test_calibration_pkl_remaps_raw_to_calibrated`** (lines 182-224 approx — runs from `def test_calibration_pkl_remaps_raw_to_calibrated(` through its closing assertion, before `def test_no_calibration_pkl_falls_back_to_raw`).

2. **`test_calibration_clipped_to_safe_range`** (lines 242-279 approx — runs from `def test_calibration_clipped_to_safe_range(` through its closing assertion, before `class TestForceReload:`).

3. **`test_force_reload_picks_up_swapped_calibrator`** (lines 289-335 approx — runs from `def test_force_reload_picks_up_swapped_calibrator(` through its closing assertion, before `def test_force_reload_returns_false_when_artifacts_missing`).

After deletion, `TestCalibration` retains: `test_no_calibration_pkl_falls_back_to_raw` + the new `test_bare_isotonic_pkl_skipped_with_warning` from Step 1.1.

`TestForceReload` retains: `test_force_reload_function_exists` + `test_force_reload_returns_false_when_artifacts_missing`.

- [ ] **Step 1.3** — Delete the three test methods.

### Step 1.4 — Apply the code change to `_try_load`

Open `backend/agents/xgb_signal.py`. Find lines 90-128 (the `if os.path.exists(_CALIBRATION_PATH):` block). Replace the entire block (everything from `if os.path.exists(_CALIBRATION_PATH):` through the `else:` clause's `logger.info` for the no-pickle case) with this:

```python
            if os.path.exists(_CALIBRATION_PATH):
                try:
                    with open(_CALIBRATION_PATH, "rb") as f:
                        obj = pickle.load(f)
                    if not (isinstance(obj, dict) and "calibrator" in obj):
                        logger.warning(
                            "xgb_signal: calibrator pkl is not the canonical "
                            '{"calibrator","feature_set"} dict shape — '
                            "skipping calibration. Legacy bare-isotonic "
                            "format dropped #311-refactor-b."
                        )
                        _calibration = None
                    else:
                        cal_set = obj.get("feature_set")
                        if cal_set is not None and cal_set != _feature_set:
                            logger.warning(
                                "xgb_signal: calibrator feature_set=%s differs from "
                                "booster feature_set=%s — skipping calibration",
                                cal_set, _feature_set,
                            )
                            _calibration = None
                        else:
                            _calibration = obj["calibrator"]
                            logger.info(
                                "xgb_signal: loaded isotonic calibrator (feature_set=%s)",
                                cal_set,
                            )
                except Exception as exc:
                    logger.exception(
                        "xgb_signal: failed to load calibrator (raw passthrough): %s",
                        exc,
                    )
                    _calibration = None
            else:
                logger.info(
                    "xgb_signal: no calibrator at %s — raw passthrough",
                    _CALIBRATION_PATH,
                )
```

Key change: the inner `if isinstance(obj, dict) and "calibrator" in obj` branch is now flipped to its negation (`if not (...)`) so the unsupported case is the early-return path. The deleted bare-isotonic `else` block at the old lines 109-122 is gone entirely.

- [ ] **Step 1.4** — Apply the code edit.

### Step 1.5 — Run the new test again; expect 1 PASS

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_xgb_signal.py::TestCalibration::test_bare_isotonic_pkl_skipped_with_warning -v
```

Expected: 1 passed.

- [ ] **Step 1.5** — Run and observe green.

### Step 1.6 — Run the full `test_xgb_signal.py` to confirm no other tests broken

```bash
../.venv/Scripts/python.exe -m pytest tests/test_xgb_signal.py -v
```

Expected: all tests pass. The previously-3-now-deleted tests no longer exist in the collection. The remaining tests should be: 2 TestFallback + 4 TestLiveModel + 3 TestModuleAttributes + 2 TestCalibration (no-pkl + new bare-rejected) + 2 TestForceReload (exists + returns-false-missing) + 6 TestV3Routing = **19 tests, all green**.

- [ ] **Step 1.6** — Verify.

### Step 1.7 — Smoke check against the live calibrator pkl

The live `backend/xgb_calibration.pkl` is currently bare-isotonic + v3 booster on disk. Confirm the new code path produces `_calibration = None` (raw passthrough) — same as before, but via the new warning message.

```bash
cd C:\Users\gl450\polymarket_app/backend
../.venv/Scripts/python.exe -c "
from agents import xgb_signal
xgb_signal._try_load()
print('calibration loaded:', xgb_signal._calibration is not None)
print('feature_set detected:', xgb_signal._feature_set)
"
```

Expected output:
- `calibration loaded: False`
- `feature_set detected: v3`
- A warning line in stderr containing "not the canonical" or "Legacy bare-isotonic format dropped".

- [ ] **Step 1.7** — Smoke verify.

### Step 1.8 — Append CHANGELOG entry

Append this EXACT block to the TOP of `C:\Users\gl450\polymarket_app\CHANGELOG.md` (above the existing top entry, after the `---` separator):

```markdown
## [Session 58.71c] — 2026-05-16 — Refactor sweep module 2: bare-isotonic calibrator removal (#311-refactor-b)

### Why
Second module of the refactor sweep. `xgb_signal._try_load` had a dual-path
calibrator loader: dict-shape `{"calibrator","feature_set"}` (canonical
since #311f) and bare-isotonic (legacy v1). The bare-isotonic branch was
~20 lines of conditional logic plus a back-compat warning path. Its only
real-world consumer is a hypothetical rollback to the v1 booster + v1
calibrator backup — a one-time event that can be handled with a 3-line
host script (documented below).

### What changed
- **`backend/agents/xgb_signal.py:_try_load`** — collapsed the
  dict-vs-bare branch into a single dict-shape check. Bare-isotonic
  pickles now log a warning and skip calibration (raw passthrough). Net:
  ~20 lines deleted, ~5 added. Same observable behavior under the
  current bare-isotonic-on-disk state (still raw passthrough); different
  warning message.
- **`backend/tests/test_xgb_signal.py`** — deleted 3 tests that exercised
  the bare-isotonic load path:
  - `test_calibration_pkl_remaps_raw_to_calibrated`
  - `test_calibration_clipped_to_safe_range`
  - `test_force_reload_picks_up_swapped_calibrator`
  Added 1 new test `test_bare_isotonic_pkl_skipped_with_warning` locking
  in the new behavior. Net: -2 tests.

### Verification
```
backend && python -m pytest tests/test_xgb_signal.py -v
=> 19 passed
backend && python -c "from agents import xgb_signal; xgb_signal._try_load(); print(xgb_signal._calibration)"
=> None  (with 'Legacy bare-isotonic format dropped' warning in log)
```

Zero live-behavior change — current `backend/xgb_calibration.pkl` is bare
isotonic (v3 refit deferred per #311-cut) so both before and after this
change produce `_calibration = None` → raw passthrough.

### Rollback to v1 booster (operator runbook)
If rolling back to the v1 booster + the v1 calibrator backup, the bare
pickle must be rewrapped into dict shape first:

```python
import pickle
from sklearn.isotonic import IsotonicRegression
iso = pickle.load(open("backend/xgb_calibration.pkl.bak_v1_20260516_182946", "rb"))
with open("backend/xgb_calibration.pkl", "wb") as f:
    pickle.dump({"calibrator": iso, "feature_set": "v1"}, f)
```

Then rename the v1 booster files back to production names (per the
#311-cut rollback procedure) and hot-reload.

---
```

- [ ] **Step 1.8** — Append the CHANGELOG entry.

### Step 1.9 — Update memory file (outside repo, not committed)

Open `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`. Find the Session 58.71 entry (Refactor sweep module 1). Append this block immediately ABOVE it (newer entries on top):

```markdown
- **Session 58.71c (2026-05-16)**: Refactor sweep #311-refactor module 2 — bare-isotonic calibrator path removal (#311-refactor-b). Deleted ~20-line bare-isotonic branch in `agents/xgb_signal._try_load`. Dict-shape `{"calibrator","feature_set"}` is now the only supported calibrator format. New policy test `test_bare_isotonic_pkl_skipped_with_warning` locks the behavior. Deleted 3 tests that exercised the bare-isotonic path (`test_calibration_pkl_remaps_raw_to_calibrated`, `test_calibration_clipped_to_safe_range`, `test_force_reload_picks_up_swapped_calibrator`). Net: -2 tests. Zero live-behavior change — current `xgb_calibration.pkl` is bare-isotonic + v3 booster which already produced raw passthrough; same after. Rollback rewrap script documented in CHANGELOG entry. Next sweep module: likely CNN_ARCH variants (glu2/glum dead in registry) or cnn_agent dead branches under MODEL_BACKEND=xgb.
```

- [ ] **Step 1.9** — Apply the memory edit. No commit (file lives outside the repo).

### Step 1.10 — Cleanup background python processes (port-8001-aware per #311-refactor-cleanup)

```powershell
$backendPid = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $backendPid -and $_.ProcessName -ne 'Coinbase AI Trader' } |
    Stop-Process -Force
```

This SKIPS the live backend (port 8001) per CLAUDE.md Shell cleanup section.

- [ ] **Step 1.10** — Run cleanup.

### Step 1.11 — Single atomic commit

```bash
cd C:\Users\gl450\polymarket_app
git add backend/agents/xgb_signal.py backend/tests/test_xgb_signal.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
refactor(#311-refactor-b): drop bare-isotonic calibrator path

Second module of #311-refactor sweep. Collapsed the dict-vs-bare branch
in xgb_signal._try_load into a single dict-shape check. Bare-isotonic
pickles now log a warning and skip calibration (raw passthrough).

Deleted 3 tests that exercised the bare-isotonic load path:
- test_calibration_pkl_remaps_raw_to_calibrated
- test_calibration_clipped_to_safe_range
- test_force_reload_picks_up_swapped_calibrator

Added 1 new test test_bare_isotonic_pkl_skipped_with_warning that locks
in the new behavior. Net: -2 tests.

Zero live-behavior change — current backend/xgb_calibration.pkl is bare
isotonic (v3 refit deferred per #311-cut) so both before and after this
change produce _calibration = None -> raw passthrough.

Rollback rewrap script for v1-booster scenarios documented in CHANGELOG.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Pre-commit hook runs the full ~1100-test suite (~5 min). All must pass — the deleted tests no longer exist; the new test was made green at Step 1.5; the code change preserves observable behavior under all currently-tested conditions.

- [ ] **Step 1.11** — Commit. Wait for pre-commit hook to finish.

### Step 1.12 — Push to origin

```bash
cd C:\Users\gl450\polymarket_app
git push
```

Expected: fast-forward push to `feat/gpu-coord-mirror`.

- [ ] **Step 1.12** — Push.

### Step 1.13 — Final verification

Re-run the smoke check from Step 1.7 to confirm the committed state still produces raw passthrough.

```bash
cd C:\Users\gl450\polymarket_app/backend
../.venv/Scripts/python.exe -c "
from agents import xgb_signal
xgb_signal._try_load()
print('calibration loaded:', xgb_signal._calibration is not None)
"
```

Expected: `calibration loaded: False` plus a "not the canonical" warning.

- [ ] **Step 1.13** — Final verify.

---

## Spec coverage check

| Spec section | Task step |
|---|---|
| 4.1 Files touched — xgb_signal.py | Step 1.4 |
| 4.1 Files touched — test_xgb_signal.py | Steps 1.1 + 1.3 |
| 4.1 Files touched — CHANGELOG.md | Step 1.8 |
| 4.1 Files touched — memory file | Step 1.9 |
| 4.2 Code change in `_try_load` | Step 1.4 (exact `if not (...)` collapse) |
| 4.3 Tests retired (3 specific tests) | Step 1.3 |
| 4.4 New test (test_bare_isotonic_pkl_skipped_with_warning) | Steps 1.1 + 1.2 + 1.5 |
| 5 Architecture (no change) | implied |
| 6 Data flow (same observable outcome) | Step 1.7 (smoke verify) |
| 7 Error handling — 5 conditions table | covered by existing + new test |
| 8 Tests — net -2 | Steps 1.1 + 1.3 |
| 9 Rollout — Phase 0 atomic commit | Step 1.11 |
| 9 Rollout — Phase 1 push | Step 1.12 |
| 9 Rollout — Phase 2 verify | Steps 1.7 + 1.13 |
| 9 Rollback rewrap script | embedded in Step 1.8 CHANGELOG entry |
| 10 Memory + CLAUDE.md sync | Steps 1.8 (CHANGELOG) + 1.9 (memory). No CLAUDE.md edit needed (invariant #13 already says dict-shape canonical) |

All spec sections have a corresponding task step.

---

## Plan complete

Saved to `docs/superpowers/plans/2026-05-16-refactor-module2-bare-isotonic-removal.md`. **1 task, 13 micro-steps, 1 atomic commit, net -2 tests, zero live-behavior change.**

Same shape as Module 1: tight scope, one commit, deletion + policy lock. Operator preference noted: use the port-8001-aware cleanup snippets (Step 1.10) — DON'T kill the backend.
