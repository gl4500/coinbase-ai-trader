# Refactor Sweep — Module 2: Bare-Isotonic Calibrator Path Removal

**Date:** 2026-05-16
**Status:** APPROVED 2026-05-16 (operator: "yes please" → "continue")
**Scope:** `backend/` only
**Branch:** continue on `feat/gpu-coord-mirror`
**Sweep position:** Module 2 of N
**Predecessor:** Module 1 (#311-refactor-a, env-var cleanup, commit `97dc8c9`)
**Successor:** Module 3 — likely CNN_ARCH variants (glu2/glum dead in registry) or cnn_agent dead branches under MODEL_BACKEND=xgb (operator picks at next brainstorm)

---

## 1. Problem

`backend/agents/xgb_signal.py:_try_load` has a dual-path calibrator loader. The canonical format since #311f is the dict `{"calibrator", "feature_set"}`. The legacy bare-isotonic format is still accepted via a back-compat branch (~20 lines + a back-compat warning log path).

Today's live state:
- v3 booster on disk (`backend/xgb_model.json`, set=v3)
- **Bare-isotonic** v1 calibrator on disk (`backend/xgb_calibration.pkl`, 2334 bytes, dated May 10) — the v3 refit was deferred per the cutover plan (~48h paper-trade window pending)
- `_try_load` detects bare-isotonic + v3 booster mismatch → skips calibration → raw passthrough

The bare-isotonic branch contributes complexity and a third "format" the loader has to reason about. Its only user is a hypothetical rollback to v1 booster, which is a one-time event that can be handled with a 3-line wrap-on-the-host script.

## 2. Goal

Delete the bare-isotonic load path entirely. Lock in the dict-shape `{"calibrator", "feature_set"}` as the only supported format. Document the rollback rewrap so future operators have a clear runbook.

## 3. Non-goals

- No change to `_try_load`'s dict-shape path (already correct).
- No change to `fit_xgb_calibration.py` (already writes dict shape via `_save_calibrator`).
- No backend restart implied — live behavior under bare-isotonic-on-disk is unchanged (calibration skipped both before and after).
- No new CLAUDE.md invariant — #13 already says "Calibrator pickle is `{"calibrator","feature_set"}` dict", which becomes literally true (was aspirational with back-compat fallback).

## 4. Approach

Single atomic commit on `feat/gpu-coord-mirror`. Pure deletion of the bare-isotonic branch + retirement of 3 tests that exercised it + addition of 1 new test locking the new behavior.

### 4.1 Files touched

| Path | Action | Diff size |
|---|---|---|
| `backend/agents/xgb_signal.py` (lines ~93-127) | EDIT — collapse the dict-vs-bare branch into single dict-shape check; bare-isotonic logs warning + skips | ~20 lines deleted, ~5 added |
| `backend/tests/test_xgb_signal.py` | DELETE 3 tests + ADD 1 new test | net ~-100 lines |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71c entry | new entry |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND — Refactor Sweep Module 2 entry | +5 lines (outside repo, no commit) |

### 4.2 Code change in `xgb_signal._try_load`

**Before (current state, lines ~85-127):**

```python
if os.path.exists(_CALIBRATION_PATH):
    try:
        with open(_CALIBRATION_PATH, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "calibrator" in obj:
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
        else:
            # Bare isotonic (legacy v1) — accept only if booster is v1
            if _feature_set == "v1":
                _calibration = obj
                logger.info(
                    "xgb_signal: loaded legacy bare-isotonic calibrator (assumed v1)",
                )
            else:
                logger.warning(
                    "xgb_signal: legacy bare-isotonic calibrator found but "
                    "booster feature_set=%s — skipping calibration",
                    _feature_set,
                )
                _calibration = None
    except Exception as exc:
        ...
else:
    logger.info(
        "xgb_signal: no calibrator at %s — raw passthrough", _CALIBRATION_PATH,
    )
```

**After:**

```python
if os.path.exists(_CALIBRATION_PATH):
    try:
        with open(_CALIBRATION_PATH, "rb") as f:
            obj = pickle.load(f)
        if not (isinstance(obj, dict) and "calibrator" in obj):
            logger.warning(
                "xgb_signal: calibrator pkl is not the canonical "
                '{"calibrator","feature_set"} dict shape — skipping calibration. '
                "Legacy bare-isotonic format dropped #311-refactor-b."
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
            "xgb_signal: failed to load calibrator (raw passthrough): %s", exc,
        )
        _calibration = None
else:
    logger.info(
        "xgb_signal: no calibrator at %s — raw passthrough", _CALIBRATION_PATH,
    )
```

Net delta in `_try_load`: 20 lines deleted, 5 lines added.

### 4.3 Tests retired

Three existing tests in `backend/tests/test_xgb_signal.py` are deleted because they specifically test the bare-isotonic load path:

| Test | Lines (approx) | What it tested |
|---|---|---|
| `test_calibration_pkl_remaps_raw_to_calibrated` | 182-224 | Bare-isotonic load + remap |
| `test_calibration_clipped_to_safe_range` | 242-279 | Bare-isotonic + output clipping |
| `test_force_reload_picks_up_swapped_calibrator` | 289-335 | Bare-isotonic swap-on-disk |

Coverage assessment after deletion:
- Calibration-actually-runs coverage: covered by `test_v3_xgb_prob_calls_tiered_history_with_pid` (the v3 booster path runs the dict-shape calibrator end-to-end when present).
- Output clipping: covered by `test_returns_float_in_range` (asserts output stays in [0.01, 0.99]).
- Hot-reload behavior: covered by `test_force_reload_function_exists` + `test_force_reload_returns_false_when_artifacts_missing`.

### 4.4 New test

```python
def test_bare_isotonic_pkl_skipped_with_warning(
    self, tmp_path, monkeypatch, fresh_xgb_module, caplog
):
    """Locks in #311-refactor-b: bare-isotonic pickle format is no longer
    supported. A bare pickle on disk is treated as "unknown shape" — skipped
    with a warning. Raw passthrough remains the failure mode."""
    import logging
    import pickle
    from sklearn.isotonic import IsotonicRegression
    import numpy as np

    _train_tiny_xgb(str(tmp_path), feature_set="v1")
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.5, 0.9])
    )
    with open(tmp_path / "xgb_calibration.pkl", "wb") as f:
        pickle.dump(iso, f)  # bare isotonic — unsupported
    monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
    monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
    monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "xgb_calibration.pkl"))

    with caplog.at_level(logging.WARNING):
        fresh_xgb_module._try_load()
    assert fresh_xgb_module._calibration is None
    assert any(
        "bare-isotonic" in r.message.lower() or "not the canonical" in r.message.lower()
        for r in caplog.records
    )
```

This test class needs to be placed in `TestCalibration` (the same class the deleted tests lived in) so the test file structure stays clean.

## 5. Architecture

No architectural change. Pure surface-level code deletion in one function.

## 6. Data flow

**Today (bare-isotonic on disk, v3 booster):**
```
.pkl bare isotonic → pickle.load() → obj is IsotonicRegression
                  → isinstance(obj, dict) is False
                  → else branch fires → "legacy bare-isotonic found but booster=v3" warning
                  → _calibration = None
                  → raw passthrough at inference
```

**After Module 2 (bare-isotonic on disk, v3 booster):**
```
.pkl bare isotonic → pickle.load() → obj is IsotonicRegression
                  → not (isinstance(obj, dict) and "calibrator" in obj)
                  → "not the canonical dict shape" warning
                  → _calibration = None
                  → raw passthrough at inference
```

**Same observable outcome.** Different warning message, fewer lines of code, fewer branches to reason about.

**After operator does the v3 calibrator refit (dict-shape on disk, v3 booster):**
```
.pkl dict {calibrator, feature_set:v3} → pickle.load() → obj is dict
                                      → isinstance(obj, dict) and "calibrator" in obj
                                      → cal_set == _feature_set
                                      → _calibration = obj["calibrator"]
                                      → calibration RUNS at inference
```

Unchanged from current dict-shape path.

## 7. Error handling

| Condition | Behavior | Test |
|---|---|---|
| Pkl missing | Log info, raw passthrough | existing `test_no_calibration_pkl_falls_back_to_raw` |
| Pkl is dict + feature_set matches | Load, use | covered by `test_v3_xgb_prob_calls_tiered_history_with_pid` |
| Pkl is dict + feature_set mismatches | Log warning, skip | existing `test_v3_skips_v1_calibrator_on_metadata_mismatch` |
| Pkl is bare isotonic (legacy on disk) | Log warning, skip — NEW BEHAVIOR | NEW `test_bare_isotonic_pkl_skipped_with_warning` |
| Pkl fails to deserialize | Log exception, skip | existing exception handler (no test, edge case) |

## 8. Tests

| File | Action | Net |
|---|---|---|
| `tests/test_xgb_signal.py` — `test_calibration_pkl_remaps_raw_to_calibrated` | DELETE | -1 |
| `tests/test_xgb_signal.py` — `test_calibration_clipped_to_safe_range` | DELETE | -1 |
| `tests/test_xgb_signal.py` — `test_force_reload_picks_up_swapped_calibrator` | DELETE | -1 |
| `tests/test_xgb_signal.py` — `test_bare_isotonic_pkl_skipped_with_warning` | NEW | +1 |
| **Net test count** | | **-2** |

Plus: existing 1100+ test suite MUST stay green (pre-commit hook).

## 9. Rollout

### Phase 0 — Single atomic commit
```bash
cd C:\Users\gl450\polymarket_app
git add backend/agents/xgb_signal.py backend/tests/test_xgb_signal.py CHANGELOG.md
git commit -m "refactor(#311-refactor-b): drop bare-isotonic calibrator path"
```

Pre-commit hook runs full suite (~5 min). On green, commit lands.

### Phase 1 — Push
```bash
git push
```

### Phase 2 — Verification (operator)
No backend restart needed. Current bare-isotonic-on-disk continues to produce raw passthrough — same as before.

```bash
.venv/Scripts/python.exe -c "
from agents import xgb_signal
xgb_signal._try_load()
print('calibration loaded:', xgb_signal._calibration is not None)
"
```

Expected: `calibration loaded: False` with a "not the canonical dict shape" warning in the log.

### Rollback (operator)
If you ever roll back to the v1 booster + the v1 calibrator backup, you'll need a one-time pickle rewrap. The CHANGELOG entry includes this exact script:

```python
import pickle
from sklearn.isotonic import IsotonicRegression
iso = pickle.load(open("backend/xgb_calibration.pkl.bak_v1_20260516_182946", "rb"))
with open("backend/xgb_calibration.pkl", "wb") as f:
    pickle.dump({"calibrator": iso, "feature_set": "v1"}, f)
```

Then rename the v1 booster files back to production names (per the #311-cut rollback procedure) and hot-reload.

## 10. Memory + CLAUDE.md sync (per CLAUDE.md rule)

Bundled into the same commit as the code changes (small scope).

- `CHANGELOG.md` — Session 58.71c entry (in same commit)
- `polymarket_app/CLAUDE.md` — no invariant change (invariant #13 already says dict shape is canonical; this commit makes that literally true)
- `memory/coinbase_trader_architecture.md` (outside repo) — Refactor Sweep Module 2 entry

## 11. Open questions

None — operator approved every clarifying question on 2026-05-16.

## 12. References

- Module 1 spec + commit: `97dc8c9` (refactor #311-refactor-a)
- Calibrator dict-shape introduction: `218a61f` (`feat(#311f)`) and `f349125` (`feat(#311-mc-wire)`)
- v3 cutover (where the current bare-isotonic-on-disk state was established): `9a5e084` (`fix(#311j) + ops(#311-cut)`)
- Backend-aware shell cleanup rule: `0ffbcf1` (`docs(#311-refactor-cleanup)`)
- CLAUDE.md invariant #13: XGB feature_set v3 (defines the canonical calibrator dict shape)
