# Refactor Sweep — Module 1: Dead Env-Var Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete 4 dead-on-arrival LLM-blend env vars (never consumed in `backend/`), tag the legitimately-gated `CNN_TRAIN_EVERY_N_SCANS` knob, lock in a "no orphan env vars" policy via test + CLAUDE.md invariant.

**Architecture:** Single atomic commit on `feat/gpu-coord-mirror`. Pure deletion of dead config fields + dead env keys + one new policy test + docstring fixes + CHANGELOG + CLAUDE.md sync. Zero live-behavior change (deleted fields were never on a code path — verified by grep).

**Tech Stack:** Python 3.11 (config.py uses `dataclasses` + `python-dotenv`), pytest. No new dependencies.

**Spec source:** `docs/superpowers/specs/2026-05-16-refactor-module1-env-cleanup-design.md`
**Branch:** `feat/gpu-coord-mirror` (continue)

---

## File map

| Path | Action | Diff scope |
|---|---|---|
| `backend/config.py` (lines 59-63) | EDIT — delete 4 fields + the "CNN/LLM blend weights" comment line | ~5 lines deleted |
| `backend/config.py` (line 64) | EDIT — replace comment with backend-gating note | ~3 lines changed |
| `backend/config.py` (module docstring) | EDIT — add policy line | ~3 lines added |
| `backend/services/hmm_regime.py` (lines 198-217) | EDIT — `regime_blend()` docstring updated | docstring-only |
| `backend/tests/test_config.py` | CREATE | ~25 LOC, 1 test class, 1 test |
| `.env` (lines 54-60) | EDIT — delete 4 env keys + 2 inter-leaving comments, update auto-train comment | ~6 lines deleted, ~1 changed |
| `.env` (top of file) | EDIT — add policy comment block | ~3 lines added |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71a entry | new entry |
| `polymarket_app/CLAUDE.md` | EDIT — add invariant #15 | +3 lines |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND — Refactor Sweep section | +5 lines (outside repo, no commit) |

---

## Task 1: Module 1 cleanup — single atomic commit

**Files:**
- Modify: `backend/config.py` (module docstring, lines 59-65 region)
- Modify: `backend/services/hmm_regime.py` (lines 198-217)
- Create: `backend/tests/test_config.py`
- Modify: `.env` (top + lines 54-61 region)
- Modify: `polymarket_app/CHANGELOG.md`
- Modify: `polymarket_app/CLAUDE.md`
- Modify: `~/.claude/projects/.../memory/coinbase_trader_architecture.md` (outside repo, not committed)

### Step 1.1 — Write the failing test FIRST (TDD red)

Create `backend/tests/test_config.py` with this exact content:

```python
"""Policy test for refactor sweep (#311-refactor-a).

Locks in: any env var defined in config.py MUST trace to a live consumer.
The four CNN_*_CNN_W / CNN_*_LLM_W env vars were dead-on-arrival (never
read anywhere in backend/) and were deleted. If anyone re-adds them
without a live consumer, this test fails.
"""
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

- [ ] **Step 1.1** — Write the file above to `backend/tests/test_config.py`.

### Step 1.2 — Run the test; expect 1 FAILURE

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Expected: `1 failed` with `AssertionError: config.cnn_trending_cnn_w was deleted #311-refactor-a ...` (the field still exists today; the test catches it).

- [ ] **Step 1.2** — Run and observe red.

### Step 1.3 — Delete the 4 dead fields from `backend/config.py`

Open `backend/config.py`. Find lines 59-63:

```python
    # CNN/LLM blend weights per regime (must sum to 1.0 each pair)
    cnn_trending_cnn_w:     float = field(default_factory=lambda: float(os.getenv("CNN_TRENDING_CNN_W",     "0.75")))
    cnn_trending_llm_w:     float = field(default_factory=lambda: float(os.getenv("CNN_TRENDING_LLM_W",     "0.25")))
    cnn_ranging_cnn_w:      float = field(default_factory=lambda: float(os.getenv("CNN_RANGING_CNN_W",      "0.40")))
    cnn_ranging_llm_w:      float = field(default_factory=lambda: float(os.getenv("CNN_RANGING_LLM_W",      "0.60")))
```

Replace these 5 lines with NOTHING (delete them).

Then find line 64 (the existing comment above `cnn_train_every_n_scans`):

```python
    # How often to auto-train (in number of scans; default 4 = ~1 hour at 15-min scan interval)
```

Replace with this 4-line block:

```python
    # Auto-train cadence in scans. Active only under MODEL_BACKEND=cnn;
    # auto-train is gated off under MODEL_BACKEND=xgb per #300, so flipping
    # this knob has no effect while the xgb backend is live.
    # Default 4 = ~1 hour at the 15-min scan interval.
```

- [ ] **Step 1.3** — Apply both edits.

### Step 1.4 — Update `config.py` module docstring

Open `backend/config.py`. The current module docstring (lines 1-4) is:

```python
"""
Central configuration loaded from .env.
All modules import the `config` singleton — never read os.environ directly.
"""
```

Replace with:

```python
"""
Central configuration loaded from .env.
All modules import the `config` singleton — never read os.environ directly.

Policy: every env var defined here MUST trace to a live consumer in backend/.
Dead entries are deleted on sight per refactor sweep policy (#311-refactor).
"""
```

- [ ] **Step 1.4** — Apply the docstring edit.

### Step 1.5 — Run the test again; expect 1 PASS

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Expected: `1 passed`.

- [ ] **Step 1.5** — Run and observe green.

### Step 1.6 — Update `regime_blend()` docstring

Open `backend/services/hmm_regime.py`. Find lines 198-207:

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

Replace with:

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

- [ ] **Step 1.6** — Apply the docstring edit.

### Step 1.7 — Edit `.env` to delete 4 env keys + add policy comment

Open `C:\Users\gl450\polymarket_app\.env`. Find lines 54-60 (use `sed -n '54,60p' .env` to confirm):

```
# CNN vs LLM blend when ADX >= ADX_TREND_THRESHOLD (trending market)
CNN_TRENDING_CNN_W=0.75
CNN_TRENDING_LLM_W=0.25
# CNN vs LLM blend when ADX < ADX_TREND_THRESHOLD (ranging market)
CNN_RANGING_CNN_W=0.40
CNN_RANGING_LLM_W=0.60
# Auto-train every N scans (4 scans × 15 min = ~1 hour)
```

Replace these 7 lines with this 2-line block:

```
# Active only under MODEL_BACKEND=cnn (auto-train gated off otherwise, #300)
```

Then find the very TOP of `.env` (the first non-blank line). Add this 3-line policy block as the new first lines of the file:

```
# All env vars in this file MUST trace to a live consumer in backend/.
# Dead entries are deleted on sight per refactor sweep policy (#311-refactor).

```

(Note the trailing blank line — keeps the existing first env var visually separated.)

- [ ] **Step 1.7** — Apply both .env edits.

### Step 1.8 — Verify no live code reads the deleted fields

```bash
cd C:\Users\gl450\polymarket_app
grep -rE "config\.cnn_trending|config\.cnn_ranging" backend/ 2>&1
```

Expected: NO output (no live code reads any of the deleted fields). If output appears, STOP — there's a hidden consumer the spec missed; reassess before committing.

- [ ] **Step 1.8** — Run grep and confirm empty output.

### Step 1.9 — Append CHANGELOG entry

Append this EXACT block to the TOP of `C:\Users\gl450\polymarket_app\CHANGELOG.md` (above the existing top entry, after the `---` separator):

```markdown
## [Session 58.71a] — 2026-05-16 — Refactor sweep module 1: dead env-var cleanup (#311-refactor-a)

### Why
First module of the refactor sweep. Investigation (grep across backend/)
revealed the 4 CNN_*_CNN_W / CNN_*_LLM_W env vars defined in config.py:60-63
are dead-on-arrival: nothing in backend/ ever reads them. `regime_blend()`
in services/hmm_regime.py uses hardcoded weights (0.75/0.25 for trending,
etc.) and was scaffolded with no config plumbing. The env vars + config
fields polluted `.env` and misled operators about what's tunable.

### What changed
- **`backend/config.py`** — deleted 4 fields (`cnn_trending_cnn_w`,
  `cnn_trending_llm_w`, `cnn_ranging_cnn_w`, `cnn_ranging_llm_w`) and
  their wrapping comment. Replaced the auto-train comment with a
  backend-gating tag. Added a policy line to the module docstring
  ("every env var MUST trace to a live consumer").
- **`backend/services/hmm_regime.py`** — `regime_blend()` docstring now
  explicitly notes weights are hardcoded (was misleading — implied
  config-driven).
- **`.env`** — deleted 4 env keys and 2 wrapping comments. Replaced the
  auto-train comment with the backend-gating tag. Added a 2-line policy
  comment block at the top of the file.
- **`backend/tests/test_config.py`** (NEW) — 1 policy test
  (`test_no_dead_llm_blend_fields`) with 4 assertions. Locks in: if
  anyone re-adds these fields without a live consumer, pre-commit fails.
- **`polymarket_app/CLAUDE.md`** — new invariant #15 documenting the
  policy.

### Verification
```
backend && python -m pytest tests/test_config.py -v
=> 1 passed
backend && grep -rE "config\.cnn_trending|config\.cnn_ranging" .
=> (empty — no live consumers)
```

Zero live-behavior change. The 4 deleted fields were never consumed anywhere.

---
```

- [ ] **Step 1.9** — Append the CHANGELOG entry.

### Step 1.10 — Add invariant #15 to CLAUDE.md

Open `C:\Users\gl450\polymarket_app\CLAUDE.md`. Find invariant #14 (the MC filter chain one added in Session 58.70). Append this NEW invariant immediately after:

```markdown
15. **Env vars MUST trace to a live consumer.** Every entry in `backend/config.py` and `.env` must be read by production code in `backend/`. Dead entries are deleted on sight per refactor sweep policy (#311-refactor). The `test_no_dead_llm_blend_fields` regression test in `backend/tests/test_config.py` enforces this for known offenders; future deletions extend that test.
```

- [ ] **Step 1.10** — Apply the edit.

### Step 1.11 — Update memory file (outside repo, not committed)

Open `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`. Find the Session 58.70 entry (most recent MC entry). Append this block immediately ABOVE it (newer entries on top):

```markdown
- **Session 58.71 (2026-05-16)**: Refactor sweep #311-refactor module 1 — dead env-var cleanup (#311-refactor-a). Deleted 4 dead LLM-blend env vars + matching config.py fields (`cnn_trending_cnn_w` etc.) — verified dead by grep before deletion. Tagged `CNN_TRAIN_EVERY_N_SCANS` with backend-gating comment. Policy test `tests/test_config.py:test_no_dead_llm_blend_fields` locks in the rule that env vars must trace to a live consumer. Zero live-behavior change. Sweep strategy: module-by-module, each module = own spec+plan+commit; module 2 likely `xgb_signal.py` legacy bare-isotonic path.
```

- [ ] **Step 1.11** — Apply the memory edit. No commit (file lives outside the repo).

### Step 1.12 — Cleanup background python processes

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

- [ ] **Step 1.12** — Run cleanup.

### Step 1.13 — Single atomic commit

```bash
cd C:\Users\gl450\polymarket_app
git add backend/config.py backend/services/hmm_regime.py \
        backend/tests/test_config.py .env CHANGELOG.md CLAUDE.md
git commit -m "$(cat <<'EOF'
refactor(#311-refactor-a): delete dead LLM-blend env vars

First module of the #311-refactor sweep. Deleted 4 env vars + config
fields that were never read anywhere in backend/ (verified by grep):
CNN_TRENDING_CNN_W, CNN_TRENDING_LLM_W, CNN_RANGING_CNN_W, CNN_RANGING_LLM_W.
regime_blend() in services/hmm_regime.py uses hardcoded weights; the
env vars were scaffolded with no plumbing.

CNN_TRAIN_EVERY_N_SCANS stays (active under MODEL_BACKEND=cnn) but
gets a backend-gating comment in both .env and config.py.

New test test_no_dead_llm_blend_fields locks in the policy: env vars
must trace to a live consumer. CLAUDE.md invariant #15 documents it.

Zero live-behavior change.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Pre-commit hook runs the full ~1100-test suite (~5 min). All tests must pass — the existing suite was green before this change, deleting unused fields can't break anything (grep verified at Step 1.8), and the one new test was made green at Step 1.5.

- [ ] **Step 1.13** — Commit. Wait for pre-commit hook to finish.

### Step 1.14 — Push to origin

```bash
cd C:\Users\gl450\polymarket_app
git push
```

Expected: fast-forward push to `feat/gpu-coord-mirror`.

- [ ] **Step 1.14** — Push.

### Step 1.15 — Verification

```bash
cd C:\Users\gl450\polymarket_app
.venv/Scripts/python.exe -c "from backend.config import config; print(hasattr(config, 'cnn_trending_cnn_w'))"
```

Expected: `False`.

```bash
cd C:\Users\gl450\polymarket_app/backend
../.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_hmm_regime.py -v
```

Expected: all pass. The hmm_regime test suite is included as a smoke check that the `regime_blend()` docstring edit didn't break anything.

- [ ] **Step 1.15** — Verify.

---

## Spec coverage check

| Spec section | Task step |
|---|---|
| 4.1 Files touched (config.py 60-63) | Step 1.3 |
| 4.1 Files touched (config.py docstring) | Step 1.4 |
| 4.1 Files touched (.env) | Step 1.7 |
| 4.1 Files touched (hmm_regime.py) | Step 1.6 |
| 4.1 Files touched (test_config.py NEW) | Step 1.1 |
| 4.1 Files touched (CHANGELOG) | Step 1.9 |
| 4.1 Files touched (CLAUDE.md) | Step 1.10 |
| 4.1 Files touched (memory) | Step 1.11 |
| 4.2 Code changes — config.py 60-65 | Step 1.3 |
| 4.2 Code changes — module docstring | Step 1.4 |
| 4.2 Code changes — .env | Step 1.7 |
| 4.2 Code changes — hmm_regime docstring | Step 1.6 |
| 4.3 New test | Steps 1.1-1.2-1.5 (TDD red→green) |
| 5 Architecture (no architectural change) | implied |
| 6 Data flow (zero live change) | implied |
| 7 Error handling | Step 1.8 (grep verify catches missed consumers) |
| 8 Tests | Step 1.5 (1 new) + Step 1.13 (full suite via pre-commit) + Step 1.15 (smoke hmm_regime) |
| 9 Rollout — Phase 0 atomic commit | Step 1.13 |
| 9 Rollout — Phase 1 push | Step 1.14 |
| 9 Rollout — Phase 2 verification | Step 1.15 |
| 10 Memory + CLAUDE.md sync | Steps 1.9, 1.10, 1.11 |

All spec sections have a corresponding task step.

---

## Plan complete

Saved to `docs/superpowers/plans/2026-05-16-refactor-module1-env-cleanup.md`. **1 task, 15 micro-steps, 1 atomic commit, 1 new test (4 assertions), zero live-behavior change.**

The whole thing is intentionally small — this module establishes the sweep workflow precedent before tackling larger surface area (xgb_signal legacy path, cnn_agent dead branches, CNN_ARCH variants, etc.) in subsequent modules.

Operator runs inline (per the v3 lesson: subagents kept dying on the pre-commit hook).
