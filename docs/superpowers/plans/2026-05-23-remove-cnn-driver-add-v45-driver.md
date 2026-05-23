# Remove CNN Driver + Add XGB v4.5 Driver Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the CNN model as a viable trade driver from the polymarket_app backend, add `MODEL_BACKEND=xgb_v45` as a v4.5-driven trade path, and make v4.5 shadow telemetry log on every scan regardless of which driver is active.

**Architecture:** `config.py` adds startup validation that rejects the deprecated `cnn` value and accepts new `xgb_v45`. `cnn_agent.generate_signal` adds a v4.5 driver branch using the `indep_thresholds` rule (matching `tools/v4_5_horizon_compare.py`). The shadow gate flips from conditional to unconditional — `xgb_prob_shadow_v4_5` runs every scan, regardless of driver. The CNN driver branch, model load, Glu1 class, `_linear` fallback, and torch import are removed.

**Tech Stack:** Python 3.11, FastAPI/asyncio backend, `xgboost`, `pytest`/`pytest-asyncio`, SQLite via aiosqlite.

**Spec:** `docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md`

**Pre-conditions (operator-confirmed before Task 1 starts):**
- Live backend on port 8001 is **paused** (`is_trading: false` or process stopped). Verify: `Invoke-WebRequest http://localhost:8001/api/status | ConvertFrom-Json` returns `is_trading: false`, OR connection refused.
- Working tree clean except this plan doc + the spec doc (both untracked / unstaged is OK).
- No training subprocesses running. Verify: `Get-Process python | Where-Object { $_.CommandLine -like '*train_xgb*' }` returns empty.
- Current branch is `master` (or whatever branch operator intends). Verify: `git status --short --branch`.

**Working directory:** `C:\Users\gl450\polymarket_app\backend\` for all pytest invocations. Use `cd backend && python -m pytest …` from the repo root.

---

## Task 0: Pre-flight verification

**Files:** None (read-only checks)

- [ ] **Step 1: Confirm 8001 paused**

Run:
```powershell
$r = try { (Invoke-WebRequest -Uri "http://localhost:8001/api/status" -UseBasicParsing -TimeoutSec 3).Content | ConvertFrom-Json } catch { $null }; if ($null -eq $r) { "8001 DOWN (good)" } elseif ($r.is_trading -eq $false) { "8001 PAUSED (good)" } else { "8001 STILL TRADING — STOP" }
```

Expected: `8001 DOWN (good)` or `8001 PAUSED (good)`. If `STILL TRADING`, stop the plan and ask operator to pause first.

- [ ] **Step 2: Confirm no training subprocesses**

Run:
```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*train_xgb*' -or $_.CommandLine -like '*train_worker*' } | Select-Object ProcessId, CommandLine | Format-List
```

Expected: empty output. If anything matches, stop.

- [ ] **Step 3: Snapshot starting HEAD SHA**

Run:
```bash
git rev-parse HEAD
```

Expected: a 40-char SHA. Save this — it's the rollback target if anything goes wrong.

- [ ] **Step 4: Sanity import check**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -c "import agents.cnn_agent, agents.xgb_signal, config; print('import OK')"
```

Expected: `import OK`. If any ImportError, fix before continuing.

---

## Task 1: Add config validation + new thresholds (spec tests 1-4)

**Files:**
- Modify: `backend/config.py` (lines 75-79 and add new fields after)
- Modify: `backend/tests/test_config.py` (add new test class)

- [ ] **Step 1: Write the 4 failing tests**

Append to `backend/tests/test_config.py`:

```python
class TestModelBackendValidation:
    """Validates the MODEL_BACKEND env-var contract after CNN deprecation (2026-05-23)."""

    def test_model_backend_cnn_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "cnn")
        from config import Config
        with pytest.raises(ValueError, match="deprecated"):
            Config()

    def test_model_backend_xgb_v45_accepted(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "xgb_v45")
        from config import Config
        cfg = Config()
        assert cfg.model_backend == "xgb_v45"

    def test_model_backend_unknown_raises(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "lstm")
        from config import Config
        with pytest.raises(ValueError, match="invalid"):
            Config()

    def test_xgb_v45_threshold_defaults(self, monkeypatch):
        monkeypatch.delenv("XGB_V45_THRESH_UP",   raising=False)
        monkeypatch.delenv("XGB_V45_THRESH_DOWN", raising=False)
        monkeypatch.setenv("MODEL_BACKEND", "xgb")  # avoid touching default flip in this test
        from config import Config
        cfg = Config()
        assert cfg.xgb_v45_thresh_up   == 0.50
        assert cfg.xgb_v45_thresh_down == 0.50
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_config.py::TestModelBackendValidation -v
```

Expected: 4 FAILs. The first two should fail with AttributeError or wrong default; the third should fail because no validation exists; the fourth should fail with `AttributeError: 'Config' object has no attribute 'xgb_v45_thresh_up'`.

- [ ] **Step 3: Implement config changes**

Edit `backend/config.py`:

Replace lines 75-79 (the existing model_backend block) with:

```python
    # ── Model backend selector ─────────────────────────────────────────────────
    # Valid values: "xgb" (v3 driver, default) | "xgb_v45" (v4.5 driver, dev).
    # Legacy "cnn" value raises ValueError at startup — CNN driver was deprecated
    # 2026-05-23. See docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md.
    model_backend:       str  = field(default_factory=lambda: _validate_backend(os.getenv("MODEL_BACKEND", "xgb").lower()))

    # ── v4.5 indep_thresholds decision rule ────────────────────────────────────
    # BUY  when p_up   > xgb_v45_thresh_up   AND p_up   >= p_down.
    # SELL when p_down > xgb_v45_thresh_down AND p_down >  p_up.
    # Defaults 0.50/0.50 match tools/v4_5_horizon_compare.py:138.
    xgb_v45_thresh_up:   float = field(default_factory=lambda: float(os.getenv("XGB_V45_THRESH_UP",   "0.50")))
    xgb_v45_thresh_down: float = field(default_factory=lambda: float(os.getenv("XGB_V45_THRESH_DOWN", "0.50")))
```

And insert this helper above the `@dataclass class Config` line (around line 14):

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

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_config.py::TestModelBackendValidation -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run full test_config.py to catch regressions**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Expected: all PASS. If `test_no_dead_llm_blend_fields` or other policy tests fail, investigate before continuing — they may need the new env vars added to whitelists.

---

## Task 2: Add indep_thresholds decision helper (spec tests 5-12)

**Files:**
- Create: `backend/tests/test_xgb_v45_decision.py`
- Modify: `backend/agents/cnn_agent.py` (add helper near other private module-level helpers)

- [ ] **Step 1: Write the 8 failing tests**

Create `backend/tests/test_xgb_v45_decision.py`:

```python
"""Tests for v4.5 indep_thresholds decision rule.

Mirrors the rule in backend/tools/v4_5_horizon_compare.py:138 exactly:
  BUY  when p_up   > thresh_up   AND p_up   >= p_down (tie -> BUY)
  SELL when p_down > thresh_down AND p_down >  p_up  (strict)
"""
import pytest
from agents.cnn_agent import _indep_thresholds_decision


THRESH = 0.50  # matches horizon_compare default


class TestIndepThresholdsDecision:

    def test_indep_strong_up(self):
        side, strength = _indep_thresholds_decision(0.10, 0.10, 0.80, THRESH, THRESH)
        assert side == "BUY"
        assert strength == 0.800

    def test_indep_strong_down(self):
        side, strength = _indep_thresholds_decision(0.80, 0.10, 0.10, THRESH, THRESH)
        assert side == "SELL"
        assert strength == 0.800

    def test_indep_neutral_dominant(self):
        side, strength = _indep_thresholds_decision(0.25, 0.50, 0.25, THRESH, THRESH)
        assert side == "HOLD"
        assert strength == 0.0

    def test_indep_both_below_threshold(self):
        side, strength = _indep_thresholds_decision(0.49, 0.02, 0.49, THRESH, THRESH)
        assert side == "HOLD"
        assert strength == 0.0

    def test_indep_p_up_at_exact_threshold_holds(self):
        # Strict-greater rule: p_up == 0.50 is NOT > 0.50
        side, strength = _indep_thresholds_decision(0.49, 0.01, 0.50, THRESH, THRESH)
        assert side == "HOLD"
        assert strength == 0.0

    def test_indep_buy_wins_tie_when_both_above_threshold(self):
        # Asymmetric tie: BUY's >= accepts, SELL's > rejects
        side, strength = _indep_thresholds_decision(0.51, 0.00, 0.51, THRESH, THRESH)
        assert side == "BUY"
        assert strength == 0.510

    def test_indep_buy_when_p_up_marginally_exceeds(self):
        side, strength = _indep_thresholds_decision(0.49, 0.00, 0.51, THRESH, THRESH)
        assert side == "BUY"
        assert strength == 0.510

    def test_indep_sell_when_p_down_marginally_exceeds(self):
        side, strength = _indep_thresholds_decision(0.51, 0.00, 0.49, THRESH, THRESH)
        assert side == "SELL"
        assert strength == 0.510
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_xgb_v45_decision.py -v
```

Expected: 8 FAILs, all with `ImportError: cannot import name '_indep_thresholds_decision' from 'agents.cnn_agent'`.

- [ ] **Step 3: Add the decision helper to `cnn_agent.py`**

Find the module-level private helpers section in `backend/agents/cnn_agent.py` (near `_mask_training_constant_channels`, before the class definition `class CoinbaseCNNAgent`). Add:

```python
def _indep_thresholds_decision(
    p_down: float, p_neutral: float, p_up: float,
    thresh_up: float, thresh_down: float,
) -> Tuple[str, float]:
    """v4.5 indep_thresholds rule. Mirrors tools/v4_5_horizon_compare.py:138.

    BUY  when p_up   > thresh_up   AND p_up   >= p_down (tie -> BUY).
    SELL when p_down > thresh_down AND p_down >  p_up  (strict).
    Else HOLD.

    Returns (side, strength) where strength is the winning class probability
    rounded to 3 decimal places, or 0.0 for HOLD.
    """
    if p_up > thresh_up and p_up >= p_down:
        return "BUY", round(p_up, 3)
    if p_down > thresh_down and p_down > p_up:
        return "SELL", round(p_down, 3)
    return "HOLD", 0.0
```

If `Tuple` isn't already imported at the top of `cnn_agent.py`, add `from typing import Tuple` (or use the existing typing import line).

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_xgb_v45_decision.py -v
```

Expected: 8 PASS.

---

## Task 3: Flip shadow gate to always-on (spec test 15)

**Files:**
- Modify: `backend/agents/cnn_agent.py:1899-1911` (shadow gate)
- Modify: `backend/tests/test_cnn_agent.py` (add `test_generate_signal_xgb_logs_v45_shadow`)

- [ ] **Step 1: Locate the existing shadow gate and similar tests**

Run:
```bash
cd backend && grep -n "xgb_shadow_v45" tests/test_cnn_agent.py | head -5
```

Use the matching `model_backend="cnn"` shadow test as a structural template for the new always-on test. Note the line where it is — we'll add the new test next to it.

- [ ] **Step 2: Write the failing test**

Append to `backend/tests/test_cnn_agent.py` in the test class that currently covers v4.5 shadow (e.g., `TestV45Shadow` or similar — check via the grep above):

```python
    @pytest.mark.asyncio
    async def test_generate_signal_xgb_logs_v45_shadow(
        self, agent_with_db, mock_coinbase, monkeypatch,
    ):
        """Under MODEL_BACKEND=xgb (v3 driver), v4.5 shadow still logs every scan."""
        monkeypatch.setattr(config, "model_backend", "xgb")

        # Mock the shadow function to return v3 prob + v4.5 3-tuple
        async def _fake_xgb_shadow(channels, pid=None):
            return (0.65, (0.10, 0.10, 0.80))

        monkeypatch.setattr(
            "agents.xgb_signal.xgb_prob_shadow_v4_5",
            lambda channels, pid=None: (0.65, (0.10, 0.10, 0.80)),
        )

        saved = {}

        async def _capture_save(**kwargs):
            saved.update(kwargs)

        monkeypatch.setattr("database.save_cnn_scan", _capture_save)

        await agent_with_db.generate_signal("BTC-USD")

        # v3 drove the decision (model_prob == 0.65 > 0.60 cnn_buy_threshold → BUY).
        # But v4.5 shadow probs must STILL be persisted, atomic 3-tuple.
        assert saved.get("xgb_prob_v4_5_down")    == pytest.approx(0.10, abs=1e-6)
        assert saved.get("xgb_prob_v4_5_neutral") == pytest.approx(0.10, abs=1e-6)
        assert saved.get("xgb_prob_v4_5_up")      == pytest.approx(0.80, abs=1e-6)
```

Note: the exact fixture names (`agent_with_db`, `mock_coinbase`) and import path for `config` should match what already exists in `test_cnn_agent.py`. Check the file's existing test structure and conform.

- [ ] **Step 3: Run test to verify it fails**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestV45Shadow::test_generate_signal_xgb_logs_v45_shadow -v
```

Expected: FAIL with `xgb_prob_v4_5_down` being `None` (because the current gate short-circuits under `model_backend == "xgb"`).

- [ ] **Step 4: Replace the shadow gate**

In `backend/agents/cnn_agent.py`, find the block at lines ~1897-1911:

```python
            # Shadow XGB probability — log every scan regardless of MODEL_BACKEND
            # so we can compare CNN vs XGB calibration on identical inputs (#181).
            if config.model_backend == "xgb":
                xgb_shadow = cnn_prob
                xgb_shadow_v45 = None
            else:
                try:
                    from agents import xgb_signal as _xgb
                    xgb_shadow, xgb_shadow_v45 = _xgb.xgb_prob_shadow_v4_5(
                        _mask_training_constant_channels(channels),
                        pid=pid,
                    )
                except Exception:
                    xgb_shadow = None
                    xgb_shadow_v45 = None
```

Replace with unconditional shadow logging:

```python
            # Always log v3 + v4.5 shadow probabilities, regardless of driver
            # (#181 + 2026-05-23 CNN deprecation). xgb_prob_shadow_v4_5 has
            # built-in isolation: v4.5 failure -> v45=None, never affects v3.
            try:
                from agents import xgb_signal as _xgb
                xgb_shadow, xgb_shadow_v45 = _xgb.xgb_prob_shadow_v4_5(
                    _mask_training_constant_channels(channels),
                    pid=pid,
                )
            except Exception:
                xgb_shadow     = None
                xgb_shadow_v45 = None
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestV45Shadow::test_generate_signal_xgb_logs_v45_shadow -v
```

Expected: PASS.

- [ ] **Step 6: Run full v4.5 shadow test class to catch regressions**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestV45Shadow -v
```

Expected: all PASS.

---

## Task 4: Verify shadow-failure isolation still holds (spec test 16)

**Files:**
- Modify: `backend/tests/test_cnn_agent.py` (add `test_generate_signal_xgb_handles_v45_shadow_failure`)

`xgb_prob_shadow_v4_5` already has the isolated try/except per invariant #17. This task confirms the always-on shadow gate doesn't break that contract.

- [ ] **Step 1: Write the failing test**

Append to the same test class as Task 3:

```python
    @pytest.mark.asyncio
    async def test_generate_signal_xgb_handles_v45_shadow_failure(
        self, agent_with_db, mock_coinbase, monkeypatch,
    ):
        """v4.5 inference failure under MODEL_BACKEND=xgb -> all 3 v45 probs NULL atomically.

        v3 driver decision must be unaffected (invariant #16, #17).
        """
        monkeypatch.setattr(config, "model_backend", "xgb")

        def _raising_shadow(channels, pid=None):
            return (0.65, None)  # xgb_prob_shadow_v4_5 already isolates v45 internally

        monkeypatch.setattr(
            "agents.xgb_signal.xgb_prob_shadow_v4_5",
            _raising_shadow,
        )

        saved = {}

        async def _capture_save(**kwargs):
            saved.update(kwargs)

        monkeypatch.setattr("database.save_cnn_scan", _capture_save)

        await agent_with_db.generate_signal("BTC-USD")

        # All three v4.5 probs NULL atomically
        assert saved.get("xgb_prob_v4_5_down")    is None
        assert saved.get("xgb_prob_v4_5_neutral") is None
        assert saved.get("xgb_prob_v4_5_up")      is None
        # v3 shadow still recorded
        assert saved.get("xgb_prob") == pytest.approx(0.65, abs=1e-6)
```

- [ ] **Step 2: Run test**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestV45Shadow::test_generate_signal_xgb_handles_v45_shadow_failure -v
```

Expected: This should already PASS due to Task 3's always-on gate routing through `xgb_prob_shadow_v4_5` which returns `(v3, None)` when v4.5 fails. If it FAILS, inspect the `save_cnn_scan` call site at `cnn_agent.py:save_cnn_scan(...)` and ensure it unpacks `xgb_shadow_v45` correctly (expects either a 3-tuple or None; never raise).

---

## Task 5: Add v4.5 driver branch in generate_signal (spec tests 13-14)

**Files:**
- Modify: `backend/agents/cnn_agent.py:1965-1973` (decision branch)
- Modify: `backend/tests/test_cnn_agent.py` (add 2 driver tests)

- [ ] **Step 1: Write the 2 failing tests**

Append to a new or existing test class in `backend/tests/test_cnn_agent.py`:

```python
class TestV45DriverPath:
    """Tests for MODEL_BACKEND=xgb_v45 driver path (added 2026-05-23)."""

    @pytest.mark.asyncio
    async def test_generate_signal_xgb_v45_driver_path(
        self, agent_with_db, mock_coinbase, monkeypatch,
    ):
        """Under MODEL_BACKEND=xgb_v45, BUY/SELL/HOLD decided by indep_thresholds on v4.5 probs."""
        monkeypatch.setattr(config, "model_backend", "xgb_v45")
        monkeypatch.setattr(config, "xgb_v45_thresh_up",   0.50)
        monkeypatch.setattr(config, "xgb_v45_thresh_down", 0.50)

        # p_up=0.80 dominant -> BUY
        monkeypatch.setattr(
            "agents.xgb_signal.xgb_prob_shadow_v4_5",
            lambda channels, pid=None: (0.55, (0.10, 0.10, 0.80)),
        )

        saved = {}

        async def _capture_save(**kwargs):
            saved.update(kwargs)

        monkeypatch.setattr("database.save_cnn_scan", _capture_save)

        result = await agent_with_db.generate_signal("BTC-USD")

        # v4.5 drove the decision
        assert result.get("side") == "BUY"
        # 3 v45 probs persisted
        assert saved.get("xgb_prob_v4_5_up")   == pytest.approx(0.80, abs=1e-6)
        assert saved.get("xgb_prob_v4_5_down") == pytest.approx(0.10, abs=1e-6)
        # v3 shadow also persisted (always-on)
        assert saved.get("xgb_prob") == pytest.approx(0.55, abs=1e-6)

    @pytest.mark.asyncio
    async def test_generate_signal_xgb_v45_holds_on_v45_failure(
        self, agent_with_db, mock_coinbase, monkeypatch,
    ):
        """Under MODEL_BACKEND=xgb_v45, v4.5 inference failure -> HOLD (never trade on garbage)."""
        monkeypatch.setattr(config, "model_backend", "xgb_v45")

        # v45 failed -> None
        monkeypatch.setattr(
            "agents.xgb_signal.xgb_prob_shadow_v4_5",
            lambda channels, pid=None: (0.55, None),
        )

        saved = {}

        async def _capture_save(**kwargs):
            saved.update(kwargs)

        monkeypatch.setattr("database.save_cnn_scan", _capture_save)

        result = await agent_with_db.generate_signal("BTC-USD")

        assert result.get("side") == "HOLD"
        # v3 prob still persisted
        assert saved.get("xgb_prob") == pytest.approx(0.55, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestV45DriverPath -v
```

Expected: Both FAIL — `MODEL_BACKEND=xgb_v45` currently doesn't trigger the v4.5 decision branch; `_cnn_prob` returns v3 prob and the existing 2-class gate fires using v3 (returns BUY or HOLD based on the 0.55 v3 prob).

- [ ] **Step 3: Add the v4.5 driver decision branch**

In `backend/agents/cnn_agent.py`, find the side-decision block at line ~1964-1973:

```python
        # ── Signal direction ──────────────────────────────────────────────────
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

Replace with branched dispatch:

```python
        # ── Signal direction ──────────────────────────────────────────────────
        if config.model_backend == "xgb_v45":
            # v4.5 3-class driver — indep_thresholds rule on (p_down, p_neutral, p_up).
            if xgb_shadow_v45 is None:
                # v4.5 inference failed — HOLD to avoid trading on garbage signal.
                side, strength = "HOLD", 0.0
            else:
                p_down, p_neutral, p_up = xgb_shadow_v45
                side, strength = _indep_thresholds_decision(
                    p_down, p_neutral, p_up,
                    thresh_up   = config.xgb_v45_thresh_up,
                    thresh_down = config.xgb_v45_thresh_down,
                )
        else:
            # MODEL_BACKEND=xgb — existing 2-class gate on v3 prob.
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

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestV45DriverPath -v
```

Expected: Both PASS.

---

## Task 6: Remove CNN driver code

**Files:**
- Modify: `backend/agents/cnn_agent.py` (delete CNN branch, model load, Glu1 class, _build_cnn, _linear, _TORCH import, self.model/self.fb)

- [ ] **Step 1: Locate every CNN-driver surface to delete**

Run:
```bash
cd backend && grep -n "_TORCH\|SignalCNNGlu1\|_build_cnn\|cnn_model_glu1\|self\.model\|self\.fb\|def _linear" agents/cnn_agent.py
```

Note every line number returned. These are the removal targets.

- [ ] **Step 2: Delete the CNN branch in `_cnn_prob`**

In `_cnn_prob` (line ~1612-1621), remove the CNN driver block. The method becomes:

```python
    def _cnn_prob(self, channels, pid: Optional[str] = None) -> float:
        # Align inference input with the training distribution — zero out the
        # channels that were constant-zero at training (P3b).
        channels = _mask_training_constant_channels(channels)
        if config.model_backend in ("xgb", "xgb_v45"):
            from agents import xgb_signal
            return xgb_signal.xgb_prob(channels, pid=pid)
        # _validate_backend guarantees this branch is unreachable.
        raise RuntimeError(f"unsupported model_backend={config.model_backend!r}")
```

- [ ] **Step 3: Delete `_linear` method**

Remove lines ~1623-1638 (the entire `@staticmethod def _linear(channels):` block).

- [ ] **Step 4: Delete `SignalCNNGlu1` class**

Search for `class SignalCNNGlu1` — delete the entire class definition.

- [ ] **Step 5: Delete `_build_cnn` factory**

Search for `def _build_cnn` — delete the function.

- [ ] **Step 6: Delete `self.model` and `self.fb` init in `__init__`**

In `CoinbaseCNNAgent.__init__`, find and delete:
- `self.model = ...` (the line that loads `cnn_model_glu1.pt` or calls `_build_cnn`)
- `self.fb = ...` (the feature-builder field, only used by `self.model.predict`)
- Any related `torch.load(...)` call
- Any `try/except` blocks that only existed to wrap the CNN model load

- [ ] **Step 7: Delete `_TORCH` import**

Search for `_TORCH =` (likely near the top of the file, around `try: import torch; _TORCH = True except ImportError: _TORCH = False`). Delete the import + guard. If `torch` is imported anywhere else in `cnn_agent.py` for non-CNN reasons, leave that import alone.

- [ ] **Step 8: Run import smoke check**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -c "import agents.cnn_agent; print('import OK')"
```

Expected: `import OK`. If `NameError` or `ImportError`, you missed a reference. Grep for the missing name and fix.

- [ ] **Step 9: Run all `cnn_agent` tests**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py -v
```

Expected: most PASS. Some tests will FAIL — those are the stale CNN tests Task 7 deletes. Note their names.

---

## Task 7: Delete stale CNN tests

**Files:**
- Modify: `backend/tests/test_cnn_agent.py` (delete stale tests)

- [ ] **Step 1: Identify stale tests**

Run:
```bash
cd backend && grep -n "SignalCNNGlu1\|_build_cnn\|\.model\.predict\|MODEL_BACKEND.*cnn" tests/test_cnn_agent.py | head -40
```

For each match, decide:
- Test references `SignalCNNGlu1` directly → DELETE
- Test references `_build_cnn` → DELETE
- Test mocks/asserts `agent.model.predict(...)` as the driver → DELETE
- Test sets `MODEL_BACKEND=cnn` to assert CNN drove the signal → DELETE
- Test sets `MODEL_BACKEND=cnn` to assert post-deprecation behavior (e.g., auto-train gating, the new ValueError) → KEEP or convert to `MODEL_BACKEND=xgb`/`xgb_v45`

- [ ] **Step 2: Delete the identified test methods/classes**

Use `Edit` tool to remove each stale block. Preserve the file's class/function structure (don't leave dangling pass statements or unused imports).

- [ ] **Step 3: Run test_cnn_agent.py to verify GREEN**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py -v
```

Expected: all PASS.

- [ ] **Step 4: Run test_xgb_signal.py for cross-check**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_xgb_signal.py -v
```

Expected: all PASS. If anything fails, the CNN driver removal touched something xgb_signal relies on — investigate.

- [ ] **Step 5: Run the policy/config tests**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```

Expected: all PASS, including `test_no_dead_llm_blend_fields` regression and `TestModelBackendValidation` from Task 1.

---

## Task 8: Update memory + architecture + thresholds memory files

**Files:**
- Modify: `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\feedback_xgb_focus_not_cnn.md`
- Modify: `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`
- Modify: `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_thresholds.md`

- [ ] **Step 1: Update `feedback_xgb_focus_not_cnn.md`**

Update the body to reflect that the driver removal is complete. Suggested edit to the "Why" section: append "**Update 2026-05-23:** CNN driver path, `_linear` fallback, `SignalCNNGlu1` class, and `_TORCH` import all removed. `MODEL_BACKEND` now validates at startup — only `xgb` (v3, default) and `xgb_v45` (v4.5) are accepted. Legacy `cnn` value raises ValueError. Class name `CoinbaseCNNAgent` + DB columns `cnn_scans/cnn_w/llm_w` + frontend filename still pending backlog task #7."

- [ ] **Step 2: Update `coinbase_trader_architecture.md`**

Find the section describing the agent pipeline / MODEL_BACKEND values. Update to reflect:
- `MODEL_BACKEND=xgb` (default) → v3 driver via `xgb_signal.xgb_prob`
- `MODEL_BACKEND=xgb_v45` → v4.5 driver via `_indep_thresholds_decision` on `xgb_prob_shadow_v4_5` 3-tuple
- v4.5 shadow now logs unconditionally on every scan
- CNN model load + Glu1 class removed
- Legacy `cnn` value raises at startup

- [ ] **Step 3: Update `coinbase_trader_thresholds.md`**

Add new thresholds section:
```
## v4.5 indep_thresholds decision rule (added 2026-05-23)
- `XGB_V45_THRESH_UP   = 0.50`  (default — matches tools/v4_5_horizon_compare.py)
- `XGB_V45_THRESH_DOWN = 0.50`  (default — same)
- BUY  when p_up   > 0.50 AND p_up   >= p_down
- SELL when p_down > 0.50 AND p_down >  p_up
- Else HOLD
```

---

## Task 9: Update CHANGELOG

**Files:**
- Modify: `C:\Users\gl450\polymarket_app\CHANGELOG.md`

- [ ] **Step 1: Read the most recent CHANGELOG entry to match format**

Run:
```bash
head -50 CHANGELOG.md
```

Note the session-naming convention (e.g., "Session 58.71X") and entry shape.

- [ ] **Step 2: Prepend a new entry**

Insert at the top of `CHANGELOG.md` (after any preamble), matching the existing entry format:

```markdown
## [Session 58.71m] — 2026-05-23 — Remove CNN driver + add XGB v4.5 driver path

**Why:** CNN was declared deprecated 2026-05-18 (`feedback_xgb_focus_not_cnn.md`). The driver branch in `_cnn_prob`, the `cnn_model_glu1.pt` checkpoint load, and the `SignalCNNGlu1` class remained as dead-but-loadable code. The dev backend on 8002 was forced to run `MODEL_BACKEND=cnn` solely to activate v4.5 shadow logging — directly contradicting "XGB only". This change removes the CNN driver fully and promotes v4.5 to a first-class driver path on 8002 for direct ROI measurement (per `feedback_roi_first_priority.md`).

**Changes:**

- **`backend/config.py`** — added `_validate_backend(value)`. Valid `MODEL_BACKEND` values now `{xgb, xgb_v45}`. Legacy `cnn` raises `ValueError` with migration message. Default flipped from `cnn` to `xgb`. Added `xgb_v45_thresh_up` / `xgb_v45_thresh_down` fields (defaults 0.50, mirror `tools/v4_5_horizon_compare.py:138`).

- **`backend/agents/cnn_agent.py`** —
  - DELETED CNN branch in `_cnn_prob` (`if _TORCH and self.model: return self.model.predict(...)`).
  - DELETED `_linear` 6-channel fallback.
  - DELETED `SignalCNNGlu1` class + `_build_cnn` factory + `_TORCH` import.
  - DELETED `self.model` + `self.fb` field init + `cnn_model_glu1.pt` checkpoint load.
  - REPLACED shadow gate with unconditional `xgb_prob_shadow_v4_5` call — v4.5 shadow now logs every scan regardless of driver (`MODEL_BACKEND`).
  - ADDED `_indep_thresholds_decision(p_down, p_neutral, p_up, thresh_up, thresh_down) -> (side, strength)` helper. Mirrors horizon_compare's indep_thresholds rule exactly (asymmetric tie: BUY's `p_up >= p_down` accepts tie, SELL's `p_down > p_up` strict).
  - ADDED v4.5 driver branch in `generate_signal`: under `MODEL_BACKEND=xgb_v45`, BUY/SELL/HOLD is decided by `_indep_thresholds_decision` on the 3-tuple from `xgb_prob_shadow_v4_5`. v4.5 failure (`None`) → HOLD.

- **`backend/tests/test_config.py`** — added `TestModelBackendValidation` (4 tests).
- **`backend/tests/test_xgb_v45_decision.py`** (NEW FILE) — 8 indep_thresholds tests covering strong/marginal/threshold-edge/tie cases.
- **`backend/tests/test_cnn_agent.py`** — added 4 tests (`test_generate_signal_xgb_v45_driver_path`, `test_generate_signal_xgb_v45_holds_on_v45_failure`, `test_generate_signal_xgb_logs_v45_shadow`, `test_generate_signal_xgb_handles_v45_shadow_failure`). Deleted stale CNN-driver tests (count: N — fill in actual after Task 7).

**Net diff:** approx −X LOC code, +Y LOC tests, +1 new test file (fill in after measurement).

**Verify:**
```bash
cd backend && python -m pytest tests/test_config.py tests/test_xgb_v45_decision.py tests/test_cnn_agent.py tests/test_xgb_signal.py -v
```

**Deployment (operator):**
1. Restart 8001 with `MODEL_BACKEND=xgb` (or default).
2. Kill 8002 (current PID is the `python.exe` listening on 8002).
3. Relaunch 8002 with `PORT=8002 MODEL_BACKEND=xgb_v45 DATABASE_URL=coinbase_dev.db`.

**Out of scope (deferred to backlog task #7, gated on shadow-week success):**
- Class rename `CoinbaseCNNAgent`
- DB column rename `cnn_scans / cnn_w / llm_w`
- Frontend file rename `CNNDashboard.tsx`
- On-disk artifact deletion `cnn_model_glu1.pt`
```

---

## Task 10: Full pytest sweep + atomic commit

**Files:** (commit collects all changes)

- [ ] **Step 1: Run full backend test suite**

Run:
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest tests/ -v --tb=short
```

Expected: all PASS. If anything fails outside the scope of this change, investigate before committing.

- [ ] **Step 2: Verify no stray `MODEL_BACKEND=cnn` consumers**

Run:
```bash
cd backend && grep -rn "MODEL_BACKEND.*cnn\|model_backend.*== ?\"cnn\"" --include="*.py" .
```

Expected: only matches in CHANGELOG (string mention) and inside the new `_validate_backend` error message. Nothing in live decision code.

- [ ] **Step 3: Stage all changes explicitly (no `git add -A`)**

Run:
```bash
git add backend/config.py backend/agents/cnn_agent.py backend/tests/test_config.py backend/tests/test_cnn_agent.py backend/tests/test_xgb_v45_decision.py CHANGELOG.md docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md docs/superpowers/plans/2026-05-23-remove-cnn-driver-add-v45-driver.md
git status --short
```

Expected: all listed files staged (status `M` or `A` in left column). No `.env`, no `.db`, no `.pt`.

- [ ] **Step 4: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(model): remove CNN driver + add MODEL_BACKEND=xgb_v45 path

CNN paper-trading path deprecated 2026-05-18 had remained as dead-but-loadable
code (cnn_model_glu1.pt load, SignalCNNGlu1 class, _linear fallback, _TORCH
import). This commit rips them all out and adds MODEL_BACKEND=xgb_v45 as a
first-class v4.5 driver path so 8002 can paper-trade on v4.5 directly (for
ROI measurement) instead of forcing CNN as the driver just to log v4.5 shadow.

Key changes:
- config.MODEL_BACKEND: valid={xgb, xgb_v45}; legacy "cnn" raises ValueError.
- cnn_agent.generate_signal: shadow gate now unconditional; v4.5 logs every
  scan regardless of driver.
- cnn_agent: new _indep_thresholds_decision helper mirrors
  tools/v4_5_horizon_compare.py's winning rule exactly.
- New thresholds: xgb_v45_thresh_up/down default 0.50.

Out of scope (deferred to backlog #7): CoinbaseCNNAgent class rename,
cnn_scans/cnn_w/llm_w DB column renames, CNNDashboard.tsx filename.

Spec: docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md
Plan: docs/superpowers/plans/2026-05-23-remove-cnn-driver-add-v45-driver.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds, pre-commit hook passes (or no hook configured).

- [ ] **Step 5: Verify commit**

Run:
```bash
git log -1 --stat
```

Expected: shows the new commit + file change summary.

- [ ] **Step 6: Stop. Wait for operator decision on push.**

Per CLAUDE.md, never push to remote without explicit operator approval. Surface the commit SHA + ask whether to push.

---

## Task 11: Operator deployment

**Files:** None (operational only)

This task is operator-driven. The plan documents the steps; the agent surfaces the commands and waits.

- [ ] **Step 1: Operator restarts 8001**

The operator restarts the live backend (via the launcher or `.venv/Scripts/python.exe main.py`). After restart, verify:

```powershell
$r = (Invoke-WebRequest -Uri "http://localhost:8001/api/status" -UseBasicParsing -TimeoutSec 3).Content | ConvertFrom-Json; $r | Select-Object is_trading, dry_run, timestamp | Format-List
```

Expected: `is_trading: true` (or whatever the operator's preference), `dry_run: true`.

- [ ] **Step 2: Operator kills 8002**

Find the 8002 PID:

```powershell
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -eq 8002 } | Select-Object OwningProcess
```

Kill it (operator runs):

```powershell
Stop-Process -Id <PID> -Force
```

- [ ] **Step 3: Operator relaunches 8002 with v4.5 driver env**

From the project root, in a new PowerShell:

```powershell
$env:PORT = "8002"
$env:MODEL_BACKEND = "xgb_v45"
$env:DATABASE_URL = "coinbase_dev.db"
cd backend
..\.venv\Scripts\python.exe main.py
```

(Or whatever the operator's preferred launcher invocation is — adjust env-var pass for cmd.exe vs PowerShell.)

- [ ] **Step 4: Verify 8002 is up and using v4.5 driver**

```powershell
$r = (Invoke-WebRequest -Uri "http://localhost:8002/api/status" -UseBasicParsing -TimeoutSec 3).Content | ConvertFrom-Json; $r | Format-List
```

Expected: HTTP 200, fields present.

After ~1 scan interval (~15 min default), check `coinbase_dev.db.cnn_scans` for non-NULL `xgb_prob_v4_5_up` values:

```bash
cd backend && ../.venv/Scripts/python.exe -c "import sqlite3; conn = sqlite3.connect('coinbase_dev.db'); print(conn.execute('SELECT COUNT(*), COUNT(xgb_prob_v4_5_up) FROM cnn_scans WHERE created_at > datetime(\"now\", \"-1 hour\")').fetchone()); conn.close()"
```

Expected: both counts equal (or v4.5 count equal to scan count) — every scan logs v4.5 shadow.

Check 8001's live db for the same — should be ≥ 1 hour of always-on v4.5 shadow probs:

```bash
cd backend && ../.venv/Scripts/python.exe -c "import sqlite3; conn = sqlite3.connect('coinbase.db'); print(conn.execute('SELECT COUNT(*), COUNT(xgb_prob_v4_5_up) FROM cnn_scans WHERE created_at > datetime(\"now\", \"-1 hour\")').fetchone()); conn.close()"
```

Expected: both counts equal (post-deploy scans have v4.5 shadow populated).

- [ ] **Step 5: Verify 8002 is making v4.5-driven trades**

After another scan interval, query `coinbase_dev.db.signals` for recent BUY/SELL entries:

```bash
cd backend && ../.venv/Scripts/python.exe -c "import sqlite3; conn = sqlite3.connect('coinbase_dev.db'); print(conn.execute('SELECT product_id, side, strength, created_at FROM signals WHERE created_at > datetime(\"now\", \"-30 minutes\") ORDER BY created_at DESC LIMIT 10').fetchall()); conn.close()"
```

Expected: non-empty list. The `strength` values should be in [0.5, 1.0) — that's the v4.5 winning-class probability (per the helper's return spec). Under the old CNN driver they would have been `(model_prob - 0.5) * 2`.

---

## Rollback

If anything goes wrong post-deploy (8001 won't restart, errors flood logs, 8002 doesn't generate sensible signals):

```bash
git revert <commit-sha>
# Restart 8001 + 8002 with old envs (MODEL_BACKEND=xgb on 8001, MODEL_BACKEND=cnn on 8002 if you still want the old shadow path).
```

`cnn_model_glu1.pt` is still on disk (this commit doesn't touch it), so the reverted code's CNN driver path works again with no manual artifact restoration.

---

## Self-review checklist (DONE during plan authoring)

**Spec coverage:** Every spec section maps to a task — config validation (Task 1), decision helper (Task 2), shadow gate flip (Task 3), shadow failure isolation (Task 4), driver branch (Task 5), CNN removal (Task 6), stale test deletion (Task 7), memory updates (Task 8), CHANGELOG (Task 9), commit (Task 10), deployment (Task 11).

**Placeholder scan:** No TBDs/TODOs in code blocks. The "N" in "Deleted stale CNN-driver tests (count: N — fill in actual after Task 7)" is intentional — actual count is unknowable until grep runs.

**Type consistency:** `_indep_thresholds_decision(p_down, p_neutral, p_up, thresh_up, thresh_down) -> Tuple[str, float]` signature consistent across Tasks 2 and 5. `xgb_shadow_v45` is `Optional[Tuple[float, float, float]]` consistently. `save_cnn_scan` kwarg names (`xgb_prob_v4_5_down/neutral/up`) match the existing DB schema (per CLAUDE.md invariant #17).

---

## Execution

**Plan complete and saved to `docs/superpowers/plans/2026-05-23-remove-cnn-driver-add-v45-driver.md`.**

Recommended: **Inline execution** — this is a single small atomic refactor with tightly-coupled changes. Subagent-per-task adds overhead without proportional value. Use `superpowers:executing-plans` if available; otherwise the agent in this session executes tasks 1-10 in order, surfacing each checkpoint to the operator, then hands off Task 11 to the operator.

If the operator prefers subagent-driven (fresh context per task with two-stage review): use `superpowers:subagent-driven-development`.
