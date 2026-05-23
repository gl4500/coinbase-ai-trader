# XGB Deployment-Aligned Scorecard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployment-aligned multi-metric scorecard (precision-at-gate, expected return per signal, paper-Sharpe, ECE) for XGB v3 driver + v4 shadow + v4.5 shadow tracks, with side-by-side retail/mid/pro fee tiers and per-fold Sharpe annualization.

**Architecture:** Four pure-function metric computers (`_precision.py`, `_expected_return.py`, `_paper_sharpe.py`, `_ece.py`) composed by a `compute_scorecard` orchestrator returning a `ScorecardReport` dataclass. CLI runner `python -m tools.scorecard` loads the existing 28-channel cache, builds OOF predictions across 5 purged-WF folds, and prints the report. Realized-return helper is extracted from `feature_set_compare.py` into `_returns.py` first (loose-coupling prep).

**Tech Stack:** Python 3.11 (`.venv/Scripts/python.exe`), numpy, xgboost, dataclasses, pytest. No new dependencies. Existing cache: `backend/cnn_dataset_cache.pt` v12 (28-channel, ~167k samples, survivorship-aware top-20). Existing CV harness: 5-fold purged walk-forward with 4h embargo (see `tools/feature_set_compare.py`).

**Spec:** `docs/superpowers/specs/2026-05-18-xgb-deployment-scorecard-design.md` (commit `9accdbe`).

**Conventions:**
- TDD red-green-refactor per `feedback_tdd_workflow` — write failing test, run to confirm RED, implement to GREEN, commit.
- `.venv/Scripts/python.exe` for all probe-tool execution per `polymarket_app_python_interpreter`.
- Push immediately after each commit per `feedback_push_on_commit`.
- Pure functions with type hints, single responsibility per `feedback_python_clean_functions`.
- XGB-side only — do not modify `cnn_agent.py` per `feedback_xgb_focus_not_cnn`.

---

## File Structure

**Create:**
- `backend/tools/_returns.py` — pure helper: realized log-returns per sample from cache + candle history
- `backend/tools/_scorecard/__init__.py` — package marker
- `backend/tools/_scorecard/_precision.py` — `precision_at_tau(scores, labels, tau) -> float`
- `backend/tools/_scorecard/_expected_return.py` — `expected_return_at_tau(scores, returns, tau, fee) -> tuple[float, int]`
- `backend/tools/_scorecard/_paper_sharpe.py` — `paper_sharpe_per_fold(scores, returns, fold_ids, fold_spans_days, tau, fee) -> tuple[float, float]`
- `backend/tools/_scorecard/_ece.py` — `expected_calibration_error(scores, labels, n_bins=10) -> float`
- `backend/tools/_scorecard/_report.py` — `ScorecardReport` dataclass + formatting
- `backend/tools/scorecard.py` — `compute_scorecard(...)` orchestrator + `__main__` CLI

**Test:**
- `backend/tests/test_returns_helper.py`
- `backend/tests/test_scorecard_precision.py`
- `backend/tests/test_scorecard_expected_return.py`
- `backend/tests/test_scorecard_paper_sharpe.py`
- `backend/tests/test_scorecard_ece.py`
- `backend/tests/test_scorecard_orchestrator.py`
- `backend/tests/test_scorecard_cli.py`

**Modify:**
- `backend/tools/feature_set_compare.py` — replace inline `_realized_returns` with import from `_returns.py` (loose-coupling prep, see Task 1)

**Output:**
- `docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md` — baseline report (written by Task 9)

---

## Task 1: Extract realized-returns helper into shared module

**Files:**
- Create: `backend/tools/_returns.py`
- Test: `backend/tests/test_returns_helper.py`
- Modify: `backend/tools/feature_set_compare.py` (replace inline `_realized_returns` with import)

**Rationale:** Per `feedback_loose_coupling`, the scorecard's expected-return and paper-Sharpe metrics need the same realized-log-return-per-sample computation that `feature_set_compare.py` already does inline. Extract once before two consumers diverge.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_returns_helper.py
import numpy as np
from tools._returns import realized_log_returns_per_sample


def test_realized_log_returns_basic():
    """Given entry close and forward close, return ln(forward/entry)."""
    entry_closes = np.array([100.0, 100.0, 100.0])
    forward_closes = np.array([101.0, 99.0, 100.5])
    result = realized_log_returns_per_sample(entry_closes, forward_closes)
    expected = np.log(np.array([1.01, 0.99, 1.005]))
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_realized_log_returns_zero_when_equal():
    """ln(1) = 0 when forward equals entry."""
    e = np.array([50.0, 75.0])
    f = np.array([50.0, 75.0])
    result = realized_log_returns_per_sample(e, f)
    np.testing.assert_allclose(result, np.zeros(2), atol=1e-12)


def test_realized_log_returns_raises_on_nonpositive():
    """Non-positive prices indicate data corruption — fail loud."""
    import pytest
    with pytest.raises(ValueError, match="non-positive"):
        realized_log_returns_per_sample(np.array([100.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="non-positive"):
        realized_log_returns_per_sample(np.array([-1.0]), np.array([100.0]))


def test_realized_log_returns_shape_mismatch():
    """Mismatched array lengths should fail immediately."""
    import pytest
    with pytest.raises(ValueError, match="shape"):
        realized_log_returns_per_sample(np.array([100.0, 101.0]), np.array([102.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_returns_helper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools._returns'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tools/_returns.py
"""Pure helper: realized log-returns per sample from entry and forward closes.

Extracted from feature_set_compare.py per loose-coupling rule so both
feature_set_compare and the scorecard share one implementation.
"""
from __future__ import annotations

import numpy as np


def realized_log_returns_per_sample(
    entry_closes: np.ndarray,
    forward_closes: np.ndarray,
) -> np.ndarray:
    """Return ln(forward_close / entry_close) elementwise.

    Args:
        entry_closes: shape (N,), close price at sample entry bar.
        forward_closes: shape (N,), close price at the bar the triple-barrier
            resolved on (TP hit, SL hit, or timeout).

    Returns:
        shape (N,) log-returns.

    Raises:
        ValueError: if shapes mismatch or any price is <= 0.
    """
    if entry_closes.shape != forward_closes.shape:
        raise ValueError(
            f"shape mismatch: entry {entry_closes.shape} vs forward {forward_closes.shape}"
        )
    if (entry_closes <= 0).any() or (forward_closes <= 0).any():
        raise ValueError("non-positive price found in entry_closes or forward_closes")
    return np.log(forward_closes / entry_closes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_returns_helper.py -v`
Expected: 4 passed.

- [ ] **Step 5: Migrate `feature_set_compare.py` to import from helper**

Find the existing `_realized_returns` inline function in `backend/tools/feature_set_compare.py` (search: `def _realized_returns`). Replace its body with a call to the helper:

```python
# In feature_set_compare.py, replace the existing _realized_returns function body with:
from tools._returns import realized_log_returns_per_sample as _realized_returns_helper

def _realized_returns(entry_closes, forward_closes):
    return _realized_returns_helper(entry_closes, forward_closes)
```

If `_realized_returns` has a different signature in `feature_set_compare.py` (e.g., takes a single sample dict), keep the wrapper but route the actual log-arithmetic through the helper. Do NOT modify the call sites — only the helper internals.

- [ ] **Step 6: Run the existing feature_set_compare tests to confirm no regression**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/ -k "feature_set" -v`
Expected: all existing feature_set tests still pass.

- [ ] **Step 7: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_returns.py backend/tests/test_returns_helper.py backend/tools/feature_set_compare.py
git commit -m "refactor: extract realized_log_returns_per_sample to _returns.py

Loose-coupling prep for scorecard implementation — both feature_set_compare
and upcoming scorecard share the same per-sample log-return computation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Task 2: Precision-at-gate metric

**Files:**
- Create: `backend/tools/_scorecard/__init__.py` (empty package marker)
- Create: `backend/tools/_scorecard/_precision.py`
- Test: `backend/tests/test_scorecard_precision.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scorecard_precision.py
import numpy as np
import pytest
from tools._scorecard._precision import precision_at_tau


def test_precision_at_tau_basic():
    """3 fire at tau=0.5, 2 of 3 are positive => precision = 2/3."""
    scores = np.array([0.9, 0.7, 0.6, 0.4, 0.3])
    labels = np.array([1, 1, 0, 0, 1])  # last positive doesn't fire
    p, n_fired = precision_at_tau(scores, labels, tau=0.5)
    assert n_fired == 3
    assert p == pytest.approx(2 / 3)


def test_precision_at_tau_no_fires_returns_nan():
    """No samples above threshold => precision undefined (NaN), n_fired=0."""
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 0])
    p, n_fired = precision_at_tau(scores, labels, tau=0.9)
    assert n_fired == 0
    assert np.isnan(p)


def test_precision_at_tau_all_fire():
    """tau=0 => all fire => precision = pos_rate."""
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 1, 1])
    p, n_fired = precision_at_tau(scores, labels, tau=0.0)
    assert n_fired == 3
    assert p == pytest.approx(2 / 3)


def test_precision_at_tau_strict_gt():
    """Threshold is strict > tau, not >=. Sample at exactly tau does NOT fire."""
    scores = np.array([0.5, 0.5, 0.51])
    labels = np.array([1, 1, 1])
    p, n_fired = precision_at_tau(scores, labels, tau=0.5)
    assert n_fired == 1
    assert p == pytest.approx(1.0)


def test_precision_at_tau_rejects_nonbinary_labels():
    """Labels must be 0/1 — fail loud on other values."""
    with pytest.raises(ValueError, match="binary"):
        precision_at_tau(np.array([0.9]), np.array([2]), tau=0.5)


def test_precision_at_tau_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        precision_at_tau(np.array([0.9, 0.5]), np.array([1]), tau=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_precision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools._scorecard'`.

- [ ] **Step 3: Create package marker**

```python
# backend/tools/_scorecard/__init__.py
"""Per-metric scorecard computers. Composed by tools/scorecard.py."""
```

- [ ] **Step 4: Write minimal implementation**

```python
# backend/tools/_scorecard/_precision.py
"""Precision at a score threshold: of signals that fire, what fraction win?"""
from __future__ import annotations

import numpy as np


def precision_at_tau(
    scores: np.ndarray,
    labels: np.ndarray,
    tau: float,
) -> tuple[float, int]:
    """Compute precision among samples with score > tau.

    Args:
        scores: shape (N,) model output probabilities.
        labels: shape (N,) binary 0/1 ground truth.
        tau: strict-greater-than threshold.

    Returns:
        (precision, n_fired). precision = NaN if n_fired == 0.

    Raises:
        ValueError: on shape mismatch or non-binary labels.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")
    uniq = np.unique(labels)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"labels must be binary 0/1, got {uniq}")

    fired = scores > tau
    n_fired = int(fired.sum())
    if n_fired == 0:
        return float("nan"), 0
    n_tp = int(labels[fired].sum())
    return n_tp / n_fired, n_fired
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_precision.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/__init__.py backend/tools/_scorecard/_precision.py backend/tests/test_scorecard_precision.py
git commit -m "feat(scorecard): precision_at_tau metric

Pure function for precision among signals firing above threshold tau.
Returns (precision, n_fired); NaN precision when no fires.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Task 3: Expected-return-per-signal metric (multi-tier fees)

**Files:**
- Create: `backend/tools/_scorecard/_expected_return.py`
- Test: `backend/tests/test_scorecard_expected_return.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scorecard_expected_return.py
import numpy as np
import pytest
from tools._scorecard._expected_return import expected_return_at_tau


def test_expected_return_basic_with_fee():
    """Fired samples return mean(r) - 2*fee."""
    scores = np.array([0.9, 0.7, 0.4])
    returns = np.array([0.02, 0.01, 0.05])  # 0.05 doesn't fire
    er, n_fired = expected_return_at_tau(scores, returns, tau=0.5, fee=0.006)
    expected = np.mean([0.02, 0.01]) - 2 * 0.006
    assert n_fired == 2
    assert er == pytest.approx(expected)


def test_expected_return_no_fires_returns_nan():
    scores = np.array([0.1, 0.2])
    returns = np.array([0.05, 0.05])
    er, n_fired = expected_return_at_tau(scores, returns, tau=0.9, fee=0.006)
    assert n_fired == 0
    assert np.isnan(er)


def test_expected_return_fee_zero():
    """fee=0 gives raw mean return on fired samples."""
    scores = np.array([0.9, 0.8])
    returns = np.array([0.03, -0.01])
    er, n_fired = expected_return_at_tau(scores, returns, tau=0.5, fee=0.0)
    assert n_fired == 2
    assert er == pytest.approx(0.01)


def test_expected_return_rejects_negative_fee():
    with pytest.raises(ValueError, match="non-negative"):
        expected_return_at_tau(np.array([0.9]), np.array([0.02]), tau=0.5, fee=-0.001)


def test_expected_return_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        expected_return_at_tau(np.array([0.9, 0.5]), np.array([0.02]), tau=0.5, fee=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_expected_return.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tools/_scorecard/_expected_return.py
"""Expected log-return per fired signal, net of round-trip fee."""
from __future__ import annotations

import numpy as np


def expected_return_at_tau(
    scores: np.ndarray,
    returns: np.ndarray,
    tau: float,
    fee: float,
) -> tuple[float, int]:
    """Mean realized log-return on samples with score > tau, minus 2 * fee.

    Args:
        scores: shape (N,).
        returns: shape (N,) realized log-returns per sample.
        tau: strict-greater-than threshold.
        fee: per-side fee (e.g., 0.006 for 0.6%). Round-trip cost = 2 * fee.

    Returns:
        (expected_return, n_fired). NaN if no signals fire.

    Raises:
        ValueError: on shape mismatch or negative fee.
    """
    if scores.shape != returns.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs returns {returns.shape}")
    if fee < 0:
        raise ValueError(f"fee must be non-negative, got {fee}")

    fired = scores > tau
    n_fired = int(fired.sum())
    if n_fired == 0:
        return float("nan"), 0
    return float(returns[fired].mean() - 2 * fee), n_fired
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_expected_return.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_expected_return.py backend/tests/test_scorecard_expected_return.py
git commit -m "feat(scorecard): expected_return_at_tau metric

Single-tier expected-return computer; multi-tier reporting handled by
the orchestrator looping over FEE_TIERS dict.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Task 4: Paper-trade Sharpe metric (per-fold annualized)

**Files:**
- Create: `backend/tools/_scorecard/_paper_sharpe.py`
- Test: `backend/tests/test_scorecard_paper_sharpe.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scorecard_paper_sharpe.py
import numpy as np
import pytest
from tools._scorecard._paper_sharpe import paper_sharpe_per_fold


def test_paper_sharpe_constant_per_signal_return():
    """If every fold has identical net returns, std across folds = 0."""
    rng = np.random.default_rng(0)
    n = 200
    scores = np.full(n, 0.9)
    returns = np.full(n, 0.01)  # constant => std=0 per fold => Sharpe inf
    fold_ids = np.tile(np.arange(5), n // 5)
    fold_spans_days = {i: 30.0 for i in range(5)}

    # constant returns => per-signal std = 0 => guard returns NaN sharpe
    mean_s, std_s, n_fired = paper_sharpe_per_fold(
        scores, returns, fold_ids, fold_spans_days, tau=0.5, fee=0.0
    )
    assert n_fired == n
    assert np.isnan(mean_s)
    assert np.isnan(std_s)


def test_paper_sharpe_positive_with_noise():
    """Positive mean and finite std => positive annualized Sharpe."""
    rng = np.random.default_rng(42)
    n_per_fold = 100
    n_folds = 5
    n = n_per_fold * n_folds
    scores = np.full(n, 0.9)
    returns = rng.normal(loc=0.005, scale=0.01, size=n)
    fold_ids = np.repeat(np.arange(n_folds), n_per_fold)
    fold_spans_days = {i: 30.0 for i in range(n_folds)}

    mean_s, std_s, n_fired = paper_sharpe_per_fold(
        scores, returns, fold_ids, fold_spans_days, tau=0.5, fee=0.0
    )
    assert n_fired == n
    assert mean_s > 0
    assert std_s >= 0
    # Sanity: per-signal Sharpe ~0.5, annualization factor sqrt(100*365/30) ~ 34.9
    # => mean_s should be in single-digits not 0.5
    assert 1.0 < mean_s < 50.0


def test_paper_sharpe_fold_with_no_fires_excluded():
    """Folds with zero signals should not contribute to mean/std."""
    n = 200
    scores = np.zeros(n)
    scores[:50] = 0.9  # only fold 0 has fires
    returns = np.full(n, 0.005)
    fold_ids = np.repeat(np.arange(5), 40)[:n]  # ensure fold 0 has at least 1
    fold_ids = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 25 + [4] * 25)
    fold_spans_days = {i: 30.0 for i in range(5)}

    # std-Sharpe per fold needs >=2 fires for sample std; fold 0 with constant
    # returns gives NaN; folds 1-4 have 0 fires. Result should be NaN.
    mean_s, std_s, n_fired = paper_sharpe_per_fold(
        scores, returns, fold_ids, fold_spans_days, tau=0.5, fee=0.0
    )
    assert n_fired == 50
    # All folds either NaN (fold 0, constant returns) or no fires (1-4) => NaN mean
    assert np.isnan(mean_s)


def test_paper_sharpe_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        paper_sharpe_per_fold(
            np.array([0.9, 0.5]),
            np.array([0.01]),
            np.array([0, 1]),
            {0: 30.0, 1: 30.0},
            tau=0.5,
            fee=0.0,
        )


def test_paper_sharpe_missing_fold_span():
    """fold_ids referring to unknown fold should fail loud."""
    with pytest.raises(KeyError):
        paper_sharpe_per_fold(
            np.array([0.9, 0.9]),
            np.array([0.01, 0.02]),
            np.array([0, 7]),  # fold 7 not in spans dict
            {0: 30.0},
            tau=0.5,
            fee=0.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_paper_sharpe.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tools/_scorecard/_paper_sharpe.py
"""Annualized paper-trade Sharpe, computed per-fold then aggregated.

Per O3 resolution in design spec: each fold gets its own per-signal Sharpe and
annualization factor sqrt(N_f) where N_f = signals/year in fold f. Aggregate
across folds as mean +/- std for an honest variance estimate.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def paper_sharpe_per_fold(
    scores: np.ndarray,
    returns: np.ndarray,
    fold_ids: np.ndarray,
    fold_spans_days: Mapping[int, float],
    tau: float,
    fee: float,
) -> tuple[float, float, int]:
    """Per-fold annualized Sharpe, then mean and std across folds.

    Args:
        scores: shape (N,).
        returns: shape (N,) realized log-returns.
        fold_ids: shape (N,) integer fold index per sample.
        fold_spans_days: fold_id -> span of that fold in days.
        tau: strict-greater-than firing threshold.
        fee: per-side fee; round-trip cost = 2*fee subtracted from each return.

    Returns:
        (mean_annual_sharpe, std_annual_sharpe, total_n_fired).
        mean/std are NaN if no fold has >=2 fires (sample std undefined).

    Raises:
        ValueError: on shape mismatch.
        KeyError: if a fold_id in the data isn't in fold_spans_days.
    """
    if not (scores.shape == returns.shape == fold_ids.shape):
        raise ValueError(
            f"shape mismatch: scores {scores.shape}, returns {returns.shape}, fold_ids {fold_ids.shape}"
        )

    fired = scores > tau
    total_n_fired = int(fired.sum())

    fold_sharpes: list[float] = []
    unique_folds = np.unique(fold_ids[fired]) if total_n_fired > 0 else np.array([], dtype=int)

    for f in unique_folds:
        f_int = int(f)
        span_days = fold_spans_days[f_int]  # KeyError surfaces loudly
        mask = fired & (fold_ids == f)
        f_returns = returns[mask] - 2 * fee
        n_f = int(mask.sum())
        if n_f < 2:
            continue  # sample std undefined with <2 obs
        mu = float(f_returns.mean())
        sigma = float(f_returns.std(ddof=1))
        if sigma == 0:
            continue  # degenerate fold, skip
        per_signal_sharpe = mu / sigma
        n_per_year = n_f * 365.0 / span_days
        annualized = per_signal_sharpe * np.sqrt(n_per_year)
        fold_sharpes.append(annualized)

    if len(fold_sharpes) == 0:
        return float("nan"), float("nan"), total_n_fired
    if len(fold_sharpes) == 1:
        return float(fold_sharpes[0]), float("nan"), total_n_fired
    arr = np.array(fold_sharpes)
    return float(arr.mean()), float(arr.std(ddof=1)), total_n_fired
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_paper_sharpe.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_paper_sharpe.py backend/tests/test_scorecard_paper_sharpe.py
git commit -m "feat(scorecard): per-fold annualized paper Sharpe

Implements O3 resolution: per-fold Sharpe with sqrt(N_f) annualization,
aggregated as mean +/- std across folds. Skips degenerate folds (<2 fires
or zero std).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Task 5: Expected calibration error (ECE)

**Files:**
- Create: `backend/tools/_scorecard/_ece.py`
- Test: `backend/tests/test_scorecard_ece.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scorecard_ece.py
import numpy as np
import pytest
from tools._scorecard._ece import expected_calibration_error


def test_ece_perfectly_calibrated():
    """Scores match empirical hit rates per bin => ECE near 0."""
    rng = np.random.default_rng(0)
    n = 10000
    scores = rng.uniform(0, 1, size=n)
    # Perfect calibration: P(label=1 | score=s) = s
    labels = (rng.uniform(0, 1, size=n) < scores).astype(int)
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece < 0.02  # finite-sample noise


def test_ece_completely_miscalibrated():
    """Model says 0.9 but truth is 0.1 hit rate => ECE high."""
    n = 1000
    scores = np.full(n, 0.9)
    labels = np.zeros(n, dtype=int)
    labels[:100] = 1  # 10% hit rate
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece == pytest.approx(0.8, abs=0.01)


def test_ece_empty_bins_skipped():
    """Bins with zero samples should not contribute or NaN out the result."""
    scores = np.array([0.05, 0.95, 0.05, 0.95])
    labels = np.array([0, 1, 0, 1])
    ece = expected_calibration_error(scores, labels, n_bins=10)
    # bins [0.0, 0.1) and [0.9, 1.0] perfectly calibrated; middle bins empty
    assert ece == pytest.approx(0.0, abs=0.06)  # scores 0.05/0.95 vs bin-mean drift


def test_ece_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        expected_calibration_error(np.array([0.5, 0.5]), np.array([1]), n_bins=10)


def test_ece_nonbinary_labels():
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error(np.array([0.5]), np.array([2]), n_bins=10)


def test_ece_invalid_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(np.array([0.5]), np.array([1]), n_bins=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_ece.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tools/_scorecard/_ece.py
"""Expected calibration error with equal-width bins on [0, 1]."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE: weighted mean of |empirical_acc - mean_score| per bin.

    Bins are equal-width on [0, 1]. Per O4 resolution in design spec, decile
    binning is safe at 167k+ samples; revisit BBQ-style adaptive binning only
    for smaller subsets.

    Args:
        scores: shape (N,) in [0, 1].
        labels: shape (N,) binary 0/1.
        n_bins: number of equal-width bins.

    Returns:
        ECE in [0, 1]; 0 = perfect calibration.

    Raises:
        ValueError: on shape mismatch, non-binary labels, or n_bins <= 0.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")
    uniq = np.unique(labels)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"labels must be binary 0/1, got {uniq}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index per sample; clamp to last bin for score == 1.0
    bin_idx = np.clip(np.digitize(scores, edges, right=False) - 1, 0, n_bins - 1)

    n = scores.shape[0]
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(labels[mask].mean())
        conf_b = float(scores[mask].mean())
        ece += (n_b / n) * abs(acc_b - conf_b)
    return ece
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_ece.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_ece.py backend/tests/test_scorecard_ece.py
git commit -m "feat(scorecard): expected calibration error (decile binning)

Equal-width decile bins on [0, 1] per O4 resolution; weighted mean of
|empirical_acc - mean_score| per bin. Empty bins skipped.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Task 6: ScorecardReport dataclass and orchestrator

**Files:**
- Create: `backend/tools/_scorecard/_report.py`
- Create: `backend/tools/scorecard.py` (module skeleton — CLI added in Task 7)
- Test: `backend/tests/test_scorecard_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scorecard_orchestrator.py
import numpy as np
import pytest
from tools.scorecard import compute_scorecard, FEE_TIERS


def _synth_dataset(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, size=n)
    labels = (rng.uniform(0, 1, size=n) < scores).astype(int)
    returns = rng.normal(loc=0.003, scale=0.01, size=n) * (2 * labels - 1)
    fold_ids = np.tile(np.arange(5), n // 5 + 1)[:n]
    fold_spans_days = {i: 30.0 for i in range(5)}
    return scores, labels, returns, fold_ids, fold_spans_days


def test_compute_scorecard_returns_report_with_all_fields():
    s, l, r, f, spans = _synth_dataset()
    report = compute_scorecard(s, l, r, f, spans)
    # Per-tau x per-tier table populated
    assert len(report.per_tau_rows) == 10  # default tau_grid length
    for row in report.per_tau_rows:
        assert "precision" in row
        assert "n_fired" in row
        # E[r] and Sharpe present per fee tier
        for tier in FEE_TIERS:
            assert f"e_return_{tier}" in row
            assert f"sharpe_mean_{tier}" in row
            assert f"sharpe_std_{tier}" in row
    # ECE is scalar
    assert isinstance(report.ece, float)
    # Recommended operating tau chosen on gate_tier (default "retail")
    assert report.recommended_operating_tau in [r["tau"] for r in report.per_tau_rows] or \
           np.isnan(report.recommended_operating_tau)
    # Gate-passed bool present
    assert isinstance(report.gates_passed, dict)
    assert {"precision", "expected_return", "paper_sharpe", "ece"} <= set(report.gates_passed.keys())


def test_compute_scorecard_invalid_gate_tier():
    s, l, r, f, spans = _synth_dataset()
    with pytest.raises(ValueError, match="gate_tier"):
        compute_scorecard(s, l, r, f, spans, gate_tier="nonexistent")


def test_compute_scorecard_recommended_tau_meets_n_fired_floor():
    """Recommended tau must have >= 100 fires per spec; else NaN."""
    s, l, r, f, spans = _synth_dataset(n=50)  # too small to ever reach 100 fires
    report = compute_scorecard(s, l, r, f, spans)
    # Either NaN (no tau has 100+) or a tau where n_fired >= 100
    matching = [row for row in report.per_tau_rows if row["tau"] == report.recommended_operating_tau]
    if not np.isnan(report.recommended_operating_tau):
        assert matching and matching[0]["n_fired"] >= 100
    else:
        # All rows below floor
        assert all(row["n_fired"] < 100 for row in report.per_tau_rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write `_report.py` dataclass**

```python
# backend/tools/_scorecard/_report.py
"""ScorecardReport: dataclass aggregating all per-tau x per-tier rows + scalars."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScorecardReport:
    """Container for one model variant's scorecard.

    per_tau_rows: list of dicts, one per tau in tau_grid. Each row has:
        tau (float), precision (float|NaN), n_fired (int),
        e_return_{tier} (float|NaN) for each fee tier,
        sharpe_mean_{tier} (float|NaN), sharpe_std_{tier} (float|NaN).
    ece: scalar in [0, 1].
    recommended_operating_tau: tau with max precision*sign(e_return_gate_tier),
        subject to n_fired >= 100. NaN if no tau qualifies.
    gates_passed: dict[str, bool] for {precision, expected_return, paper_sharpe, ece}.
    """
    per_tau_rows: list[dict[str, Any]]
    ece: float
    recommended_operating_tau: float
    gates_passed: dict[str, bool]
    pos_rate: float
    gate_tier: str
```

- [ ] **Step 4: Write `scorecard.py` orchestrator**

```python
# backend/tools/scorecard.py
"""XGB deployment-aligned scorecard orchestrator.

Composes per-metric computers from tools._scorecard.* into a single
ScorecardReport for a model variant. CLI runner appended in Task 7.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from tools._scorecard._ece import expected_calibration_error
from tools._scorecard._expected_return import expected_return_at_tau
from tools._scorecard._paper_sharpe import paper_sharpe_per_fold
from tools._scorecard._precision import precision_at_tau
from tools._scorecard._report import ScorecardReport

FEE_TIERS: Mapping[str, float] = {"retail": 0.006, "mid": 0.0025, "pro": 0.0005}
DEFAULT_TAU_GRID: tuple[float, ...] = (
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
)
N_FIRED_FLOOR = 100  # per spec: recommended tau must fire >= 100 signals


def compute_scorecard(
    scores: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    fold_ids: np.ndarray,
    fold_spans_days: Mapping[int, float],
    *,
    fee_tiers: Mapping[str, float] = FEE_TIERS,
    gate_tier: str = "retail",
    tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
    n_ece_bins: int = 10,
) -> ScorecardReport:
    """Compute the full deployment-aligned scorecard for one model variant.

    Args:
        scores: (N,) score-class probabilities (e.g., p_up).
        labels: (N,) binary 0/1 ground truth (UP barrier hit).
        returns: (N,) realized log-return per sample.
        fold_ids: (N,) integer fold index 0..K-1.
        fold_spans_days: fold_id -> span in days.
        fee_tiers: name -> per-side fee. Default retail/mid/pro.
        gate_tier: which tier's metrics evaluate the hard gates.
        tau_grid: thresholds to sweep.
        n_ece_bins: equal-width bins for ECE.

    Returns:
        ScorecardReport.

    Raises:
        ValueError: if gate_tier not in fee_tiers.
    """
    if gate_tier not in fee_tiers:
        raise ValueError(f"gate_tier {gate_tier!r} not in fee_tiers {list(fee_tiers)}")

    pos_rate = float(labels.mean())
    per_tau_rows: list[dict] = []

    for tau in tau_grid:
        prec, n_fired = precision_at_tau(scores, labels, tau)
        row: dict = {"tau": float(tau), "precision": prec, "n_fired": n_fired}
        for tier_name, fee in fee_tiers.items():
            er, _ = expected_return_at_tau(scores, returns, tau, fee)
            sh_mean, sh_std, _ = paper_sharpe_per_fold(
                scores, returns, fold_ids, fold_spans_days, tau, fee
            )
            row[f"e_return_{tier_name}"] = er
            row[f"sharpe_mean_{tier_name}"] = sh_mean
            row[f"sharpe_std_{tier_name}"] = sh_std
        per_tau_rows.append(row)

    ece = expected_calibration_error(scores, labels, n_bins=n_ece_bins)

    # Recommended tau: highest precision among rows with n_fired >= floor
    # AND positive expected return at gate_tier.
    rec_tau: float = float("nan")
    eligible = [
        row for row in per_tau_rows
        if row["n_fired"] >= N_FIRED_FLOOR
        and not np.isnan(row[f"e_return_{gate_tier}"])
        and row[f"e_return_{gate_tier}"] > 0
        and not np.isnan(row["precision"])
    ]
    if eligible:
        best = max(eligible, key=lambda r: r["precision"])
        rec_tau = best["tau"]

    # Gate evaluation on gate_tier
    if np.isnan(rec_tau):
        gates = {"precision": False, "expected_return": False, "paper_sharpe": False, "ece": ece < 0.05}
    else:
        op = next(r for r in per_tau_rows if r["tau"] == rec_tau)
        gates = {
            "precision": op["precision"] >= pos_rate + 0.03,
            "expected_return": op[f"e_return_{gate_tier}"] > 0,
            "paper_sharpe": (
                not np.isnan(op[f"sharpe_mean_{gate_tier}"])
                and op[f"sharpe_mean_{gate_tier}"] > 0
            ),
            "ece": ece < 0.05,
        }

    return ScorecardReport(
        per_tau_rows=per_tau_rows,
        ece=ece,
        recommended_operating_tau=rec_tau,
        gates_passed=gates,
        pos_rate=pos_rate,
        gate_tier=gate_tier,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_orchestrator.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_report.py backend/tools/scorecard.py backend/tests/test_scorecard_orchestrator.py
git commit -m "feat(scorecard): compute_scorecard orchestrator + ScorecardReport

Composes four per-metric computers across tau grid and fee tiers; emits
ScorecardReport with per-tau rows, scalar ECE, recommended operating tau
(precision-max subject to n_fired>=100 and positive E[r] at gate_tier),
and hard-gate bools per spec.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Task 7 — CORRECTED 2026-05-19: v3 driver scorecard CLI

> The original Task 7/8/9 below this corrected section are **SUPERSEDED**. They
> were written against false premises discovered during execution. The decisive
> one (found when the first smoke run crashed): **the v3 booster is not trained
> on `cnn_dataset_cache.pt` at all.** `tools.train_xgb.train_xgb_v3` reads
> per-pid OHLCV parquets, builds tiered candle slices (micro 60 / meso 168 /
> macro 336), extracts features via `extract_features(feature_set='v3')`, and
> labels each sample **`close[t+4] > close[t]`** — a naive 4-bar-ahead direction.
> The ±1% triple-barrier belongs to the CNN cache, NOT to v3 XGB. Consequences:
>
> 1. The v3 scorecard must rebuild samples from parquets (mirror `train_xgb_v3`),
>    not load the cache. The cache is read only for the survivorship-aware
>    top-20 pid ranking.
> 2. Realized return per v3 signal is the plain 4-bar forward log-return
>    `ln(close[t+4]/close[t])` — there are no barriers to replay. (A
>    `_barrier_replay.py` written against the wrong premise was deleted.)
> 3. `purged_walk_forward_splits` already lives standalone in
>    `tools/walk_forward.py` — import it; there is no reusable `_train_fold_xgb`.
> 4. One harness cannot cover v3/v4/v4.5: v4/v4.5 use different extractors
>    (`extract_v4` / `extract_v4_5`) and labels — deferred to Tasks 7b/7c.
>
> Task 7 v1 implements **`--track v3` only**. `--track v4` / `v4.5` raise
> `NotImplementedError`.
>
> **Spec note:** the design spec's "Val fold convention" (~167k cache samples)
> is wrong for v3 — that count is the CNN cache. v3 sample count depends on
> `sample_step` over the top-20 pids' parquets. The spec should be amended.

**Files:**
- Create: `backend/tools/_scorecard/_cv_harness.py` — v3 sample building from parquets + per-fold OOF prediction
- Modify: `backend/tools/scorecard.py` — append the `--track v3` CLI
- Test: `backend/tests/test_scorecard_cli.py`

**v3 facts (from `tools/train_xgb.py:train_xgb_v3`):** per pid, read parquet, require ≥336 bars, roll one sample every `sample_step` bars from bar 336; tiers `{micro: records[t-60:t], meso: records[t-168:t], macro: records[t-336:t]}`; `extract_features(tiers, 'v3')` → 350 features; label `1 if close[t+4] > close[t] else 0`. Booster: `binary:logistic`, `max_depth=4/min_child_weight=1/subsample=0.7/colsample_bytree=0.8/lr=0.05`, 200 rounds, `feature_weights_v3` on the DMatrix (invariant #13). Realized return = `ln(close[t+4]/close[t])`.

- [ ] **Step 1: `_cv_harness.py`** — `V3Dataset` dataclass `(X[N,350], y, entry_ts, entry_close, exit_close, pid)`; `top_n_pids_from_cache(cache_path)` (cache read only for survivorship-aware top-20 ranking); `build_v3_samples(pids, parquet_dir, sample_step)` mirrors `train_xgb_v3` and records entry/exit close; `train_fold_v3(X_tr, y_tr, X_va)` trains a **fresh** booster per fold on the pre-extracted features (params above, `feature_weights_v3`); `oof_predict_v3(ds)` runs the 5-fold purged-WF loop → `(scores, fold_ids, fold_spans_days)`.

- [ ] **Step 2: CLI in `scorecard.py`** — `_load_v3_track(cache, parquet_dir, sample_step)` composes harness + `realized_log_returns_per_sample(entry_close, exit_close)`; `main()` argparse `--track/--cache/--parquet-dir/--sample-step/--gate-tier`, raises `NotImplementedError` for v4/v4.5, prints `_format_report` with an OOF-mean-AUC sanity line.

- [ ] **Step 3: Verify (requires a training/backend pause — see `feedback_no_pytest_during_trading`)**
  - `pytest tests/test_scorecard_cli.py -v -m "not slow"` → 3 passed (help, missing-track, v4-not-implemented)
  - `python -m tools.scorecard --track v3` → exit 0, full report
  - **AUC sanity anchor:** the report's "OOF mean AUC" — compare against the documented v3 ceiling in `xgb_feature_optimization_findings.md`. A materially different number means the per-fold booster config has diverged from `train_xgb_v3`.

- [ ] **Step 4: Commit** — `_cv_harness.py`, `scorecard.py`, `test_scorecard_cli.py`, `CHANGELOG.md`, in one commit; push.

## Task 8 — CORRECTED 2026-05-19: v3 smoke run

Verification only, **v3 track only**. During a training/backend pause: run `python -m tools.scorecard --track v3 --gate-tier retail` from `backend/`, capture the report. Acceptance: exit 0; 10-row per-tau table with no unexpected all-NaN columns; ECE in `[0,1]`; recommended tau is a grid value or NaN; 4 gate booleans; OOF AUC consistent with the documented v3 ceiling. Then `pytest tests/test_scorecard_cli.py -v -m slow`. v4/v4.5 smoke deferred to 7b/7c. No commit.

## Task 9 — CORRECTED 2026-05-19: persist v3 baseline

Create `docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md` with the v3 report, the four hard-gate outcomes, the AUC-vs-deployment-metric interpretation, and the gate-retirement recommendation (keep 0.55 AUC / retire / run alongside). v4 and v4.5 sections are placeholders marked "pending Tasks 7b/7c". Commit + push.

## Follow-up — Tasks 7b / 7c (v4 and v4.5 tracks)

Backlog. Each needs its own harness: load OHLCV from `data/history/<pid>.parquet`, rebuild samples via `extract_v4` / `extract_v4_5`, per-fold purged-WF retrain. 7c (v4.5) additionally sweeps 3 horizons × 3 decision rules and scores on `p_up`. Spec these before implementing — they are not mechanical extensions of the v3 CLI.

---

## Task 7 (SUPERSEDED 2026-05-19 — see corrected section above): CLI runner with cache loading and OOF prediction

**Files:**
- Modify: `backend/tools/scorecard.py` (append `__main__` block + loader)
- Test: `backend/tests/test_scorecard_cli.py`

**Rationale:** Wires the orchestrator to the actual cache and an existing trained model. Re-uses the same 5-fold purged-WF harness already used by `tools/feature_set_compare.py` and other probes (per `xgb_probe_results_log.md`).

- [ ] **Step 1: Read the cache-loading + CV pattern from existing probe**

Open `backend/tools/feature_set_compare.py` and identify:
- The function that loads `cnn_dataset_cache.pt` into `(X, y, pid_array, entry_ts_array, entry_close_array, forward_close_array)`. Call it `_load_cache_pooled` (rename if it has another name in the file).
- The function that builds 5-fold purged-WF splits with 4h embargo. Call it `_purged_wf_folds`.
- The function that trains XGBoost on a fold and returns OOF predictions. Call it `_train_fold_xgb`.

If these functions live in `feature_set_compare.py`, **extract them** into `backend/tools/_cv_harness.py` first (loose-coupling — the scorecard CLI shouldn't import from a sibling probe). If they're already in a shared helper module, import from there.

Spec the extraction in a 1-task interlude before Step 2 of this task if the existing structure mixes the CV harness with feature_set_compare-specific logic. Otherwise proceed.

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_scorecard_cli.py
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
PYTHON = BACKEND.parent / ".venv" / "Scripts" / "python.exe"


def test_scorecard_cli_help():
    """--help should exit 0 and mention --track."""
    result = subprocess.run(
        [str(PYTHON), "-m", "tools.scorecard", "--help"],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--track" in result.stdout
    assert "--cache" in result.stdout


def test_scorecard_cli_missing_track_arg():
    """Running without --track should fail with usage error."""
    result = subprocess.run(
        [str(PYTHON), "-m", "tools.scorecard"],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


@pytest.mark.slow
def test_scorecard_cli_v3_smoke():
    """Smoke run on the real cache — should print a report and exit 0.

    Marked slow because it loads the full ~167k-sample cache and trains 5 XGB
    folds. Skip in fast CI; run manually with: pytest -m slow.
    """
    cache_path = BACKEND / "cnn_dataset_cache.pt"
    if not cache_path.exists():
        pytest.skip(f"cache not present at {cache_path}")
    result = subprocess.run(
        [str(PYTHON), "-m", "tools.scorecard", "--track", "v3"],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        timeout=900,  # 15 min cap
    )
    assert result.returncode == 0
    # Report contains expected headers
    assert "precision" in result.stdout.lower()
    assert "ece" in result.stdout.lower()
    assert "recommended" in result.stdout.lower()
```

- [ ] **Step 3: Run test to verify the non-slow tests fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_cli.py -v -m "not slow"`
Expected: FAIL — `tools.scorecard` has no `__main__` handler.

- [ ] **Step 4: Append CLI to `scorecard.py`**

Add to the bottom of `backend/tools/scorecard.py`:

```python
# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _load_track_predictions(track: str, cache_path: str) -> dict:
    """Load (scores, labels, returns, fold_ids, fold_spans_days) for one track.

    Re-uses the cache-loading and CV-harness primitives from _cv_harness.py
    (extracted from feature_set_compare in Task 7 Step 1 prep). For each track:
      - v3: load v3 booster, predict OOF across 5 folds
      - v4: load v4 booster (binary), predict OOF
      - v4.5: load v4.5 multiclass booster, return p_up softmax slice for
              each (horizon, decision_rule) cell — for CLI v1 print the
              h24/argmax_margin cell as default; full 9-cell sweep deferred
              to per-cell --cell flag in a follow-up.
    """
    from tools._cv_harness import (
        load_cache_pooled,
        purged_wf_folds,
        train_fold_xgb_v3,
        train_fold_xgb_v4,
        train_fold_xgb_v45,
    )
    from tools._returns import realized_log_returns_per_sample

    cache = load_cache_pooled(cache_path)
    folds = purged_wf_folds(cache, n_splits=5, embargo_hours=4)

    n = len(cache.y)
    scores_oof = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    fold_spans_days: dict[int, float] = {}

    trainer = {"v3": train_fold_xgb_v3, "v4": train_fold_xgb_v4, "v4.5": train_fold_xgb_v45}[track]

    for f_idx, (train_idx, val_idx) in enumerate(folds):
        scores_oof[val_idx] = trainer(cache, train_idx, val_idx)
        fold_ids[val_idx] = f_idx
        span = (cache.entry_ts[val_idx].max() - cache.entry_ts[val_idx].min()) / 86400.0
        fold_spans_days[f_idx] = float(span)

    # Drop never-assigned samples (purge zone)
    keep = fold_ids >= 0
    returns = realized_log_returns_per_sample(
        cache.entry_close[keep], cache.forward_close[keep]
    )
    return {
        "scores": scores_oof[keep],
        "labels": cache.y[keep].astype(int),
        "returns": returns,
        "fold_ids": fold_ids[keep],
        "fold_spans_days": fold_spans_days,
    }


def _format_report(report: ScorecardReport, track: str) -> str:
    lines = [
        f"=== Scorecard for track={track} (gate_tier={report.gate_tier}) ===",
        f"pos_rate: {report.pos_rate:.4f}",
        f"ECE: {report.ece:.4f}  (gate <0.05 => {'PASS' if report.gates_passed['ece'] else 'FAIL'})",
        f"Recommended operating tau: {report.recommended_operating_tau}",
        "",
        "Per-tau table (E[r] and Sharpe shown for retail / mid / pro tiers):",
        f"{'tau':>6}  {'prec':>6}  {'n_fired':>8}  "
        f"{'E[r]_R':>9}  {'E[r]_M':>9}  {'E[r]_P':>9}  "
        f"{'Sh_R':>7}  {'Sh_M':>7}  {'Sh_P':>7}",
    ]
    for row in report.per_tau_rows:
        lines.append(
            f"{row['tau']:>6.2f}  "
            f"{row['precision']:>6.3f}  {row['n_fired']:>8d}  "
            f"{row['e_return_retail']:>9.5f}  {row['e_return_mid']:>9.5f}  {row['e_return_pro']:>9.5f}  "
            f"{row['sharpe_mean_retail']:>7.3f}  {row['sharpe_mean_mid']:>7.3f}  {row['sharpe_mean_pro']:>7.3f}"
        )
    lines.append("")
    lines.append("Gates passed: " + ", ".join(f"{k}={v}" for k, v in report.gates_passed.items()))
    return "\n".join(lines)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="XGB deployment-aligned scorecard runner")
    parser.add_argument("--track", required=True, choices=["v3", "v4", "v4.5"],
                        help="Which XGB head to score")
    parser.add_argument("--cache", default="cnn_dataset_cache.pt",
                        help="Path to cnn_dataset_cache.pt (default: ./cnn_dataset_cache.pt)")
    parser.add_argument("--gate-tier", default="retail", choices=list(FEE_TIERS),
                        help="Fee tier used to evaluate hard gates (default: retail)")
    args = parser.parse_args()

    data = _load_track_predictions(args.track, args.cache)
    report = compute_scorecard(
        data["scores"], data["labels"], data["returns"],
        data["fold_ids"], data["fold_spans_days"],
        gate_tier=args.gate_tier,
    )
    print(_format_report(report, args.track))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Extract CV harness from feature_set_compare if needed**

If Task 7 Step 1 found that `_load_cache_pooled`, `_purged_wf_folds`, and per-track XGB trainers live inside `feature_set_compare.py`, create `backend/tools/_cv_harness.py` with their pure-function form and update `feature_set_compare.py` to import from it. Write `backend/tests/test_cv_harness.py` first with at least: cache-load smoke (skip if file missing), fold-count and embargo-property tests on synthetic data, trainer-returns-1D-array shape test.

If the trainers don't exist yet (i.e., previously the booster was trained inline per probe), create minimal `train_fold_xgb_v3`, `train_fold_xgb_v4`, `train_fold_xgb_v45` that load the *existing trained boosters* (`backend/xgb_model.json`, etc.) and use them to predict on `val_idx` rather than re-train. **Re-training is intentionally not in scope** — the scorecard measures the existing deployed models, not fresh ones.

Adjust the test trainer-shape test accordingly: `assert preds.shape == (len(val_idx),) and preds.dtype == np.float64 and (preds >= 0).all() and (preds <= 1).all()`.

Commit this interlude as its own commit before Step 6:

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_cv_harness.py backend/tests/test_cv_harness.py backend/tools/feature_set_compare.py
git commit -m "refactor: extract _cv_harness.py from feature_set_compare for scorecard reuse

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

- [ ] **Step 6: Run the non-slow CLI tests**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_cli.py -v -m "not slow"`
Expected: 2 passed.

- [ ] **Step 7: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/scorecard.py backend/tests/test_scorecard_cli.py
git commit -m "feat(scorecard): CLI runner --track {v3,v4,v4.5}

Loads cache, builds OOF predictions across 5 purged-WF folds using the
existing trained booster (no retraining), runs compute_scorecard, prints
formatted report. v4.5 defaults to h24/argmax_margin cell; full 9-cell
sweep deferred to follow-up.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Task 8 (SUPERSEDED 2026-05-19 — see "Task 8 — CORRECTED" above): Run smoke test on real cache

**Files:** none modified — verification only.

- [ ] **Step 1: Confirm cache exists**

Run: `ls -la C:/Users/gl450/polymarket_app/backend/cnn_dataset_cache.pt`
Expected: file present, size > 100MB.

- [ ] **Step 2: Run scorecard CLI on v3 track**

Run: `cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m tools.scorecard --track v3 --gate-tier retail 2>&1 | tee /tmp/scorecard_v3.log`
Expected: exit 0, prints a per-tau table with 10 rows. Wall time 5–15 minutes depending on cache size.

Acceptance check:
- Per-tau table prints with all columns populated (no `nan` in `precision` for taus where `n_fired > 0`)
- ECE prints as a value in `[0.0, 1.0]`
- `Recommended operating tau:` line shows either a number from `tau_grid` or `nan`
- `Gates passed:` line shows 4 boolean entries

If any column is all-NaN unexpectedly, stop and investigate before proceeding to Task 9.

- [ ] **Step 3: Run on v4 and v4.5**

Run: `cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m tools.scorecard --track v4 2>&1 | tee /tmp/scorecard_v4.log`
Run: `cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m tools.scorecard --track v4.5 2>&1 | tee /tmp/scorecard_v45.log`
Expected: both exit 0 with same column structure.

- [ ] **Step 4: Run the slow smoke test from pytest**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_scorecard_cli.py -v -m slow`
Expected: `test_scorecard_cli_v3_smoke` passes.

- [ ] **Step 5: Commit nothing — this task only verifies**

No commit required for Task 8. Move to Task 9.

---

## Task 9 (SUPERSEDED 2026-05-19 — see "Task 9 — CORRECTED" above): Persist baseline results to spec-sibling doc

**Files:**
- Create: `docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md`

- [ ] **Step 1: Compile baseline doc**

Create `docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md` with this template (fill in the actual numbers from `/tmp/scorecard_*.log`):

```markdown
# XGB Scorecard Baseline Results — 2026-05-18

**Spec:** `2026-05-18-xgb-deployment-scorecard-design.md` (commit `9accdbe`)
**Implementation:** plan `2026-05-18-xgb-deployment-scorecard.md`
**Cache:** `cnn_dataset_cache.pt` v12 (28-channel, ~167k samples, survivorship-aware top-20)
**CV:** 5-fold purged WF, 4h embargo
**Gate tier:** retail (0.6% per side)

## v3 driver (binary p_up)

[Paste the formatted output from `/tmp/scorecard_v3.log` here.]

## v4 shadow (binary p_up, OHLCV+BB added)

[Paste from `/tmp/scorecard_v4.log` here.]

## v4.5 shadow (3-class, h24/argmax_margin cell)

[Paste from `/tmp/scorecard_v45.log` here.]

## Cross-track comparison

| Track | Recommended τ | Precision | E[r]_retail | Sharpe_retail | ECE | All gates passed |
|---|---|---|---|---|---|---|
| v3 | ... | ... | ... | ... | ... | ... |
| v4 | ... | ... | ... | ... | ... | ... |
| v4.5 | ... | ... | ... | ... | ... | ... |

## Interpretation

[1–3 paragraphs:
 - Does any track pass all 4 hard gates at retail?
 - If yes, is it the same track that wins on AUC currently in shadow?
 - If no, what's the closest miss? Which gate is the blocker (calibration, precision, fees)?
 - Recommendation: keep 0.55 AUC gate, retire it, or run alongside? ]

## Follow-up tasks (to backlog)

- v4.5 full 9-cell sweep (h24/h72/h168 × argmax_margin/indep_thresholds/net_direction)
- SELL-side scorecard (deferred per spec O5)
- Per-product / per-regime breakdown (if cross-track winner is unclear at the pooled level)
```

- [ ] **Step 2: Fill in numbers from the log files**

Open each `/tmp/scorecard_*.log` and copy the formatted output blocks into the corresponding sections. Build the cross-track comparison table by reading the "Recommended operating tau" and per-tau row for each track.

- [ ] **Step 3: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md
git commit -m "docs: scorecard baseline results for v3, v4, v4.5 shadow

Captures first run of deployment-aligned scorecard against all 3 active
XGB heads. Cross-track comparison + gate-pass summary. Informs the
follow-up decision on whether to retire the 0.55 AUC promotion gate.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

---

## Done criteria

All these must be true:

- [ ] 7 new test files all pass: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_returns_helper.py tests/test_scorecard_*.py -v`
- [ ] CLI runs end-to-end on all 3 tracks: `python -m tools.scorecard --track {v3,v4,v4.5}`
- [ ] Baseline results doc committed with real numbers, not placeholders
- [ ] No new dependencies added to `requirements.txt`
- [ ] `cnn_agent.py` not modified (per `feedback_xgb_focus_not_cnn`)
- [ ] All commits pushed to `feat/gpu-coord-mirror`
- [ ] Memory updated: append a session-log note to `coinbase_trader_session_log.md` summarizing scorecard ship and pointing to the baseline-results doc
