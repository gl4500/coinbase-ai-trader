import numpy as np
import pytest

from tools.scorecard import FEE_TIERS, compute_scorecard


def _synth_dataset(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, size=n)
    labels = (rng.uniform(0, 1, size=n) < scores).astype(int)
    returns = rng.normal(loc=0.003, scale=0.01, size=n) * (2 * labels - 1)
    fold_ids = np.tile(np.arange(5), n // 5 + 1)[:n]
    fold_spans_days = {i: 30.0 for i in range(5)}
    return scores, labels, returns, fold_ids, fold_spans_days


def test_compute_scorecard_returns_report_with_all_fields():
    s, labels, r, f, spans = _synth_dataset()
    report = compute_scorecard(s, labels, r, f, spans)
    assert len(report.per_tau_rows) == 10
    for row in report.per_tau_rows:
        assert "precision" in row
        assert "n_fired" in row
        for tier in FEE_TIERS:
            assert f"e_return_{tier}" in row
            assert f"sharpe_mean_{tier}" in row
            assert f"sharpe_std_{tier}" in row
    assert isinstance(report.ece, float)
    rec = report.recommended_operating_tau
    taus = [r["tau"] for r in report.per_tau_rows]
    assert rec in taus or np.isnan(rec)
    assert isinstance(report.gates_passed, dict)
    assert {"precision", "expected_return", "paper_sharpe", "ece"} <= set(
        report.gates_passed.keys()
    )


def test_compute_scorecard_invalid_gate_tier():
    s, labels, r, f, spans = _synth_dataset()
    with pytest.raises(ValueError, match="gate_tier"):
        compute_scorecard(s, labels, r, f, spans, gate_tier="nonexistent")


def test_compute_scorecard_recommended_tau_meets_n_fired_floor():
    """Recommended tau must have >= 100 fires per spec; else NaN."""
    s, labels, r, f, spans = _synth_dataset(n=50)
    report = compute_scorecard(s, labels, r, f, spans)
    rec = report.recommended_operating_tau
    matching = [row for row in report.per_tau_rows if row["tau"] == rec]
    if not np.isnan(rec):
        assert matching and matching[0]["n_fired"] >= 100
    else:
        assert all(row["n_fired"] < 100 for row in report.per_tau_rows)
