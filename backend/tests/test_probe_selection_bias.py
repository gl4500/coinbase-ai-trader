import pytest
from tools.probe_selection_bias import iid_auc_se, fold_level_se
from tools.probe_selection_bias import (
    expected_max_under_null, deflated_probability,
)


def test_iid_auc_se_balanced():
    # SE = sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))
    se = iid_auc_se(81951, 85982)
    assert se == pytest.approx(0.00141, abs=5e-5)


def test_iid_auc_se_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        iid_auc_se(0, 100)


def test_fold_level_se_known():
    # sample std of the logged baseline purged-WF folds
    se = fold_level_se([0.516, 0.507, 0.527, 0.523, 0.529])
    assert se == pytest.approx(0.00899, abs=1e-4)


def test_fold_level_se_rejects_too_few():
    with pytest.raises(ValueError, match="2 folds"):
        fold_level_se([0.52])


def test_expected_max_grows_with_n():
    lo = expected_max_under_null(10, 0.009, 0.5)
    hi = expected_max_under_null(200, 0.009, 0.5)
    assert 0.5 < lo < hi


def test_expected_max_known_value():
    # N=100, se=0.009, center=0.5 -> 0.5 + 0.009 * E[max of 100 normals ~2.53]
    em = expected_max_under_null(100, 0.009, 0.5)
    assert em == pytest.approx(0.5228, abs=5e-4)


def test_expected_max_rejects_small_n():
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_under_null(1, 0.009, 0.5)


def test_deflated_probability_high_when_far_above_floor():
    # observed AUC 0.60 is far above any best-of-N noise floor
    p = deflated_probability(0.60, 100, 0.009, 0.5)
    assert p > 0.99


def test_deflated_probability_half_at_floor():
    floor = expected_max_under_null(50, 0.01, 0.0)
    p = deflated_probability(floor, 50, 0.01, 0.0)
    assert p == pytest.approx(0.5, abs=1e-9)


def test_deflated_probability_low_when_below_floor():
    # a Delta of 0 against a positive best-of-N noise floor
    p = deflated_probability(0.0, 100, 0.009, 0.0)
    assert p < 0.5


from tools.probe_selection_bias import TRIALS, N_TIERS, analyze


def test_trials_table_shape():
    assert len(TRIALS) == 17                       # channel-candidate trials
    for t in TRIALS:
        assert set(t) == {"name", "delta", "passed"}
    # exactly the two RSI-rank runs are marked passed
    assert sum(1 for t in TRIALS if t["passed"]) == 2


def test_analyze_row_structure():
    rows = analyze()
    # 2 Track-A observed values x 3 N tiers x 2 noise scales = 12
    # + Track B: 1 x 3 x 2 = 6  -> 18 rows
    assert len(rows) == 18
    for r in rows:
        assert {"track", "observed_label", "observed", "n_trials", "noise",
                "se", "center", "expected_max", "deflated_prob"} <= set(r)
        assert 0.0 <= r["deflated_prob"] <= 1.0


def test_analyze_iid_inflates_confidence_vs_fold():
    rows = analyze()
    # for Track A best AUC at N=100, the iid noise scale gives a much higher
    # deflated probability than the honest fold-level scale
    def cell(noise):
        return next(r for r in rows
                    if r["track"].startswith("A") and r["n_trials"] == 100
                    and r["noise"] == noise
                    and "best" in r["observed_label"])
    assert cell("iid")["deflated_prob"] > cell("fold")["deflated_prob"]


from tools.probe_selection_bias import render_report, verdict_line


def test_verdict_line_reflects_headline_probability():
    # headline = Track A, best documented AUC, fold noise, N=100
    assert "marginal" in verdict_line(0.73).lower()
    assert "noise" in verdict_line(0.20).lower()       # below floor
    assert "real" in verdict_line(0.99).lower()        # clears the bar


def test_render_report_structure():
    rows = analyze()
    doc = render_report(rows)
    assert "# Probe Selection-Bias Meta-Analysis" in doc
    assert "## Trial table" in doc
    assert "## Verdict" in doc
    # every trial name appears in the rendered trial table
    for t in TRIALS:
        assert t["name"] in doc
    # both noise scales are reported
    assert "fold" in doc and "iid" in doc
