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
