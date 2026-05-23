import pytest
from tools.probe_selection_bias import iid_auc_se, fold_level_se


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
