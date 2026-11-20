"""Tests for tools.strategy_discovery.mine_profiles (Phase 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.strategy_discovery.mine_profiles import (
    apply_deflation,
    bootstrap_ci,
    leaf_metrics,
    leaf_qualifies,
    long_shot_band,
    pick_best_hyperparams,
)


def test_deflation_factor_applied_to_reported_profit():
    deflated, infl = apply_deflation(raw=0.072, inner_cv_se=0.015, n_combos=9)
    assert infl == pytest.approx(0.015 * math.sqrt(2 * math.log(9)), rel=1e-9)
    assert deflated == pytest.approx(0.072 - infl, rel=1e-9)


def test_long_shot_band_triggers_bootstrap():
    assert long_shot_band(avg_win=0.16, avg_loss=-0.05, win_rate=0.75) is True
    assert long_shot_band(avg_win=0.14, avg_loss=-0.05, win_rate=0.75) is False
    assert long_shot_band(avg_win=0.16, avg_loss=-0.08, win_rate=0.75) is False
    assert long_shot_band(avg_win=0.16, avg_loss=-0.05, win_rate=0.69) is False


def test_leaf_metrics_computes_win_loss_dd_sortino():
    trades = np.array([0.08, 0.08, -0.05, 0.08, 0.08, -0.05])
    m = leaf_metrics(trades)
    assert m["trade_count"] == 6
    assert m["win_rate"] == pytest.approx(4 / 6)
    assert m["avg_win"] == pytest.approx(0.08)
    assert m["avg_loss"] == pytest.approx(-0.05)
    assert m["cumulative_profit_raw"] == pytest.approx(0.22, abs=1e-9)
    assert m["max_dd"] == pytest.approx(0.05, abs=1e-9)
    assert m["sortino"] == pytest.approx(0.22 / 6 / 0.05, rel=1e-6)


def test_q0_gates_applied_to_deflated_profit():
    fold_metrics = [
        {"avg_win": 0.08, "avg_loss": -0.08, "max_dd": 0.20, "deflated_profit": 0.05},  # PASS
        {"avg_win": 0.06, "avg_loss": -0.09, "max_dd": 0.25, "deflated_profit": 0.03},  # PASS
        {
            "avg_win": 0.07,
            "avg_loss": -0.11,
            "max_dd": 0.22,
            "deflated_profit": 0.04,
        },  # FAIL (avg_loss < -0.10)
        {"avg_win": 0.09, "avg_loss": -0.07, "max_dd": 0.18, "deflated_profit": 0.06},  # PASS
        {"avg_win": 0.10, "avg_loss": -0.06, "max_dd": 0.15, "deflated_profit": 0.07},  # PASS
    ]
    n_pass = sum(leaf_qualifies(m) for m in fold_metrics)
    assert n_pass == 4
    from tools.strategy_discovery.mine_profiles import _Q0_MIN_FOLDS

    assert n_pass >= _Q0_MIN_FOLDS


def test_pick_best_hyperparams_picks_max_inner_cv():
    inner_scores = {
        (3, 20): 0.012,
        (3, 50): 0.018,
        (3, 100): 0.005,
        (5, 20): 0.022,
        (5, 50): 0.019,
        (5, 100): 0.011,
        (7, 20): 0.025,
        (7, 50): 0.020,
        (7, 100): 0.015,
    }
    chosen_depth, chosen_min_leaf, raw_max, inner_se = pick_best_hyperparams(inner_scores)
    assert chosen_depth == 7
    assert chosen_min_leaf == 20
    assert raw_max == pytest.approx(0.025, rel=1e-9)
    expected_se = float(np.std(list(inner_scores.values()), ddof=1))
    assert inner_se == pytest.approx(expected_se, rel=1e-9)


def test_bootstrap_ci_returns_95pct_band_on_resampled_trades():
    trades = np.full(50, 0.01)
    rng = np.random.default_rng(7)
    lower, upper = bootstrap_ci(trades, n_iter=500, rng=rng)
    assert lower == pytest.approx(0.50, abs=1e-6)
    assert upper == pytest.approx(0.50, abs=1e-6)
    mixed = np.concatenate([np.full(30, 0.10), np.full(20, -0.05)])
    lo2, hi2 = bootstrap_ci(mixed, n_iter=500, rng=rng)
    assert lo2 < 2.0 < hi2
    assert (hi2 - lo2) > 0.1


def test_mine_profiles_for_pid_horizon_returns_qualifying_leaves_only(tmp_path):
    """Integration: build a synthetic Phase 2 parquet where a clear cohort exists,
    run mine_profiles_for_pid_horizon end-to-end (on CPU), and assert at least
    one qualifying profile comes back."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.tools.strategy_discovery.endpoint_publication_fixture import (
        publish_endpoints_for_frame,
    )
    from tools.strategy_discovery.mine_profiles import mine_profiles_for_pid_horizon

    rng = np.random.default_rng(101)
    n = 2000
    ts_ms = np.arange(n, dtype="int64") * 3_600_000
    feat_0 = rng.uniform(0.0, 1.0, size=n)
    feat_1 = rng.uniform(0.0, 1.0, size=n)
    labels = np.where(feat_0 > 0.5, 0.08, -0.02).astype("float64")
    # The mined horizon's tail cannot carry a label: its exit bar does not exist. The original
    # fixture labelled every row, which no producer can do and which the publication would
    # reject.
    labels_h24 = np.where(np.arange(n) + 24 < n, labels, np.nan)
    df = pd.DataFrame(
        {
            "ts": ts_ms,
            "source_row_id": np.arange(n, dtype="int64"),
            "open": np.full(n, 1.0),
            "high": np.full(n, 1.0),
            "low": np.full(n, 1.0),
            "close": np.full(n, 1.0),
            "market_cap": np.full(n, 1e9),
            "fdv": np.full(n, 2e9),
            "fdv_over_mc": np.full(n, 2.0),
            "circ_over_total": np.full(n, 0.5),
            "vol_24h": np.full(n, 1e7),
            "vol_over_mc": np.full(n, 0.01),
            "price_over_ema20": feat_0,
            "price_over_ema50": np.full(n, 1.0),
            "price_over_ema200": np.full(n, 1.0),
            "ret_1h_sign": np.full(n, 0.0),
            "ret_24h_sign": feat_1,
            "ret_7d_sign": np.full(n, 0.0),
            "atr14_pct": np.full(n, 0.02),
            "label_h1": labels,
            "label_h4": labels,
            "label_h24": labels_h24,
            "label_h72": labels,
            "label_h168": labels,
        }
    )
    parquet_path = tmp_path / "FOO-USD.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)
    # Publish the endpoints beside the parquet, exactly as Phase 2 does. Without this the
    # miner refuses the product, which is the correct behaviour and was how this test caught
    # the new requirement.
    publish_endpoints_for_frame(tmp_path, "FOO-USD", df, horizons=[24])

    profiles = mine_profiles_for_pid_horizon(
        pid="FOO-USD",
        horizon=24,
        parquet_path=parquet_path,
        device="cpu",
        seed=42,
    )
    assert len(profiles) >= 1
    assert all(p.n_folds_evaluated == 5 for p in profiles)
    assert all(p.validation_version == "chronological_distinct_folds_v2" for p in profiles)
    winners = [p for p in profiles if p.avg_win >= 0.05 and p.cumulative_profit_deflated > 0]
    assert len(winners) >= 1, (
        f"no winners; got profiles: {[(p.avg_win, p.cumulative_profit_deflated) for p in profiles]}"
    )


_STUB_HORIZON = 168


def _stub_mining_frame(monkeypatch, timestamps, labels=None, *, tmp_path=None, publish=True):
    """A frame Phase 2 could actually have written, plus its publication.

    Differences from the original, all forced by the contract the miner now enforces:
      * `source_row_id`, `close`, `high` and `low` exist -- row identity and the frame binding
        both need them.
      * the last `_STUB_HORIZON` labels are NaN, because a label needs its exit bar to exist.
        The old fixture labelled every row including the last, which no producer can do.
      * endpoints are published beside the parquet, so `mine_profiles_for_pid_horizon` has an
        artifact to validate against instead of a bare frame.

    `publish=False` keeps a bare frame for the tests that must reach a guard firing BEFORE the
    publication is consulted.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.tools.strategy_discovery.endpoint_publication_fixture import (
        publish_endpoints_for_frame,
    )
    from tools.strategy_discovery import mine_profiles as miner

    n = len(timestamps)
    frame = pd.DataFrame({name: np.arange(n, dtype=float) for name in miner._FEATURE_COLUMNS})
    frame["ts"] = timestamps
    frame["source_row_id"] = np.arange(n, dtype="int64")
    for name in ("close", "high", "low"):
        frame[name] = np.full(n, 1.0)
    base = np.arange(n, dtype=float) if labels is None else np.asarray(labels, dtype=float)
    tail = np.where(np.arange(n) + _STUB_HORIZON < n, base, np.nan)
    frame[f"label_h{_STUB_HORIZON}"] = tail
    monkeypatch.setattr(pq, "read_table", lambda _: pa.Table.from_pandas(frame))

    if not publish or tmp_path is None:
        return miner, "unused"
    parquet_path = tmp_path / "TEST.parquet"
    publish_endpoints_for_frame(tmp_path, "TEST", frame, horizons=[_STUB_HORIZON])
    return miner, parquet_path


def test_miner_rejects_rows_out_of_published_order_instead_of_sorting_them(monkeypatch):
    """This test used to assert that the miner SORTED a reversed frame before building
    tensors. The contract is now stronger and the sort is gone.

    A reordered parquet is not the artifact Phase 2 published: `build_data_id` hashes the
    timestamps and price arrays in frame order, so a reordered frame cannot carry a valid
    publication at all. Sorting it into shape would repair a corrupted artifact silently,
    which is the class of behaviour this whole effort removes. Rejection is the correct
    outcome, and it fires before any endpoint is even consulted.
    """
    timestamps = np.arange(300, dtype="int64")[::-1] * 3_600_000
    miner, parquet_path = _stub_mining_frame(monkeypatch, timestamps, publish=False)

    with pytest.raises(ValueError, match="ascending"):
        miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu")


@pytest.mark.parametrize("kind", ["duplicate", "subhour", "null", "fractional"])
def test_miner_rejects_invalid_hourly_timestamps(monkeypatch, kind):
    ts = np.arange(300, dtype="int64") * 3_600_000
    if kind == "duplicate":
        ts[1] = ts[0]
    elif kind == "subhour":
        ts[1] = ts[0] + 1
    else:
        ts = ts.astype(float)
        ts[1] = np.nan if kind == "null" else ts[1] + 0.5
    miner, parquet_path = _stub_mining_frame(monkeypatch, ts, publish=False)
    with pytest.raises(ValueError, match="timestamp"):
        miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu")


def test_miner_skips_outer_folds_without_usable_inner_history(monkeypatch, tmp_path, caplog):
    """300 RETAINED rows: enough history to pass the 200-row floor, not enough for 3 usable
    inner folds at horizon 168.

    The source frame is 300 + 168 rows because the fixture now NaNs the unlabelable tail, as a
    real producer must. Codex 29950d93 caught that leaving it at 300 SOURCE rows retained only
    132, so the miner returned at the `n < 200` guard and this test passed without ever
    reaching the inner-fold guard it exists to check. The diagnostic is asserted for the same
    reason: a silent early return must not be able to masquerade as this guard again.
    """
    rows = 300 + _STUB_HORIZON
    miner, parquet_path = _stub_mining_frame(
        monkeypatch, np.arange(rows, dtype="int64") * 3_600_000, tmp_path=tmp_path
    )

    def forbidden(**kwargs):
        pytest.fail("must not fit a tree without usable inner folds")

    monkeypatch.setattr(miner, "fit_tree", forbidden)
    with caplog.at_level("WARNING"):
        assert miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu") == []
    assert "require 5 outer and 3 inner folds" in caplog.text, (
        f"expected the fold-history diagnostic, got: {caplog.text!r}. A 'fewer than 200 labeled "
        f"rows' message means the test returned before the guard it is about."
    )


def test_miner_training_prefix_preserves_feature_label_alignment(monkeypatch, tmp_path):
    """Training slices stay PREFIXES, so a feature row and its label keep the same index.

    The frame is ascending now rather than reversed -- the reversed version tested the sort
    that no longer exists -- so the alignment claim is asserted directly: column 0 of the
    features equals the labels over the same prefix, and the first row really is row 0.
    """
    timestamps = np.arange(12000, dtype="int64") * 3_600_000
    miner, parquet_path = _stub_mining_frame(monkeypatch, timestamps, tmp_path=tmp_path)

    class Captured(Exception):
        pass

    def capture(features, labels, next_eligible, **kwargs):
        assert len(features) < len(labels)
        np.testing.assert_array_equal(
            features[:, 0].cpu().numpy(), labels[: len(features)].cpu().numpy()
        )
        assert features[0, 0].item() == 0.0
        raise Captured

    monkeypatch.setattr(miner, "fit_tree", capture)
    with pytest.raises(Captured):
        miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu")


@pytest.mark.parametrize("n", [900, 1000, 6000])
def test_miner_requires_complete_comparable_fold_history(monkeypatch, caplog, n, tmp_path):
    miner, parquet_path = _stub_mining_frame(
        monkeypatch, np.arange(n, dtype="int64") * 3_600_000, tmp_path=tmp_path
    )

    def forbidden(**kwargs):
        pytest.fail("must reject insufficient fold history before fitting")

    monkeypatch.setattr(miner, "fit_tree", forbidden)
    assert miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu") == []
    assert "insufficient history" in caplog.text.lower()
    assert "5 outer" in caplog.text


@pytest.mark.parametrize("passing_periods", [1, 3, 4, 5])
def test_multiple_leaves_in_one_period_count_as_one_fold(monkeypatch, passing_periods, tmp_path):
    import torch

    from tools.strategy_discovery.profit_tree import TreeNode

    miner, parquet_path = _stub_mining_frame(
        monkeypatch, np.arange(12000, dtype="int64") * 3_600_000, tmp_path=tmp_path
    )

    # Four distinct leaves share the actual root-direction identity. Their
    # success in one period must never stand in for four independent periods.
    def split(feature, left, right):
        return TreeNode(feature=feature, threshold=0.5, left=left, right=right)

    tree = split(
        0,
        split(1, split(2, TreeNode(), TreeNode()), split(2, TreeNode(), TreeNode())),
        TreeNode(),
    )
    monkeypatch.setattr(miner, "fit_tree", lambda **kwargs: tree)
    monkeypatch.setattr(miner, "_DEPTH_GRID", (3,))
    monkeypatch.setattr(miner, "_MIN_LEAF_GRID", (20,))
    monkeypatch.setattr(miner, "pick_best_hyperparams", lambda _: (3, 20, 0.1, 0.0))
    monkeypatch.setattr(
        miner, "_assign_leaves", lambda root, rows: [i % 4 for i in range(len(rows))]
    )
    monkeypatch.setattr(miner, "walk_and_sum", lambda *args: torch.tensor([0.1]))
    replay_calls = 0

    def replay(*args):
        nonlocal replay_calls
        period = replay_calls // 4
        replay_calls += 1
        return [0.1] if period < passing_periods else [-0.2]

    monkeypatch.setattr(miner, "_replay_trades", replay)
    profiles = miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu")
    assert replay_calls == 20
    if passing_periods < 4:
        assert profiles == [], "several leaves from one period cannot satisfy the four-fold gate"
    else:
        assert len(profiles) == 1
        assert profiles[0].n_folds_passed_q0 == passing_periods
        assert profiles[0].n_folds_passed_q0 <= profiles[0].n_folds_evaluated
        binding = profiles[0].rule_binding
        assert binding["profile_leaf_id"] == 0
        assert binding["source_leaf_id"] == 3
        assert binding["source_outer_fold"] == passing_periods - 1
        assert binding["pid"] == "TEST"
        assert binding["horizon"] == 168
        assert binding["feature_schema"] == list(miner._FEATURE_COLUMNS)
        assert len(binding["source_tree_digest"]) == 64


def test_miner_rejects_changed_feature_dtype_before_fitting(monkeypatch, tmp_path):
    miner, parquet_path = _stub_mining_frame(
        monkeypatch, np.arange(12000, dtype="int64") * 3_600_000, tmp_path=tmp_path
    )
    original_tensor = miner.torch.tensor

    def downcast_features(*args, **kwargs):
        tensor = original_tensor(*args, **kwargs)
        return tensor.float() if tensor.ndim == 2 else tensor

    def forbidden(**kwargs):
        pytest.fail("must not fit under an unsupported comparison dtype")

    monkeypatch.setattr(miner.torch, "tensor", downcast_features)
    monkeypatch.setattr(miner, "fit_tree", forbidden)
    with pytest.raises(ValueError, match="float64"):
        miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu")


def test_miner_rejects_gapped_source_before_row_count_labels_can_overlap(monkeypatch):
    ts = np.arange(300, dtype="int64") * 7_200_000
    miner, parquet_path = _stub_mining_frame(monkeypatch, ts, publish=False)
    with pytest.raises(ValueError, match="contiguous hourly"):
        miner.mine_profiles_for_pid_horizon("TEST", 168, parquet_path, device="cpu")
