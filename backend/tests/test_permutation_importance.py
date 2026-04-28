"""Tests for tools.permutation_importance — channel permutation primitives.

Permutation importance shuffles one channel of an [N, C, T] tensor and measures
how much val-loss changes. The two scramble modes answer different questions:
  - "sample" axis: reorder rows for one channel only (preserves within-window
    pattern, asks if the channel's level/scale matters at all)
  - "time"   axis: shuffle the T timesteps within each sample for one channel
    (preserves value distribution per row, asks if the channel's temporal
    pattern matters)

These tests pin the primitives so the importance numbers we publish are
defensible — a bug in the shuffle invalidates every channel's delta.
"""
import math
import pytest

torch = pytest.importorskip("torch")

from tools.permutation_importance import (  # noqa: E402
    permute_channel_cross_sample,
    permute_channel_within_sample_time,
    weighted_bce_with_logits,
)


@pytest.fixture
def x_fixture():
    """Deterministic [N=8, C=3, T=4] tensor with channel-distinct value ranges
    so we can detect cross-channel contamination."""
    torch.manual_seed(0)
    n, c, t = 8, 3, 4
    base = torch.arange(n * c * t, dtype=torch.float32).view(n, c, t)
    return base


class TestPermuteCrossSample:
    def test_target_channel_rows_are_a_permutation(self, x_fixture):
        x = x_fixture
        out = permute_channel_cross_sample(x, ch=1, generator=torch.Generator().manual_seed(7))
        # Channel 1 row-set must equal the original row-set (just reordered)
        before = {tuple(row.tolist()) for row in x[:, 1, :]}
        after  = {tuple(row.tolist()) for row in out[:, 1, :]}
        assert before == after

    def test_other_channels_unchanged(self, x_fixture):
        x = x_fixture
        out = permute_channel_cross_sample(x, ch=1, generator=torch.Generator().manual_seed(7))
        for ch in (0, 2):
            assert torch.equal(out[:, ch, :], x[:, ch, :]), \
                f"channel {ch} must be untouched when permuting channel 1"

    def test_input_not_mutated(self, x_fixture):
        x = x_fixture
        x_before = x.clone()
        _ = permute_channel_cross_sample(x, ch=0, generator=torch.Generator().manual_seed(7))
        assert torch.equal(x, x_before), "permute must not mutate the input tensor"

    def test_at_least_some_rows_actually_move(self, x_fixture):
        # With N=8 the chance of identity perm is 1/8! ≈ 2.5e-5 — safe to assert
        # at least one row differs at the picked seed.
        x = x_fixture
        out = permute_channel_cross_sample(x, ch=0, generator=torch.Generator().manual_seed(3))
        diffs = (out[:, 0, :] != x[:, 0, :]).any(dim=1).sum().item()
        assert diffs >= 1, "expected at least one row to be reordered"


class TestPermuteWithinSampleTime:
    def test_per_row_value_multiset_preserved(self, x_fixture):
        x = x_fixture
        out = permute_channel_within_sample_time(
            x, ch=2, generator=torch.Generator().manual_seed(11)
        )
        for i in range(x.shape[0]):
            before = sorted(x[i, 2, :].tolist())
            after  = sorted(out[i, 2, :].tolist())
            assert before == after, \
                f"sample {i}: timestep shuffle must preserve channel value multiset"

    def test_other_channels_unchanged(self, x_fixture):
        x = x_fixture
        out = permute_channel_within_sample_time(
            x, ch=2, generator=torch.Generator().manual_seed(11)
        )
        for ch in (0, 1):
            assert torch.equal(out[:, ch, :], x[:, ch, :])

    def test_at_least_some_timesteps_move(self, x_fixture):
        # Per-row T=4 → identity perm prob 1/4!=1/24; over 8 rows the chance
        # ALL 8 are identity is (1/24)^8 — essentially zero.
        x = x_fixture
        out = permute_channel_within_sample_time(
            x, ch=0, generator=torch.Generator().manual_seed(5)
        )
        n_changed = (out[:, 0, :] != x[:, 0, :]).any(dim=1).sum().item()
        assert n_changed >= 1


class TestWeightedBCE:
    def test_matches_manual_for_uniform_weights(self):
        # Uniform weights → weighted mean reduces to plain mean
        logits  = torch.tensor([[0.5], [-1.0], [2.0], [0.0]])
        targets = torch.tensor([[1.0], [0.0],  [1.0], [0.0]])
        weights = torch.ones_like(targets)

        manual = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="mean"
        ).item()
        ours = weighted_bce_with_logits(logits, targets, weights)
        assert math.isclose(ours, manual, rel_tol=1e-6, abs_tol=1e-7)

    def test_weights_actually_reweight(self):
        # Two samples; if we down-weight the high-loss one, total drops.
        logits  = torch.tensor([[5.0], [0.0]])      # second is high-loss vs target=1
        targets = torch.tensor([[1.0], [1.0]])
        equal   = weighted_bce_with_logits(logits, targets, torch.ones_like(targets))
        skewed  = weighted_bce_with_logits(
            logits, targets, torch.tensor([[10.0], [1.0]])
        )
        assert skewed < equal, \
            "weighting the easy sample 10× should drag the weighted mean down"

    def test_zero_weight_sum_does_not_explode(self):
        # Edge case: if all weights are zero, fall back to safe small denom
        logits  = torch.tensor([[0.0], [0.0]])
        targets = torch.tensor([[1.0], [0.0]])
        out = weighted_bce_with_logits(
            logits, targets, torch.zeros_like(targets)
        )
        assert math.isfinite(out), "must not return inf/nan when weights sum to 0"


class TestNoOpDelta:
    """Sanity: shuffling a constant channel with itself produces no change in
    forward output, hence delta = 0. Catches regressions where the permutation
    accidentally also touches an unrelated channel."""

    def test_constant_channel_shuffle_yields_identical_tensor(self):
        # Channel 1 is all-zero; shuffling it must produce a tensor equal to input
        x = torch.zeros(4, 3, 5)
        x[:, 0, :] = torch.arange(20, dtype=torch.float32).view(4, 5)
        x[:, 2, :] = torch.arange(20, 40, dtype=torch.float32).view(4, 5)
        out_s = permute_channel_cross_sample(x, ch=1, generator=torch.Generator().manual_seed(0))
        out_t = permute_channel_within_sample_time(x, ch=1, generator=torch.Generator().manual_seed(0))
        assert torch.equal(out_s, x)
        assert torch.equal(out_t, x)
