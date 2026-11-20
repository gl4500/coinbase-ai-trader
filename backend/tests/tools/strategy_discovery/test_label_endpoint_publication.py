"""Endpoint publication: the simulation publishes the exit it already chose.

`_simulate_one` computes which of three branches fired — stop, trail, horizon — and
then throws that away, returning only a PnL. Three downstream components each
re-derived it from a different clock, and on gapped data they disagreed. This slice
makes the simulation publish it instead.

**Publication changes no label value.** Every expected PnL below is derived by hand
from the documented exit policy, so these tests pin the scalar behaviour independently
of the refactor: if the shared internal result drifts from the scalar path, or the
scalar path drifts from its own formula, they fail.

What publication does **not** do, and these tests must not be read as doing:

* it does not fix the contemporaneous ATR (tracked unresolved as
  `intrabar-atr-causality-followup`) — the trail threshold still reads the current
  bar's ATR, so trail labels still depend on information from after their decision
  point;
* it does not change fill assumptions, intrabar ordering, or replay semantics;
* it does not make eligibility or replay metrics comparable to their old values —
  those were computed from the wrong clock and *will* change once consumers read
  endpoints. Only the label scalars are preserved.
"""

import math
import os
import sys

import pandas as pd
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.labels import (  # noqa: E402
    simulate_dynamic_exit_labels,
    simulate_labels_with_endpoints,
)

_H = 3_600_000
_FEE = 0.012
_STOP = 0.08
_FLOOR = 0.06


def _frame(rows):
    """rows: list of (high, low, close, atr14_pct). Contiguous hourly bars."""
    return pd.DataFrame(
        {
            "ts": [i * _H for i in range(len(rows))],
            "open": [r[2] for r in rows],
            "high": [r[0] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[2] for r in rows],
            "atr14_pct": [r[3] for r in rows],
        }
    )


# Hand-derived scenarios. Each comment states the arithmetic so a failure is readable.

# Stop: entry close 100, stop at 100*(1-0.08)=92. Bar 1 low 91 <= 92 triggers.
# exit price is the ASSUMED STOP LEVEL 92, not the observed low.
# return = 92/100 - 1 - 0.012 = -0.092
_STOP_ROWS = [
    (100.0, 100.0, 100.0, _FLOOR),
    (101.0, 91.0, 95.0, _FLOOR),
    (101.0, 100.0, 100.0, _FLOOR),
]
_STOP_LABEL = 92.0 / 100.0 - 1.0 - _FEE

# Trail: no stop (low 100 vs stop 92). peak rises to bar 1 high 110; atr_pct = 0.06;
# 100/110 - 1 = -0.0909 <= -0.06 triggers. exit = 110*(1-0.06) = 103.4
# return = 103.4/100 - 1 - 0.012 = 0.022
_TRAIL_ROWS = [
    (100.0, 100.0, 100.0, _FLOOR),
    (110.0, 100.0, 105.0, _FLOOR),
    (111.0, 109.0, 110.0, _FLOOR),
]
_TRAIL_LABEL = 110.0 * (1.0 - _FLOOR) / 100.0 - 1.0 - _FEE

# Horizon h=2: no stop, no trail. Exit at closes[2] = 101.5
# return = 101.5/100 - 1 - 0.012 = 0.003
_HORIZON_ROWS = [
    (100.0, 100.0, 100.0, _FLOOR),
    (101.0, 99.5, 100.5, _FLOOR),
    (102.0, 100.0, 101.5, _FLOOR),
]
_HORIZON_LABEL = 101.5 / 100.0 - 1.0 - _FEE


def _labels(df, horizon, **kw):
    return simulate_dynamic_exit_labels(df, horizons=[horizon], **kw)[f"label_h{horizon}"]


# ── scalar parity: the public API and its values are untouched ────────────────


def test_public_scalar_api_still_returns_label_columns():
    out = simulate_dynamic_exit_labels(_frame(_HORIZON_ROWS), horizons=[1, 2])
    assert "label_h1" in out.columns and "label_h2" in out.columns
    # the input frame's own columns survive
    for col in ("ts", "open", "high", "low", "close", "atr14_pct"):
        assert col in out.columns


@pytest.mark.parametrize(
    "rows,horizon,expected",
    [
        (_STOP_ROWS, 2, _STOP_LABEL),
        (_TRAIL_ROWS, 2, _TRAIL_LABEL),
        (_HORIZON_ROWS, 2, _HORIZON_LABEL),
    ],
)
def test_scalar_label_values_are_unchanged(rows, horizon, expected):
    """Hand-derived from the exit policy, so this pins the scalar behaviour
    independently of the refactor rather than against the refactor's own output."""
    assert _labels(_frame(rows), horizon).iloc[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("rows,horizon", [(_STOP_ROWS, 2), (_TRAIL_ROWS, 2), (_HORIZON_ROWS, 2)])
def test_the_endpoint_path_and_the_scalar_path_agree_exactly(rows, horizon):
    """INTERNAL CONSISTENCY ONLY. These two paths now share one implementation, so
    this proves they agree with each other and says NOTHING about whether legacy
    values were preserved. Exact legacy parity is pinned separately below, against
    values captured from the pre-refactor code."""
    df = _frame(rows)
    scalar = _labels(df, horizon)
    labelled, endpoints = simulate_labels_with_endpoints(
        df, horizons=[horizon], product_id="BTC-USD"
    )
    assert labelled[f"label_h{horizon}"].equals(scalar)
    by_entry = {e.entry_row_id: e for e in endpoints if e.horizon == horizon}
    # Exact: the endpoint carries the very float the scalar path produced.
    assert float(by_entry[0].label_value).hex() == float(scalar.iloc[0]).hex()


# ── EXACT legacy parity, pinned from the pre-refactor implementation ─────────
#
# Captured by running `simulate_dynamic_exit_labels` from labels.py AT f2c7927 --
# before the shared-result refactor -- and recording `float.hex()` of every value.
# These are the only bit-for-bit evidence in this file: `pytest.approx` carries a
# default RELATIVE tolerance and is not exactness, and comparing the refactor's two
# paths to each other proves only that they share code.

_LEGACY_LABEL_HEX = {
    "stop": ["-0x1.78d4fdf3b6457p-4", "nan", "nan"],
    "trail": ["0x1.6872b020c4983p-6", "nan", "nan"],
    "horizon": ["0x1.89374bc6a7e18p-9", "nan", "nan"],
    "cap": ["0x1.89374bc6a7e18p-9", "nan", "nan"],
    "atr_nan": ["0x1.6872b020c4983p-6", "nan", "nan"],
    "tail_stop": ["nan", "nan"],
}

_ATR_NAN_ROWS = [
    (100.0, 100.0, 100.0, _FLOOR),
    (110.0, 100.0, 105.0, float("nan")),
    (111.0, 109.0, 110.0, _FLOOR),
]
_TAIL_STOP_ROWS = [(100.0, 100.0, 100.0, _FLOOR), (101.0, 50.0, 60.0, _FLOOR)]

_PARITY_CASES = {
    "stop": (_STOP_ROWS, 2, {}),
    "trail": (_TRAIL_ROWS, 2, {}),
    "horizon": (_HORIZON_ROWS, 2, {}),
    "cap": (_HORIZON_ROWS, 5, {"max_hold_bars": 2}),
    "atr_nan": (_ATR_NAN_ROWS, 2, {}),
    "tail_stop": (_TAIL_STOP_ROWS, 2, {}),
}


def _hexes(series):
    return ["nan" if math.isnan(v) else float(v).hex() for v in series]


@pytest.mark.parametrize("case", sorted(_PARITY_CASES))
def test_legacy_scalar_labels_are_preserved_bit_for_bit(case):
    """Every label value, including its NaN availability mask, identical to the
    pre-refactor implementation's output at the bit level."""
    rows, horizon, kw = _PARITY_CASES[case]
    out = simulate_dynamic_exit_labels(_frame(rows), horizons=[horizon], **kw)
    assert _hexes(out[f"label_h{horizon}"]) == _LEGACY_LABEL_HEX[case]


@pytest.mark.parametrize("case", sorted(_PARITY_CASES))
def test_the_publication_path_reproduces_the_legacy_labels_bit_for_bit(case):
    """The endpoint-producing path must also land on the legacy values exactly —
    otherwise publication would have changed research output while claiming not to."""
    rows, horizon, kw = _PARITY_CASES[case]
    labelled, _ = simulate_labels_with_endpoints(
        _frame(rows), horizons=[horizon], **kw, product_id="BTC-USD"
    )
    assert _hexes(labelled[f"label_h{horizon}"]) == _LEGACY_LABEL_HEX[case]


# ── the endpoint itself ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rows,horizon,kind,offset,basis",
    [
        (_STOP_ROWS, 2, "stop", 1, "assumed_stop_level"),
        (_TRAIL_ROWS, 2, "trail", 1, "assumed_trail_level"),
        (_HORIZON_ROWS, 2, "horizon", 2, "bar_close"),
    ],
)
def test_the_published_endpoint_names_the_branch_that_fired(rows, horizon, kind, offset, basis):
    _, endpoints = simulate_labels_with_endpoints(
        _frame(rows), horizons=[horizon], product_id="BTC-USD"
    )
    e = next(e for e in endpoints if e.entry_row_id == 0 and e.horizon == horizon)
    assert e.exit_kind == kind
    assert e.exit_price_basis == basis
    assert e.exit_row_id == offset
    assert e.bars_held == offset


def test_a_triggered_exit_ends_before_the_horizon():
    """The whole point: an early exit is published as early, not as the nominal
    horizon. Deriving it from the horizon is the arithmetic being removed."""
    _, endpoints = simulate_labels_with_endpoints(
        _frame(_STOP_ROWS), horizons=[2], product_id="BTC-USD"
    )
    e = next(e for e in endpoints if e.entry_row_id == 0)
    assert e.bars_held == 1 < 2


def test_the_cap_binds_when_smaller_than_the_horizon():
    """`horizon_cap = min(horizon, max_hold_bars)`."""
    _, endpoints = simulate_labels_with_endpoints(
        _frame(_HORIZON_ROWS), horizons=[5], max_hold_bars=2, product_id="BTC-USD"
    )
    e = next(e for e in endpoints if e.entry_row_id == 0)
    assert e.exit_kind == "horizon"
    assert e.bars_held == 2


def test_published_clocks_come_from_the_frame_and_the_declared_duration():
    _, endpoints = simulate_labels_with_endpoints(
        _frame(_HORIZON_ROWS), horizons=[2], product_id="BTC-USD"
    )
    e = next(e for e in endpoints if e.entry_row_id == 0)
    assert e.entry_bar_start == 0
    assert e.exit_bar_start == 2 * _H
    assert e.bar_duration_ms == _H
    assert e.entry_available_at == _H
    assert e.exit_observable_at == 3 * _H


def test_trail_endpoints_declare_their_unevidenced_ordering_assumption():
    """The trail raises the peak from a bar's high and then compares it against the
    same bar's low. That ordering is not established by the data."""
    _, endpoints = simulate_labels_with_endpoints(
        _frame(_TRAIL_ROWS), horizons=[2], product_id="BTC-USD"
    )
    e = next(e for e in endpoints if e.entry_row_id == 0)
    assert e.intrabar_order_assumption == "high_before_low"
    assert e.intrabar_timing_known is False


@pytest.mark.parametrize("rows,horizon", [(_STOP_ROWS, 2), (_HORIZON_ROWS, 2)])
def test_non_trail_endpoints_declare_no_ordering_assumption(rows, horizon):
    _, endpoints = simulate_labels_with_endpoints(
        _frame(rows), horizons=[horizon], product_id="BTC-USD"
    )
    e = next(e for e in endpoints if e.entry_row_id == 0)
    assert e.intrabar_order_assumption is None


def test_every_endpoint_carries_the_unclearable_causality_blocker():
    """The ATR defect is unresolved (`intrabar-atr-causality-followup`). Publication
    does not touch it, so every record of this label version declares it."""
    from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER

    _, endpoints = simulate_labels_with_endpoints(
        _frame(_HORIZON_ROWS), horizons=[1, 2], product_id="BTC-USD"
    )
    assert endpoints
    for e in endpoints:
        assert CAUSALITY_BLOCKER in e.blockers


# ── unavailable labels get a disposition, never a fabricated endpoint ────────


def test_the_tail_precheck_is_preserved_even_when_a_stop_would_have_fired():
    """`last_idx >= n` returns NaN BEFORE any bar is evaluated. Row 0's bar 1 low of
    50 would have stopped instantly, and the label is still unavailable. Changing
    that would alter label values, which this slice must not do."""
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 50.0, 60.0, _FLOOR)])
    labels = _labels(df, 2)
    assert math.isnan(labels.iloc[0])


def test_an_unavailable_label_publishes_no_endpoint_and_is_counted():
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 50.0, 60.0, _FLOOR)])
    labelled, endpoints = simulate_labels_with_endpoints(df, horizons=[2], product_id="BTC-USD")
    assert math.isnan(labelled["label_h2"].iloc[0])
    assert not [e for e in endpoints if e.entry_row_id == 0 and e.horizon == 2]


def test_unavailable_rows_are_reported_as_a_named_disposition():
    """A missing endpoint must be visible as a count with a reason, not inferred from
    a gap in the output."""
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 100.0, 100.0, _FLOOR)])
    _, endpoints, dispositions = simulate_labels_with_endpoints(
        df, horizons=[2], product_id="BTC-USD", with_dispositions=True
    )
    assert dispositions["insufficient_horizon"] >= 1
    assert sum(dispositions.values()) + len(endpoints) == len(df) * 1


# ── the legacy non-finite ATR fallback is preserved exactly ──────────────────


def test_a_non_finite_atr_still_falls_back_to_the_floor():
    """Legacy behaviour: a NaN ATR becomes `atr_trail_floor`. The label must equal
    the label of an identical frame whose ATR is the floor outright."""
    nan_rows = [
        (100.0, 100.0, 100.0, _FLOOR),
        (110.0, 100.0, 105.0, float("nan")),
        (111.0, 109.0, 110.0, _FLOOR),
    ]
    floor_rows = _TRAIL_ROWS
    assert _labels(_frame(nan_rows), 2).iloc[0] == pytest.approx(
        _labels(_frame(floor_rows), 2).iloc[0], abs=1e-12
    )


def test_the_non_finite_atr_fallback_still_publishes_a_trail_endpoint():
    nan_rows = [
        (100.0, 100.0, 100.0, _FLOOR),
        (110.0, 100.0, 105.0, float("nan")),
        (111.0, 109.0, 110.0, _FLOOR),
    ]
    _, endpoints = simulate_labels_with_endpoints(
        _frame(nan_rows), horizons=[2], product_id="BTC-USD"
    )
    e = next(e for e in endpoints if e.entry_row_id == 0)
    assert e.exit_kind == "trail"
    assert e.bars_held == 1


# ── published endpoints satisfy the validator that already exists ────────────


def test_published_endpoints_validate_against_the_existing_validator():
    """Publication and validation must agree, or one of them is wrong. This is the
    cross-check that a self-consistent producer cannot give itself."""
    from tools.strategy_discovery.endpoint_records import (
        ExpectedEndpointContext,
        endpoint_digest,
        validate_endpoint,
    )

    df = _frame(_HORIZON_ROWS)
    _, endpoints = simulate_labels_with_endpoints(df, horizons=[2], product_id="BTC-USD")
    source = {int(i): int(t) for i, t in enumerate(df["ts"])}
    for e in endpoints:
        validate_endpoint(
            e,
            source_bar_starts=source,
            expected=ExpectedEndpointContext(
                product_id=e.product_id,
                horizon=e.horizon,
                data_id=e.data_id,
                label_version=e.label_version,
                cost_version=e.cost_version,
                config_id=e.config_id,
                label_value=e.label_value,
                bar_duration_ms=e.bar_duration_ms,
                max_hold_bars=e.max_hold_bars,
                digest=endpoint_digest(e),
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Defects found by Codex executing the in-progress publication path.
#
# Both are the same shape once more: a value was TRUSTED where it should have been
# validated. An exit offset was taken as proof of availability without checking the
# PnL was finite, and a timestamp was coerced by `to_numpy(dtype="int64")` — which
# TRUNCATES — before anything checked it was a whole number.
# ─────────────────────────────────────────────────────────────────────────────


def test_a_nonfinite_label_publishes_no_endpoint():
    """A NaN close yields a NaN PnL through the HORIZON branch, which still reports an
    exit offset — so availability checked on the offset alone published a record
    carrying `label_value = NaN`."""
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 99.0, float("nan"), _FLOOR)])
    labelled, endpoints = simulate_labels_with_endpoints(df, horizons=[1], product_id="BTC-USD")
    assert math.isnan(labelled["label_h1"].iloc[0])
    assert endpoints == []


def test_nonfinite_and_insufficient_horizon_are_counted_separately():
    """Different causes: one is a boundary condition, the other means the input was
    unusable. Collapsing them would hide bad data behind an expected tail count."""
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 99.0, float("nan"), _FLOOR)])
    _, _, dispositions = simulate_labels_with_endpoints(
        df, horizons=[1], product_id="BTC-USD", with_dispositions=True
    )
    assert dispositions["nonfinite_label"] == 1  # row 0: NaN close
    assert dispositions["insufficient_horizon"] == 1  # row 1: horizon runs off the end


@pytest.mark.parametrize("bad_ts", [[0.5, _H + 0.5], [0.0, 1.5]])
def test_fractional_timestamps_are_refused_not_truncated(bad_ts):
    """`to_numpy(dtype="int64")` truncates, so 0.5 would have been published as 0 —
    a bar start that does not describe the frame, passing every internal check."""
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 99.0, 100.5, _FLOOR)])
    df["ts"] = bad_ts
    with pytest.raises(ValueError, match="ts"):
        simulate_labels_with_endpoints(df, horizons=[1], product_id="BTC-USD")


def test_non_chronological_timestamps_are_refused():
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 99.0, 100.5, _FLOOR)])
    df["ts"] = [_H, 0]
    with pytest.raises(ValueError, match="increasing"):
        simulate_labels_with_endpoints(df, horizons=[1], product_id="BTC-USD")


def test_duplicate_timestamps_are_refused():
    df = _frame([(100.0, 100.0, 100.0, _FLOOR), (101.0, 99.0, 100.5, _FLOOR)])
    df["ts"] = [_H, _H]
    with pytest.raises(ValueError, match="increasing"):
        simulate_labels_with_endpoints(df, horizons=[1], product_id="BTC-USD")


@pytest.mark.parametrize("bad", ["UNKNOWN", "", "   ", 7, None])
def test_a_placeholder_product_identity_cannot_be_published(bad):
    """A default placeholder must never become an artifact's identity."""
    with pytest.raises(ValueError, match="product_id"):
        simulate_labels_with_endpoints(_frame(_HORIZON_ROWS), horizons=[2], product_id=bad)


def test_the_candidate_frame_carries_its_own_row_identity():
    """Later filtering must have an identity to preserve rather than reconstructing
    positions after a reindex."""
    labelled, _ = simulate_labels_with_endpoints(
        _frame(_HORIZON_ROWS), horizons=[2], product_id="BTC-USD"
    )
    assert list(labelled["source_row_id"]) == [0, 1, 2]


# ── types validated BEFORE coercion ──────────────────────────────────────────


@pytest.mark.parametrize("horizon", [True, False, 2.0, "2", 0, -1, None])
def test_a_coercible_horizon_is_refused(horizon):
    """`int(True)` is 1, so a bool horizon would silently become horizon 1 and be
    bound into the artifact identity as if it were a real parameter."""
    with pytest.raises(ValueError, match="horizon"):
        simulate_labels_with_endpoints(
            _frame(_HORIZON_ROWS), horizons=[horizon], product_id="BTC-USD"
        )


@pytest.mark.parametrize("cap", [True, 2.0, "2", 0, -1, None])
def test_a_coercible_max_hold_bars_is_refused(cap):
    with pytest.raises(ValueError, match="max_hold_bars"):
        simulate_labels_with_endpoints(
            _frame(_HORIZON_ROWS), horizons=[2], product_id="BTC-USD", max_hold_bars=cap
        )


@pytest.mark.parametrize("field", ["stop_loss_pct", "atr_trail_floor", "round_trip_fee"])
@pytest.mark.parametrize("value", [True, "0.05", None, float("nan"), float("inf")])
def test_a_coercible_or_non_finite_exit_parameter_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        simulate_labels_with_endpoints(
            _frame(_HORIZON_ROWS), horizons=[2], product_id="BTC-USD", **{field: value}
        )


def test_a_padded_product_id_is_refused():
    """The validator requires unpadded identities; the producer must not emit one it
    would then reject."""
    with pytest.raises(ValueError, match="product_id"):
        simulate_labels_with_endpoints(_frame(_HORIZON_ROWS), horizons=[2], product_id=" BTC-USD ")


@pytest.mark.parametrize("recipe", ["", "   ", 7, None])
def test_an_unnamed_feature_recipe_is_refused(recipe):
    """The ATR recipe is part of what produced every trail exit; it cannot be blank."""
    with pytest.raises(ValueError, match="feature_recipe"):
        simulate_labels_with_endpoints(
            _frame(_HORIZON_ROWS), horizons=[2], product_id="BTC-USD", feature_recipe=recipe
        )
