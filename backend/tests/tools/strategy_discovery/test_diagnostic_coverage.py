"""Requested-pair coverage cannot use the survivor list as its denominator."""

import pytest

from tools.strategy_discovery.diagnostic_coverage import declare_pairs, summarize_coverage


def record(pid, horizon, status="completed", reason_code=None, observed=None):
    return {
        "pid": pid,
        "horizon": horizon,
        "status": status,
        "reason_code": reason_code,
        "observed": observed or {},
    }


def test_request_generator_is_materialized_before_cross_product():
    pairs = declare_pairs(["ETH-USD", "BTC-USD", "BTC-USD"], (h for h in [1, 4, 1]))
    assert pairs == (("BTC-USD", 1), ("BTC-USD", 4), ("ETH-USD", 1), ("ETH-USD", 4))


@pytest.mark.parametrize("horizon", [True, False, 1.0, 1.5, "1", 0, -1, None])
def test_request_rejects_coerced_horizons(horizon):
    with pytest.raises(ValueError, match="horizon"):
        declare_pairs(["BTC-USD"], [horizon])


@pytest.mark.parametrize("pids,horizons", [([], [1]), (["BTC-USD"], []), ([""], [1])])
def test_empty_requests_cannot_be_vacuously_complete(pids, horizons):
    with pytest.raises(ValueError):
        declare_pairs(pids, horizons)


def test_missing_product_removes_neither_horizon_from_denominator():
    pairs = declare_pairs(["BTC-USD", "MISSING-USD"], [1, 4])
    rows = [record("BTC-USD", h) for h in [1, 4]]
    with pytest.raises(ValueError, match="missing"):
        summarize_coverage(pairs, rows)
    rows += [
        record("MISSING-USD", h, "excluded", "missing_input", {"exists": False}) for h in [1, 4]
    ]
    summary = summarize_coverage(pairs, rows)
    assert summary.requested_count == 4
    assert summary.completed_count == 2
    assert summary.excluded_count == 2
    assert summary.dispositions_complete is True
    assert summary.evaluation_validated is False
    assert summary.deployment_eligible is False


def test_interrupted_run_preserves_pending_and_running_pairs():
    pairs = declare_pairs(["BTC-USD"], [1, 4, 24])
    summary = summarize_coverage(
        pairs,
        [record("BTC-USD", 1), record("BTC-USD", 4, "running"), record("BTC-USD", 24, "pending")],
    )
    assert summary.dispositions_complete is False
    assert summary.pending_count == summary.running_count == 1
    assert "incomplete_pair_dispositions" in summary.blockers


def test_terminal_errors_are_accounted_but_never_validated_evidence():
    pairs = declare_pairs(["BTC-USD"], [1])
    summary = summarize_coverage(
        pairs, [record("BTC-USD", 1, "error", "invalid_input", {"rows": 3})]
    )
    assert summary.dispositions_complete is True
    assert summary.error_count == 1
    assert summary.evaluation_validated is False


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unexpected",
        "boolean_horizon",
        "missing_reason",
        "missing_facts",
        "completed_reason",
        "unknown_status",
        "nonfinite_fact",
    ],
)
def test_invalid_dispositions_fail_closed(mutation):
    pairs = declare_pairs(["BTC-USD"], [1])
    rows = [record("BTC-USD", 1)]
    if mutation == "duplicate":
        rows *= 2
    elif mutation == "unexpected":
        rows[0]["pid"] = "ETH-USD"
    elif mutation == "boolean_horizon":
        rows[0]["horizon"] = True
    elif mutation == "missing_reason":
        rows[0].update(status="excluded", observed={"rows": 0})
    elif mutation == "missing_facts":
        rows[0].update(status="error", reason_code="bad_input")
    elif mutation == "completed_reason":
        rows[0]["reason_code"] = "bad_input"
    elif mutation == "unknown_status":
        rows[0]["status"] = "success"
    else:
        rows[0]["observed"] = {"rows": float("nan")}
    with pytest.raises(ValueError):
        summarize_coverage(pairs, rows)


def test_even_all_completed_claims_do_not_validate_fold_or_leaf_evidence():
    pairs = declare_pairs(["BTC-USD"], [1])
    summary = summarize_coverage(pairs, [record("BTC-USD", 1)])
    assert summary.dispositions_complete is True
    assert summary.completed_count == 1
    assert summary.evaluation_validated is False
    assert "fold_leaf_evidence_not_validated" in summary.blockers


@pytest.mark.parametrize(
    "changes",
    [
        {"blockers": ()},
        {"requested_count": 100},
        {"pending_count": 1, "completed_count": 0},
        {"completed_count": -1, "pending_count": 2},
        {"requested_count": True},
        {"completed_count": 1.0},
        {"dispositions_complete": 1},
        {"blockers": ["fold_leaf_evidence_not_validated"]},
        {"completed_count": 0, "excluded_count": 1},
        {"completed_count": 0, "error_count": 1},
    ],
)
def test_direct_summary_constructor_rejects_inconsistent_claims(changes):
    from tools.strategy_discovery.diagnostic_coverage import CoverageSummary

    values = dict(
        requested_count=1,
        pending_count=0,
        running_count=0,
        completed_count=1,
        excluded_count=0,
        error_count=0,
        dispositions_complete=True,
        blockers=("fold_leaf_evidence_not_validated",),
    )
    values.update(changes)
    with pytest.raises(ValueError):
        CoverageSummary(**values)
