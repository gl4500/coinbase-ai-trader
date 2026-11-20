"""Tests for tools.strategy_discovery.build_phase4 (Phase 4 driver)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.build_phase4 import build_phase4


def _write_minimal_phase3(phase3_dir: Path, pids=None):
    _PROFILE_COLUMNS = [
        "pid",
        "horizon",
        "leaf_id",
        "rule_path_summary",
        "cumulative_profit_raw",
        "cumulative_profit_deflated",
        "deflation_pp",
        "win_rate",
        "avg_win",
        "avg_loss",
        "max_dd",
        "sortino",
        "trade_count",
        "n_folds_passed_q0",
        "chosen_depth",
        "chosen_min_leaf",
        "bootstrap_triggered",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "n_combos_searched",
        "inner_cv_se",
        "schema_version",
    ]
    rows = [
        (
            "BTC-USD",
            24,
            0,
            "price_over_ema20 > 1.0",
            0.10,
            0.06,
            0.04,
            0.6,
            0.08,
            -0.04,
            0.20,
            1.2,
            50,
            5,
            5,
            50,
            False,
            None,
            None,
            9,
            0.015,
            1,
        ),
        (
            "ETH-USD",
            24,
            1,
            "price_over_ema20 > 1.0",
            0.08,
            0.05,
            0.03,
            0.55,
            0.07,
            -0.04,
            0.18,
            1.1,
            40,
            5,
            5,
            50,
            False,
            None,
            None,
            9,
            0.012,
            1,
        ),
        (
            "SOL-USD",
            24,
            2,
            "price_over_ema20 > 1.0",
            0.06,
            0.04,
            0.02,
            0.50,
            0.06,
            -0.05,
            0.22,
            1.0,
            35,
            4,
            5,
            50,
            False,
            None,
            None,
            9,
            0.010,
            1,
        ),
    ]
    if pids is not None:
        rows = [row for row in rows if row[0] in pids]
    df = pd.DataFrame(rows, columns=_PROFILE_COLUMNS)
    df["schema_version"] = 3
    df["rule_version"] = "profile_rule_binding_v1"
    df["rule_digest"] = None
    df["validation_version"] = "chronological_distinct_folds_v2"
    df["n_folds_evaluated"] = 5
    phase3_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False), phase3_dir / "profiles_h24.parquet"
    )
    sidecar = {
        "BTC-USD__0": "price_over_ema20 > 1.0",
        "ETH-USD__1": "price_over_ema20 > 1.0",
        "SOL-USD__2": "price_over_ema20 > 1.0",
    }
    if pids is not None:
        sidecar = {key: text for key, text in sidecar.items() if key.split("__")[0] in pids}
    from tests.tools.strategy_discovery.rule_fixtures import write_bound_sidecar_fixture

    write_bound_sidecar_fixture(phase3_dir / "rule_paths_h24.json", sidecar)


def _write_minimal_phase2(phase2_dir: Path, pids, *, publish_endpoints: bool = True):
    """Phase 2 output as the producer writes it: the parquet AND its endpoint publication.

    `publish_endpoints=False` reproduces a pre-endpoint artifact, which Phase 4 must report as
    unpublished rather than replay. Publishing is the default because without it these tests
    ran on an EMPTY universe -- every product excluded for a missing sidecar -- and passed
    while exercising nothing. That was verified by instrumenting the loader: 0 loaded, 3
    excluded.
    """
    from tools.strategy_discovery.build_phase2 import _publish_endpoints
    from tools.strategy_discovery.endpoint_dataset import build_data_id
    from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER, LabelEndpoint
    from tools.strategy_discovery.labels import COST_VERSION, LABEL_VERSION

    phase2_dir.mkdir(parents=True, exist_ok=True)
    horizon, bar = 24, 3_600_000
    exit_config = {
        "stop_loss_pct": 0.08,
        "atr_trail_floor": 0.06,
        "max_hold_bars": 168,
        "round_trip_fee": 0.012,
    }
    for pid in pids:
        n = 100
        starts = (np.arange(n, dtype="int64") * bar).tolist()
        labels = [0.05 if row + horizon < n else float("nan") for row in range(n)]
        df = pd.DataFrame(
            {
                "ts": starts,
                "source_row_id": list(range(n)),
                "close": [1.0] * n,
                "high": [1.0] * n,
                "low": [1.0] * n,
                "atr14_pct": [0.06] * n,
                "price_over_ema20": [1.5] * n,
                "vol_over_mc": [0.01] * n,
                "label_h24": labels,
            }
        )
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False), phase2_dir / f"{pid}.parquet"
        )
        if not publish_endpoints:
            continue
        data_id = build_data_id(
            product_id=pid,
            bar_duration_ms=bar,
            timestamps=starts,
            closes=df["close"].tolist(),
            highs=df["high"].tolist(),
            lows=df["low"].tolist(),
            atr_pcts=df["atr14_pct"].tolist(),
            feature_recipe="atr14_pct_wilder_v1",
            config=exit_config,
        )
        records = [
            LabelEndpoint(
                product_id=pid,
                horizon=horizon,
                data_id=data_id,
                label_version=LABEL_VERSION,
                cost_version=COST_VERSION,
                config_id=data_id,
                label_value=float(value),
                entry_row_id=row,
                exit_row_id=row + horizon,
                bars_held=horizon,
                max_hold_bars=168,
                entry_bar_start=starts[row],
                exit_bar_start=starts[row + horizon],
                bar_duration_ms=bar,
                entry_available_at=starts[row] + bar,
                exit_observable_at=starts[row + horizon] + bar,
                exit_kind="horizon",
                exit_price_basis="bar_close",
                intrabar_timing_known=True,
                intrabar_order_assumption=None,
                blockers=(CAUSALITY_BLOCKER,),
            )
            for row, value in enumerate(labels)
            if value == value
        ]
        # Published through the REAL producer helper, so the sidecar schema cannot drift from
        # what the adapter reads.
        _publish_endpoints(phase2_dir, pid, records, [horizon])


def test_sweeps_all_three_caps_writes_three_deployments(tmp_path: Path):
    from tools.strategy_discovery.build_phase4 import build_phase4

    phase3_dir = tmp_path / "phase3"
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase4"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])
    cards = build_phase4(
        phase3_dir=phase3_dir,
        phase2_dir=phase2_dir,
        output_dir=output_dir,
        caps=[3, 4, 5],
        beam_width=3,
        pool_size=3,
        bootstrap_iter=50,
        seed=42,
        horizons=[24],
    )
    assert set(cards.keys()) == {3, 4, 5}
    for cap in [3, 4, 5]:
        assert (output_dir / f"deployment_n{cap}.json").exists()


def test_writes_scorecard_md_and_telemetry_parquet(tmp_path: Path):
    from tools.strategy_discovery.build_phase4 import build_phase4

    phase3_dir = tmp_path / "phase3"
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase4"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])
    build_phase4(
        phase3_dir=phase3_dir,
        phase2_dir=phase2_dir,
        output_dir=output_dir,
        caps=[3],
        beam_width=3,
        pool_size=3,
        bootstrap_iter=50,
        seed=42,
        horizons=[24],
    )
    assert (output_dir / "scorecard.md").exists()
    # Telemetry parquet may or may not have content (depending on whether any trades fired)
    # but the file should be writable; if no telemetry, skip the assertion
    tele_path = output_dir / "portfolio_telemetry_n3.parquet"
    if tele_path.exists():
        df = pq.read_table(tele_path).to_pandas()
        # If exists, must have schema_version column
        assert "schema_version" in df.columns


def test_main_returns_zero_on_at_least_one_passing_cap(tmp_path: Path, monkeypatch):
    from tools.strategy_discovery import build_phase4 as bp4

    # Mock build_phase4 to return a card with overall_pass=True
    def fake_build(*args, **kwargs):
        from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
        from tools.strategy_discovery.scorecard import CapScorecard

        return {
            3: CapScorecard(
                cap=3,
                metrics=PortfolioMetrics(),
                k_evaluated=0,
                inflation=0.0,
                gates={},
                overall_pass=True,
                selected_profiles=[],
            )
        }

    monkeypatch.setattr(bp4, "build_phase4", fake_build)
    rc = bp4.main(
        [
            "--phase3-dir",
            str(tmp_path),
            "--phase2-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--caps",
            "3",
        ]
    )
    assert rc == 0


def test_passing_research_artifact_cannot_authorize_deployment(tmp_path):
    from tools.strategy_discovery.build_phase4 import _write_deployment_json
    from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
    from tools.strategy_discovery.scorecard import CapScorecard, evaluate_cap_gates

    metrics = PortfolioMetrics(
        cumulative_profit_raw=0.3,
        cumulative_profit_deflated=0.2,
        max_dd=0.1,
        sortino=2,
        trade_count=100,
    )
    gates, passed = evaluate_cap_gates(metrics)
    assert passed
    card = CapScorecard(3, metrics, 100, 0.1, gates, passed, [])
    path = tmp_path / "deployment_n3.json"
    _write_deployment_json(card, path)
    payload = json.loads(path.read_text())
    assert payload["evaluation_scope"] == "research_selection"
    assert payload["deployment_eligible"] is False
    assert payload["deployment_blockers"]
    assert payload["gates"]["overall"] == "pass"
    assert payload["gates"]["scope"] == "research_only"


# ── the universe is reported, and a corrupt artifact stops the run ────────────


def test_a_product_without_published_endpoints_is_reported_not_silently_dropped(tmp_path: Path):
    """Codex 56aeec3b / b1a08054. An unpublished product may be excluded -- one product should
    not abort a sweep over many -- but the thinner universe must be VISIBLE, not merely
    inferable from a smaller profit number."""
    phase3_dir, phase2_dir, output_dir = tmp_path / "p3", tmp_path / "p2", tmp_path / "out"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD"])
    _write_minimal_phase2(phase2_dir, ["SOL-USD"], publish_endpoints=False)

    build_phase4(phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir, caps=[2])

    payload = json.loads((output_dir / "deployment_n2.json").read_text(encoding="utf-8"))
    universe = payload["universe"]
    assert set(universe["requested_products"]) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    assert "SOL-USD" not in universe["evaluated_products"]
    assert universe["excluded_products"]["SOL-USD"] == "endpoint_sidecar_missing"
    assert universe["evaluated_product_count"] == 2
    assert universe["requested_product_count"] == 3


def test_a_corrupt_published_artifact_stops_the_run(tmp_path: Path):
    """The counterpart. A published artifact that does not describe its own frame must RAISE
    naming the product: excluding it would shrink the optimisation universe while still writing
    a scorecard that looks successful, which is the silent thinning §9.3 forbids.

    No scorecard may be written for the affected cap.
    """
    from tools.strategy_discovery.build_phase4 import EndpointArtifactError

    phase3_dir, phase2_dir, output_dir = tmp_path / "p3", tmp_path / "p2", tmp_path / "out"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])

    # the artifact stays internally coherent; only the FRAME moves, so nothing but the
    # identity recompute can notice
    frame = pq.read_table(phase2_dir / "BTC-USD.parquet").to_pandas()
    frame.loc[0, "close"] = float(frame.loc[0, "close"]) + 1.0
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), phase2_dir / "BTC-USD.parquet"
    )

    with pytest.raises(EndpointArtifactError, match="BTC-USD"):
        build_phase4(phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir, caps=[2])
    assert not (output_dir / "deployment_n2.json").exists(), (
        "a run that hit a corrupt artifact must not leave an apparently successful scorecard"
    )


# -- a product excluded for its FEATURES leaves the profile pool too -----------------
# Codex R2-78. The profile filter used to run only when some product was unpublished, so a
# product with missing or empty features kept its profiles while its frame was dropped. No
# unpublished product is present in these runs: the feature exclusion must be enough.


def _write_empty_phase2(phase2_dir: Path, pid: str) -> None:
    phase2_dir.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame(
        {
            "ts": pd.Series([], dtype="int64"),
            "source_row_id": pd.Series([], dtype="int64"),
            "close": pd.Series([], dtype="float64"),
            "price_over_ema20": pd.Series([], dtype="float64"),
            "label_h24": pd.Series([], dtype="float64"),
        }
    )
    pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), phase2_dir / f"{pid}.parquet")


@pytest.mark.parametrize("bad_features", ["missing", "empty"])
def test_a_product_without_usable_features_leaves_the_pool_and_the_rest_replays(
    tmp_path: Path, bad_features
):
    phase3_dir, phase2_dir, output_dir = tmp_path / "p3", tmp_path / "p2", tmp_path / "out"
    _write_minimal_phase3(phase3_dir, pids={"BTC-USD", "ETH-USD"})
    _write_minimal_phase2(phase2_dir, ["BTC-USD"])
    if bad_features == "empty":
        _write_empty_phase2(phase2_dir, "ETH-USD")
    assert not (phase2_dir / "ETH-USD.endpoints.json").exists()

    build_phase4(phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir, caps=[1])

    payload = json.loads((output_dir / "deployment_n1.json").read_text(encoding="utf-8"))
    universe = payload["universe"]
    assert universe["requested_products"] == ["BTC-USD", "ETH-USD"]
    assert universe["evaluated_products"] == ["BTC-USD"]
    assert universe["excluded_products"] == {"ETH-USD": "phase2_features_missing_or_empty"}
    assert universe["requested_product_count"] == 2
    assert universe["evaluated_product_count"] == 1
    assert [p["pid"] for p in payload["profiles"]] == ["BTC-USD"]
    assert payload["portfolio_metrics"]["trade_count"] > 0, "the GOOD product must really replay"


def test_a_corrupt_artifact_still_raises_beside_a_feature_exclusion(tmp_path: Path):
    """The exclusion path must not swallow the fail-closed one."""
    from tools.strategy_discovery.build_phase4 import EndpointArtifactError

    phase3_dir, phase2_dir, output_dir = tmp_path / "p3", tmp_path / "p2", tmp_path / "out"
    _write_minimal_phase3(phase3_dir, pids={"BTC-USD", "ETH-USD"})
    _write_minimal_phase2(phase2_dir, ["BTC-USD"])
    frame = pq.read_table(phase2_dir / "BTC-USD.parquet").to_pandas()
    frame.loc[0, "close"] = float(frame.loc[0, "close"]) + 1.0
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), phase2_dir / "BTC-USD.parquet"
    )

    with pytest.raises(EndpointArtifactError, match="BTC-USD"):
        build_phase4(phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir, caps=[1])
    assert not (output_dir / "deployment_n1.json").exists()


def test_labels_and_endpoints_from_the_real_producer_reach_the_replay(tmp_path: Path):
    """Codex 71e39e92: full label-producer integration, not a hand-built record set.

    The labels AND the endpoints come from `simulate_labels_with_endpoints` on real price
    movement, are published through the producer's own `_publish_endpoints`, and are then
    loaded, validated and replayed by Phase 4. Nothing in the chain is synthesised by the test
    except the prices themselves.

    Non-vacuity is ASSERTED here rather than checked by instrumentation: the universe must be
    fully evaluated, and the replay must actually select trades.
    """
    from tools.strategy_discovery.build_phase2 import _publish_endpoints
    from tools.strategy_discovery.labels import simulate_labels_with_endpoints

    phase3_dir, phase2_dir, output_dir = tmp_path / "p3", tmp_path / "p2", tmp_path / "out"
    _write_minimal_phase3(phase3_dir)
    phase2_dir.mkdir(parents=True, exist_ok=True)

    horizon, bar, n = 24, 3_600_000, 120
    for pid in ("BTC-USD", "ETH-USD", "SOL-USD"):
        # real movement, so stops and trails can genuinely fire rather than every exit
        # landing on the nominal horizon
        closes = [100.0 + 4.0 * np.sin(i / 7.0) + i * 0.05 for i in range(n)]
        source = pd.DataFrame(
            {
                "ts": (np.arange(n, dtype="int64") * bar).tolist(),
                "open": closes,
                "high": [c * 1.004 for c in closes],
                "low": [c * 0.996 for c in closes],
                "close": closes,
                "atr14_pct": [0.06] * n,
            }
        )
        labelled, endpoints = simulate_labels_with_endpoints(
            source, horizons=[horizon], product_id=pid
        )
        assert endpoints, "the producer must emit endpoints for this frame"
        # the rule feature the Phase 3 profiles read
        labelled["price_over_ema20"] = [1.5] * n
        labelled["vol_over_mc"] = [0.01] * n
        pq.write_table(
            pa.Table.from_pandas(labelled, preserve_index=False), phase2_dir / f"{pid}.parquet"
        )
        _publish_endpoints(phase2_dir, pid, endpoints, [horizon])

    cards = build_phase4(
        phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir, caps=[2]
    )

    payload = json.loads((output_dir / "deployment_n2.json").read_text(encoding="utf-8"))
    universe = payload["universe"]
    assert universe["evaluated_product_count"] == 3, "every product must reach the replay"
    assert universe["excluded_products"] == {}
    # the replay did real work: trades were selected from producer-generated endpoints
    assert payload["portfolio_metrics"]["trade_count"] > 0
    assert cards is not None


def test_the_published_universe_is_asserted_not_merely_instrumented(tmp_path: Path):
    """Codex 71e39e92. The earlier vacuity -- every product excluded for a missing sidecar,
    both output tests green on an EMPTY universe -- was caught by instrumenting the loader by
    hand. An assertion belongs in the suite, or the same hole reopens unnoticed."""
    phase3_dir, phase2_dir, output_dir = tmp_path / "p3", tmp_path / "p2", tmp_path / "out"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])

    build_phase4(phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir, caps=[3])

    payload = json.loads((output_dir / "deployment_n3.json").read_text(encoding="utf-8"))
    universe = payload["universe"]
    assert universe["evaluated_product_count"] == 3
    assert universe["excluded_products"] == {}
    assert payload["portfolio_metrics"]["trade_count"] > 0, (
        "a green output test on an empty universe proves nothing"
    )
