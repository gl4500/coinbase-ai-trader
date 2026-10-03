import json

import numpy as np
import pandas as pd
import pytest

from tools.vol_target import prereg as P
from tools.vol_target import screen as V

DAY = 86400


def _raw(first="2016-05-18", last="2026-10-02", drop=(), seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range(first, last, freq="D")
    days = days[~days.isin(pd.to_datetime(list(drop)))]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days))))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "start": ((days - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(
                "int64"
            ),
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": 1.0,
            "page": 0,
        }
    )


CONS = {p: {"base_increment": 1e-8, "base_min": 1e-8, "quote_min": 1.0} for p in P.PRODUCTS}


@pytest.fixture(scope="module")
def report():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(V.P, "BOOT_RESAMPLES", 200)  # speed only; the bootstrap never gates
        return V.evaluate({p: _raw(seed=i) for i, p in enumerate(P.PRODUCTS)}, CONS)


def test_evaluate_runs_both_periods_every_scenario(report):
    assert report["periods"]["dev"] == ("2016-08-30", "2025-04-13")
    assert set(report["results"]["block"]["scenarios"]) == {s.name for s in P.SCENARIOS}
    assert report["verdict"]["verdict"] in {"KILL", "INCONCLUSIVE", "PASS_TO_FORWARD"}


def test_first_executions_follow_the_frozen_calendar(report):
    d = report["results"]
    assert d["dev"]["scenarios"]["P0"]["calendar"]["first_execute"] == "2016-09-05"
    assert d["dev"]["scenarios"]["D1"]["calendar"]["first_execute"] == "2016-09-06"
    assert d["block"]["scenarios"]["P0"]["calendar"]["first_execute"] == "2025-04-14"
    assert d["block"]["scenarios"]["D1"]["calendar"]["first_execute"] == "2025-04-15"


def test_fixed_diagnostic_never_gates(report):
    s = report["results"]["block"]["scenarios"]["P0"]
    assert "fixed50" in s and "passes" in s and "fixed50" not in json.dumps(s["passes"])


def test_coverage_counts_valid_p0_block_decisions_per_sleeve(report):
    cov = report["block_valid_decisions"]
    assert set(cov) == set(P.PRODUCTS) and all(v >= 70 for v in cov.values())


def test_a_data_gap_beyond_tolerance_is_inconclusive_data():
    gap = [f"2020-01-{d:02d}" for d in range(1, 10)]
    r = V.evaluate({p: _raw(drop=gap) for p in P.PRODUCTS}, CONS)
    assert r["verdict"]["reason"] == "data"


def test_import_refuses_a_snapshot_whose_manifest_is_not_the_locked_one(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="lock"):
        V.import_snapshot(src, tmp_path / "out")


def test_import_verifies_every_raw_file_and_copies_bytes(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "BTC-USD.raw.parquet").write_bytes(b"btc")
    (src / "ETH-USD.raw.parquet").write_bytes(b"eth")
    man = {
        "products": {
            "BTC-USD": {"sha256": V.S._sha(src / "BTC-USD.raw.parquet")},
            "ETH-USD": {"sha256": "sha256:" + "0" * 64},
        }
    }
    (src / "manifest.json").write_text(json.dumps(man))
    monkeypatch.setattr(V.P, "SOURCE_SNAPSHOT_LOCK", V.S._sha(src / "manifest.json"))
    with pytest.raises(RuntimeError, match="ETH-USD"):
        V.import_snapshot(src, out)
    assert not (out / "manifest.json").exists()  # nothing published on failure
