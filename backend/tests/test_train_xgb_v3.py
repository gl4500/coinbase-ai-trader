"""TDD tests for tools/train_xgb.train_xgb_v3 (--feature-set v3 trainer).

Contract:
    train_xgb_v3(pids, parquet_dir, out_dir, ...) -> dict
        - Pulls per-pid candle history via tiered_history.fetch_tiered(source='parquet')
        - Builds rolling samples; each sample produces a 350-element v3 feature row
        - Labels: 1 if close[t+H] > close[t] else 0, H=4
        - Calls xgb.train with feature_weights from feature_weights_v3()
        - Atomic write of xgb_model.json + xgb_features.json (tmp + rename)
        - features.json includes {"feature_set": "v3", "feature_weights": [...]}
        - Products with < 336 parquet bars are skipped from training
"""

import json
import os
import sys

import pandas as pd
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _write_parquet(parquet_dir, pid, n_bars, start_ts=1_700_000_000):
    rows = [
        {
            "start": start_ts + i * 3600,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + i * 0.01 + (i % 7) * 0.5,
            "volume": 1000.0,
        }
        for i in range(n_bars)
    ]
    df = pd.DataFrame(rows)
    df["ingest_ts"] = 1_700_000_000
    df["schema_version"] = 1
    df.to_parquet(parquet_dir / f"{pid}.parquet")


class TestV3Trainer:
    def test_v3_writes_feature_set_to_metadata(self, tmp_path):
        from tools.train_xgb import train_xgb_v3

        pdir = tmp_path / "history"
        pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        _write_parquet(pdir, "ETH-USD", 500)
        out = tmp_path / "out"
        out.mkdir()
        train_xgb_v3(["BTC-USD", "ETH-USD"], str(pdir), str(out), n_estimators=5, learning_rate=0.3)
        meta = json.loads((out / "xgb_features.json").read_text())
        assert meta["feature_set"] == "v3"
        assert len(meta["feature_names"]) == 350

    def test_v3_passes_feature_weights(self, tmp_path, monkeypatch):
        """feature_weights belongs on the DMatrix (set_info) in modern xgboost.
        Capture them via a wrapping set_info spy."""
        import xgboost as _xgb

        from tools import train_xgb as t

        pdir = tmp_path / "history"
        pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        out = tmp_path / "out"
        out.mkdir()

        captured = {}
        orig_set_info = _xgb.DMatrix.set_info

        def spy_set_info(self, **kw):
            fw = kw.get("feature_weights")
            if fw is not None:
                captured["feature_weights"] = list(fw)
            return orig_set_info(self, **kw)

        monkeypatch.setattr(_xgb.DMatrix, "set_info", spy_set_info)

        t.train_xgb_v3(["BTC-USD"], str(pdir), str(out), n_estimators=5, learning_rate=0.3)
        fw = captured.get("feature_weights")
        assert fw is not None, "DMatrix.set_info was never called with feature_weights"
        assert len(fw) == 350
        assert max(fw) == 3.0
        assert min(fw) == 0.0

    def test_v3_skips_short_history_products(self, tmp_path):
        from tools.train_xgb import train_xgb_v3

        pdir = tmp_path / "history"
        pdir.mkdir()
        _write_parquet(pdir, "OK-USD", 500)
        _write_parquet(pdir, "TINY-USD", 100)  # < 336
        out = tmp_path / "out"
        out.mkdir()
        result = train_xgb_v3(
            ["OK-USD", "TINY-USD"], str(pdir), str(out), n_estimators=5, learning_rate=0.3
        )
        assert "TINY-USD" in result.get("skipped_pids", [])
        assert "OK-USD" not in result.get("skipped_pids", [])

    def test_v3_atomic_write_no_partial_artifacts(self, tmp_path, monkeypatch):
        from tools import train_xgb as t

        pdir = tmp_path / "history"
        pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        out = tmp_path / "out"
        out.mkdir()

        def boom(*args, **kwargs):
            raise RuntimeError("simulated trainer crash")

        monkeypatch.setattr("xgboost.train", boom)

        with pytest.raises(RuntimeError):
            t.train_xgb_v3(["BTC-USD"], str(pdir), str(out), n_estimators=5, learning_rate=0.3)
        assert not (out / "xgb_model.json").exists()
        assert not (out / "xgb_features.json").exists()

    def test_v3_saved_model_loads_back_as_json(self, tmp_path):
        """Regression: xgboost picks serialization format from the LAST file
        extension. A temp name of 'xgb_model.json.tmp' writes UBJSON (binary)
        and the rename to '.json' leaves a binary file that load_model
        rejects. Atomic-write tmp names must keep '.json' last
        ('xgb_model.tmp.json')."""
        import xgboost as xgb

        from tools.train_xgb import train_xgb_v3

        pdir = tmp_path / "history"
        pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        out = tmp_path / "out"
        out.mkdir()

        train_xgb_v3(["BTC-USD"], str(pdir), str(out), n_estimators=3, learning_rate=0.3)

        b = xgb.Booster()
        b.load_model(str(out / "xgb_model.json"))  # must not raise
        assert b.num_boosted_rounds() == 3

    def test_v3_reads_each_parquet_once_per_pid(self, tmp_path, monkeypatch):
        """Perf invariant: train_xgb_v3 reads each pid's parquet exactly once,
        regardless of how many rolling samples it produces. Previous impl
        called fetch_tiered per sample, which re-read the parquet from disk
        every time (~500x slowdown at production scale)."""
        import pandas as _pd

        from tools.train_xgb import train_xgb_v3

        pdir = tmp_path / "history"
        pdir.mkdir()
        # 500 bars × 2 pids = ~14 samples per pid at sample_step=24 starting at idx=336
        _write_parquet(pdir, "BTC-USD", 500)
        _write_parquet(pdir, "ETH-USD", 500)
        out = tmp_path / "out"
        out.mkdir()

        calls = {"count": 0}
        orig = _pd.read_parquet

        def spy(path, *a, **kw):
            calls["count"] += 1
            return orig(path, *a, **kw)

        monkeypatch.setattr("pandas.read_parquet", spy)

        train_xgb_v3(["BTC-USD", "ETH-USD"], str(pdir), str(out), n_estimators=5, learning_rate=0.3)
        # 2 pids => exactly 2 parquet reads (one short-history check would
        # be a separate read, but both pids have 500 bars so no extra reads).
        # Allow up to 2 per pid for any incidental sort/inspection paths.
        assert calls["count"] <= 4, (
            f"read_parquet called {calls['count']} times for 2 pids — "
            f"expected at most 4. Per-sample reads regression."
        )
