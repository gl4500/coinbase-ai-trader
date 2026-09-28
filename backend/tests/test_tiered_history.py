"""TDD tests for services/tiered_history.py — XGB feature_set v3 data layer.

Contract:
    fetch_tiered(pid, source, now_ts) -> {"micro": List[Candle],
                                           "meso":  List[Candle],
                                           "macro": List[Candle]}

micro = last 60 hourly bars
meso  = last 168 hourly bars
macro = last 336 hourly bars

Short-history return: any tier whose underlying series is shorter than its
required length is returned as []. Caller (_extract_v3) interprets [] as
"fill that tier's slots with 0.0".
"""

import os
import sqlite3
import sys

import pandas as pd
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _candle(start_ts, close=100.0):
    return {
        "start": start_ts,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1000.0,
    }


@pytest.fixture
def parquet_dir(tmp_path):
    d = tmp_path / "history"
    d.mkdir()
    return d


def _write_parquet(parquet_dir, pid, n_bars, start_ts=1_700_000_000):
    rows = [_candle(start_ts + i * 3600, close=100.0 + i * 0.1) for i in range(n_bars)]
    df = pd.DataFrame(rows)
    df["ingest_ts"] = 1_700_000_000
    df["schema_version"] = 1
    df.to_parquet(parquet_dir / f"{pid}.parquet")


@pytest.fixture
def sqlite_db(tmp_path):
    path = tmp_path / "coinbase.db"
    c = sqlite3.connect(path)
    c.execute("""
        CREATE TABLE candles (
            id INTEGER PRIMARY KEY, product_id TEXT, start REAL,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        )""")
    c.commit()
    c.close()
    return path


def _seed_sqlite(db_path, pid, n_bars, start_ts=1_700_000_000):
    c = sqlite3.connect(db_path)
    for i in range(n_bars):
        ts = start_ts + i * 3600
        c.execute(
            "INSERT INTO candles (product_id, start, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, ts, 100.0, 101.0, 99.0, 100.0 + i * 0.1, 1000.0),
        )
    c.commit()
    c.close()


# ──────────────────────────────────────────────────────────────────────
class TestSliceContracts:
    def test_returns_three_keys(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert set(result.keys()) == {"micro", "meso", "macro"}

    def test_micro_returns_last_60_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["micro"]) == 60

    def test_meso_returns_last_168_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["meso"]) == 168

    def test_macro_returns_last_336_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["macro"]) == 336

    def test_chronological_order_ascending(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        for tier in ("micro", "meso", "macro"):
            starts = [c["start"] for c in result[tier]]
            assert starts == sorted(starts), f"{tier} not sorted ascending"


class TestShortHistory:
    def test_short_history_returns_empty_list_for_macro(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "NEW-USD", 200)  # < 336
        result = fetch_tiered("NEW-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert len(result["meso"]) == 168
        assert len(result["micro"]) == 60

    def test_short_history_meso_empty_macro_empty(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "TINY-USD", 100)  # < 168
        result = fetch_tiered("TINY-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert result["meso"] == []
        assert len(result["micro"]) == 60

    def test_only_micro_history(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "FRESH-USD", 70)
        result = fetch_tiered("FRESH-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert result["meso"] == []
        assert len(result["micro"]) == 60

    def test_parquet_missing_returns_all_empty(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        result = fetch_tiered("MISSING-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result == {"micro": [], "meso": [], "macro": []}


class TestSourceDispatch:
    def test_source_live_reads_sqlite_first(self, sqlite_db):
        from services.tiered_history import fetch_tiered

        _seed_sqlite(sqlite_db, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="live", db_path=str(sqlite_db))
        assert len(result["macro"]) == 336

    def test_source_live_falls_back_to_parquet_for_prefix(self, sqlite_db, parquet_dir):
        from services.tiered_history import fetch_tiered

        _seed_sqlite(sqlite_db, "BTC-USD", 100)  # SQLite has 100, < 336
        _write_parquet(parquet_dir, "BTC-USD", 400)  # parquet has 400
        result = fetch_tiered(
            "BTC-USD",
            source="live",
            db_path=str(sqlite_db),
            parquet_dir=str(parquet_dir),
        )
        assert len(result["macro"]) == 336

    def test_unknown_source_raises(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        with pytest.raises(ValueError, match="unknown source"):
            fetch_tiered("BTC-USD", source="bogus", parquet_dir=str(parquet_dir))

    def test_source_live_reads_prod_schema_with_start_time_column(self, tmp_path):
        """Regression: production candles table uses 'start_time', not 'start'.
        Caught during cutover smoke when xgb_signal v3 path raised
        'no such column: start' on the live DB."""
        from services.tiered_history import fetch_tiered

        prod_db = tmp_path / "prod.db"
        c = sqlite3.connect(prod_db)
        c.execute("""
            CREATE TABLE candles (
                id INTEGER PRIMARY KEY, product_id TEXT, start_time INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL
            )""")
        for i in range(400):
            c.execute(
                "INSERT INTO candles (product_id, start_time, open, high, low, close, volume)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", 1_700_000_000 + i * 3600, 100.0, 101.0, 99.0, 100.0 + i * 0.1, 1000.0),
            )
        c.commit()
        c.close()
        result = fetch_tiered("BTC-USD", source="live", db_path=str(prod_db))
        assert len(result["macro"]) == 336
        # Aliased: returned dict still uses 'start' key for parity with parquet
        assert "start" in result["macro"][0]


class TestNowTsFilter:
    def test_now_ts_excludes_future_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered

        _write_parquet(parquet_dir, "BTC-USD", 400, start_ts=1_700_000_000)
        cutoff = 1_700_000_000 + 100 * 3600
        result = fetch_tiered(
            "BTC-USD",
            source="parquet",
            parquet_dir=str(parquet_dir),
            now_ts=cutoff,
        )
        for tier in ("micro", "meso", "macro"):
            for c in result[tier]:
                assert c["start"] < cutoff, f"{tier} contains future bar"


# ---------------------------------------------------------------------------------------
# Closed-bar replay safety (2026-09-28)
#
# `_read_parquet`/`_read_sqlite` filter `start < now_ts` -- on the bar's START only. A bar
# that began before `now_ts` but had not yet CLOSED is therefore included.
#
# Live that is correct and harmless: the store only holds data up to now, so the newest bar
# is genuinely partial. In REPLAY from a completed parquet that same bar carries its final
# high/low/close/volume, so up to 59 minutes of future information enters the features -- at
# a 15-minute scan cadence. Every retrospective re-score built this way is invalid.
#
# `closed_only=True` makes the as-of semantics explicit. It is OPT-IN: the default is
# unchanged, so no live path moves. Flipping the default would change what the replay path
# computes and is a separate, operator-owned decision.

_BAR = 3600.0


def _write_hourly_parquet(tmp_path, pid, n=400, first_start=0.0):
    import pandas as pd

    rows = [
        {
            "start": first_start + i * _BAR,
            "open": 10.0 + i,
            "high": 11.0 + i,
            "low": 9.0 + i,
            "close": 10.5 + i,
            "volume": 100.0 + i,
        }
        for i in range(n)
    ]
    d = tmp_path / "hist"
    d.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f"{pid}.parquet")
    return str(d)


def _write_hourly_sqlite(tmp_path, pid, n=400, first_start=0.0):
    import sqlite3

    p = tmp_path / "candles.db"
    c = sqlite3.connect(str(p))
    c.execute(
        "CREATE TABLE IF NOT EXISTS candles (product_id TEXT, start_time REAL, "
        "open REAL, high REAL, low REAL, close REAL, volume REAL)"
    )
    c.executemany(
        "INSERT INTO candles VALUES (?,?,?,?,?,?,?)",
        [
            (pid, first_start + i * _BAR, 10.0 + i, 11.0 + i, 9.0 + i, 10.5 + i, 100.0 + i)
            for i in range(n)
        ],
    )
    c.commit()
    c.close()
    return str(p)


def test_default_still_admits_the_unclosed_bar(tmp_path):
    """Characterisation of the defect, so the behaviour is pinned rather than assumed.
    now_ts lands 10 minutes into the bar starting at 399*3600; that bar is still returned."""
    from services.tiered_history import fetch_tiered

    pdir = _write_hourly_parquet(tmp_path, "AAA-USD")
    forming_start = 399 * _BAR
    now = forming_start + 600.0
    tiers = fetch_tiered("AAA-USD", source="parquet", now_ts=now, parquet_dir=pdir)
    assert tiers["micro"][-1]["start"] == forming_start


def test_closed_only_excludes_the_bar_that_had_not_closed(tmp_path):
    from services.tiered_history import fetch_tiered

    pdir = _write_hourly_parquet(tmp_path, "AAA-USD")
    forming_start = 399 * _BAR
    now = forming_start + 600.0
    tiers = fetch_tiered(
        "AAA-USD", source="parquet", now_ts=now, parquet_dir=pdir, closed_only=True
    )
    assert tiers["micro"][-1]["start"] == forming_start - _BAR


def test_a_bar_closing_exactly_at_now_is_still_usable(tmp_path):
    """Boundary: a bar that ended precisely at the decision instant WAS fully observed.
    Excluding it would discard real information and quietly shorten every window."""
    from services.tiered_history import fetch_tiered

    pdir = _write_hourly_parquet(tmp_path, "AAA-USD")
    last_closed = 398 * _BAR
    now = last_closed + _BAR
    tiers = fetch_tiered(
        "AAA-USD", source="parquet", now_ts=now, parquet_dir=pdir, closed_only=True
    )
    assert tiers["micro"][-1]["start"] == last_closed


def test_the_default_is_byte_identical_to_passing_false(tmp_path):
    """Non-vacuity for the opt-in claim: no live path can move because the default changed."""
    from services.tiered_history import fetch_tiered

    pdir = _write_hourly_parquet(tmp_path, "AAA-USD")
    now = 399 * _BAR + 600.0
    implicit = fetch_tiered("AAA-USD", source="parquet", now_ts=now, parquet_dir=pdir)
    explicit = fetch_tiered(
        "AAA-USD", source="parquet", now_ts=now, parquet_dir=pdir, closed_only=False
    )
    assert implicit == explicit


def test_closed_only_applies_to_the_sqlite_source_too(tmp_path):
    """The live source reads SQLite; a fix that only covered parquet would leave the leak
    reachable through the other reader."""
    from services.tiered_history import fetch_tiered

    db = _write_hourly_sqlite(tmp_path, "AAA-USD")
    pdir = _write_hourly_parquet(tmp_path, "BBB-USD")  # unrelated, keeps the fallback empty
    forming_start = 399 * _BAR
    now = forming_start + 600.0
    leaky = fetch_tiered("AAA-USD", source="live", now_ts=now, db_path=db, parquet_dir=pdir)
    safe = fetch_tiered(
        "AAA-USD", source="live", now_ts=now, db_path=db, parquet_dir=pdir, closed_only=True
    )
    assert leaky["micro"][-1]["start"] == forming_start
    assert safe["micro"][-1]["start"] == forming_start - _BAR


def test_closed_only_without_an_as_of_instant_is_refused(tmp_path):
    """With no now_ts there is no decision instant, so "closed" has no referent. Silently
    doing nothing would hand the caller a false guarantee."""
    from services.tiered_history import fetch_tiered

    pdir = _write_hourly_parquet(tmp_path, "AAA-USD")
    with pytest.raises(ValueError):
        fetch_tiered("AAA-USD", source="parquet", now_ts=None, parquet_dir=pdir, closed_only=True)
