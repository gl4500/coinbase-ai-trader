# Marketcap Bronze Schema v2 — Add `volume_24h` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the bronze marketcap parquet schema with a `volume_24h` column populated from the existing CoinGecko + CoinPaprika API responses, then operator-backfill all 49 tracked products.

**Architecture:** Single atomic commit on `feat/gpu-coord-mirror`. Schema lives in `backend/tools/build_marketcap_parquet.py` (`_SCHEMA_VERSION`, `_SCHEMA`, `_save_marketcap_history`, `_load_marketcap_history`); both fetchers (`backend/services/coingecko_marketcap.py`, `coinpaprika_marketcap.py`) get their return shape extended from `(ts_ms, market_cap)` to `(ts_ms, market_cap, volume_24h)`; cache layer (`backend/services/marketcap_history_cache.py`) writes the new column and triggers full refetch when on-disk parquet is schema-v1. Zero extra API calls — `total_volumes` is already in the CoinGecko `/market_chart/range` response and `volume` is in the CoinPaprika `/ohlcv/historical` rows.

**Tech Stack:** Python 3.11, pyarrow (existing dep), httpx (existing dep), pytest + pytest-asyncio.

**Spec source:** `docs/superpowers/specs/2026-05-16-marketcap-bronze-volume-design.md`
**Branch:** `feat/gpu-coord-mirror` (continue)

---

## File map

| Path | Action | Diff scope |
|---|---|---|
| `backend/tools/build_marketcap_parquet.py` | EDIT — bump `_SCHEMA_VERSION` 1→2; add `volume_24h` field to `_SCHEMA`; extend `_save_marketcap_history` + `_load_marketcap_history` + `rows_from_history` to carry `volume_24h` | ~20 lines |
| `backend/services/coingecko_marketcap.py:fetch_marketcap_history` (lines 205-268) | EDIT — also parse `total_volumes` from response; return `List[Tuple[int, float, float]]` | ~25 lines |
| `backend/services/coinpaprika_marketcap.py:fetch_marketcap_history` (lines 112-end) | EDIT — also parse `volume` field per row; return `List[Tuple[int, float, float]]` | ~15 lines |
| `backend/services/marketcap_history_cache.py` | EDIT — handle `volume_24h` in merge dict; bump return tuple to `(ts_ms, mc, volume_24h)`; new `_schema_is_stale` check that triggers full refetch when cached rows have `schema_version < _SCHEMA_VERSION` | ~25 lines |
| `backend/tests/test_coingecko_marketcap.py` | EXTEND — `test_coingecko_parses_volume_24h` + `test_coingecko_handles_missing_total_volumes` | +60 LOC |
| `backend/tests/test_coinpaprika_marketcap.py` | EXTEND — `test_coinpaprika_parses_volume_24h` | +35 LOC |
| `backend/tests/test_marketcap_history_cache.py` | EXTEND — `test_cache_v1_parquet_triggers_full_refetch` | +50 LOC |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71i entry | new |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND — marketcap Step A entry | outside repo |

---

## Coordination with concurrent subagents

Two background subagents may be mid-flight on this branch:
- **Subagent 4b** owns `backend/agents/cnn_agent.py`, `backend/tests/test_cnn_agent.py`, `backend/tests/test_config.py`
- **Subagent 6** has staged work (probe scripts) but is blocked behind 4b

This Step A touches **only** `backend/services/`, `backend/tools/`, `backend/tests/test_coingecko_marketcap.py`, `backend/tests/test_coinpaprika_marketcap.py`, `backend/tests/test_marketcap_history_cache.py`, and `CHANGELOG.md`. No conflict with 4b/6's owned files.

**CHANGELOG.md is the only shared file.** Before staging CHANGELOG.md edits (Step 1.8 below), run:

```bash
git fetch origin
git log --oneline origin/feat/gpu-coord-mirror -3
```

If origin tip has moved past `866d3ce` (the Module 4a commit at plan-write time), pull --rebase before committing:

```bash
git pull --rebase origin feat/gpu-coord-mirror
```

If `CHANGELOG.md` merges cleanly, continue. If conflict, keep both entries (Step A's at top — newer-on-top convention).

---

## Task 1: Bronze schema v2 — single atomic commit

**Files:** all listed above.

### Step 1.1 — Write the 4 failing tests FIRST (TDD red)

Append to `backend/tests/test_coingecko_marketcap.py`:

```python
# ── volume_24h extraction (Step A: marketcap bronze v2) ──────────────────────


class TestVolume24hExtraction:
    @pytest.mark.asyncio
    async def test_coingecko_parses_volume_24h(self, monkeypatch):
        """Fetcher must extract total_volumes alongside market_caps.

        CoinGecko /coins/{id}/market_chart/range returns prices, market_caps,
        and total_volumes as parallel arrays. The old parser ignored
        total_volumes; the new contract returns (ts_ms, market_cap, volume_24h)
        tuples.
        """
        import services.coingecko_marketcap as cg

        fake_body = {
            "prices":        [[1746921600000, 100.0], [1747008000000, 101.0]],
            "market_caps":   [[1746921600000, 1.0e9], [1747008000000, 1.01e9]],
            "total_volumes": [[1746921600000, 5.0e7], [1747008000000, 6.0e7]],
        }

        class _FakeResp:
            status_code = 200
            def json(self): return fake_body

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return _FakeResp()

        monkeypatch.setattr(cg, "_coinbase_to_cg_id", lambda pid: "bitcoin")
        monkeypatch.setattr(cg.httpx, "AsyncClient", _FakeClient)

        rows = await cg.fetch_marketcap_history("BTC-USD", 1746921600000, 1747008000000)
        assert len(rows) == 2
        ts0, mc0, vol0 = rows[0]
        assert mc0 == 1.0e9
        assert vol0 == 5.0e7
        ts1, mc1, vol1 = rows[1]
        assert vol1 == 6.0e7

    @pytest.mark.asyncio
    async def test_coingecko_handles_missing_total_volumes(self, monkeypatch, caplog):
        """Missing/empty total_volumes → all rows get volume_24h=0.0 + warning."""
        import logging
        import services.coingecko_marketcap as cg

        fake_body = {
            "market_caps":   [[1746921600000, 1.0e9]],
            "total_volumes": [],
        }

        class _FakeResp:
            status_code = 200
            def json(self): return fake_body

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return _FakeResp()

        monkeypatch.setattr(cg, "_coinbase_to_cg_id", lambda pid: "bitcoin")
        monkeypatch.setattr(cg.httpx, "AsyncClient", _FakeClient)

        with caplog.at_level(logging.WARNING):
            rows = await cg.fetch_marketcap_history("BTC-USD", 0, 9999999999999)
        assert rows == [(1746921600000, 1.0e9, 0.0)]
        assert any(
            "total_volumes" in r.message.lower() or "volume_24h" in r.message.lower()
            for r in caplog.records
        )
```

Append to `backend/tests/test_coinpaprika_marketcap.py`:

```python
# ── volume_24h extraction (Step A: marketcap bronze v2) ──────────────────────


class TestVolume24hExtraction:
    @pytest.mark.asyncio
    async def test_coinpaprika_parses_volume_24h(self, monkeypatch):
        """Fetcher must map the `volume` field from each historical row
        to a `volume_24h` value in the returned tuple."""
        import services.coinpaprika_marketcap as cp

        fake_rows = [
            {"time_open": "2026-05-10T00:00:00Z", "market_cap": 1.0e9, "volume": 5.0e7},
            {"time_open": "2026-05-11T00:00:00Z", "market_cap": 1.01e9, "volume": 6.0e7},
        ]

        class _FakeResp:
            status_code = 200
            def json(self): return fake_rows

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, *a, **kw): return _FakeResp()

        monkeypatch.setattr(cp, "_coinbase_to_cp_id", lambda pid: "btc-bitcoin")
        monkeypatch.setattr(cp.httpx, "AsyncClient", _FakeClient)

        rows = await cp.fetch_marketcap_history("BTC-USD", 0, 9999999999999)
        assert len(rows) == 2
        # Returned shape: (ts_ms, market_cap, volume_24h)
        assert all(len(r) == 3 for r in rows)
        assert rows[0][1] == 1.0e9
        assert rows[0][2] == 5.0e7
        assert rows[1][2] == 6.0e7
```

Append to `backend/tests/test_marketcap_history_cache.py`:

```python
# ── schema v1 → v2 auto-upgrade (Step A: marketcap bronze v2) ────────────────


class TestSchemaV1AutoUpgrade:
    @pytest.mark.asyncio
    async def test_cache_v1_parquet_triggers_full_refetch(self, tmp_path, monkeypatch):
        """A parquet on disk with schema_version=1 must trigger a full refetch
        even if its ingest_ts is fresh — because v1 lacks the new volume_24h
        column required by downstream consumers."""
        import time
        from tools.build_marketcap_parquet import _save_marketcap_history, _SCHEMA_VERSION
        from services.marketcap_history_cache import fetch_marketcap_history_cached

        assert _SCHEMA_VERSION >= 2, (
            "Plan requires _SCHEMA_VERSION bumped to 2 before this test passes"
        )

        # Pre-seed parquet with v1 schema (force schema_version=1, omit volume_24h)
        path = tmp_path / "BTC-USD.parquet"
        now = int(time.time())
        # _save_marketcap_history with explicit schema_version=1 to simulate stale state
        _save_marketcap_history(
            str(path),
            [{
                "start": 1746921600,
                "market_cap": 1.0e9,
                "fdv": 1.0e9,
                "ingest_ts": now,            # fresh — would normally hit cache
                "schema_version": 1,         # but v1 → must refetch
            }],
            now_ts=now,
        )

        # Spy on the underlying fetcher; return new (ts, mc, vol) tuples
        calls = {"n": 0}
        async def fake_fetch(pid, start_ms, end_ms):
            calls["n"] += 1
            return [(1746921600000, 1.0e9, 5.0e7)]
        monkeypatch.setattr(
            "services.marketcap_history_cache.fetch_marketcap_history",
            fake_fetch,
        )

        out = await fetch_marketcap_history_cached(
            "BTC-USD", 1746921600000, 1747008000000, str(tmp_path)
        )
        assert calls["n"] == 1, "v1 parquet should trigger full refetch"

        # Verify parquet was rewritten at v2 with volume_24h
        from tools.build_marketcap_parquet import _load_marketcap_history
        rewritten = _load_marketcap_history(str(path))
        assert len(rewritten) >= 1
        assert rewritten[0]["schema_version"] == 2
        assert "volume_24h" in rewritten[0]
        assert rewritten[0]["volume_24h"] == 5.0e7
```

- [ ] **Step 1.1** — Append all four tests to the three test files.

### Step 1.2 — Run; expect 4 FAILED (collection or assertion errors)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_coingecko_marketcap.py::TestVolume24hExtraction tests/test_coinpaprika_marketcap.py::TestVolume24hExtraction tests/test_marketcap_history_cache.py::TestSchemaV1AutoUpgrade -v
```

Expected: 4 FAILED. The coingecko/coinpaprika tests unpack 3-tuples but the current fetchers return 2-tuples (`ValueError: not enough values to unpack`). The cache test asserts `_SCHEMA_VERSION >= 2` which is currently 1.

- [ ] **Step 1.2** — Run and observe red.

### Step 1.3 — Bump schema in `build_marketcap_parquet.py`

Edit `backend/tools/build_marketcap_parquet.py`.

Find at line 46-53:

```python
_SCHEMA_VERSION = 1
_SCHEMA = pa.schema([
    pa.field("start",          pa.int64()),
    pa.field("market_cap",     pa.float64()),
    pa.field("fdv",            pa.float64()),
    pa.field("ingest_ts",      pa.int64()),
    pa.field("schema_version", pa.int32()),
])
```

Replace with:

```python
_SCHEMA_VERSION = 2  # was 1; v2 adds volume_24h (Step A, 2026-05-16)
_SCHEMA = pa.schema([
    pa.field("start",          pa.int64()),
    pa.field("market_cap",     pa.float64()),
    pa.field("fdv",            pa.float64()),
    pa.field("volume_24h",     pa.float64()),
    pa.field("ingest_ts",      pa.int64()),
    pa.field("schema_version", pa.int32()),
])
```

Find `_load_marketcap_history` at line 61-82. Replace the row dict construction (lines 72-81):

```python
        r = {
            "start":      int(rows["start"][i]),
            "market_cap": float(rows["market_cap"][i]),
            "fdv":        float(rows["fdv"][i]),
        }
        if has_ingest and rows["ingest_ts"][i] is not None:
            r["ingest_ts"] = int(rows["ingest_ts"][i])
        if has_sv and rows["schema_version"][i] is not None:
            r["schema_version"] = int(rows["schema_version"][i])
        out.append(r)
```

With (add `has_vol` detection + conditional read so old v1 parquets without the column still load):

```python
        r = {
            "start":      int(rows["start"][i]),
            "market_cap": float(rows["market_cap"][i]),
            "fdv":        float(rows["fdv"][i]),
        }
        # volume_24h is v2-only — old v1 parquets lack the column entirely.
        if "volume_24h" in rows and rows["volume_24h"][i] is not None:
            r["volume_24h"] = float(rows["volume_24h"][i])
        if has_ingest and rows["ingest_ts"][i] is not None:
            r["ingest_ts"] = int(rows["ingest_ts"][i])
        if has_sv and rows["schema_version"][i] is not None:
            r["schema_version"] = int(rows["schema_version"][i])
        out.append(r)
```

Find `_save_marketcap_history` at line 85-129. In the inner merge (lines 103-117), insert volume_24h after `fdv`:

```python
        merged: Dict = {
            "start":      start,
            "market_cap": mc,
            "fdv":        fdv,
        }
```

→

```python
        merged: Dict = {
            "start":      start,
            "market_cap": mc,
            "fdv":        fdv,
            "volume_24h": float(r.get("volume_24h", 0.0)),
        }
```

And in the table construction (lines 119-128), add the column between `fdv` and `ingest_ts`:

```python
    table = pa.table(
        {
            "start":          [r["start"]      for r in ordered],
            "market_cap":     [r["market_cap"] for r in ordered],
            "fdv":            [r["fdv"]        for r in ordered],
            "volume_24h":     [float(r.get("volume_24h", 0.0)) for r in ordered],
            "ingest_ts":      [int(r.get("ingest_ts", now_ts)) for r in ordered],
            "schema_version": [int(r.get("schema_version", _SCHEMA_VERSION)) for r in ordered],
        },
        schema=_SCHEMA,
    )
```

Find `rows_from_history` at line 132. Read the signature; if it currently takes `history: Iterable[Tuple[int, float]]`, extend the signature to also accept volume:

```python
def rows_from_history(
    history: Iterable[Tuple[int, float, float]],
    fdv_history: Optional[Iterable[Tuple[int, float]]] = None,
) -> List[Dict]:
```

In the body, when building each row dict, include `volume_24h` from the third tuple element. Use `history[i][2]` instead of unpacking — or update the unpack to `for ts_ms, mc, vol in history`.

- [ ] **Step 1.3** — Apply all four edits to `build_marketcap_parquet.py`.

### Step 1.4 — Extend `coingecko_marketcap.py:fetch_marketcap_history`

Edit `backend/services/coingecko_marketcap.py` lines 253-268. Find:

```python
    raw = body.get("market_caps") if isinstance(body, dict) else None
    if not isinstance(raw, list):
        return []

    rows: List[Tuple[int, float]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            ts = int(entry[0])
            mc = float(entry[1])
        except (TypeError, ValueError):
            continue
        rows.append((ts, mc))
    rows.sort(key=lambda r: r[0])
    return rows
```

Replace with:

```python
    if not isinstance(body, dict):
        return []
    mc_raw  = body.get("market_caps")
    vol_raw = body.get("total_volumes")
    if not isinstance(mc_raw, list) or not mc_raw:
        return []

    if not isinstance(vol_raw, list) or not vol_raw:
        logger.warning(
            "coingecko_marketcap: total_volumes missing/empty pid=%s — volume_24h=0.0",
            product_id,
        )
        vol_raw = []

    # Index volume by timestamp for safe lookup (CG arrays usually align but
    # we don't depend on equal lengths).
    vol_by_ts: dict = {}
    for entry in vol_raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            vol_by_ts[int(entry[0])] = float(entry[1])
        except (TypeError, ValueError):
            continue

    rows: List[Tuple[int, float, float]] = []
    for entry in mc_raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            ts = int(entry[0])
            mc = float(entry[1])
        except (TypeError, ValueError):
            continue
        vol = vol_by_ts.get(ts, 0.0)
        rows.append((ts, mc, vol))
    rows.sort(key=lambda r: r[0])
    return rows
```

Also update the function's return-type annotation at the `def fetch_marketcap_history(...) -> List[Tuple[int, float]]:` line — find the signature and change `List[Tuple[int, float]]` → `List[Tuple[int, float, float]]`.

- [ ] **Step 1.4** — Apply the edit.

### Step 1.5 — Extend `coinpaprika_marketcap.py:fetch_marketcap_history`

Read `backend/services/coinpaprika_marketcap.py` lines 112 onward to find the body's row-extraction loop. It currently builds `(ts, mc)` tuples. Edit so that for each historical row dict from the response (typically with keys `time_open`/`time_close`/`market_cap`/`volume`/`price`):

- Extract `volume` field with `.get("volume", 0.0)` (default 0.0 on missing).
- Append `(ts_ms, market_cap, volume_24h)` tuple.

Update the function's return-type annotation: `List[Tuple[int, float]]` → `List[Tuple[int, float, float]]`.

Concretely, find the row append (likely something like `rows.append((ts_ms, mc))`) and change to `rows.append((ts_ms, mc, float(row.get("volume", 0.0) or 0.0)))`. Wrap with try/except to keep the existing skip-on-bad-row behavior.

If the CoinPaprika response shape is harder to read than this brief suggests, open the file with the Read tool first and inspect the full row-parse loop before editing.

- [ ] **Step 1.5** — Apply the edit.

### Step 1.6 — Extend `marketcap_history_cache.py` for v2 + schema-check

Edit `backend/services/marketcap_history_cache.py`. Three changes:

**Change A** — add a schema staleness check above the existing freshness checks. Insert after `_covers_range` (around line 51):

```python
def _schema_is_stale(rows: List[dict]) -> bool:
    """Returns True if any cached row has schema_version < _SCHEMA_VERSION.

    Triggers full refetch even if ingest_ts is fresh — v1 parquets lack the
    volume_24h column required by Step A's downstream consumers.
    """
    if not rows:
        return True
    return any(int(r.get("schema_version", 0)) < _SCHEMA_VERSION for r in rows)
```

**Change B** — `use_cache` gating (lines 71-75). Find:

```python
    use_cache = (
        cached
        and _is_fresh(cached, now_ts, refresh_secs)
        and _covers_range(cached, end_ms)
    )
```

Replace with:

```python
    use_cache = (
        cached
        and not _schema_is_stale(cached)
        and _is_fresh(cached, now_ts, refresh_secs)
        and _covers_range(cached, end_ms)
    )
```

**Change C** — merge dict + return tuple now carry `volume_24h`. The merge block (lines 79-96) builds dict rows for the parquet. Update:

```python
        for r in cached:
            merged_by_start[int(r["start"])] = {
                "start": int(r["start"]),
                "market_cap": float(r["market_cap"]),
                "fdv": float(r.get("fdv", r["market_cap"])),
                "ingest_ts": int(r.get("ingest_ts", now_ts)),
                "schema_version": int(r.get("schema_version", _SCHEMA_VERSION)),
            }
        for ts_ms, mc in fresh:
            start = (int(ts_ms) // 1000 // _BAR_SECS) * _BAR_SECS
            merged_by_start[start] = {
                "start": start,
                "market_cap": float(mc),
                "fdv": float(mc),
                "ingest_ts": now_ts,
                "schema_version": _SCHEMA_VERSION,
            }
```

Replace with (carry volume_24h on both paths, unpack 3-tuple from `fresh`):

```python
        for r in cached:
            merged_by_start[int(r["start"])] = {
                "start": int(r["start"]),
                "market_cap": float(r["market_cap"]),
                "fdv": float(r.get("fdv", r["market_cap"])),
                "volume_24h": float(r.get("volume_24h", 0.0)),
                "ingest_ts": int(r.get("ingest_ts", now_ts)),
                "schema_version": int(r.get("schema_version", _SCHEMA_VERSION)),
            }
        for ts_ms, mc, vol in fresh:
            start = (int(ts_ms) // 1000 // _BAR_SECS) * _BAR_SECS
            merged_by_start[start] = {
                "start": start,
                "market_cap": float(mc),
                "fdv": float(mc),
                "volume_24h": float(vol),
                "ingest_ts": now_ts,
                "schema_version": _SCHEMA_VERSION,
            }
```

The return loop (lines 103-108) still returns `(ts_ms, market_cap)` 2-tuples. Update to 3-tuples:

```python
    out: List[Tuple[int, float, float]] = []
    for r in cached:
        ts_ms = int(r["start"]) * 1000
        if start_ms <= ts_ms <= end_ms:
            out.append((ts_ms, float(r["market_cap"]), float(r.get("volume_24h", 0.0))))
    out.sort(key=lambda x: x[0])
    return out
```

Update the function's return type annotation: `List[Tuple[int, float]]` → `List[Tuple[int, float, float]]`.

- [ ] **Step 1.6** — Apply all three changes.

### Step 1.7 — Run; expect 4 GREEN + no regressions in the 3 test files

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_coingecko_marketcap.py tests/test_coinpaprika_marketcap.py tests/test_marketcap_history_cache.py -v
```

Expected: all tests pass. New: `TestVolume24hExtraction` (2 tests in coingecko, 1 in coinpaprika), `TestSchemaV1AutoUpgrade` (1 test). Existing tests must still pass — if any old test asserted on a 2-tuple shape, update it to 3-tuple (probably 0-2 such tests).

If existing tests fail with `ValueError: not enough values to unpack (expected 2, got 3)`, find them and update the unpack to `(ts, mc, vol)`.

- [ ] **Step 1.7** — Run and observe green.

### Step 1.8 — COORDINATION CHECKPOINT before staging CHANGELOG

```bash
git fetch origin
git log --oneline origin/feat/gpu-coord-mirror -3
```

If origin tip has moved past `866d3ce`:

```bash
git pull --rebase origin feat/gpu-coord-mirror
```

If `CHANGELOG.md` conflict, resolve by keeping BOTH entries (yours at top, then theirs). If you can't resolve cleanly, abort (`git rebase --abort`) and report — don't force the issue.

- [ ] **Step 1.8** — Sync with origin.

### Step 1.9 — Append CHANGELOG entry

Append at TOP of `CHANGELOG.md` (above current top entry):

```markdown
## [Session 58.71i] — 2026-05-16 — Marketcap bronze schema v2: volume_24h (#marketcap-A)

### Why
The XGB v3 feature extractor currently has no marketcap-related channels.
Step A of a 3-step buildout (Step A: bronze schema; Step B: channel wiring;
Step C: retrain) extends the existing parquet-backed marketcap cache to
include 24h trading volume — which CoinGecko `/coins/{id}/market_chart/range`
and CoinPaprika `/coins/{cp_id}/ohlcv/historical` already return in the
same responses the current parsers ignore.

### What changed
- **`backend/tools/build_marketcap_parquet.py`** — bumped `_SCHEMA_VERSION`
  1→2, added `volume_24h` field to `_SCHEMA`, extended
  `_save_marketcap_history` / `_load_marketcap_history` / `rows_from_history`
  to carry the new column. v1 parquets still load (missing column → key omitted).
- **`backend/services/coingecko_marketcap.py:fetch_marketcap_history`** —
  parses `total_volumes` alongside `market_caps`, returns
  `List[Tuple[int, float, float]]` (ts_ms, market_cap, volume_24h). Missing
  total_volumes → logs warning + fills 0.0.
- **`backend/services/coinpaprika_marketcap.py:fetch_marketcap_history`** —
  parses `volume` field per row, same return shape.
- **`backend/services/marketcap_history_cache.py`** — new
  `_schema_is_stale()` check forces full refetch when on-disk parquet is v1
  (auto-upgrade); merge logic carries `volume_24h`; return tuple shape
  bumped to `(ts_ms, mc, volume_24h)`.
- **`backend/tests/test_coingecko_marketcap.py`** — +2 tests
  (`test_coingecko_parses_volume_24h`,
  `test_coingecko_handles_missing_total_volumes`).
- **`backend/tests/test_coinpaprika_marketcap.py`** — +1 test
  (`test_coinpaprika_parses_volume_24h`).
- **`backend/tests/test_marketcap_history_cache.py`** — +1 test
  (`test_cache_v1_parquet_triggers_full_refetch`).

### Verification
```
backend && python -m pytest tests/test_coingecko_marketcap.py \
                            tests/test_coinpaprika_marketcap.py \
                            tests/test_marketcap_history_cache.py -v
=> all passed (4 new + existing)
```

Zero extra API calls per pid (volume was already in the response). Bronze
parquets upgrade lazily on next cache hit per pid; operator preflight below
forces a one-shot full upgrade across all 49 tracked products.

### Operator preflight (run once after this commit)

```bash
cd backend
../.venv/Scripts/python.exe -m tools.build_marketcap_parquet --all-tracked
```

CoinGecko free tier rate-limits ~30 req/min; 49 calls ≈ 100 sec wall time.
After completion: all 49 tracked pids have parquet at schema_version=2 with
`volume_24h` populated.

### Step B preview
Wire `volume_24h` into `tools/xgb_features.py` v3 extractor as new
channel(s). Bump `N_CHANNELS = 28 → 30+`. Retrain booster (Step C).
Will be its own brainstorm cycle.

---
```

- [ ] **Step 1.9** — Append the CHANGELOG entry.

### Step 1.10 — Update memory file

Append above the current top entry in `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`:

```markdown
- **Session 58.71i (2026-05-16)**: Marketcap bronze schema v2 — `volume_24h` (#marketcap-A). Extended `backend/tools/build_marketcap_parquet.py` (`_SCHEMA_VERSION` 1→2, added `volume_24h` field), both fetchers (`coingecko_marketcap.py`, `coinpaprika_marketcap.py`) to parse volume from existing endpoints (zero extra API calls), and the cache layer (`marketcap_history_cache.py`) for auto-upgrade (`_schema_is_stale` triggers full refetch when v1 parquet on disk). Return tuple shape: `(ts_ms, market_cap)` → `(ts_ms, market_cap, volume_24h)`. 4 new tests. Operator preflight (`tools.build_marketcap_parquet --all-tracked`) backfills all 49 tracked pids at v2 (~100 sec wall time). Step A of 3-step marketcap channel buildout (A=bronze schema, B=channel wiring in `xgb_features.py`, C=v3 retrain on N_CHANNELS bump).
```

- [ ] **Step 1.10** — Apply.

### Step 1.11 — Port-aware cleanup (per CLAUDE.md)

```powershell
$backendPid = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $backendPid -and $_.ProcessName -ne 'Coinbase AI Trader' } |
    Stop-Process -Force
```

**Do NOT use blanket `Stop-Process python -Force`** — that kills the live backend on port 8001.

- [ ] **Step 1.11** — Run cleanup.

### Step 1.12 — Stage + commit

```bash
cd C:\Users\gl450\polymarket_app
git add backend/tools/build_marketcap_parquet.py \
        backend/services/coingecko_marketcap.py \
        backend/services/coinpaprika_marketcap.py \
        backend/services/marketcap_history_cache.py \
        backend/tests/test_coingecko_marketcap.py \
        backend/tests/test_coinpaprika_marketcap.py \
        backend/tests/test_marketcap_history_cache.py \
        CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(marketcap-A): bronze schema v2 — add volume_24h

Step A of marketcap channel buildout. Extends bronze parquet schema with
volume_24h column populated from existing API responses (zero extra API
calls — total_volumes is already in CoinGecko /market_chart/range, and
volume is in CoinPaprika /ohlcv/historical).

build_marketcap_parquet:
- _SCHEMA_VERSION 1 -> 2.
- _SCHEMA gains volume_24h pa.float64 field.
- _save_marketcap_history + _load_marketcap_history + rows_from_history
  carry the new column (v1 parquets without it still load).

coingecko_marketcap.fetch_marketcap_history:
- Parses total_volumes alongside market_caps.
- Returns (ts_ms, market_cap, volume_24h) tuples.
- Missing/empty total_volumes -> volume_24h=0.0 + warning.

coinpaprika_marketcap.fetch_marketcap_history:
- Parses volume per row, same tuple shape.

marketcap_history_cache:
- New _schema_is_stale() check: parquet at schema_version<2 triggers full
  refetch on next access, auto-upgrading the file.
- Merge logic + return tuple carry volume_24h.

4 new tests (2 coingecko, 1 coinpaprika, 1 cache auto-upgrade).

Operator preflight (run once after this commit) backfills all 49 tracked
products at v2 in ~100 sec:
  cd backend && python -m tools.build_marketcap_parquet --all-tracked

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Pre-commit hook runs the full ~1100-test suite (~5 min). All must pass — the 4 new tests were made green at Step 1.7; the existing tests don't depend on the old 2-tuple shape (verified at Step 1.7).

- [ ] **Step 1.12** — Commit; wait for pre-commit hook to finish.

### Step 1.13 — Push

```bash
cd C:\Users\gl450\polymarket_app
git push
```

- [ ] **Step 1.13** — Push.

### Step 1.14 — Operator preflight backfill

```bash
cd backend
../.venv/Scripts/python.exe -m tools.build_marketcap_parquet --all-tracked
```

If the `--all-tracked` flag doesn't exist on the current backfill tool, run it per-pid in a loop:

```bash
.venv/Scripts/python.exe -c "
import sqlite3
c = sqlite3.connect('backend/coinbase.db')
pids = [r[0] for r in c.execute('SELECT product_id FROM products WHERE is_tracked=1')]
print(' '.join(pids))
" | tr ' ' '\n' | while read pid; do
    .venv/Scripts/python.exe -m backend.tools.build_marketcap_parquet --pid "$pid"
    sleep 2  # respect CG rate limit
done
```

Expected: 49 parquets at `backend/data/marketcap/<pid>.parquet`, all at schema_version=2, all with `volume_24h` column populated.

- [ ] **Step 1.14** — Run preflight. Verify with:

```bash
.venv/Scripts/python.exe -c "
import os, pandas as pd
pdir = 'backend/data/marketcap'
files = sorted(os.listdir(pdir))
print(f'parquet files: {len(files)}')
for f in files[:3]:
    df = pd.read_parquet(os.path.join(pdir, f))
    print(f'  {f}: rows={len(df)} schema_v={int(df[\"schema_version\"].iloc[0])} cols={df.columns.tolist()}')
"
```

Expected: ≥45 parquets (49 minus any that legitimately have no CoinGecko mapping), schema_v=2, `volume_24h` in cols.

### Step 1.15 — Verify backend still up

```bash
curl -sS -m 3 http://localhost:8001/api/status
```

Should return 200 with `is_trading:true`. The marketcap work is data-only; the live scan loop is unaffected.

- [ ] **Step 1.15** — Verify.

---

## Spec coverage check

| Spec section | Task step |
|---|---|
| 4.1 Files touched — build_marketcap_parquet.py | Step 1.3 |
| 4.1 Files touched — coingecko_marketcap.py | Step 1.4 |
| 4.1 Files touched — coinpaprika_marketcap.py | Step 1.5 |
| 4.1 Files touched — marketcap_history_cache.py | Step 1.6 |
| 4.1 Files touched — 3 test files | Step 1.1 |
| 4.1 Files touched — CHANGELOG | Step 1.9 |
| 4.1 Files touched — memory | Step 1.10 |
| 4.2 Code changes (CoinGecko parser w/ vol_by_ts indexing) | Step 1.4 |
| 4.2 Code changes (CoinPaprika parser) | Step 1.5 |
| 4.2 Code changes (cache merge + return) | Step 1.6 |
| 4.2 Code changes (build_marketcap_parquet schema bump) | Step 1.3 |
| 4.3 New tests (4 total) | Steps 1.1 (write) + 1.2 (red) + 1.7 (green) |
| 5 Architecture diagram | implied by code changes |
| 6 Data flow | Step 1.4 + 1.6 |
| 7 Error handling (8 conditions) | covered by tests + try/except patterns in Step 1.4-1.5 |
| 8 Tests count (+4 new) | Step 1.1 |
| 9 Rollout — atomic commit | Step 1.12 |
| 9 Rollout — push | Step 1.13 |
| 9 Rollout — operator preflight | Step 1.14 |
| 9 Rollout — verification | Step 1.15 |
| 10 Memory + CLAUDE.md sync (CHANGELOG only — no invariant) | Steps 1.9 + 1.10 |

All spec sections have a corresponding task step. CLAUDE.md not edited — spec section 10 explicitly says "no invariant change."

---

## Plan complete

Saved to `docs/superpowers/plans/2026-05-16-marketcap-bronze-volume.md`. **1 task, 15 micro-steps, 1 atomic commit + 1 operator preflight, +4 tests, zero extra API calls.**

Same shape as Module 1/2/4a: tight scope, single commit. Plus a coordination checkpoint (Step 1.8) to handle the parallel subagents (4b on `cnn_agent.py`, 6 blocked) potentially landing CHANGELOG entries.
