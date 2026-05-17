# Marketcap Bronze Schema v2 — Add `volume_24h`

**Date:** 2026-05-16
**Status:** APPROVED 2026-05-16 (operator: "approved" × 2)
**Scope:** `backend/services/` + `backend/tools/` + `backend/tests/` + `backend/data/marketcap/` (host-side parquets)
**Branch:** continue on `feat/gpu-coord-mirror`
**Position:** Step A of a 3-step marketcap channel buildout
- **Step A (this spec):** extend bronze schema with `volume_24h`, backfill all 49 tracked pids
- Step A.2 (future): supply data (`circulating_supply`, `total_supply`) once snapshot-vs-historical decision is revisited
- Step B (future): wire marketcap data into `tools/xgb_features.py` feature extractor; bump `N_CHANNELS = 28 → 30+`
- Step C (future): retrain v3 booster on extended feature space

**Predecessors:** v3 cutover (#311-cut), MC CIFilter live (#311-mc-sync), Session 58.68 marketcap bronze cache (#284/#285)

---

## 1. Problem

Three marketcap channels would help the XGB v3 booster differentiate small-cap chop from large-cap signal. The bronze data pipeline (`services/coingecko_marketcap.py`, `services/coinpaprika_marketcap.py`, `services/marketcap_history_cache.py`) exists but its schema is incomplete:

```
current bronze schema v1:
  start (int unix), market_cap (float USD), fdv (float USD),
  ingest_ts (int unix), schema_version (int = 1)
```

Two gaps:
1. **No `volume_24h` column.** The CoinGecko `/coins/{id}/market_chart/range` endpoint returns `prices`, `market_caps`, AND `total_volumes` in the same response — but the current parser only reads `market_caps`. CoinPaprika `/coins/{id}/ohlcv/historical` returns a `volume` field per row that's also ignored. Adding the column is **free in API cost** (existing call returns the data).
2. **Coverage: only 3 of 49 tracked products have parquet files** (BTC, ETH, SOL). 46 missing.

This Step A spec only addresses `volume_24h` + the backfill. Supply data (`circulating_supply`, `total_supply`) deferred to Step A.2 because supply requires a separate `/coins/{id}` endpoint, is current-snapshot-only (no historical supply API), and the snapshot-vs-historical semantics need their own design pass.

## 2. Goal

After this spec ships:
- Bronze parquet schema is **v2**: adds `volume_24h` per row.
- All **49 tracked products** have a parquet file at `backend/data/marketcap/<pid>.parquet`.
- Both CoinGecko and CoinPaprika fetchers parse and return `volume_24h`.
- Cache layer detects schema mismatch (v1 on disk vs v2 expected) and triggers full refetch automatically.
- Tests lock in: missing/null `total_volumes` produces `volume_24h = 0.0` with warning logged.

Step B (channel wiring) is then unblocked across all 49 products, not just 3.

## 3. Non-goals

- No supply columns (circulating, total, max) — deferred to Step A.2.
- No changes to the daily granularity (still 1 row per UTC day).
- No changes to `tools/xgb_features.py` or `agents/cnn_agent.py` — those are Step B.
- No new dependencies (CoinGecko + CoinPaprika clients already present).
- No retraining of the v3 booster — that requires Step B + C.
- No removal or migration of historical bronze data already on disk (the 3 v1 parquets get rewritten in-place when first re-fetched).

## 4. Approach

Single atomic commit on `feat/gpu-coord-mirror`. Extend both fetchers to parse the volume field from their respective endpoints. Bump `schema_version` constant 1→2 in the cache. Add cache logic: parquet whose `schema_version < 2` triggers full refetch on next access. 4 new tests. Followed by an operator preflight that runs the backfill across all 49 tracked products (~100 sec wall time at CoinGecko's free-tier rate limit).

### 4.1 Files touched

| Path | Action | Diff scope |
|---|---|---|
| `backend/services/coingecko_marketcap.py:fetch_marketcap_history` | EDIT — also parse `total_volumes`, return rows with `volume_24h` field | ~20 lines added |
| `backend/services/coinpaprika_marketcap.py:fetch_marketcap_history` | EDIT — parse `volume` from each row, map to `volume_24h` | ~15 lines added |
| `backend/services/marketcap_history_cache.py` | EDIT — bump `_CURRENT_SCHEMA_VERSION` constant to 2; detect v1 parquet on disk + trigger full refetch; write `volume_24h` to parquet | ~10 lines |
| `backend/tools/build_marketcap_parquet.py` | EDIT — pass through `volume_24h` to parquet writer | ~5 lines |
| `backend/tests/test_coingecko_marketcap.py` | EXTEND — `test_coingecko_parses_volume_24h`, `test_coingecko_handles_missing_total_volumes` | +2 tests |
| `backend/tests/test_coinpaprika_marketcap.py` | EXTEND — `test_coinpaprika_parses_volume_24h` | +1 test |
| `backend/tests/test_marketcap_history_cache.py` | EXTEND — `test_cache_v1_parquet_triggers_full_refetch` | +1 test |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71i entry | new |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND — marketcap Step A entry | outside repo |

### 4.2 Code changes

**`coingecko_marketcap.py:fetch_marketcap_history` (current behavior)**:

```python
body = await response.json()
raw = body.get("market_caps") if isinstance(body, dict) else None
if not raw:
    return []
return [(int(ts), float(mc)) for ts, mc in raw]
```

**After edit**:

```python
body = await response.json()
if not isinstance(body, dict):
    return []
mc_raw  = body.get("market_caps")  or []
vol_raw = body.get("total_volumes") or []
if not mc_raw:
    return []
# Zip by index — CoinGecko returns matching timestamps. If lengths differ, use min.
n = min(len(mc_raw), len(vol_raw)) if vol_raw else 0
if not vol_raw or n == 0:
    logger.warning("CoinGecko %s: total_volumes missing/empty — volume_24h=0", pid)
    return [(int(ts), float(mc), 0.0) for ts, mc in mc_raw]
if len(mc_raw) != len(vol_raw):
    logger.warning("CoinGecko %s: mc_raw len=%d vs vol_raw len=%d — using min",
                   pid, len(mc_raw), len(vol_raw))
return [
    (int(mc_raw[i][0]), float(mc_raw[i][1]), float(vol_raw[i][1]))
    for i in range(n)
]
```

Return type changes from `List[Tuple[int, float]]` to `List[Tuple[int, float, float]]`.

**`coinpaprika_marketcap.py`** (similar pattern): each row dict gains a `volume_24h` extraction with default `0.0` on missing key.

**`marketcap_history_cache.py`**:

```python
_CURRENT_SCHEMA_VERSION = 2  # was 1

def _cached_parquet_is_stale(df: pd.DataFrame) -> bool:
    """Returns True if the parquet on disk needs full refetch.

    Triggers:
      - schema_version < current (NEW): v1 parquets lack volume_24h
      - ingest_ts older than refresh_secs (existing)
    """
    if df.empty:
        return True
    file_version = int(df["schema_version"].iloc[0])
    if file_version < _CURRENT_SCHEMA_VERSION:
        return True
    # ... existing TTL check
```

**`build_marketcap_parquet.py`** — `_write_parquet` gets `volume_24h` column added:

```python
df = pd.DataFrame(rows, columns=["start", "market_cap", "volume_24h"])
df["fdv"] = df["market_cap"]  # CoinGecko doesn't return FDV in this endpoint —
                              # existing fallback (use market_cap as proxy)
df["ingest_ts"] = int(time.time())
df["schema_version"] = _CURRENT_SCHEMA_VERSION
df.to_parquet(out_path)
```

### 4.3 New tests

**`test_coingecko_parses_volume_24h`** (`test_coingecko_marketcap.py`):

```python
async def test_coingecko_parses_volume_24h(monkeypatch):
    """Fetcher must extract total_volumes alongside market_caps."""
    fake_response = {
        "prices":        [[1746921600000, 100.0]],
        "market_caps":   [[1746921600000, 1.0e9]],
        "total_volumes": [[1746921600000, 5.0e7]],
    }
    # ... mock httpx.AsyncClient response, call fetch_marketcap_history
    rows = await fetch_marketcap_history("BTC-USD", start_ms=..., end_ms=...)
    assert len(rows) == 1
    ts, mc, vol = rows[0]
    assert mc == 1.0e9
    assert vol == 5.0e7
```

**`test_coingecko_handles_missing_total_volumes`**:

```python
async def test_coingecko_handles_missing_total_volumes(monkeypatch, caplog):
    fake_response = {"market_caps": [[1746921600000, 1.0e9]], "total_volumes": []}
    rows = await fetch_marketcap_history(...)
    assert rows == [(1746921600, 1.0e9, 0.0)]
    assert any("total_volumes missing/empty" in r.message for r in caplog.records)
```

**`test_coinpaprika_parses_volume_24h`** — analogous with the CP response shape.

**`test_cache_v1_parquet_triggers_full_refetch`** (`test_marketcap_history_cache.py`):

```python
async def test_cache_v1_parquet_triggers_full_refetch(tmp_path, monkeypatch):
    """v1 parquet on disk must trigger full refetch (schema_version mismatch)."""
    # Pre-seed parquet with schema v1 (no volume_24h column)
    df = pd.DataFrame({
        "start": [1746921600], "market_cap": [1.0e9], "fdv": [1.0e9],
        "ingest_ts": [int(time.time())],  # fresh (would normally hit cache)
        "schema_version": [1],
    })
    df.to_parquet(tmp_path / "BTC-USD.parquet")

    # Spy on the fetcher
    calls = {"n": 0}
    async def fake_fetch(*args, **kw):
        calls["n"] += 1
        return [(1746921600, 1.0e9, 5.0e7)]
    monkeypatch.setattr("services.coingecko_marketcap.fetch_marketcap_history", fake_fetch)

    rows = await fetch_marketcap_history_cached("BTC-USD", ..., parquet_dir=str(tmp_path))
    assert calls["n"] == 1, "v1 parquet should trigger full refetch"
    # New file should be v2
    df2 = pd.read_parquet(tmp_path / "BTC-USD.parquet")
    assert int(df2["schema_version"].iloc[0]) == 2
    assert "volume_24h" in df2.columns
```

## 5. Architecture

No architectural change. Pure data-layer extension. The cache contract stays "ask, get parquet, ignore source." The schema-version detector is a new private branch in `_cached_parquet_is_stale`.

```
Caller (future Step B feature extractor)
         │
         ▼
marketcap_history_cache.fetch_marketcap_history_cached(pid, start, end)
         │
         ├─ Read existing parquet (if any)
         │  ├─ schema_version < 2 → full refetch (NEW)
         │  ├─ ingest_ts > 24h old → refetch + merge (existing)
         │  └─ else → return cached rows
         │
         ▼ on refetch
coingecko_marketcap.fetch_marketcap_history(pid, start, end)
         │
         └─ returns List[Tuple[ts, market_cap, volume_24h]]   ← NEW shape
         │
         ▼
Write parquet with columns:
   start, market_cap, fdv, volume_24h, ingest_ts, schema_version=2
```

## 6. Data flow

**Before:** single CoinGecko call returns `market_caps`. Volume in response (`total_volumes`) is ignored. Parquet has no volume column.

**After:** same single call. Parser also extracts `total_volumes`. Returns 3-tuples. Cache writes `volume_24h` to parquet. Schema bumps to v2. On next access by an existing v1 parquet, the cache detects the mismatch and triggers a full refetch, transparently upgrading the file.

Zero extra API calls per pid.

## 7. Error handling

| Condition | Behavior | Test |
|---|---|---|
| CoinGecko returns valid `total_volumes` | Parsed; `volume_24h` populated | `test_coingecko_parses_volume_24h` |
| CoinGecko returns `total_volumes: null` or empty list | All rows get `volume_24h = 0.0`; warning logged | `test_coingecko_handles_missing_total_volumes` |
| CoinGecko returns mismatched lengths (`mc_raw` vs `vol_raw`) | Zip uses `min(len)`; trailing rows dropped; warning logged | covered by min() logic; manual edge-case verify |
| CoinPaprika row missing `volume` key | `volume_24h = 0.0` for that row | `test_coinpaprika_parses_volume_24h` |
| HTTP 429 (rate limit) | Existing retry logic in fetchers (unchanged) handles | existing tests cover |
| v1 parquet on disk on first access post-commit | Cache detects, refetches, overwrites in-place | `test_cache_v1_parquet_triggers_full_refetch` |
| Backfill fails partway through 49 pids | Per-pid independent; operator reruns failed pids; preflight is idempotent | manual operator step |
| FDV column already wasn't real (was MC proxy in current code) | Unchanged behavior; `fdv` column still populated with `market_cap` value | n/a |

## 8. Tests

| File | Action | Net |
|---|---|---|
| `tests/test_coingecko_marketcap.py` | EXTEND | +2 tests |
| `tests/test_coinpaprika_marketcap.py` | EXTEND | +1 test |
| `tests/test_marketcap_history_cache.py` | EXTEND | +1 test |
| **Total new** | | **+4 tests** |

Plus: the existing 1100+ test suite MUST stay green (pre-commit hook).

## 9. Rollout

### Phase 0 — Atomic commit
```bash
cd C:\Users\gl450\polymarket_app
git add backend/services/coingecko_marketcap.py \
        backend/services/coinpaprika_marketcap.py \
        backend/services/marketcap_history_cache.py \
        backend/tools/build_marketcap_parquet.py \
        backend/tests/test_coingecko_marketcap.py \
        backend/tests/test_coinpaprika_marketcap.py \
        backend/tests/test_marketcap_history_cache.py \
        CHANGELOG.md
git commit -m "feat(marketcap-A): bronze schema v2 — add volume_24h"
```

Pre-commit hook runs full suite (~5 min). On green, commit lands.

### Phase 1 — Push
```bash
git push
```

### Phase 2 — Operator backfill (~100 sec wall time)

```bash
cd backend
../.venv/Scripts/python.exe -m tools.build_marketcap_parquet --all-tracked
```

This:
- Reads the tracked-products list from `coinbase.db` (49 pids).
- For each pid, calls `coingecko_marketcap.fetch_marketcap_history` for 365-day range.
- Writes/overwrites `backend/data/marketcap/<pid>.parquet` at schema v2.
- Existing 3 parquets (BTC/ETH/SOL) get rewritten with the new `volume_24h` column.
- Missing 46 pids get created.
- Per-pid independent; one failure doesn't block others.

### Phase 3 — Verification

```bash
.venv/Scripts/python.exe -c "
import os, pandas as pd
pdir = 'backend/data/marketcap'
files = sorted(os.listdir(pdir))
for f in files[:3]:
    df = pd.read_parquet(os.path.join(pdir, f))
    print(f'{f}: rows={len(df)} schema_v={int(df[\"schema_version\"].iloc[0])} cols={df.columns.tolist()}')
print(f'TOTAL parquets: {len(files)}')
"
```

Expected: 49 parquets, all schema_v=2, all have `volume_24h` column.

### Rollback

`git revert <commit>`. Parquets remain at v2 on disk (no harm — the v1 code path will treat v2 as "modern" and not refetch). To fully roll back parquet data, the operator can `rm backend/data/marketcap/*.parquet` and rerun the v1 backfill from the reverted code.

## 10. Memory + CLAUDE.md sync (per CLAUDE.md rule)

Bundled into the same commit:
- `CHANGELOG.md` — Session 58.71i entry
- `polymarket_app/CLAUDE.md` — no invariant change (no behavior contract added; this is data-layer extension)
- `memory/coinbase_trader_architecture.md` (outside repo) — marketcap pipeline Step A entry

## 11. Open questions

None — operator approved every clarifying question on 2026-05-16.

## 12. References

- Session 58.68 (`f2fe8c8`): marketcap bronze cache + probe `--source` flag (#284/#285)
- CoinGecko `/coins/{id}/market_chart/range` endpoint: returns prices, market_caps, total_volumes parallel arrays
- CoinPaprika `/coins/{cp_id}/ohlcv/historical` endpoint: returns per-row dict with `volume` field
- `services/marketcap_history_cache.py`: 109 LOC cache layer (Session 58.68)
- `xgb_feature_optimization_findings.md` memory: notes that adding new inputs (e.g. OKX OI) was already the path to push AUC past the 0.5284 baseline — marketcap is the same theory
