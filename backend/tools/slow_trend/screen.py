"""Preregistered slow-trend screen. See the plan's Preregistration section.

python -m tools.slow_trend.screen fetch          # raw snapshot + manifest (refuses overwrite)
python -m tools.slow_trend.screen lock           # write snapshot.lock (commit it before run)
python -m tools.slow_trend.screen run [--replay] # frozen grid -> ledger + report
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tools.slow_trend import daily_bars as D
from tools.slow_trend import gates as G
from tools.slow_trend import metrics as M
from tools.slow_trend import prereg as P
from tools.slow_trend.rule import decisions, desired_state
from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve, trend_target

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
OUT = BACKEND / "data" / "research" / "slow_trend"
LOCK = Path(__file__).with_name("snapshot.lock")
BLOCKS = (P.BOOT_BLOCK_WEEKS, *P.BOOT_SENS_WEEKS)


# ── pure evaluation ────────────────────────────────────────────────────────────


def dev_start_from(cal: dict, n: int, last: str) -> Optional[str]:
    ok = None
    for c in cal.values():
        v = decisions(c["close"], n).notna()
        ok = v if ok is None else (ok & v)
    days = ok[ok].index
    days = days[days <= pd.Timestamp(last)]
    return str(days[0].date()) if len(days) else None


def _summ(results: list, initial: float) -> dict:
    eq = sum(r.equity for r in results)
    tv = sum(r.terminal_value for r in results)
    ex = sum(r.exec_fees for r in results)
    tf = sum(r.terminal_fee for r in results)
    return {
        "terminal_value": tv,
        "net_return": tv / initial - 1,
        "max_drawdown": M.max_drawdown(eq, initial),
        "exec_fees": ex,
        "terminal_fee": tf,
        "total_fees": ex + tf,
        "slippage_cost": sum(r.slippage_cost for r in results),
        "unliquidatable": [r.unliquidatable for r in results],
        "residual_units": [r.residual_units for r in results],
        "residual_marked_value": sum(r.residual_marked_value for r in results),
        "entries": sum(r.entries for r in results),
        "exits": sum(r.exits for r in results),
        "round_trips": sum(r.round_trips for r in results),
        "skipped": sum(r.skipped for r in results),
        "skipped_exits": sum(r.skipped_exits for r in results),
        "executed_turnover": sum(r.traded_notional for r in results) / initial,
        "exposure_mean": float(np.mean([r.exposure for r in results])),
        "boundary_returns": M.boundary_returns(eq, initial),
        "per_sleeve": {
            pid: {
                "terminal_value": r.terminal_value,
                "net_return": r.terminal_value / r.initial - 1,
                "max_drawdown": M.max_drawdown(r.equity, r.initial),
                "exposure": r.exposure,
                "round_trips": r.round_trips,
                "stale_mark_days": r.stale_mark_days,
                "max_stale_run": r.max_stale_run,
            }
            for pid, r in zip(P.PRODUCTS, results, strict=True)
        },
        "_equity": eq,
    }


def run_period(cal: dict, start: str, end: str, products: dict) -> dict:
    diagnostics = {}
    for pid in P.PRODUCTS:
        dec = decisions(cal[pid]["close"], P.SMA_LEN).loc[start:end]
        diagnostics[pid] = {"suppressed_decision_days": int(dec.isna().sum())}
    out = {"start": start, "end": end, "diagnostics": diagnostics, "scenarios": {}}
    for sc in P.SCENARIOS:
        costs = Costs(sc.entry_fee, sc.exit_fee, sc.slip)
        legs = {"trend": [], "buy_hold": [], "dca52": []}
        for pid in P.PRODUCTS:
            full, prod = cal[pid], products[pid]
            bars = full.loc[start:end]
            # warmup from history, THEN slice, THEN hold state: the period starts flat
            dec = decisions(full["close"], P.SMA_LEN).loc[start:end]
            tgt = trend_target(desired_state(dec), sc.delay)
            legs["trend"].append(run_sleeve(bars, tgt, P.SLEEVE_USD, costs, prod))
            hold = pd.Series(True, index=bars.index)
            legs["buy_hold"].append(run_sleeve(bars, hold, P.SLEEVE_USD, costs, prod))
            legs["dca52"].append(run_dca(bars, P.SLEEVE_USD, P.DCA_TRANCHES, costs, prod))
        s = {k: _summ(v, P.INITIAL_USD) for k, v in legs.items()}
        s["cash"] = {"terminal_value": P.INITIAL_USD, "net_return": 0.0, "max_drawdown": 0.0}
        s["passes"] = {
            **G.passes(s["trend"], s["buy_hold"], P.INITIAL_USD, P.G2_DRAWDOWN_RATIO),
            "affected": G.affected(s["trend"], s["buy_hold"]),
        }
        wt = M.weekly_returns(s["trend"]["_equity"])
        for name, key in (("buy_hold", "bh"), ("dca52", "dca52")):
            wc = M.weekly_returns(s[name]["_equity"])
            s[f"ci_vs_{key}"] = {
                str(b): M.paired_block_ci(wt, wc, b, P.BOOT_RESAMPLES, P.BOOT_SEED) for b in BLOCKS
            }
        s["ci_vs_cash"] = {
            str(b): M.paired_block_ci(wt, wt * 0.0, b, P.BOOT_RESAMPLES, P.BOOT_SEED)
            for b in BLOCKS
        }
        for k in ("trend", "buy_hold", "dca52"):
            s[k].pop("_equity")
        out["scenarios"][sc.name] = s
    return out


def _inadequate(report: dict, why: str) -> dict:
    report["data_checks"]["inadequate_because"] = why
    report["verdict"] = G.verdict(False, None, 0)
    return report


def evaluate(raw: dict, constraints: dict) -> dict:
    report = {"data_checks": {"terminal": {}}, "dev_start": None, "results": None}
    firsts = {p: D.first_day(raw[p]) for p in P.PRODUCTS}
    report["data_checks"]["first_day"] = firsts
    if any(f is None for f in firsts.values()):
        return _inadequate(report, "no_aligned_rows")
    common_first = max(firsts.values())
    report["data_checks"]["common_first"] = common_first
    if common_first > P.DEV_END:
        return _inadequate(report, "no_development_coverage")
    windows = {"dev": (common_first, P.DEV_END), "block": (P.BLOCK_START, P.BLOCK_END)}
    audits = {
        w: {p: D.audit(raw[p], a, b, P.MAX_MISSING_DAYS) for p in P.PRODUCTS}
        for w, (a, b) in windows.items()
    }
    report["data_checks"]["audits"] = audits
    if not all(v["adequate"] for w in audits.values() for v in w.values()):
        return _inadequate(report, "audit")
    # rows outside [common_first, BLOCK_END] are unused and excluded BEFORE normalisation
    cal = {
        p: D.to_calendar(
            D.normalise(D.window(raw[p], common_first, P.BLOCK_END)), common_first, P.BLOCK_END
        )
        for p in P.PRODUCTS
    }
    checks = report["data_checks"]
    for p in P.PRODUCTS:
        checks["terminal"][p] = {
            d: bool(pd.notna(cal[p].loc[d, "close"])) for d in (P.DEV_END, P.BLOCK_END)
        }
    if not all(all(t.values()) for t in checks["terminal"].values()):
        return _inadequate(report, "terminal_close_missing")
    dev_start = dev_start_from(cal, P.SMA_LEN, P.DEV_END)
    report["dev_start"] = dev_start
    if dev_start is None:
        return _inadequate(report, "no_common_valid_sma_before_dev_end")
    products = {p: Product(**constraints[p]) for p in P.PRODUCTS}
    periods = {"dev": (dev_start, P.DEV_END), "block": (P.BLOCK_START, P.BLOCK_END)}
    res = {n: run_period(cal, a, b, products) for n, (a, b) in periods.items()}
    passes = {n: {s: r["scenarios"][s]["passes"] for s in r["scenarios"]} for n, r in res.items()}
    rt = res["block"]["scenarios"]["P0"]["trend"]["round_trips"]
    report.update(periods=periods, results=res, verdict=G.verdict(True, passes, rt))
    return report


# ── freeze mechanics ───────────────────────────────────────────────────────────


def ensure_new_snapshot(out: Path) -> None:
    if (out / "manifest.json").exists():
        raise FileExistsError(
            f"{out} already holds a snapshot; a new snapshot is a new "
            "preregistration - move the old directory aside deliberately"
        )


def experiment_identity(prereg_sha: str, manifest_sha: str) -> str:
    """Stable across commits: HEAD is attempt provenance, never part of the permission key."""
    return hashlib.sha256(f"{prereg_sha}|{manifest_sha}".encode()).hexdigest()[:16]


def run_mode(
    entries: list, experiment_id: str, replay: bool = False, new_prereg: bool = False
) -> str:
    mine = [e["status"] for e in entries if e.get("experiment_id") == experiment_id]
    if "completed" in mine:
        if replay:
            return "replay"
        raise RuntimeError("experiment already completed: rerun only as --replay")
    if mine:
        return "retry"  # an earlier attempt started or failed; a new commit does not reset it
    others_done = any(
        e["status"] == "completed" and e.get("experiment_id") != experiment_id for e in entries
    )
    if others_done:
        if not new_prereg:
            raise RuntimeError(
                "a different preregistration already completed; pass "
                "--new-preregistration explicitly"
            )
        return "new_preregistration"
    return "first"


def last_attempt(entries: list, experiment_id: str) -> Optional[int]:
    seqs = [
        e["attempt"]
        for e in entries
        if e.get("experiment_id") == experiment_id and e["status"] == "started"
    ]
    return seqs[-1] if seqs else None


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(ledger: Path, entry: dict) -> None:
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def _fetch() -> None:
    import httpx

    ensure_new_snapshot(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    start = int(pd.Timestamp(P.FETCH_FROM).timestamp())
    end = int(pd.Timestamp(P.BLOCK_END).timestamp()) + D.DAY
    manifest = {"fetched_at": _now(), "source": D.PUBLIC_BASE, "products": {}}
    async with httpx.AsyncClient() as client:
        get = D.public_getter(client)
        for pid in P.PRODUCTS:
            raw = await D.fetch_daily(pid, start, end, get)
            path = OUT / f"{pid}.raw.parquet"
            raw.to_parquet(path, index=False)
            manifest["products"][pid] = {
                "sha256": _sha(path),
                "rows": len(raw),
                "requests": raw.attrs["requests"],
                "pages_with_rows": int(raw["page"].nunique()),
                "first_day": D.first_day(raw),
                "constraints": D.product_constraints(await get(f"/products/{pid}")),
            }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _lock() -> None:
    LOCK.write_text(_sha(OUT / "manifest.json") + "\n")
    print(f"wrote {LOCK}; commit it before `run`")


def safe_overlap(hourly_path: Path, raw: pd.DataFrame, first: str) -> dict:
    """Informational only. Never raises: the bytes are read ONCE, then hashed and parsed from
    that same capture, and the error path does no file I/O."""
    if not hourly_path.exists():
        return {"status": "absent"}
    digest = None
    try:
        data = hourly_path.read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        cal = D.to_calendar(D.normalise(D.window(raw, first, P.BLOCK_END)), first, P.BLOCK_END)
        out = D.hourly_overlap(pd.read_parquet(io.BytesIO(data)), cal)
        return {"status": "ok", "sha256": digest, **out}
    except Exception as exc:
        return {"status": "error", "sha256": digest, "error": repr(exc)}


def _text_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")  # identity must not depend on checkout


def text_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(_text_bytes(path)).hexdigest()


def source_digest(paths: list, root: Path = REPO) -> str:
    """Digest of the measurement source actually on disk, recorded per attempt."""
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        h.update(str(path.relative_to(root)).replace("\\", "/").encode() + b"\0")
        h.update(_text_bytes(path) + b"\0")
    return "sha256:" + h.hexdigest()


def _source_files() -> list:
    return sorted(Path(__file__).parent.glob("*.py")) + [BACKEND / "clients" / "coinbase_client.py"]


def replay_label(entries: list, experiment_id: str, source_sha: str):
    """`replay` when the completed attempt ran the same source; otherwise `corrected_replay`,
    linked to that first completed attempt. Both reports are kept."""
    done = [
        e for e in entries if e.get("experiment_id") == experiment_id and e["status"] == "completed"
    ]
    first = done[0]
    label = "replay" if first.get("source_sha256") == source_sha else "corrected_replay"
    return label, first.get("attempt")


def _run(replay: bool, new_prereg: bool) -> None:
    dirty = _git("status", "--porcelain", "--", *P.FREEZE_PATHS)
    if dirty:
        sys.exit(f"refusing to run: uncommitted changes\n{dirty}")
    manifest_sha = _sha(OUT / "manifest.json")
    if not LOCK.exists() or LOCK.read_text().strip() != manifest_sha:
        sys.exit("snapshot.lock missing or does not match the manifest")
    manifest = json.loads((OUT / "manifest.json").read_text())
    head, prereg_sha = _git("rev-parse", "HEAD"), text_digest(Path(P.__file__))
    exp_id = experiment_identity(prereg_sha, manifest_sha)
    ledger = OUT / "runs.jsonl"
    entries = (
        [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
        if ledger.exists()
        else []
    )
    mode = run_mode(entries, exp_id, replay=replay, new_prereg=new_prereg)
    src = source_digest(_source_files())
    corrects = None
    if mode == "replay":
        mode, corrects = replay_label(entries, exp_id, src)
    attempt = 1 + max((e.get("attempt", 0) for e in entries), default=0)
    base = {
        "experiment_id": exp_id,
        "attempt": attempt,
        "head": head,
        "mode": mode,
        "source_sha256": src,
        "replays_attempt": corrects,
    }
    _append(
        ledger, {**base, "status": "started", "parent": last_attempt(entries, exp_id), "at": _now()}
    )
    try:
        raw, constraints = {}, {}
        for pid, m in manifest["products"].items():
            path = OUT / f"{pid}.raw.parquet"
            if _sha(path) != m["sha256"]:
                raise RuntimeError(f"snapshot {pid} does not match its manifest digest")
            raw[pid], constraints[pid] = pd.read_parquet(path), m["constraints"]
        report = evaluate(raw, constraints)
    except Exception as exc:
        _append(ledger, {**base, "status": "failed", "error": repr(exc), "at": _now()})
        raise
    first = report["data_checks"].get("common_first")
    report["hourly_overlap_informational"] = (
        {
            pid: safe_overlap(BACKEND / "data" / "history" / f"{pid}.parquet", raw[pid], first)
            for pid in P.PRODUCTS
        }
        if first
        else {}
    )
    report.update(
        experiment_id=exp_id,
        attempt=attempt,
        mode=mode,
        head=head,
        source_sha256=src,
        replays_attempt=corrects,
        prereg_sha256=prereg_sha,
        manifest_sha256=manifest_sha,
    )
    name = f"report_{exp_id}_a{attempt}_{mode}.json"
    (OUT / name).write_text(json.dumps(report, indent=2, default=str))
    _append(
        ledger,
        {**base, "status": "completed", "report": name, "verdict": report["verdict"], "at": _now()},
    )
    print(json.dumps({"verdict": report["verdict"], "mode": mode, "report": name}, indent=2))


if __name__ == "__main__":
    cmd, flags = (sys.argv[1] if len(sys.argv) > 1 else ""), set(sys.argv[2:])
    if cmd == "fetch":
        asyncio.run(_fetch())
    elif cmd == "lock":
        _lock()
    elif cmd == "run":
        _run(replay="--replay" in flags, new_prereg="--new-preregistration" in flags)
    else:
        sys.exit(
            "usage: python -m tools.slow_trend.screen "
            "{fetch|lock|run [--replay] [--new-preregistration]}"
        )
