"""Preregistered volatility-target screen. See the plan's Preregistration section.

python -m tools.vol_target.screen import <slow_trend_out>   # copy + verify the locked snapshot
python -m tools.vol_target.screen lock                      # write snapshot.lock (commit it)
python -m tools.vol_target.screen run [--replay] [--new-preregistration]
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.slow_trend import daily_bars as D
from tools.slow_trend import gates as G
from tools.slow_trend import metrics as M
from tools.slow_trend import screen as S
from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve
from tools.vol_target import prereg as P
from tools.vol_target.fsim import run_weight_sleeve
from tools.vol_target.signal import schedule
from tools.vol_target.verdict import verdict

BACKEND = Path(__file__).resolve().parents[2]
OUT = BACKEND / "data" / "research" / "vol_target"
LOCK = Path(__file__).with_name("snapshot.lock")
BLOCKS = (P.BOOT_BLOCK_WEEKS, *P.BOOT_SENS_WEEKS)
RAW = tuple(f"{p}.raw.parquet" for p in P.PRODUCTS)
# The experiment identity covers the INHERITED preregistration too (Codex N2): a committed change
# to a shared fee/gate/date constant must not silently reuse this experiment's identity.
PREREG_FILES = (Path(P.__file__), BACKEND / "tools" / "slow_trend" / "prereg.py")
# Family record (Codex N3): the second candidate evaluated on the same snapshot and block.
FAMILY = {
    "candidate_ordinal": 2,
    "prior": {
        "candidate": "slow_trend SMA100",
        "experiment_id": "2252e434a40e11f6",
        "report": "report_2252e434a40e11f6_a1_first.json",
        "verdict": "KILL",
        "reason": "primary_failed",
    },
    "block_status": "previously observed retrospective evaluation, reused for a second candidate",
}


def prereg_digest(paths: list, root: Path) -> str:
    return S.source_digest(list(paths), root)


def _realised_vol(equity: pd.Series) -> float:
    r = np.log(equity[equity > 0]).diff().dropna()
    return float(r.std(ddof=1) * math.sqrt(P.ANNUALISATION)) if len(r) > 1 else float("nan")


def _summ_w(results: list) -> dict:
    eq = sum(r.equity for r in results)
    tv = sum(r.terminal_value for r in results)
    ex, tf = sum(r.exec_fees for r in results), sum(r.terminal_fee for r in results)
    return {
        "terminal_value": tv,
        "net_return": tv / P.INITIAL_USD - 1,
        "max_drawdown": M.max_drawdown(eq, P.INITIAL_USD),
        "total_fees": ex + tf,
        "slippage_cost": sum(r.slippage_cost for r in results),
        "unliquidatable": [r.unliquidatable for r in results],
        "residual_marked_value": sum(r.residual_marked_value for r in results),
        "rebalance_turnover_excl_terminal": sum(r.traded_notional for r in results) / P.INITIAL_USD,
        "realised_vol": _realised_vol(eq),
        "boundary_returns": M.boundary_returns(eq, P.INITIAL_USD),
        "per_sleeve": {
            pid: {
                "terminal_value": r.terminal_value,
                "net_return": r.terminal_value / r.initial - 1,
                "max_drawdown": M.max_drawdown(r.equity, r.initial),
                "realised_vol": _realised_vol(r.equity),
                "exposure": r.exposure,
                "buys": r.buys,
                "sells": r.sells,
                "size_skipped": r.size_skipped,
                "missed_open": r.missed_open,
                "within_deadband": r.within_deadband,
                "valid_decisions": r.valid_decisions,
                "invalid_decisions": r.invalid_decisions,
                "executed": r.executed,
                "capped_fraction_of_valid_decisions": (
                    r.capped / r.valid_decisions if r.valid_decisions else None
                ),
                "decision_log": r.decision_log,
                "executed_weights": r.executed_weights,
                "stale_mark_days": r.stale_mark_days,
                "max_stale_run": r.max_stale_run,
            }
            for pid, r in zip(P.PRODUCTS, results, strict=True)
        },
        "_equity": eq,
    }


def run_period(cal: dict, start: str, end: str, products: dict) -> dict:
    out = {"start": start, "end": end, "scenarios": {}}
    for sc in P.SCENARIOS:
        costs = Costs(sc.entry_fee, sc.exit_fee, sc.slip)
        legs = {"vol_target": [], "fixed50": [], "buy_hold": [], "dca52": []}
        first_exec = []
        for pid in P.PRODUCTS:
            bars, prod = cal[pid].loc[start:end], products[pid]
            sch = schedule(cal[pid]["close"], start, end, sc.delay)
            first_exec.append(sch["execute"].min())
            fixed = sch.assign(target=P.FIXED_DIAG_WEIGHT)
            legs["vol_target"].append(
                run_weight_sleeve(bars, sch, P.SLEEVE_USD, costs, prod, P.DEADBAND)
            )
            legs["fixed50"].append(
                run_weight_sleeve(bars, fixed, P.SLEEVE_USD, costs, prod, P.DEADBAND)
            )
            hold = pd.Series(True, index=bars.index)
            legs["buy_hold"].append(run_sleeve(bars, hold, P.SLEEVE_USD, costs, prod))
            legs["dca52"].append(run_dca(bars, P.SLEEVE_USD, P.DCA_TRANCHES, costs, prod))
        s = {k: _summ_w(legs[k]) for k in ("vol_target", "fixed50")}
        s.update({k: S._summ(legs[k], P.INITIAL_USD) for k in ("buy_hold", "dca52")})
        s["cash"] = {"terminal_value": P.INITIAL_USD, "net_return": 0.0, "max_drawdown": 0.0}
        s["calendar"] = {"first_execute": str(min(first_exec).date())}
        s["passes"] = {
            **G.passes(s["vol_target"], s["buy_hold"], P.INITIAL_USD, P.G2_DRAWDOWN_RATIO),
            "affected": G.affected(s["vol_target"], s["buy_hold"]),
        }
        wt = M.weekly_returns(s["vol_target"]["_equity"])
        for name, key in (("buy_hold", "bh"), ("dca52", "dca52"), ("fixed50", "fixed50")):
            wc = M.weekly_returns(s[name]["_equity"])
            s[f"ci_vs_{key}"] = {
                str(b): M.paired_block_ci(wt, wc, b, P.BOOT_RESAMPLES, P.BOOT_SEED) for b in BLOCKS
            }
        for k in ("vol_target", "fixed50", "buy_hold", "dca52"):
            s[k].pop("_equity")
        out["scenarios"][sc.name] = s
    return out


def _inadequate(report: dict, why: str) -> dict:
    report["data_checks"]["inadequate_because"] = why
    report["verdict"] = verdict(False, None, {})
    return report


def evaluate(raw: dict, constraints: dict) -> dict:
    report = {"family": FAMILY, "data_checks": {"terminal": {}}, "results": None}
    firsts = {p: D.first_day(raw[p]) for p in P.PRODUCTS}
    report["data_checks"]["first_day"] = firsts
    if any(f is None for f in firsts.values()):
        return _inadequate(report, "no_aligned_rows")
    common_first = max(firsts.values())
    report["data_checks"]["common_first"] = common_first
    if common_first > P.DEV_START:
        return _inadequate(report, "no_development_coverage")
    windows = {"dev": (P.DEV_START, P.DEV_END), "block": (P.BLOCK_START, P.BLOCK_END)}
    # "input" audits everything the signal CONSUMES (warmup from common_first), not only the
    # scored dates (Codex N1); conflicting duplicates are rejected here, before normalise.
    audited = {"input": (common_first, P.DEV_END), **windows}
    audits = {
        w: {p: D.audit(raw[p], a, b, P.MAX_MISSING_DAYS) for p in P.PRODUCTS}
        for w, (a, b) in audited.items()
    }
    report["data_checks"]["audits"] = audits
    if not all(v["adequate"] for w in audits.values() for v in w.values()):
        return _inadequate(report, "audit")
    cal = {
        p: D.to_calendar(
            D.normalise(D.window(raw[p], common_first, P.BLOCK_END)), common_first, P.BLOCK_END
        )
        for p in P.PRODUCTS
    }
    for p in P.PRODUCTS:
        report["data_checks"]["terminal"][p] = {
            d: bool(pd.notna(cal[p].loc[d, "close"])) for d in (P.DEV_END, P.BLOCK_END)
        }
    if not all(all(t.values()) for t in report["data_checks"]["terminal"].values()):
        return _inadequate(report, "terminal_close_missing")
    products = {p: Product(**constraints[p]) for p in P.PRODUCTS}
    res = {n: run_period(cal, a, b, products) for n, (a, b) in windows.items()}
    passes = {n: {s: r["scenarios"][s]["passes"] for s in r["scenarios"]} for n, r in res.items()}
    p0 = res["block"]["scenarios"]["P0"]["vol_target"]["per_sleeve"]
    coverage = {p: p0[p]["valid_decisions"] for p in P.PRODUCTS}
    report.update(
        periods=windows,
        results=res,
        block_valid_decisions=coverage,
        verdict=verdict(True, passes, coverage),
    )
    return report


def import_snapshot(src: Path, out: Path) -> dict:
    """Copy the slow-trend LOCKED snapshot; verify the manifest against the frozen lock and
    every raw file against the manifest BEFORE publishing anything."""
    if S._sha(src / "manifest.json") != P.SOURCE_SNAPSHOT_LOCK:
        raise RuntimeError(
            "source manifest does not match the frozen slow-trend lock (SOURCE_SNAPSHOT_LOCK)"
        )
    manifest = json.loads((src / "manifest.json").read_text())
    for pid in P.PRODUCTS:
        if S._sha(src / f"{pid}.raw.parquet") != manifest["products"][pid]["sha256"]:
            raise RuntimeError(f"snapshot {pid} does not match its manifest digest")
    S.ensure_new_snapshot(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in RAW:
        shutil.copyfile(src / name, out / name)
    shutil.copyfile(src / "manifest.json", out / "manifest.json")  # last: publication marker
    return manifest


def _source_files() -> list:
    here = Path(__file__).parent
    return (
        sorted(here.glob("*.py"))
        + sorted((here.parent / "slow_trend").glob("*.py"))
        + [BACKEND / "clients" / "coinbase_client.py"]
    )


def _lock() -> None:
    sha = S._sha(OUT / "manifest.json")
    if sha != P.SOURCE_SNAPSHOT_LOCK:
        sys.exit("imported manifest is not the locked slow-trend snapshot")
    LOCK.write_text(sha + "\n")
    print(f"wrote {LOCK}; commit it before `run`")


def _run(replay: bool, new_prereg: bool) -> None:
    dirty = S._git("status", "--porcelain", "--", *P.FREEZE_PATHS)
    if dirty:
        sys.exit(f"refusing to run: uncommitted changes\n{dirty}")
    manifest_sha = S._sha(OUT / "manifest.json")
    if not LOCK.exists() or LOCK.read_text().strip() != manifest_sha:
        sys.exit("snapshot.lock missing or does not match the manifest")
    manifest = json.loads((OUT / "manifest.json").read_text())
    head, prereg_sha = S._git("rev-parse", "HEAD"), prereg_digest(PREREG_FILES, S.REPO)
    exp_id = S.experiment_identity(prereg_sha, manifest_sha)
    ledger = OUT / "runs.jsonl"
    entries = (
        [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
        if ledger.exists()
        else []
    )
    mode = S.run_mode(entries, exp_id, replay=replay, new_prereg=new_prereg)
    src = S.source_digest(_source_files())
    corrects = None
    if mode == "replay":
        mode, corrects = S.replay_label(entries, exp_id, src)
    attempt = 1 + max((e.get("attempt", 0) for e in entries), default=0)
    base = {
        "experiment_id": exp_id,
        "attempt": attempt,
        "head": head,
        "mode": mode,
        "source_sha256": src,
        "replays_attempt": corrects,
        "family": FAMILY,
    }
    S._append(
        ledger,
        {**base, "status": "started", "parent": S.last_attempt(entries, exp_id), "at": S._now()},
    )
    try:
        raw, constraints = {}, {}
        for pid, m in manifest["products"].items():
            path = OUT / f"{pid}.raw.parquet"
            if S._sha(path) != m["sha256"]:
                raise RuntimeError(f"snapshot {pid} does not match its manifest digest")
            raw[pid], constraints[pid] = pd.read_parquet(path), m["constraints"]
        report = evaluate(raw, constraints)
    except Exception as exc:
        S._append(ledger, {**base, "status": "failed", "error": repr(exc), "at": S._now()})
        raise
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
    S._append(
        ledger,
        {
            **base,
            "status": "completed",
            "report": name,
            "verdict": report["verdict"],
            "at": S._now(),
        },
    )
    print(json.dumps({"verdict": report["verdict"], "mode": mode, "report": name}, indent=2))


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "import" and args:
        print(json.dumps(import_snapshot(Path(args[0]), OUT), indent=2))
    elif cmd == "lock":
        _lock()
    elif cmd == "run":
        flags = set(args)
        _run(replay="--replay" in flags, new_prereg="--new-preregistration" in flags)
    else:
        sys.exit(
            "usage: python -m tools.vol_target.screen "
            "{import <slow_trend_out>|lock|run [--replay] [--new-preregistration]}"
        )
