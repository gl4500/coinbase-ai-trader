"""Local-data inventory for the strategy-discovery rebuild (Phase 1).

Audits existing on-disk data BEFORE any new CoinPaprika API calls (per the
data-first directive in the 2026-05-23 brainstorm spec). Produces:

  - scan_history_parquets(dir)    -> {pid: {first_ts, last_ts, n_rows}}
  - scan_marketcap_parquets(dir)  -> {pid: {first_ts, last_ts, n_rows}}
  - scan_1m_dir(dir)              -> {pid: n_files} (returns {} when empty)
  - inventory_report(...)         -> Markdown string
  - run() / main()                CLI: write Markdown report + JSON sidecar

Pure stdlib + pyarrow. No HTTP calls. No model loading. No DB access.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import pyarrow.parquet as pq

# Directory layout (relative to repo root or BACKEND env)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_HISTORY_DIR = os.path.join(BACKEND_DIR, "data", "history")
DEFAULT_MARKETCAP_DIR = os.path.join(BACKEND_DIR, "data", "marketcap")
DEFAULT_1M_DIR = os.path.join(BACKEND_DIR, "data", "history", "1m")


def _is_pid_parquet(filename: str) -> bool:
    """A pid parquet is *.parquet that does NOT start with '__' (CLAUDE.md inv #8)."""
    return filename.endswith(".parquet") and not filename.startswith("__")


def _pid_from_filename(filename: str) -> str:
    """Strip .parquet suffix, restore '/' from '_' (matches build_marketcap_parquet)."""
    name = filename[: -len(".parquet")]
    return name  # we use '-' separators (e.g. BTC-USD), never '/', so no restore needed


def scan_history_parquets(directory: str) -> Dict[str, Dict[str, int]]:
    """Scan a directory of *.parquet OHLCV files and return per-pid coverage stats.

    Schema assumed: a 'start' column in epoch seconds. Files starting with '__'
    (e.g. __MACRO__.parquet) are skipped. Missing directory returns {}.
    """
    if not os.path.isdir(directory):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for entry in sorted(os.listdir(directory)):
        if not _is_pid_parquet(entry):
            continue
        path = os.path.join(directory, entry)
        try:
            table = pq.read_table(path, columns=["start"])
        except Exception:
            continue
        starts = table.column("start").to_pylist()
        if not starts:
            continue
        pid = _pid_from_filename(entry)
        out[pid] = {
            "first_ts": int(min(starts)),
            "last_ts": int(max(starts)),
            "n_rows": len(starts),
        }
    return out


def scan_marketcap_parquets(directory: str) -> Dict[str, Dict[str, int]]:
    """Same shape as scan_history_parquets but for marketcap bronze parquets."""
    return scan_history_parquets(directory)  # identical scan — both keyed on 'start'


def scan_1m_dir(directory: str) -> Dict[str, int]:
    """Scan a 1-minute candles directory laid out as <dir>/<pid>/*.parquet.

    Returns {pid: n_files} for each pid subdirectory. Pids with no files yet
    return n_files=0 (subdir exists but empty — SP1 infrastructure shipped
    but populated status varies). Missing directory returns {}.
    """
    if not os.path.isdir(directory):
        return {}
    out: Dict[str, int] = {}
    for entry in sorted(os.listdir(directory)):
        sub = os.path.join(directory, entry)
        if not os.path.isdir(sub):
            continue
        n = sum(1 for f in os.listdir(sub) if f.endswith(".parquet"))
        out[entry] = n
    return out


def _ts_to_iso_date(ts: int) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _days_span(first_ts: int, last_ts: int) -> int:
    return max(0, (int(last_ts) - int(first_ts)) // 86400)


def inventory_report(
    *,
    history: Dict[str, Dict[str, int]],
    marketcap: Dict[str, Dict[str, int]],
    minute1: Dict[str, int],
) -> str:
    """Render the three inventory dicts as a single Markdown report."""
    lines: list[str] = []
    lines.append("# Local Data Inventory")
    lines.append("")
    lines.append("Generated for the 2026-05-23 strategy-discovery rebuild data-first directive.")
    lines.append("")

    # --- 1-hour OHLCV ---
    lines.append("## 1-hour OHLCV (`backend/data/history/`)")
    lines.append("")
    lines.append(f"Total: {len(history)} pids")
    lines.append("")
    lines.append("| pid | first | last | days | rows |")
    lines.append("|---|---|---|---:|---:|")
    for pid in sorted(history.keys()):
        h = history[pid]
        days = _days_span(h["first_ts"], h["last_ts"])
        lines.append(
            f"| {pid} | {_ts_to_iso_date(h['first_ts'])} | "
            f"{_ts_to_iso_date(h['last_ts'])} | {days} | {h['n_rows']:,} |"
        )
    lines.append("")

    # --- Marketcap (CoinPaprika tokenomic) ---
    lines.append("## CoinPaprika tokenomic (`backend/data/marketcap/`)")
    lines.append("")
    lines.append(f"Total: {len(marketcap)} pids")
    lines.append("")
    if marketcap:
        lines.append("| pid | first | last | days | rows |")
        lines.append("|---|---|---|---:|---:|")
        for pid in sorted(marketcap.keys()):
            m = marketcap[pid]
            days = _days_span(m["first_ts"], m["last_ts"])
            lines.append(
                f"| {pid} | {_ts_to_iso_date(m['first_ts'])} | "
                f"{_ts_to_iso_date(m['last_ts'])} | {days} | {m['n_rows']:,} |"
            )
    else:
        lines.append("(none)")
    lines.append("")

    # --- 1-minute OHLCV ---
    lines.append("## 1-minute OHLCV (`backend/data/history/1m/`)")
    lines.append("")
    lines.append(f"Total: {len(minute1)} pid directories")
    lines.append("")
    if minute1:
        lines.append("| pid | n_files |")
        lines.append("|---|---:|")
        for pid in sorted(minute1.keys()):
            lines.append(f"| {pid} | {minute1[pid]} |")
    else:
        lines.append("(no 1m directory or empty)")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI: write Markdown report + JSON sidecar to docs/superpowers/specs/.

    The JSON sidecar is the structured data Task 5 (universe curation) consumes.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Audit local data for strategy-discovery rebuild.")
    parser.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--marketcap-dir", default=DEFAULT_MARKETCAP_DIR)
    parser.add_argument("--minute1-dir", default=DEFAULT_1M_DIR)
    parser.add_argument(
        "--out-md",
        default=os.path.join(
            BACKEND_DIR, "..", "docs", "superpowers", "specs", "2026-05-23-data-inventory-report.md"
        ),
    )
    parser.add_argument(
        "--out-json",
        default=os.path.join(
            BACKEND_DIR, "..", "docs", "superpowers", "specs", "2026-05-23-data-inventory.json"
        ),
    )
    args = parser.parse_args()

    history = scan_history_parquets(args.history_dir)
    marketcap = scan_marketcap_parquets(args.marketcap_dir)
    minute1 = scan_1m_dir(args.minute1_dir)

    md = inventory_report(history=history, marketcap=marketcap, minute1=minute1)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"history": history, "marketcap": marketcap, "minute1": minute1}, f, indent=2)

    print(f"  history:   {len(history)} pids", flush=True)
    print(f"  marketcap: {len(marketcap)} pids", flush=True)
    print(f"  1m dirs:   {len(minute1)} pids", flush=True)
    print(f"  wrote: {args.out_md}", flush=True)
    print(f"  wrote: {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
