"""Probe OKX for which Coinbase alt/meme pids have a SWAP listing.

Standalone diagnostic for #211 (Ch 27 OI fix decision). The current
_PRODUCT_TO_OKX map in services/okx_oi_history.py only covers 30
top-cap products; #210 found that 17/20 survivorship-aware pids
(alts/memes) get all-zero OI because their tickers are not in the
map. This probe queries OKX's public /api/v5/public/instruments
endpoint and tells us which of a candidate Coinbase ticker list
actually has a `<TICKER>-USDT-SWAP` instrument live on OKX, so we
know which entries can be safely added to the map.

Output: one line per candidate pid showing FOUND/MISSING and the
exact OKX instId we'd use. No production code is touched.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable, Set

import httpx

# Top-20 survivorship-aware pids that #210 audit flagged as all-zero OI
# (everything except the 3 already-mapped DOT/LINK/AVAX). Extra candidates
# from the wider scan list are also probed so we don't have to re-run.
_DEFAULT_CANDIDATES = [
    # All-zero per #210 audit
    "PENGU-USD", "JTO-USD", "POPCAT-USD", "BONK-USD", "NKN-USD",
    "AIOZ-USD", "ZK-USD", "JASMY-USD", "TRU-USD", "SKL-USD",
    "PEPE-USD", "FET-USD", "MOODENG-USD", "XCN-USD", "ONDO-USD",
    "ALGO-USD", "LRDS-USD",
    # Wider survivorship scan candidates — speculative
    "ZORA-USD", "WIF-USD", "RENDER-USD", "FLOKI-USD", "WLD-USD",
    "BERA-USD", "ENA-USD", "STRK-USD", "TON-USD", "JUP-USD",
]


def _candidate_to_inst(product_id: str) -> str:
    """Coinbase BTC-USD → OKX BTC-USDT-SWAP, naively."""
    base = product_id.split("-")[0]
    return f"{base}-USDT-SWAP"


def fetch_swap_inst_ids(timeout: float = 15.0) -> Set[str]:
    """One call to OKX's /public/instruments?instType=SWAP — returns the
    set of all live SWAP instIds. ~700 entries. Empty set on failure."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://www.okx.com/api/v5/public/instruments",
                params={"instType": "SWAP"},
            )
            if resp.status_code != 200:
                print(f"OKX HTTP {resp.status_code}", file=sys.stderr)
                return set()
            payload = resp.json()
            if payload.get("code") != "0":
                print(f"OKX code={payload.get('code')} "
                      f"msg={payload.get('msg')!r}", file=sys.stderr)
                return set()
            return {row["instId"] for row in payload.get("data", [])
                    if isinstance(row, dict) and "instId" in row}
    except Exception as e:
        print(f"OKX request failed: {e}", file=sys.stderr)
        return set()


def report(candidates: Iterable[str]) -> None:
    print("Fetching OKX SWAP listing index...", flush=True)
    t0 = time.time()
    live = fetch_swap_inst_ids()
    print(f"  {len(live):,} SWAP instruments live "
          f"(fetched in {time.time() - t0:.1f}s)", flush=True)
    if not live:
        print("Empty listing — probably geo-blocked or rate-limited.",
              flush=True)
        return

    print(f"\nProbing {len(list(candidates))} candidate Coinbase pids:",
          flush=True)
    found, missing = [], []
    for pid in candidates:
        inst = _candidate_to_inst(pid)
        if inst in live:
            found.append((pid, inst))
        else:
            missing.append((pid, inst))

    print(f"\n[FOUND] {len(found)} pids with live OKX SWAP:")
    for pid, inst in found:
        print(f"    {pid:<14} -> {inst}")
    print(f"\n[MISSING] {len(missing)} pids without OKX SWAP:")
    for pid, inst in missing:
        print(f"    {pid:<14} -> {inst}")

    if found:
        print("\nDiff for services/okx_oi_history.py _PRODUCT_TO_OKX:")
        for pid, inst in found:
            print(f'    "{pid}":   "{inst}",')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pids", nargs="*", default=None,
        help="Candidate Coinbase pids to probe (default: built-in list).",
    )
    args = parser.parse_args()
    candidates = args.pids if args.pids else _DEFAULT_CANDIDATES
    report(candidates)


if __name__ == "__main__":
    main()
