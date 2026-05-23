"""CoinPaprika historical marketcap fetcher (#293) — free-tier sibling of
services/coingecko_marketcap (#260) that does NOT require an API key.

Why a second source: CoinGecko's free tier began returning 401 on its
historical market_chart/range endpoint (probe #260d/e/f, 2026-05-09).
CoinPaprika offers the same shape of data (one daily row per coin with a
market_cap field) at 25,000 requests/day with no key.

Public API mirrors coingecko_marketcap so the probe harness can swap providers
without changing downstream code:

    await fetch_marketcap_history(pid: str, start_ms: int, end_ms: int)
        -> list[(ts_ms, market_cap)] sorted ascending

    _coinbase_to_cp_id(pid)   -> Optional[str]    ("BTC-USD" -> "btc-bitcoin")

Endpoint (free tier — `coins/{id}/ohlcv/historical` is paywalled, this one
isn't):
    GET https://api.coinpaprika.com/v1/tickers/{cp_id}/historical
        ?start=YYYY-MM-DD&interval=1d
    Returns one row per UTC day with: timestamp, price, volume_24h, market_cap.
    Free tier window is rolling ~12 months; older starts return HTTP 402.

Strict causality: same EOD-UTC stamping as CoinGecko, so the same 1-day lag
applies at align time. This module returns raw rows; alignment lives in the
probe / parquet writer.

Kill switch: env COINPAPRIKA_DISABLED=1 short-circuits without an HTTP call.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.coinpaprika.com/v1"
_HISTORY_URL = f"{_BASE}/tickers/{{cp_id}}/historical"
_TICKER_URL  = f"{_BASE}/tickers/{{cp_id}}"

_SCHEMA_VERSION = 1


# ── Coinbase product → CoinPaprika id mapping ───────────────────────────────
# CoinPaprika ids follow the pattern <ticker>-<slug>. Verified live 2026-05-10.
_PRODUCT_TO_CP_ID = {
    # ── Original 28 entries (verified 2026-05-10) ───────────────────────────
    "BTC-USD":     "btc-bitcoin",
    "ETH-USD":     "eth-ethereum",
    "SOL-USD":     "sol-solana",
    "XRP-USD":     "xrp-xrp",
    "BNB-USD":     "bnb-binance-coin",
    "ADA-USD":     "ada-cardano",
    "AVAX-USD":    "avax-avalanche",
    "LINK-USD":    "link-chainlink",
    "DOT-USD":     "dot-polkadot",
    "DOGE-USD":    "doge-dogecoin",
    "ARB-USD":     "arb-arbitrum",
    "ALGO-USD":    "algo-algorand",
    "ONDO-USD":    "ondo-ondo",
    "FET-USD":     "fet-fetch-ai",
    "PEPE-USD":    "pepe-pepe",
    "BONK-USD":    "bonk-bonk1",
    "POPCAT-USD":  "popcat-popcat",
    "JTO-USD":     "jto-jito",
    "PENGU-USD":   "pengu-pudgy-penguins",
    "ZK-USD":      "zk-zksync",
    "TRU-USD":     "tru-truefi-token",
    "SKL-USD":     "skl-skale-network",
    "JASMY-USD":   "jasmy-jasmycoin",
    "NKN-USD":     "nkn-nkn",
    "AIOZ-USD":    "aioz-aioz-network",
    "MOODENG-USD": "moodeng-moo-deng",
    "LRDS-USD":    "lrds-lordlabs",
    "XCN-USD":     "xcn-onyxcoin",
    # Added 2026-05-23 for strategy-discovery rebuild Phase 1 Task 4
    # (candidate pool extension for universe curation).
    # Sourced from Task 1 inventory (top-N pids by row count),
    # cross-referenced against CoinPaprika /v1/coins.
    # Stables/wrapped/gold-pegged tokens excluded (no tokenomic signal).
    "A8-USD":      "a8-ancient8",
    "AAVE-USD":    "aave-new",
    "ABT-USD":     "abt-arcblock",
    "AERO-USD":    "aero-aerodrome-finance",
    "AKT-USD":     "akt-akash-network",
    "ALEPH-USD":   "aleph-alephim",
    "ALICE-USD":   "alice-my-neighbor-alice",
    "APE-USD":     "ape-apecoin",
    "API3-USD":    "api3-api3",
    "APR-USD":     "apr-apriori",
    "APT-USD":     "apt-aptos",
    "ASTER-USD":   "aster-aster",
    "ATOM-USD":    "atom-cosmos",
    "AUDIO-USD":   "audio-audius",
    "AXL-USD":     "ico-axelar",
    "AXS-USD":     "axs-axie-infinity",
    "BARD-USD":    "bard-lombard",
    "BCH-USD":     "bch-bitcoin-cash",
    "BERA-USD":    "bera-berachain",
    "BIO-USD":     "bio-bio-protocol",
    "BLUR-USD":    "blur-blur",
    "BLZ-USD":     "blz-bluzelle",
    "BOBA-USD":    "boba-boba-network",
    "BTRST-USD":   "btrst-braintrust",
    "CFG-USD":     "cfg-centrifuge",
    "CHZ-USD":     "chz-chiliz",
    "CLANKER-USD": "clanker-tokenbot",
    "COMP-USD":    "comp-compoundd",
    "CRO-USD":     "cro-cryptocom-chain",
    "CRV-USD":     "crv-curve-dao-token",
    "CTSI-USD":    "ctsi-cartesi",
    "DASH-USD":    "dash-dash",
    "DEXT-USD":    "dext-dextools",
    "DRIFT-USD":   "drift-drift-protocol",
    "EDGE-USD":    "edge-edgex",
    "EIGEN-USD":   "eigen-eigenlayer",
    "ELA-USD":     "ela-elastos",
    "ENA-USD":     "ena-ethena",
    "ETC-USD":     "etc-ethereum-classic",
    "EUL-USD":     "eul-euler",
    "FARM-USD":    "farm-harvest-finance",
    "FARTCOIN-USD": "fartcoin-fartcoin",
    "FIDA-USD":    "fida-bonfida",
    "FIL-USD":     "fil-filecoin",
    "FIS-USD":     "fis-stafi",
    "FLOCK-USD":   "flock-flockio",
    "FLOW-USD":    "flow-flow",
    "FORT-USD":    "fort-forta",
    "FORTH-USD":   "forth-ampleforth-governance-token",
    "GFI-USD":     "gfch-goldfinch",
    "GMT-USD":     "gmt-stepn",
    "GNO-USD":     "gno-gnosis",
    "GRT-USD":     "grt-the-graph",
    "GTC-USD":     "gtc-gitcoin",
    "HBAR-USD":    "hbar-hedera-hashgraph",
    "HIGH-USD":    "high-highstreet-token",
    "HYPE-USD":    "hype-hyperliquid",
    "HYPER-USD":   "hyper-hyperlane",
    "ICP-USD":     "icp-internet-computer",
    "IMX-USD":     "imx-immutable-x",
    "INDEX-USD":   "index-index-cooperative",
    "INJ-USD":     "inj-injective-protocol",
    "INX-USD":     "inx-infinex",
    "IO-USD":      "io-ionet",
    "IOTX-USD":    "iotx-iotex",
    "IP-USD":      "ip-story",
    "IRYS-USD":    "irys-irys",
    "KITE-USD":    "kite-kite-ai",
    "KNC-USD":     "knc-kyber-network",
    "KSM-USD":     "ksm-kusama",
    "KTA-USD":     "kta-keeta",
    "L3-USD":      "l3-layer3",
    "LAYER-USD":   "layer-solayer",
    "LCX-USD":     "lcx-lcx",
    "LDO-USD":     "ldo-lido-dao",
    "LRC-USD":     "lrc-loopring",
    "LTC-USD":     "ltc-litecoin",
    "MASK-USD":    "mask-mask-network",
    "MATH-USD":    "math-math",
    "MET-USD":     "met-meteora",
    "METIS-USD":   "metis-metis-token",
    "MLN-USD":     "mln-enzyme",
    "MON-USD":     "mon-monad",
    "MORPHO-USD":  "morpho-morpho",
    "NCT-USD":     "nct-polyswarm",
    "NEAR-USD":    "near-near-protocol",
    "NMR-USD":     "nmr-numeraire",
    "NOM-USD":     "nom-nomina",
    "OP-USD":      "op-optimism",
    "ORCA-USD":    "orca-orca",
    "OSMO-USD":    "osmo-osmosis",
    "PENDLE-USD":  "pendle-pendle",
    "PERP-USD":    "perp-perpetual-protocol",
    "PIRATE-USD":  "pirate-pirate-nation-token",
    "PLU-USD":     "plu-pluton",
    "PLUME-USD":   "plume-plume",
    "PNG-USD":     "png-pangolin",
    "PNUT-USD":    "pnut-peanut-the-squirrel",
    "POL-USD":     "pol-polygon-ecosystem-token",
    "POLS-USD":    "pols-polkastarter",
    "PRCL-USD":    "prcl-parcl",
    "PRIME-USD":   "prime-echelon-prime",
    "PRO-USD":     "pro-propy",
    "PROMPT-USD":  "prompt-wayfinder",
    "PROVE-USD":   "prove-succinct",
    "PUMP-USD":    "pump-pumpfun",
    "QNT-USD":     "qnt-quant",
    "RAD-USD":     "rad-radicle",
    "RARE-USD":    "rare-superrare",
    "RAVE-USD":    "rave-ravedao",
    "RECALL-USD":  "recall-recall-network",
    "RED-USD":     "red-redstone",
    "RENDER-USD":  "rndr-render-token",
    "REQ-USD":     "req-request-network",
    "RLS-USD":     "rls-rayls",
    "RSC-USD":     "rsc-researchcoin",
    "SAFE-USD":    "safe-safe",
    "SAPIEN-USD":  "sapien-sapien",
    "SD-USD":      "sd-stader",
    "SEAM-USD":    "seam-seamless",
    "SEI-USD":     "sei-sei",
    "SHIB-USD":    "shib-shiba-inu",
    "SKR-USD":     "skr-seeker",
    "SKY-USD":     "sky-sky",
    "SPK-USD":     "spk-spark",
    "SPX-USD":     "spx-spx6900",
    "SQD-USD":     "sqd-sqd",
    "STORJ-USD":   "storj-storj",
    "STRK-USD":    "strk-starknet",
    "STX-USD":     "stx-stacks",
    "SUI-USD":     "sui-sui",
    "SUP-USD":     "sup-superfluid",
    "SUPER-USD":   "super-superfarm",
    "SXT-USD":     "sxt-space-and-time",
    "SYND-USD":    "synd-syndicate",
    "SYRUP-USD":   "syrup-syrup-token",
    "TAO-USD":     "tao-bittensor",
    "THQ-USD":     "thq-theoriq",
    "TIA-USD":     "tia-celestia",
    "TIME-USD":    "time-wonderland",
    "TON-USD":     "toncoin-the-open-network",
    "TRAC-USD":    "trac-origintrail",
    "TRB-USD":     "trb-tellor",
    "TREE-USD":    "tree-tree-capital",
    "TROLL-USD":   "troll-troll2",
    "TRUMP-USD":   "trump-official-trump",
    "UNI-USD":     "uni-uniswap",
    "UP-USD":      "up-unitas",
    "USELESS-USD": "useless-useless-coin",
    "VVV-USD":     "vvv-venice-token",
    "W-USD":       "w-wormhole",
    "WIF-USD":     "wif-dogwifcoin",
    "WLD-USD":     "wld-worldcoin",
    "WLFI-USD":    "wlfi-official-world-liberty-financial",
    "XAN-USD":     "xan-anoma",
    "XLM-USD":     "xlm-stellar",
    "ZEC-USD":     "zec-zcash",
    "ZEN-USD":     "zen-horizen",
    "ZETA-USD":    "zeta-zetachain",
    "ZKP-USD":     "zkp-panther-protocol",
    "ZORA-USD":    "zora-zora",
}


def _coinbase_to_cp_id(product_id: Optional[str]) -> Optional[str]:
    """Coinbase product_id -> CoinPaprika coin id, or None if unmapped."""
    if not product_id:
        return None
    return _PRODUCT_TO_CP_ID.get(product_id)


def _is_disabled() -> bool:
    return os.environ.get("COINPAPRIKA_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ms_to_iso_date(ms: int) -> str:
    """Epoch ms -> YYYY-MM-DD UTC."""
    d = _dt.datetime.fromtimestamp(int(ms) / 1000, tz=_dt.timezone.utc)
    return d.strftime("%Y-%m-%d")


def _iso_to_ms(iso: str) -> Optional[int]:
    """CoinPaprika time_open like '2025-01-01T00:00:00Z' -> epoch ms.
    Returns None if unparseable."""
    if not iso:
        return None
    s = iso.replace("Z", "+00:00")
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


async def fetch_marketcap_history(
    product_id: str, start_ms: int, end_ms: int
) -> List[Tuple[int, float, float]]:
    """Daily marketcap timeseries for one Coinbase pid.

    Returns list of (ts_ms, market_cap, volume_24h) sorted ascending. Skips
    rows with a missing or non-positive market_cap. Empty list when:
      - pid is unmapped
      - COINPAPRIKA_DISABLED=1
      - HTTP non-200 or transport error
      - response shape unexpected

    Step A (2026-05-16): return tuple shape extended to carry volume_24h
    parsed from the response's `volume_24h` field. Missing -> 0.0 fill.
    """
    if _is_disabled():
        return []

    cp_id = _coinbase_to_cp_id(product_id)
    if cp_id is None:
        return []

    params = {
        "start":    _ms_to_iso_date(start_ms),
        "interval": "1d",
    }
    url = _HISTORY_URL.format(cp_id=cp_id)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, params=params)
    except Exception as e:
        logger.warning(
            "coinpaprika_marketcap history HTTP error pid=%s: %r", product_id, e
        )
        return []

    if resp.status_code != 200:
        logger.warning(
            "coinpaprika_marketcap history non-200 pid=%s status=%d",
            product_id, resp.status_code,
        )
        return []

    try:
        body = resp.json()
    except Exception:
        return []

    if not isinstance(body, list):
        return []

    rows: List[Tuple[int, float, float]] = []
    for entry in body:
        if not isinstance(entry, dict):
            continue
        mc = entry.get("market_cap")
        if mc is None:
            continue
        try:
            mc_f = float(mc)
        except (TypeError, ValueError):
            continue
        if mc_f <= 0.0:
            continue
        ts_ms = _iso_to_ms(entry.get("timestamp"))
        if ts_ms is None:
            continue
        try:
            vol_f = float(entry.get("volume_24h") or 0.0)
        except (TypeError, ValueError):
            vol_f = 0.0
        rows.append((ts_ms, mc_f, vol_f))
    rows.sort(key=lambda r: r[0])
    return rows


async def fetch_supply_snapshot(
    product_id: str,
) -> Optional[Tuple[float, float, Optional[float]]]:
    """Current-ticker supply snapshot for one Coinbase pid.

    Returns (circulating_supply, total_supply, max_supply_or_None) or None on
    any failure (unmapped pid, disabled, non-200, malformed body, missing
    circulating/total).

    max_supply is None for tokens with no fixed cap (e.g. ETH) — the
    distinction matters because it changes how FDV is interpreted downstream.

    Endpoint: GET /v1/tickers/{cp_id}  (no key, free tier).
    """
    if _is_disabled():
        return None

    cp_id = _coinbase_to_cp_id(product_id)
    if cp_id is None:
        return None

    url = _TICKER_URL.format(cp_id=cp_id)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
    except Exception as e:
        logger.warning(
            "coinpaprika_marketcap ticker HTTP error pid=%s: %r", product_id, e
        )
        return None

    if resp.status_code != 200:
        logger.warning(
            "coinpaprika_marketcap ticker non-200 pid=%s status=%d",
            product_id, resp.status_code,
        )
        return None

    try:
        body = resp.json()
    except Exception:
        return None

    if not isinstance(body, dict):
        return None

    try:
        circ  = float(body["circulating_supply"])
        total = float(body["total_supply"])
    except (KeyError, TypeError, ValueError):
        return None

    max_raw = body.get("max_supply")
    max_supply: Optional[float]
    if max_raw is None:
        max_supply = None
    else:
        try:
            max_supply = float(max_raw)
        except (TypeError, ValueError):
            max_supply = None

    return (circ, total, max_supply)
