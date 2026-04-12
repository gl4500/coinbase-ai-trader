"""
Coinbase Advanced Trade API client — CDP JWT auth (current standard as of Feb 2025).

Supports both CDP key formats:

  OLD format (portal.cdp.coinbase.com, downloaded JSON has 'name' field):
    COINBASE_API_KEY_NAME=organizations/xxx/apiKeys/yyy
    COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----\\n...\\n-----END EC PRIVATE KEY-----\\n
    → Uses ES256 JWT

  NEW format (portal.cdp.coinbase.com, downloaded JSON has 'id' field):
    COINBASE_API_KEY_NAME=<uuid from 'id' field>
    COINBASE_API_PRIVATE_KEY=<base64 string from 'privateKey' field>
    → Uses EdDSA JWT (Ed25519)

JWT tokens expire after 2 minutes — this client regenerates them per-request.
"""
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from config import config

logger = logging.getLogger(__name__)

_BASE = "https://api.coinbase.com/api/v3/brokerage"

# Products whose order book endpoint returned 404 — skip future calls
_NO_BOOK: set = set()


# ── JWT auth ──────────────────────────────────────────────────────────────────

def _load_private_key(private_key_str: str):
    """
    Load a Coinbase CDP private key in either format:
      - PEM  (old format): starts with '-----BEGIN ...'  → EC P-256, use ES256
      - Raw base64 (new format): 64-byte Ed25519 key     → use EdDSA
    Returns (private_key_object, algorithm_string).
    """
    import base64
    from cryptography.hazmat.primitives import serialization

    key_str = private_key_str.replace("\\n", "\n").strip()

    # ── Old format: PEM ───────────────────────────────────────────────────────
    if key_str.startswith("-----BEGIN"):
        private_key = serialization.load_pem_private_key(key_str.encode(), password=None)
        return private_key, "ES256"

    # ── New format: raw base64 ────────────────────────────────────────────────
    try:
        raw = base64.b64decode(key_str)
    except Exception:
        raise ValueError(
            "COINBASE_API_PRIVATE_KEY is not valid PEM or base64.\n"
            "Set it to the exact 'privateKey' value from the JSON downloaded\n"
            "at portal.cdp.coinbase.com → API Keys."
        )

    if len(raw) == 64:
        # Ed25519: CDP provides 64-byte key (32-byte seed + 32-byte public)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private_key = Ed25519PrivateKey.from_private_bytes(raw[:32])
        return private_key, "EdDSA"

    if len(raw) == 32:
        # Raw EC P-256 private scalar
        from cryptography.hazmat.primitives.asymmetric.ec import derive_private_key, SECP256R1
        private_key = derive_private_key(int.from_bytes(raw, "big"), SECP256R1())
        return private_key, "ES256"

    raise ValueError(
        f"Unrecognised private key: {len(raw)} raw bytes. "
        "Expected PEM, 64-byte Ed25519, or 32-byte EC P-256."
    )


def _generate_jwt(method: str, path: str) -> str:
    """Generate a short-lived JWT for one Coinbase API request."""
    try:
        import jwt as pyjwt

        key_name = config.coinbase_api_key
        private_key, algorithm = _load_private_key(config.coinbase_api_secret)

        uri = f"{method} api.coinbase.com{path}"
        now = int(time.time())
        token = pyjwt.encode(
            {
                "sub": key_name,
                "iss": "cdp",
                "nbf": now,
                "exp": now + 120,
                "uri": uri,
            },
            private_key,
            algorithm=algorithm,
            headers={"kid": key_name, "nonce": uuid.uuid4().hex},
        )
        return token
    except Exception as e:
        logger.error(f"JWT generation failed: {e}")
        raise RuntimeError(
            "Could not generate Coinbase JWT. "
            "Check COINBASE_API_KEY_NAME and COINBASE_API_PRIVATE_KEY in .env"
        ) from e


def _auth_headers(method: str, path: str) -> Dict[str, str]:
    token = _generate_jwt(method, path)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _get(path: str, params: Optional[Dict] = None) -> Any:
    full = f"/api/v3/brokerage{path}"
    url  = f"https://api.coinbase.com{full}"
    hdrs = _auth_headers("GET", full) if config.has_credentials else {}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=hdrs, params=params)
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, payload: Dict) -> Any:
    full = f"/api/v3/brokerage{path}"
    url  = f"https://api.coinbase.com{full}"
    body = json.dumps(payload)
    hdrs = _auth_headers("POST", full)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=hdrs, content=body)
        resp.raise_for_status()
        return resp.json()


# ── Products ──────────────────────────────────────────────────────────────────

async def get_products(product_type: str = "SPOT") -> List[Dict]:
    """Fetch all products, paginating through every page via cursor."""
    all_products: List[Dict] = []
    cursor: Optional[str] = None
    for _ in range(10):  # safety cap — 10 pages × 250 = 2500 products max
        params: Dict[str, Any] = {"product_type": product_type, "limit": 250}
        if cursor:
            params["cursor"] = cursor
        data = await _get("/products", params)
        batch = data.get("products", [])
        all_products.extend(batch)
        pagination = data.get("pagination", {})
        if pagination.get("has_next") and pagination.get("next_cursor"):
            cursor = pagination["next_cursor"]
        else:
            break
    return all_products


async def get_product(product_id: str) -> Optional[Dict]:
    try:
        return await _get(f"/products/{product_id}")
    except Exception as e:
        logger.warning(f"get_product({product_id}): {e}")
        return None


async def get_best_bid_ask(product_ids: List[str]) -> Dict[str, Dict]:
    """Batch fetch best bid/ask.  Returns dict keyed by product_id."""
    try:
        # Coinbase accepts repeated query params
        params = [("product_ids", pid) for pid in product_ids]
        full   = "/api/v3/brokerage/best_bid_ask"
        url    = f"https://api.coinbase.com{full}"
        hdrs   = _auth_headers("GET", full) if config.has_credentials else {}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=hdrs, params=params)
            resp.raise_for_status()
            data = resp.json()
        result: Dict[str, Dict] = {}
        for entry in data.get("pricebooks", []):
            pid  = entry.get("product_id", "")
            bids = entry.get("bids", [])
            asks = entry.get("asks", [])
            bid  = float(bids[0]["price"]) if bids else None
            ask  = float(asks[0]["price"]) if asks else None
            result[pid] = {
                "bid":   bid,
                "ask":   ask,
                "price": (bid + ask) / 2 if bid and ask else bid or ask,
            }
        return result
    except Exception as e:
        logger.warning(f"get_best_bid_ask: {e}")
        return {}


# ── Candles ───────────────────────────────────────────────────────────────────

async def get_candles(
    product_id:  str,
    granularity: str = "ONE_HOUR",
    limit:       int = 100,
) -> List[Dict]:
    """
    Fetch OHLCV candles.
    Granularity options: ONE_MINUTE FIVE_MINUTE FIFTEEN_MINUTE THIRTY_MINUTE
                         ONE_HOUR TWO_HOUR SIX_HOUR ONE_DAY
    Returns list sorted oldest → newest with keys: start, open, high, low, close, volume.
    """
    secs  = {"ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
             "THIRTY_MINUTE": 1800, "ONE_HOUR": 3600,
             "TWO_HOUR": 7200, "SIX_HOUR": 21600, "ONE_DAY": 86400}
    end   = int(time.time())
    start = end - secs.get(granularity, 3600) * min(limit, 300)
    try:
        data = await _get(
            f"/products/{product_id}/candles",
            {"start": str(start), "end": str(end), "granularity": granularity},
        )
        candles = []
        for c in data.get("candles", []):
            candles.append({
                "start":  int(c["start"]),
                "open":   float(c["open"]),
                "high":   float(c["high"]),
                "low":    float(c["low"]),
                "close":  float(c["close"]),
                "volume": float(c["volume"]),
            })
        return sorted(candles, key=lambda x: x["start"])
    except Exception as e:
        logger.warning(f"get_candles({product_id}): {e}")
        return []


# ── Order book ────────────────────────────────────────────────────────────────

async def get_orderbook(product_id: str, limit: int = 10) -> Dict:
    if product_id in _NO_BOOK:
        return {"bids": [], "asks": []}
    try:
        data = await _get(f"/products/{product_id}/book", {"limit": str(limit)})
        pb   = data.get("pricebook", {})
        return {
            "bids": [{"price": float(b["price"]), "size": float(b["size"])}
                     for b in pb.get("bids", [])],
            "asks": [{"price": float(a["price"]), "size": float(a["size"])}
                     for a in pb.get("asks", [])],
        }
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e):
            _NO_BOOK.add(product_id)
            logger.debug(f"get_orderbook({product_id}): no book available — skipping future calls")
        else:
            logger.debug(f"get_orderbook({product_id}): {e}")
        return {"bids": [], "asks": []}


# ── Accounts ──────────────────────────────────────────────────────────────────

async def get_accounts() -> List[Dict]:
    try:
        data = await _get("/accounts")
        return [
            {
                "uuid":      a.get("uuid"),
                "currency":  a.get("currency"),
                "available": float(a.get("available_balance", {}).get("value", 0)),
                "hold":      float(a.get("hold", {}).get("value", 0)),
            }
            for a in data.get("accounts", [])
        ]
    except Exception as e:
        logger.warning(f"get_accounts: {e}")
        return []


async def get_usd_balance() -> float:
    accounts = await get_accounts()
    for a in accounts:
        if a["currency"] in ("USD", "USDC"):
            return a["available"]
    return 0.0


# ── Orders ────────────────────────────────────────────────────────────────────

async def place_limit_order(
    product_id:  str,
    side:        str,
    base_size:   float,
    limit_price: float,
    post_only:   bool = False,
) -> Dict:
    return await _post("/orders", {
        "client_order_id": str(uuid.uuid4()),
        "product_id":      product_id,
        "side":            side.upper(),
        "order_configuration": {
            "limit_limit_gtc": {
                "base_size":   str(round(base_size, 8)),
                "limit_price": str(round(limit_price, 2)),
                "post_only":   post_only,
            }
        },
    })


async def place_market_order(
    product_id: str,
    side:       str,
    quote_size: float,
) -> Dict:
    return await _post("/orders", {
        "client_order_id": str(uuid.uuid4()),
        "product_id":      product_id,
        "side":            side.upper(),
        "order_configuration": {
            "market_market_ioc": {
                "quote_size": str(round(quote_size, 2)),
            }
        },
    })


async def cancel_orders(order_ids: List[str]) -> Dict:
    return await _post("/orders/batch_cancel", {"order_ids": order_ids})


async def get_orders(
    product_id:   Optional[str]       = None,
    order_status: Optional[List[str]] = None,
    limit:        int                  = 100,
) -> List[Dict]:
    params: Dict[str, Any] = {"limit": str(limit)}
    if product_id:
        params["product_id"] = product_id
    if order_status:
        params["order_status"] = order_status
    try:
        data = await _get("/orders/historical/batch", params)
        return data.get("orders", [])
    except Exception as e:
        logger.warning(f"get_orders: {e}")
        return []
