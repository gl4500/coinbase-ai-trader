"""Input identity for published label endpoints.

`data_id` binds **everything the exit decision reads**, not merely the prices. That
distinction is load-bearing: `labels._simulate_one` consumes `atr_pcts[i]` directly to
set its trail threshold, so an OHLC-only hash would let the ATR column — the very
column carrying the known contemporaneous-causality defect — be altered or recomputed
without invalidating a single binding.

Identical prices on different clocks must not share an identity either, so the ordered
timestamps, the row count, the product and the **declared** bar duration are bound too.
A declared duration rather than an inferred one, because the next row may itself be
missing and spacing therefore cannot establish a bar's length.

Encoding is deterministic by construction: every float goes in as `float.hex()`, which
round-trips exactly, and non-finite values become explicit tokens rather than reaching
a JSON encoder that would either fail or emit a non-standard literal. A NaN in the ATR
column is legitimate input — the simulation falls back to its floor — so it must hash
consistently rather than abort.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

DATA_ID_VERSION = "endpoint_data_id_v1"

_NON_FINITE = {"nan": "nan", "inf": "inf", "-inf": "-inf"}


def _encode_float(value: Any) -> str:
    """Exact, deterministic, and explicit about non-finite values."""
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return float.hex(number)


def _encode_floats(values: Sequence[Any], field: str) -> list:
    if values is None:
        raise ValueError(f"{field} is required for the input identity")
    return [_encode_float(v) for v in values]


def _encode_ints(values: Sequence[Any], field: str) -> list:
    out = []
    for value in values:
        number = int(value)
        if number != value:
            raise ValueError(f"{field} must contain whole numbers, got {value!r}")
        out.append(number)
    return out


def build_data_id(
    *,
    product_id: str,
    bar_duration_ms: int,
    timestamps: Sequence[int],
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    atr_pcts: Sequence[float],
    feature_recipe: str,
    config: Mapping[str, Any],
) -> str:
    """Digest the identity of the inputs a published endpoint depends on.

    Covers the product, the declared bar duration, the ordered timestamps and row
    count, every array the simulation consumes — **including the ATR** — the feature
    recipe that produced the ATR, and the exit config that shapes every exit.

    Two frames with identical prices but different timestamps, a different declared
    duration, a different ATR column, a different feature recipe, or a different exit
    config all receive different identities.
    """
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValueError("product_id must be a nonempty string")
    if type(bar_duration_ms) is not int or bar_duration_ms <= 0:
        raise ValueError("bar_duration_ms must be a positive int, declared not inferred")
    if not isinstance(feature_recipe, str) or not feature_recipe.strip():
        raise ValueError("feature_recipe must be a nonempty string")
    if not isinstance(config, Mapping) or not config:
        raise ValueError("config is required: the exit parameters shape every endpoint")

    lengths = {len(timestamps), len(closes), len(highs), len(lows), len(atr_pcts)}
    if len(lengths) != 1:
        raise ValueError("every consumed array must have the same length as the frame")

    body = {
        "version": DATA_ID_VERSION,
        "product_id": product_id,
        "bar_duration_ms": bar_duration_ms,
        "row_count": len(closes),
        "feature_recipe": feature_recipe,
        "timestamps": _encode_ints(timestamps, "timestamps"),
        "closes": _encode_floats(closes, "closes"),
        "highs": _encode_floats(highs, "highs"),
        "lows": _encode_floats(lows, "lows"),
        # Bound deliberately: the trail threshold reads this column directly, so an
        # identity without it would not notice the input most likely to change.
        "atr_pcts": _encode_floats(atr_pcts, "atr_pcts"),
        "config": {key: _encode_float(config[key]) for key in sorted(config)},
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
