"""Turn "what was loaded" into one stable, diagnosable identity string.

Blocker 1 of the 2026-09-28 controlling document: `cnn_scans` and `trades` carry no model or
config identity, so no stored number can be attributed to a version. That is why a
1,582-trade PnL figure silently mixed model eras, and why the peer session's rule — "agent tag
is not model provenance" — has to be enforced by data rather than by memory.

Pure: reads only the files it is given, and nothing else. No database, no clock, no globals.
Persisting the result is a separate concern, because a schema migration carries a different
kind of risk than a hash.

Design stance: **refuse rather than guess.** A fingerprint that is wrong is worse than no
fingerprint, because it invites attribution that cannot be justified. So a missing file, an
empty input set, or a directory raises instead of yielding a plausible-looking digest.

LIMITATION, found by running this against the live artifacts and worth stating loudly: this
identifies what was **configured**, not what actually **ran**. The live config reports
`MC_FILTERS=ci`, yet commit 589b571 deleted the import that registered that filter, so for two
months `ci` was requested and never executed. A config fingerprint would have stamped
"ci requested" on every one of those runs and looked perfectly consistent. Provenance by
configuration is necessary and NOT sufficient: pair it with an effective-behaviour attestation
such as `agents.mc.registry.chain_health()`, which reports what resolved rather than what was
asked for.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping

_READ_CHUNK = 1 << 20


def _file_digest(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"provenance: artifact not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"provenance: not a file: {path}")
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return {"path": os.path.basename(path), "sha256": h.hexdigest(), "bytes": size}


def _typed(value: Any) -> str:
    """Encode a config value with its TYPE, so `1`, `True`, `1.0` and `"1"` cannot collide.

    `True == 1` in Python, and `str()` flattens `1` and `1.0` on some paths, so a naive
    encoding lets a bool alias an int. The peer session found that exact aliasing in another
    artifact path, where `schema_version=True` compared equal to `1`.
    """
    return "%s:%r" % (type(value).__name__, value)


def artifact_fingerprint(paths: List[str], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Identity of a set of model artifacts plus the config that governs their use.

    Returns `{"digest", "artifacts", "config"}`. `digest` is `sha256:<hex>` over the per-file
    digests and the typed config pairs; `artifacts` lists each file's own digest and size so a
    mismatch can be localised rather than merely detected — a bare digest tells you two runs
    differ and never why.

    Order-independent in `paths`: supplying the same artifacts in a different order must not
    mint a new identity, or routine refactors would look like model changes.

    The CALLER chooses which config keys matter. That keeps this function honest — it cannot
    know which settings reach inference — and makes the choice reviewable at the callsite
    instead of buried here.
    """
    if not paths:
        raise ValueError("provenance: refusing to fingerprint an empty artifact set")

    artifacts = sorted((_file_digest(p) for p in paths), key=lambda d: d["sha256"])
    config_pairs = sorted((str(k), _typed(v)) for k, v in dict(config).items())

    h = hashlib.sha256()
    for entry in artifacts:
        h.update(entry["sha256"].encode("ascii"))
        h.update(b"\x00")
    h.update(b"\x01")
    h.update(json.dumps(config_pairs, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    return {
        "digest": "sha256:" + h.hexdigest(),
        "artifacts": artifacts,
        "config": dict(config),
    }
