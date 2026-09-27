"""Pure requested-pair accounting; no fold validation, I/O, or selection authority.

The caller must declare pairs from the original request before input filtering.
This module cannot reconstruct or attest an omitted original request.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

Pair = tuple[str, int]
_STATUSES = {"pending", "running", "completed", "excluded", "error"}
_FIELDS = {"pid", "horizon", "status", "reason_code", "observed"}


def _pair(pid, horizon) -> Pair:
    if not isinstance(pid, str) or not pid.strip() or pid != pid.strip():
        raise ValueError("product identity must be a nonempty unpadded string")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer without coercion")
    return pid, horizon


def declare_pairs(pids: Iterable[str], horizons: Iterable[int]) -> tuple[Pair, ...]:
    """Materialize once, validate, then deduplicate the complete request product."""
    if isinstance(pids, (str, bytes)) or isinstance(horizons, (str, bytes)):
        raise ValueError("products and horizons must be collections")
    products, periods = list(pids), list(horizons)
    if not products or not periods:
        raise ValueError("empty research request cannot establish coverage")
    return tuple(sorted({_pair(pid, horizon) for pid in products for horizon in periods}))


@dataclass(frozen=True)
class CoverageSummary:
    requested_count: int
    pending_count: int
    running_count: int
    completed_count: int
    excluded_count: int
    error_count: int
    dispositions_complete: bool
    blockers: tuple[str, ...]

    def __post_init__(self):
        counts = (
            self.pending_count,
            self.running_count,
            self.completed_count,
            self.excluded_count,
            self.error_count,
        )
        if (
            type(self.requested_count) is not int
            or self.requested_count <= 0
            or any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != self.requested_count
        ):
            raise ValueError("coverage counts must be nonnegative integers summing to request")
        complete = not (self.pending_count or self.running_count)
        if type(self.dispositions_complete) is not bool or self.dispositions_complete != complete:
            raise ValueError("disposition completeness contradicts pending/running counts")
        if not isinstance(self.blockers, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.blockers
        ):
            raise ValueError("blockers must be an immutable tuple of names")
        required = {"fold_leaf_evidence_not_validated"}
        if not complete:
            required.add("incomplete_pair_dispositions")
        if self.excluded_count:
            required.add("excluded_pairs")
        if self.error_count:
            required.add("errored_pairs")
        if not required.issubset(self.blockers):
            raise ValueError("coverage summary is missing required evidence blockers")

    @property
    def evaluation_validated(self) -> bool:
        return False  # Counts and lifecycle claims are not fold/leaf evidence.

    @property
    def deployment_eligible(self) -> bool:
        return False


def summarize_coverage(requested: Iterable[Pair], records: Iterable[dict]) -> CoverageSummary:
    """Require exactly one valid disposition per declared pair, including pending.

    A completed count records producer claims only. Terminal errors/exclusions can
    make accounting complete while leaving evaluation incomplete or invalid.
    """
    expected = set()
    for value in requested:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("invalid requested pair")
        pair = _pair(*value)
        if pair in expected:
            raise ValueError("duplicate requested pair")
        expected.add(pair)
    if not expected:
        raise ValueError("missing requested pair declaration")
    seen = set()
    counts = Counter()
    for row in records:
        if not isinstance(row, dict) or set(row) != _FIELDS:
            raise ValueError("invalid pair disposition fields")
        pair = _pair(row["pid"], row["horizon"])
        if pair not in expected:
            raise ValueError("unexpected pair disposition")
        if pair in seen:
            raise ValueError("duplicate pair disposition")
        status, reason, facts = row["status"], row["reason_code"], row["observed"]
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("invalid pair status")
        if not isinstance(facts, dict):
            raise ValueError("observed facts must be an object")
        try:
            json.dumps(facts, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("observed facts must be finite JSON data") from exc
        if status in {"excluded", "error"}:
            if not isinstance(reason, str) or not reason.strip() or not facts:
                raise ValueError("excluded/error pair requires reason code and observed facts")
        elif reason is not None:
            raise ValueError("nonfailure disposition cannot carry a failure reason")
        seen.add(pair)
        counts[status] += 1
    if seen != expected:
        raise ValueError("missing pair dispositions; absence is not success")
    complete = not (counts["pending"] or counts["running"])
    blockers = ["fold_leaf_evidence_not_validated"]
    if not complete:
        blockers.append("incomplete_pair_dispositions")
    if counts["excluded"]:
        blockers.append("excluded_pairs")
    if counts["error"]:
        blockers.append("errored_pairs")
    return CoverageSummary(
        requested_count=len(expected),
        pending_count=counts["pending"],
        running_count=counts["running"],
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        error_count=counts["error"],
        dispositions_complete=complete,
        blockers=tuple(blockers),
    )
