# Provenance Part 2 — Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Every NEW `cnn_scans` row and every NEW `trades` entry row carries the identity of
the model artifacts and decision config that produced it; historical rows stay NULL.

**Architecture:** four loosely coupled layers. `xgb_signal` fingerprints exactly what it loaded
(per load, so `force_reload` mints a new identity). `mc.registry` reports the EFFECTIVE filter
chain. `provenance.decision_provenance` (pure) combines driver + shadow fingerprints with the
typed decision config into one `sha256:` digest. `database` persists the digest on rows plus a
`model_provenance` registry (digest → detail) so a digest can be traced to files and values.
`cnn_agent` threads it through `save_cnn_scan` and `book.buy → open_trade`.

**Spec:** blocker 1 of `docs/specs/2026-09-27-strategy-evidence-and-decision.md` ("no model hash,
config, or scan_id on trade rows, so no number is attributable to a version"); part 1 is
`services/provenance.artifact_fingerprint` (CHANGELOG 58.95).

## Global Constraints

- Additive schema only: new nullable columns `cnn_scans.model_provenance`,
  `trades.model_provenance` (CREATE + `ALTER TABLE ... ADD COLUMN` migration), new table
  `model_provenance`. No backfill — historical rows are unattributed, never guessed.
- A malformed digest is REJECTED (`ValueError`) by the writers, never stored: a wrong identity
  is worse than none.
- No driver loaded ⇒ provenance `None` (unattributed). Never the string "unknown".
- Provenance computation must never raise into the scan loop (invariant #14 style): failure ⇒
  `None` + `logger.exception`.
- Closes inherit the entry row's provenance; `close_trade`'s already-closed `UNKNOWN` insert
  stays NULL.

## Decision config fingerprinted (the CALLER chooses, per part 1)

`model_backend`, `cnn_buy_threshold`, `cnn_sell_threshold`, `xgb_v45_thresh_up`,
`xgb_v45_thresh_down`, `mc_filters_effective` (from the registry, not `MC_FILTERS`),
`_CNN_MAX_FRAC`, `_CNN_STOP_LOSS_PCT`, `_CNN_ATR_TRAIL_MULT/MIN/MAX`, `_CNN_MAX_HOLD_SECS`,
`_P_DOWN_EXIT_THRESHOLD`, `_P_DOWN_STALE_MS`, and `exit_thresholds` `FEE_RATE`,
`GIVEBACK_FRAC`, `LARGE_POSITION_FRAC`, `LARGE_POSITION_FLOOR`, `MAX_DOLLAR_GIVEBACK_FRAC`,
`MAX_LOSS_FRAC_OF_CAPITAL`.

## Review Focus

1. `force_reload` must clear the old fingerprint — a stale identity after a model swap is the
   exact failure this exists to prevent. → `test_force_reload_mints_new_fingerprint`.
2. Calibrator present on disk but REJECTED (feature_set mismatch) must not enter the
   fingerprint. → `test_rejected_calibrator_not_fingerprinted`.
3. `MC_FILTERS=ci` with `ci` unregistered must fingerprint as an EMPTY effective chain.
   → `test_effective_chain_excludes_unregistered`.
4. Writers reject `"unknown"` / truncated digests. → `test_*_rejects_malformed_provenance`.
5. Provenance failure never breaks a scan. → `test_provenance_failure_saves_scan_unattributed`.

## Tasks

1. **xgb_signal fingerprints** — `_fingerprint_v3`, `_fingerprint_v45`, `loaded_fingerprints()`;
   set on successful load (calibrator included only if actually adopted), cleared on attempt
   start and in `force_reload`. Tests in `tests/test_xgb_signal.py`.
2. **Effective MC chain** — `registry.effective_filter_names() -> List[str]`. Tests in
   `tests/agents/mc/`.
3. **Pure composition** — `provenance.decision_provenance(fingerprints, config) -> Optional[dict]`
   and `provenance.validate_digest(value) -> str`. Tests in `tests/test_provenance.py`.
4. **Persistence** — columns + registry table + `record_provenance`, `save_cnn_scan` and
   `open_trade(..., model_provenance=None)` validation. Tests in
   `tests/test_provenance_persistence.py`.
5. **Wiring** — `CoinbaseCNNAgent._current_provenance()`; scan dict + `book.buy(...,
   model_provenance=)`; registry row written when the digest changes. Tests in
   `tests/test_provenance_persistence.py`. Docs: CHANGELOG, CLAUDE.md invariant.

## Known limits

- File bytes are hashed right after load; a swap inside that window could mismatch. A later
  `force_reload` re-fingerprints.
- Identifies what was CONFIGURED and LOADED, not every code path that ran; code changes are
  identified by git, not here.
