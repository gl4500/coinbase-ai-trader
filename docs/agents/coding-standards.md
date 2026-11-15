# polymarket_app — Python function standard (normative)

Mandatory reference for `AGENTS.md` (Code Style). Read it before writing, specifying or reviewing
Python in this repository. Operator standard, 2026-05-17, set while reviewing the XGB v4 spec, which
described public functions and a 250-line training script without showing how they decompose.

## The rule

1. **Type hints on every signature**, return type included. No `Any` unless genuinely unknown. Use
   `from __future__ import annotations` for forward references.
2. **Pure functions over in-place buffer mutation.** Don't write `helper(input, out_buffer, offset)`
   that mutates `out_buffer[offset:offset+10]`. Return the computed values and let the caller assemble
   them. (v3's `_stats_from_candles(candles, stat_offset, out)` is the anti-example.)
3. **Single responsibility per function.** `_slope`, `_pct_rank` and `_delta_at` each do one thing and
   are independently testable. A god-function such as a 250-line `main()` decomposes into named helpers
   (`_load_x`, `_build_y`, `_train_z`, `_save_artifacts`, with `main` as orchestrator).
4. **Derived constants over hard-coded duplicates.** `N_CHANNELS = len(_CHANNEL_FIELDS)`, not
   `N_CHANNELS = 5`, so adding a channel cannot leave the count out of date.
5. **Explicit input-contract docstrings.** Say what raises, what returns zero or empty, and what is
   required. A reader should not need the body to know the contract.
6. **No module-level mutable state.** No globals modified at run time. Constants are fine; mutable
   singletons are not.
7. **Public vs private clearly marked.** `_leading_underscore` for internal helpers. Public API at the
   top of the module after constants, helpers below.

## Where it applies

- **Specs:** every new module gets a "concrete function signatures" section: signatures, docstrings
  and decomposition. A spec that only says "~250 LOC, mirrors the structure of X" is incomplete.
- **Implementation:** follow the rule as written, even where nearby existing code is messier.
- **Review:** flag violations as findings and propose smaller helpers. Existing code is not swept
  as a side effect of an unrelated task; a sweep is its own task.
