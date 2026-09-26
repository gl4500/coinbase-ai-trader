# Codex prerequisite review and maker fallback fixes

Branch: fix/validated-labels-maker-fallback, explicitly stacked on completed 7bd327c.
No push, merge, deployment, live database migration or backend restart is authorized by this work.

## Issue register
TaskCreate is unavailable; this ordered register preserves the repository Find-List-Fix workflow.

1. DONE (4 regression failures before fix; 27 tests pass): database.py pending selection and v2 write accessors mutate pending legacy rows. Filter and guard by current label_version; replace the contradictory tracker test.
2. DONE (tiny-price regression reproduced; 10 tracker tests pass): outcome_tracker.py rounds stored endpoint prices to six decimals. Preserve measured precision and prove tiny-price return reconstruction.
3. DONE (2 regression failures reproduced; 21 diagnostics tests pass): diagnostics.py scores prematurely resolved rows without a target-time gate and drops injected now. Gate current metrics/funnel on maturity and pass now through.
4. DONE (30 initial failures; 55 execution/client tests pass): order_executor.py cancel failure/partial-fill fallback can duplicate exposure. Require positive cancellation result plus final order reconciliation; never replace unknown or partially filled orders automatically. Preserve original order identity and report reconciliation needs. Validate placement responses.
5. DONE (11 invalid-price failures reproduced; 40 label/tracker tests pass): outcome_labels.py permits nonfinite/nonpositive target prices into scores. Reject malformed price inputs as non-scoring UNAVAILABLE.
6. DONE (full suite, scoped lint/format and mandatory commit hook pass): full non-slow suite, scoped lint/format, final review and handoff.

## Review evidence
The original six prerequisite test files pass independently: 69 passed in 5.64s.
This does not validate the claimed legacy immutability: the original tracker test explicitly expected legacy pending rows to be converted to v2.

## Maker design
Cancel acknowledgement initiates cancellation; it is not proof of final order state.
Read the exact order by ID, require terminal CANCELLED and explicit zero fills before any market replacement.
Conservatively stop on partial fills rather than automatically topping up: current callers cannot account for split fills safely.
Full-fill races return the original maker order; ambiguous states return reconciliation_required with that ID.
No blind retries after an uncertain market placement.

Official references:
- https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/orders/cancel-order
- https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/orders/get-order
- https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/orders/create-order

## Scope and remaining work

The completed patch is a prerequisite repair and maker duplicate-order safeguard,
not evidence of a profitable strategy or a complete exchange accounting system.
Partial fills deliberately leave a reconciliation requirement; callers currently
close/update the paper book before confirmed exchange execution and may ignore
returned error details. They must be repaired before live promotion. Remaining
order: shared executor lifecycle (finding 1), entry/exit routing symmetry and
fill-confirmed position accounting (findings 2/3), then immutable signal/order/fill
provenance and fee-aware held-out strategy evaluation.

The as-of gate prevents future-target rows entering metrics. Diagnostics still
use the existing 60-second cache; this is not a historical database replay API.
Labels still depend on the quality and completeness of locally stored candles.
No winning-strategy claim follows from these changes.

Repository bookkeeping: no TaskCreate/Skill invocation tools are exposed in this
session, so tasks are recorded here and relevant installed SKILL.md instructions
were read directly. The implementation note above mirrors the new CLAUDE.md rule
inside the allowed project scope. No external Claude memory files were modified.
Test cleanup is limited to this task's processes/artifacts; blanket Python process
termination would disrupt the live backend or the session bridge and is not used.

## Final validation

- Original prerequisite modules before changes: 69 passed.
- Final full suite: `python -m pytest backend/tests -m "not slow and not integration" -q --tb=short`
  -> 1414 passed, 65 skipped, 1 deselected, 1 xfailed, 2 xpassed,
  14 sklearn feature-name warnings, 318.06 seconds. No failures.
- This adds 50 collected cases over the 7bd327c prerequisite branch.
- Ruff check and format check: all 14 changed Python files pass.
- git diff --check: clean.
- Expected red-to-green evidence is retained in the ignored main-project
  `.coordination/` logs: legacy, precision, diagnostics, maker, invalid-price.
- The mandatory pre-commit hook repeated the same suite successfully: 1414 passed,
  65 skipped, 1 deselected, 1 xfailed, 2 xpassed; commit allowed.

Claude reported PR #58 now exists for 7bd327c and PR #57 for the macro-regime
branch. These repairs are a local follow-up on a separate branch; neither PR was
modified by Codex. Claude was notified that PR #58 lacks the review corrections.
