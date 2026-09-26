# Position and order lifecycle contract

**Status:** specification. No live behaviour changes here, and none are authorised by this document.
**Purpose:** the shared contract that execution findings 2 and 3 need before they can be fixed safely,
plus the two taker-surface defects found while reviewing.
**Authors:** drafted by the Claude session, reviewed and materially corrected by the parallel Codex
session over the shared session link on 2026-09-26. The implementation then went through two
further adversarial review rounds which found ten defects in it — six of presence-versus-validity
and four of state-versus-evidence consistency — recorded in
`backend/tests/test_position_lifecycle_adversarial.py`.

---

## 0. The pattern this exists to fix

Four separate places in the execution layer conflate **accepted** with **confirmed** — they treat "the
API call returned" as "the exchange did what we asked":

| Site | Conflation | Status |
|---|---|---|
| `order_executor.execute_maker_signal` | market fallback fired after an unconfirmed cancel; partial fill read as no fill | **fixed** (PR #60) |
| `order_executor.execute_market_order` | absent `success_response` → `order_id="unknown"`, written `status="live"`, returned `success: True` | open |
| `order_executor.cancel_order` | marks `canceled` and returns success without inspecting per-order results | open |
| `cnn_agent._check_risk_exits` / `exit_watcher.on_price_tick` | paper book closed **before** the exchange confirms; failure swallowed | open (finding 3) |

Four instances is a systemic pattern, not four unrelated bugs. Point-fixing each one leaves the next
caller free to reintroduce it, which is why the contract is the artifact rather than a fifth patch.

An `order_id` of `"unknown"` deserves specific mention: it is worse than a recorded failure, because the
row can never be matched to a fill. It is permanently unreconcilable by construction.

---

## 1. Two state machines, not one

The single biggest correction to the first draft: **order lifecycle and position exposure are separate
machines.** Collapsing them produces the nonsensical claim that a rejected *exit* means there is no
*position* — when in fact the position is still fully held and still at risk.

- An **order** is an instruction with a terminal outcome.
- A **position** is exposure that exists until something closes it.

A terminal order state never directly implies a position state. It is *evidence* that the position
machine consumes.

---

## 2. Order states

| State | Terminal | Meaning |
|---|---|---|
| `INTENT_CREATED` | no | We have decided to place an order and persisted that decision, **before** any network call. Carries a `client_order_id`. |
| `SUBMITTING` | no | A submission is in flight. No response yet. |
| `ACCEPTED` | no | The exchange acknowledged the order and returned **its own** identifier. |
| `WORKING_PARTIAL` | no | Partially filled, remainder still live on the book. |
| `FILLED` | yes | Fully filled. |
| `SETTLED_PARTIAL` | yes | Partially filled, no remainder working (cancelled or expired remainder). |
| `CANCELLED` | yes | Removed from the book with **zero** fills. |
| `REJECTED` | yes | The exchange refused it. No fill ever existed. |
| `UNKNOWN` | no | We cannot establish the state. Not a failure — an absence of knowledge. |

Three distinctions that the current code does not make:

1. **`CANCELLED` is not `REJECTED`.** A successful cancellation is us getting what we asked for; a
   rejection is the exchange refusing. Conflating them hides which side failed.
2. **`WORKING_PARTIAL` is not `SETTLED_PARTIAL`.** A live remainder can still fill, so any top-up sized
   against the current fill is racing that remainder. Only the terminal form has a stable remainder.
3. **`UNKNOWN` is a first-class state.** The present code has no way to say "I don't know", so it
   guesses — and every guess in the four sites above guesses *success*.

### `INTENT_CREATED` and the stable `client_order_id`

The first draft began at `ACCEPTED`, which leaves the crash-and-timeout window unmodelled: if the
process dies between submission and response, restart has no record that an order may exist, and the
natural recovery — submit again — doubles exposure. This is the same failure shape as finding 4.

Therefore: **persist `INTENT_CREATED` with a generated `client_order_id` before the network call.** On
restart, any `INTENT_CREATED` or `SUBMITTING` row is a reconciliation question, never a resubmission
trigger. The `client_order_id` is what makes the question answerable, because it lets us ask the
exchange "did you ever see this?" rather than guessing from timestamps.

---

## 3. Position states

| State | Meaning |
|---|---|
| `FLAT` | No exposure. |
| `OPENING` | An entry is in progress; exposure may or may not exist yet. |
| `OPEN` | Exposure confirmed at the intended size. |
| `OPEN_WITH_RESIDUAL` | Exposure confirmed, but smaller than intended (partial entry) or larger than zero after a partial exit. |
| `CLOSING` | An exit is in progress; exposure still exists until proven otherwise. |
| `CLOSED` | Flat, confirmed. **No qualifier — `CLOSED` always means zero exposure.** |
| `RECONCILIATION_REQUIRED` | Local belief and exchange truth disagree, or truth is unknown. |

`OPEN_WITH_RESIDUAL` replaces the first draft's `CLOSED_PARTIAL`. The old name implied closure had
happened; what actually exists is *remaining exposure*, which is the fact a risk system must act on.
Naming a state after the action that partly happened rather than the exposure that remains is how a
position gets forgotten.

### Legal position transitions

```
FLAT                    -> OPENING
OPENING                 -> OPEN | OPEN_WITH_RESIDUAL | FLAT | RECONCILIATION_REQUIRED
OPEN                    -> CLOSING
OPEN_WITH_RESIDUAL      -> CLOSING
CLOSING                 -> CLOSED | OPEN | OPEN_WITH_RESIDUAL | RECONCILIATION_REQUIRED
any                     -> RECONCILIATION_REQUIRED
RECONCILIATION_REQUIRED -> OPEN | OPEN_WITH_RESIDUAL | CLOSED | FLAT   (reconciler only, §7)
```

Two consequences worth stating because the current code violates both:

- `OPENING -> FLAT` is how a rejected **entry** resolves: no exposure was ever created.
- `CLOSING -> OPEN` is how a rejected **exit** resolves: the position is *still held*. It must never
  become `CLOSED`, and it must never be expressed as "no position". This is finding 3.

---

## 4. Evidence required for each transition

A transition without its evidence is not a transition. It is `RECONCILIATION_REQUIRED`.

| Entering | Required evidence |
|---|---|
| `INTENT_CREATED` | `client_order_id`, product, side, intended size, `execution_mode` (`paper`\|`live`), `run_id`, `config_version`, `model_hash`, `created_at` |
| `SUBMITTING` | the above plus `submitted_at` |
| `ACCEPTED` | exchange `order_id`, `accepted_at` |
| `FILLED` | exchange `order_id`, terminal status, `filled_size`, `avg_fill_price`, `fee`, `fill_id`(s), `observed_at` |
| `WORKING_PARTIAL` | confirmed fill evidence (`filled_size`, `avg_fill_price`, `fill_id`(s)) **and** positive evidence that the remainder is still live. **Not** a terminal status — requiring one here contradicted this state's own non-terminality in the first draft. |
| `SETTLED_PARTIAL` | confirmed fill evidence **plus** terminal confirmation that the remainder is gone (cancelled or expired), and the resulting `remaining_size` of zero working |
| `CANCELLED` | terminal status **and** explicit zero `filled_size` **and** zero `filled_value` |
| `REJECTED` | terminal status plus the exchange's reason |
| `OPEN` / `OPEN_WITH_RESIDUAL` / `CLOSED` | the order evidence above, plus the resulting position size |

**A placement or cancellation may be reported as success only when the exchange identifies the
order.** `"unknown"` is an error path, never an identifier. This closes `execute_market_order` and
`cancel_order` directly.

**Identification is necessary but not sufficient.** Three things must stay distinct, and collapsing any
two of them is the same class of bug as the four sites in §0:

| | Means | Does **not** mean |
|---|---|---|
| **Submission accepted** | the exchange has the order and named it | anything about fills |
| **Cancellation acknowledged** | the exchange received the cancel *request* | the order is off the book |
| **Cancellation terminally confirmed** | the order is terminal with a known final `filled_size` | — |

A cancellation **acknowledgement** never implies flat exposure and never grants permission to replace
the order. Only terminal confirmation does. PR #60 already enforces this in the maker path; the
contract generalises it so the next caller cannot reintroduce it.

---

## 5. Invariants

- **I1 — The paper book may not record a close until `CLOSED`.** Today `book.sell()` runs before
  `execute_live_exit` and the exception is swallowed, so a failed live exit leaves the ledger flat while
  the exchange still holds the position. Under this contract the position enters `CLOSING`, and only
  confirmed terminal fill evidence reaches `CLOSED`. *(finding 3)*
- **I2 — A live account must always be able to reach `CLOSING`.** A routing flag may choose maker
  versus taker; it may never decide whether an exit is *attempted*. Entries routing live while exits
  no-op is what makes an unexitable live position possible. *(finding 2)*
- **I3 — No automated action on a partial until split-fill accounting exists.** Promoted from PR #60's
  maker policy to the position layer, which is where it belongs.
- **I4 — No automatic retry while order or remainder state is `UNKNOWN`.** Retrying an unknown is how
  exposure doubles.
- **I5 — `RECONCILIATION_REQUIRED` blocks new entries for that product** and is never cleared by a
  retry. Only §7 clears it.
- **I6 — Idempotency is keyed on the exchange `fill_id`.** Reprocessing the same fill must be a no-op.
  Sequence numbers and timestamps are not identities.
- **I7 — Quantities compare as `Decimal` at the product's increment.** Never float equality. "Filled
  size equals held size" is meaningless without the increment.
- **I8 — Every row carries `run_id`, `model_hash`, `config_version` and `client_order_id`.**
  Exchange `order_id` and `fill_id` are **nullable until observed** and must never be fabricated or
  defaulted: an `INTENT_CREATED` row cannot carry an identifier the exchange has not yet issued, and
  the first draft's demand that *every* row carry them contradicted its own §4 evidence table. A
  placeholder in either field is the `"unknown"` defect wearing a different hat. Without the rest,
  the accounting reconciliation stays archaeology — the same provenance §5 of
  `2026-09-26-accounting-reconciliation.md` already requires.

---

## 6. Enable / disable / in-flight transition contract

Recorded separately at Codex's request, because bundling it into the late-binding containment fix
(#61) would have widened that change unsafely.

**What may change on enable/disable:**

- routing mode (maker versus taker)
- whether *new entries* are permitted

**What must survive across enable/disable/re-enable:**

- drawdown counters and the halt flag
- paper balance
- every open position's state and evidence

This is currently violated: `enable_trading` constructs a **new** `OrderExecutor`, so
`_dry_run_balance`, `_halted`, `_day_start_balance` and `_week_start_balance` reset on every cycle. PR
#61 fixed *which* executor the automated paths observe; it deliberately did **not** fix state
continuity, and that remains open, pinned by
`test_replacing_the_executor_still_discards_risk_state`.

**In-flight orders:** a mode change must never apply to an order already in `SUBMITTING` or `ACCEPTED`.
An order is routed by the mode in force when its `INTENT_CREATED` was written, and that mode is recorded
on the row. Disabling trading must not orphan an in-flight order — it continues to a terminal state and
its position consequence is applied normally, because the exposure is real whether or not we currently
want new exposure.

**Risk exits are never gated by the entry switch.** Verified already true on the scan path:
`_scan_cycle` withholds the executor from the entry leg when `is_trading` is false but passes it
unconditionally to `_check_risk_exits`. The contract makes that explicit rather than incidental.

---

## 7. Reconciler

Does not exist today. Nothing in the contract works without it.

**Deterministic auto-confirmation is allowed.** Where exchange evidence uniquely determines the state —
a terminal status with a matching `fill_id` and a quantity equal at the product increment — the
reconciler records the transition itself, citing that evidence. This is mechanical, not judgement, and
requiring a human for it would guarantee the reconciler is bypassed.

**Discretionary overrides require explicit, auditable authorisation.** Anything not uniquely determined
— conflicting records, missing fills, a quantity mismatch beyond the increment — needs a recorded
decision with an actor and a reason.

**It must run before trading is enabled.** The current database is what an unreconciled start looks
like: 52 open `trades` rows, 3 rows in `positions`, an empty CNN `positions_json`, 12 products with open
trades in no state at all, and TECH state frozen since 2026-05-17 still claiming 38 positions.

**There is nothing to reconcile against yet.** `orders` has **zero** rows, no `filled_size`, no
`fee_paid`, and `trades` has no order or fill column. Persisting intents, orders and fills is a
prerequisite for the reconciler, not a later refinement.

---

## 8. Sequencing

Findings 2 and 3 **cannot** be fixed before the reconciler exists, because both are about trusting
exchange truth over local belief, and today no exchange truth is recorded.

1. Provenance columns; intent, order and fill persistence *(no behaviour change)*
2. The pure transition validator — this contract as executable rules, no DB, no network
3. The reconciler in **report-only** mode
4. The state machine behind a flag, writing states without gating behaviour
5. I1 and I2 as behaviour changes, once 1–4 are proven
6. `execute_market_order` and `cancel_order` brought under §4's identification rule

**Step 6 is not gated on steps 1–5.** Those two repairs are *fail-closed*: they convert a false success
into an explicit error and place no orders that were not already being placed. They are independently
testable and reviewable without changing deployed routing, so gating them behind reconciliation would
delay a safety improvement for no benefit. What **is** gated on reconciliation is the full
exit-accounting integration — I1 and I2 — because those depend on exchange truth that is not yet
recorded.

Explicitly **out of scope** of this document: any change to live routing, the paper book, thresholds,
models, or strategy.

---

## 8b. Trusted caller boundary — what the validator cannot check

Added after two adversarial review rounds against the implementation, because the
rules below are weaker than they look and a reader should not mistake them for
verification.

The validator requires a linked `order_id` and a non-empty `fill_ids` list before
it will move a position, and requires a real boolean for the reconciler's
`exchange_terminal`. **None of that is verification.** This module cannot confirm
that an identifier corresponds to anything the exchange ever issued, because no
persisted correlation exists: `orders` has zero rows, `trades` has no order or
fill column, and nothing cross-checks a claimed `fill_id` against a real fill.

So "linked" currently means **"the caller supplied a well-formed identifier"**,
nothing stronger. A caller that fabricates `order_id="ex-1"` and
`fill_ids=["f-1"]` will pass every check here. That is acceptable only because the
validator is one layer inside a system whose next step (§8 step 1) is to persist
intents, orders and fills and make the correlation real. Until then this is a
trusted-caller boundary, and the trust is doing load-bearing work.

Two consequences:

- Do not cite a green validator as evidence that execution state is correct.
- The reconciler (§7) cannot be built on the validator alone. It needs the
  persisted records, which is why §8 puts persistence first.

## 9. What the validator implements

A pure function over `(current_state, event, evidence) -> next_state`, raising on an illegal transition
and on evidence that fails §4. No database, no clock, no network — so the rules above are provable in
isolation before anything persists or acts on them.

Deliberately **not** in the validator: what to *do* in a state. It answers only whether a transition is
legal and adequately evidenced.

**What its tests do and do not establish.** Passing validator tests demonstrate that the transition
rules above hold as written. They are **not** evidence of end-to-end execution correctness: they
exercise no exchange, no database, no concurrency and no clock. Treating a green validator suite as
proof that execution is safe would repeat the mistake this whole document exists to correct.
