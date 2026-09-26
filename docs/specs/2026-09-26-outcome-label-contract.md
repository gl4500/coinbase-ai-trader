# Outcome-label contract — `signal_outcomes` label version 2

**Status:** specification (Task 2). Implementation in Task 3.
**Supersedes:** the implicit, undocumented behaviour of `OutcomeTracker.check_pending` (label version 1).
**Input evidence:** `docs/audits/2026-09-26-strategy-audit-report.md` §1.

---

## 0. Why this document exists

Label version 1 has no specification. Its behaviour was whatever `check_pending` happened to do, and
what it did was: *fetch the latest available price at the moment the resolver ran, and call that the
4-hour outcome.* The audit measured the consequence — mean resolution delay **45.75 h**, median
17.45 h, p90 140.91 h, max 495.32 h, with 57.76% of 134,181 resolved rows resolved more than six
hours late. The recorded 22.03% BUY WIN fraction is therefore not a 4-hour predictive score.

Version 2 fixes the timing and, equally important, **writes down what the label means** so no future
reader has to infer it from code.

---

## 1. Prediction target and intended horizon

**Target:** the signed, path-independent **endpoint return of the product's hourly close over a
fixed 4-bar window**, measured from an executable entry reference.

- Horizon `H_BARS = 4` hourly candles (4 h), stored per row rather than implied by a global constant,
  so a future change of horizon cannot silently reinterpret old rows.
- The label is an **economic outcome diagnostic**. It is explicitly *not* any model's training
  target (see section 7).

## 2. Entry-price convention

Entry is the **open of the first hourly candle that starts strictly after the signal timestamp**:

    E = floor_to_hour(signal_time) + 3600      # entry candle bucket-open
    entry_price_v2 = candle[E].open

Rationale: the signal fires at an arbitrary intra-hour moment against a live quote. That quote is not
an executable price at any defined later instant, and using it silently mixes a mid-bar quote with a
bar close. The next bar open is the earliest reference an executor could realistically have taken,
and it is the convention the audit used for its reconstruction, so the two stay comparable.

The original `entry_price` column is **left untouched** — it remains the recorded scan quote. The v2
entry reference is stored separately. The two differ, and that difference is itself information.

## 3. Target price: which completed candle

    exit_candle_start = E + (H_BARS - 1) * 3600      # = E + 3h
    target_price      = candle[exit_candle_start].close
    target_time       = E + H_BARS * 3600            # exit candle's closing instant

### Candle timestamp semantics

`candles.start_time` (and the client's `start` field) is the **bucket-open** epoch in seconds. A
candle stamped `S` covers the half-open interval `[S, S + 3600)` and its close is final only at
`S + 3600`.

**A candle is usable only when `now >= S + 3600`.** This is the rule version 1 violated:
`coinbase_client.get_candles(pid, "ONE_HOUR", limit=1)` returns the newest bucket, which is normally
the *in-progress* hour, so even a promptly resolved row could be labelled from an incomplete candle.

### Boundary behaviour

- Entry uses `>` not `>=`: a signal landing exactly on an hour boundary `T` takes the candle at
  `T + 3600`, never `T`. A signal at `T` cannot be assumed to precede the trades inside bar `T`.
- `target_time` is the exit candle's close instant. A row becomes **eligible** at
  `now >= target_time` — the moment the exit bar is final, not one bar later.
- The span from entry open to exit close is exactly `H_BARS` hours by construction. Defining the
  window in *bars* rather than seconds avoids the sub-hour quantisation drift a
  `signal_time + 14400` definition produces.

## 4. Three distinct timestamps — never conflated

| Concept | Column | Meaning |
|---|---|---|
| **Target time** | `target_time` | When the outcome *should* be measured: the exit candle's close instant. Fixed at record time; never moves. |
| **Price observation time** | `price_observed_at` | The close instant of the candle actually used. Under this contract it equals `target_time` whenever resolution succeeds; recorded independently so any divergence is visible rather than assumed away. |
| **Processing time** | `processed_at` | When the resolver ran. Carries **no** pricing meaning. |

Version 1's `checked_at` is a processing timestamp that the dashboard read as though it were a price
timestamp. Version 2 keeps `checked_at` for legacy rows and adds the three fields above.

Also persisted: `entry_candle_start`, `exit_candle_start`, `price_source`, `label_version`,
`resolve_attempts`, `unresolved_reason`.

## 5. Return direction, units, thresholds

    raw_return    = (target_price - entry_price_v2) / entry_price_v2
    signed_return = raw_return       if side == 'BUY'
                  = -raw_return      if side == 'SELL'

- **Units: decimal fraction.** `0.0123` means +1.23%. Not percentage points, not dollars. Version 1
  mixed dollar P&L, fraction, and percentage-point columns across tables; this column is a fraction,
  always.
- A SELL is scored as a short: the signal is correct when price falls.

| Outcome | Condition |
|---|---|
| `WIN` | `signed_return > +0.005` |
| `LOSS` | `signed_return < -0.005` |
| `NEUTRAL` | otherwise (the ±0.5% dead zone, inclusive of both bounds) |

The ±0.5% band is carried over from version 1 deliberately, so a v1-vs-v2 comparison isolates the
*timing* fix instead of confounding it with a threshold change. The band is **gross** — it ignores
fees. At the audit's 0.60%/side stress assumption a +0.5% "WIN" is a net loss, so WIN means "moved
favourably by more than 50 bp", never "profitable".

## 6. Missing data, retries, terminal states

The resolver **never** substitutes the current market price for a missing historical price. If either
the entry or the exit candle is absent from the local store:

1. The row stays **unresolved** (`outcome IS NULL`) and `resolve_attempts` increments.
2. It is retried on later passes, up to `MAX_RESOLVE_ATTEMPTS = 5`.
3. A row becomes terminally **`UNAVAILABLE`** when *either* `resolve_attempts >= 5` *or*
   `now > target_time + UNAVAILABLE_GRACE_SECS` (7 days), with `unresolved_reason` set to
   `missing_entry_candle`, `missing_exit_candle`, or `invalid_entry_price`.
4. `UNAVAILABLE` is a terminal, **non-scoring** state. It is not WIN, LOSS, or NEUTRAL, and it is
   excluded from every accuracy denominator (Task 4).

The grace window exists because the local candle store has genuine gaps — 356 products over
2025-04-10 to present, hourly, containing one 6-hour gap and two multi-week gaps. Backfill can close
a gap days later, so a row is given time before being written off, but not unbounded time.

### Idempotency

- Resolution writes with `UPDATE ... WHERE id = ? AND outcome IS NULL`, so a completed label can
  never be silently overwritten by a retry or a concurrent pass.
- Re-running the resolver over resolved rows is a no-op. It cannot duplicate rows, because resolution
  is an UPDATE on an existing primary key, never an INSERT.
- Legacy (version 1) rows are never recomputed or rewritten.

## 7. Relationship to each model's actual training target

**The v2 label is not any model's training target.** Stated precisely, because calling a different
target "calibration" is exactly the error the audit identified:

| Consumer | Target | Path dependence | Window | Classes |
|---|---|---|---|---|
| CNN / XGB v3 / v4 (`cnn_agent._triple_barrier_label`) | First touch of `high >= entry*(1+0.01)` or `low <= entry*(1-0.01)`; at the time barrier, sign of close move outside a `label_thresh` dead zone | **Yes** — intrabar highs/lows, first-touch ordering | `forward_hours` bars | binary 1/0, `None` dropped |
| XGB v4.5 (`tools/train_xgb_v4_5._triple_barrier_label_3class`) | First touch of `close >= entry*(1+label_thresh)` / `<= entry*(1-label_thresh)`, else NEUTRAL on timeout | **Yes** — on closes | `forward_hours`, operator-supplied per run (no default) | 3-class DOWN/NEUTRAL/UP |
| CNN auto-train loop | `_FORWARD_HOURS = 4`, "close higher 4 hours ahead" | via the shared triple-barrier labeler | 4 bars | binary |
| **`signal_outcomes` v2 (this contract)** | **Endpoint return vs ±0.5%** | **No** — endpoints only | **4 bars** | WIN/LOSS/NEUTRAL |

Consequences that bind the diagnostics work:

1. **Probability calibration against this label is invalid.** A model trained on first touch of ±1%
   on intrabar extremes is not predicting a 4-bar endpoint move of ±0.5%. Version 1's
   confidence-decile "calibration" table compares those two different things. Calibration may be
   reported only when the evaluated label matches the scoring model's own target and horizon.
2. **Even the horizon is not shared.** v4.5 takes `forward_hours` per run with no default; v3/v4 take
   it from the dataset schema. One 4-bar label cannot be the target of all of them at once.
3. `source='CNN'` is a historical book name. The audit found no per-row immutable model identifier,
   so rows cannot be attributed to a model version. Until a model hash is persisted per signal, any
   per-model claim from this table is unsupported — which is why Task 6 specifies run ID and model
   hash as required provenance.

What the v2 label *can* support: "did the signal's stated direction move favourably by more than
50 bp over the next four completed hours, measured from the next bar open." Nothing more.

## 8. Explicitly out of scope

- No change to model thresholds, training, or strategy behaviour.
- No recomputation of legacy labels.
- No claim that a favourable label implies profitability. Signal-label accuracy and executed-trade
  P&L are separate quantities, reported separately (Task 4).
