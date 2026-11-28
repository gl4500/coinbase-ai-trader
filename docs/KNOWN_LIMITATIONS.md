# Known limitations of the legacy trading system

Status as of 2026-10-03, from the Claude/Codex consensus (archived at
`C:\Users\gl450\analysis_archive\future_plans_debate_2026-10-03\consensus.md`). Investment in the
legacy CNN/XGB bot is **frozen**. It keeps running as a paper diagnostic, and these limitations
are documented rather than fixed unless a named consumer triggers the fix. **Do not treat the
legacy ledger, its features or its helpers as evidence of a deployable edge.**

| # | Limitation | Evidence | Status / trigger |
|---|---|---|---|
| 1 | **The live-history merge drops every live row.** `seen` is built from `all_candles` itself, so `[c for c in all_candles if c["start"] not in seen]` is always empty. Whenever SQLite holds fewer than 336 bars AND a non-empty Parquet prefix exists, features come from the Parquet prefix only (with no Parquet prefix, the live rows are kept). | `backend/services/tiered_history.py:134-135` (verified by reading) | Fix when a named approved consumer adopts `tiered_history`. The fix changes 8001 features, so it is an operator decision. |
| 2 | **`macro_signals` has latent defects.** `oi_usd` is raw contracts; "BTC dominance" is BTC's share of Binance futures volume; fetch failures default to 0/1/50 while `fetch_ok=True`; Binance is geo-blocked here (HTTP 451). | `backend/services/macro_signals.py:219-245, 300-320` | **No consumer outside its tests** (verified on the root checkout and `origin/main`). Fix only if something adopts it. |
| 3 | **The paper ledger is gross and unexecutable.** Fills are at the last trade price (`ws_subscriber` ticker `price`), so no spread is charged; there is no fee term; the `orders` table is empty. The account's real tier is 0.50% maker / 0.90% taker. | `services/ws_subscriber.py:27-28`, `agents/cnn_agent.py` (`_CNNBook`) | Documented. Every paper P&L figure is before fees and spread. |
| 4 | **The shared `agent_state` row.** State is keyed only by agent (`CNN`), so a second backend on the same DB overwrites the paper book. | `backend/database.py` (`agent_state`) | Mitigated by the run-isolation rule in `CLAUDE.md`. The single portfolio owner is TRIGGER-GATED (a forward-paper engine). |
| 5 | **Historical replay can leak future data.** `tiered_history` filters on bar START; in replay a completed bar's full OHLC is admitted. | CLAUDE.md invariant 22 | Pass `closed_only=True` in every historical reconstruction. |
| 6 | **The live SQLite `candles` table stores the OPEN hour, frozen.** Coinbase's hourly candles include the in-progress hour, and `INSERT OR IGNORE` keeps that first partial version for good (Codex inventory P1). **Measured 2026-10-03** (read-only, last 48 h, 288 closed hours, 6 products): stored volume is a **median 3.4%** of the final (p10 0.5%, p90 13%), close is off by a median 26.5 bps (max 501 bps), and the high/low range is narrower than final in **100%** of bars. The Parquet history is mostly final (P2: about 9 of the last 300 h per product differ, ~3%). So the live bot's SQLite-based features are computed on truncated bars, while training used mostly-final Parquet: a train/serve skew. | `database.py` candle `INSERT OR IGNORE`; `market_scanner.py:132-134`; `history_backfill.py:250-256`; `C:\Users\gl450\analysis_archive\do_now_2026-10-03\writer_inventory.md` (outside the repo): P1 (SQLite `INSERT OR IGNORE`) / P2 (Parquet backfill skips known starts) | **Confirmed active-writer defect.** Originals preserved (snapshot `20261003T195004Z`). A writer fix changes 8001's behaviour → operator decision. Do not use the SQLite `candles` table as research data. |
| 7 | **No demonstrated strategy edge.** The ABSTAIN verdict (58.90); the SMA100 screen KILL (58.98); the live bot loses before fees. | CHANGELOG 58.90, 58.98 | Not to be funded. New candidates go through preregistered one-shot screens. |

Data preservation: see the **Run isolation & data retention** section of `CLAUDE.md`, and
`backend/tools/data_snapshot`.

## Codex live-guard observation (SWCFG-D4, 2026-11-28)

The native hook UI was reported by the operator/pm-claude as showing one installed,
active `PreToolUse` hook. That observation establishes the UI state at that time;
it does not establish that a particular shell invocation reached the guard.
The 2026-11-21 nonexistent-PID probe was not denied. No real process was terminated.
Its cause remains unresolved; do not treat the guard as observed enforcement on
that invocation, or repeat a termination probe to investigate it.

The suspected `exec_command` versus `Bash` matcher mismatch is contradicted by the
[official hook tool-coverage documentation](https://learn.chatgpt.com/docs/hooks#tool-coverage)
and the source matching the locally installed `codex-cli 0.171.0`:

- [ExecCommandHandler](https://github.com/openai/codex/blob/ccf9be8cc458bd6b8336fe84d68d8893f2691a5e/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs)
  constructs a `Bash` hook identity and puts `args.cmd` in `tool_input.command`.
- [HookToolName](https://github.com/openai/codex/blob/ccf9be8cc458bd6b8336fe84d68d8893f2691a5e/codex-rs/core/src/tools/hook_names.rs)
  defines that shell identity as `Bash`.

The current `.codex/hooks.json` matcher `^Bash$` and the guard's `command` payload
reader match that contract. Replacing the matcher with `^exec_command$` would not
match this documented shell-hook name. Retain the existing matcher unless direct
version/path-specific payload evidence supports a different change.

A non-destructive in-process verification supplied a documented `Bash` payload,
injected training/live-port probes and no test window. The matcher matched, and
`live_guard.main` returned exit 2 with a JSON deny decision. The command in that
payload was data only: no shell command or real training/port probe was executed.
This checks the guard when given the documented input, not native hook dispatch.

Remaining evidence needed: correlate an actual harmless invocation with its native
hook event/payload, loaded client revision, hook result and dispatch path (including
nested code-mode calls). Do not infer the historical payload from the model-visible
tool name. Tool hooks have coverage limits and are not a complete enforcement boundary.
Changing hook scripts/configuration or trust to capture that evidence needs the
operator's separate authorization. No hook/trust/global settings change is made by
this documentation update, and it does not close the native-dispatch investigation.
