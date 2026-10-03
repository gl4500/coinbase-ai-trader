# Known limitations of the legacy trading system

Status as of 2026-10-03, from the Claude/Codex consensus (archived at
`C:\Users\gl450\analysis_archive\future_plans_debate_2026-10-03\consensus.md`). Investment in the
legacy CNN/XGB bot is **frozen**. It keeps running as a paper diagnostic, and these limitations
are documented rather than fixed unless a named consumer triggers the fix. **Do not treat the
legacy ledger, its features or its helpers as evidence of a deployable edge.**

| # | Limitation | Evidence | Status / trigger |
|---|---|---|---|
| 1 | **The live-history merge drops every live row.** `seen` is built from `all_candles` itself, so `[c for c in all_candles if c["start"] not in seen]` is always empty. Whenever SQLite holds fewer than 336 bars, features come from the Parquet prefix only. | `backend/services/tiered_history.py:134-135` (verified by reading) | Fix when a named approved consumer adopts `tiered_history`. The fix changes 8001 features, so it is an operator decision. |
| 2 | **`macro_signals` has latent defects.** `oi_usd` is raw contracts; "BTC dominance" is BTC's share of Binance futures volume; fetch failures default to 0/1/50 while `fetch_ok=True`; Binance is geo-blocked here (HTTP 451). | `backend/services/macro_signals.py:219-245, 300-320` | **No consumer outside its tests** (verified on the root checkout and `origin/main`). Fix only if something adopts it. |
| 3 | **The paper ledger is gross and unexecutable.** Fills are at the last trade price (`ws_subscriber` ticker `price`), so no spread is charged; there is no fee term; the `orders` table is empty. The account's real tier is 0.50% maker / 0.90% taker. | `services/ws_subscriber.py:27-28`, `agents/cnn_agent.py` (`_CNNBook`) | Documented. Every paper P&L figure is before fees and spread. |
| 4 | **The shared `agent_state` row.** State is keyed only by agent (`CNN`), so a second backend on the same DB overwrites the paper book. | `backend/database.py` (`agent_state`) | Mitigated by the run-isolation rule in `CLAUDE.md`. The single portfolio owner is TRIGGER-GATED (a forward-paper engine). |
| 5 | **Historical replay can leak future data.** `tiered_history` filters on bar START; in replay a completed bar's full OHLC is admitted. | CLAUDE.md invariant 22 | Pass `closed_only=True` in every historical reconstruction. |
| 6 | **Provisional candles may persist as final.** `INSERT OR IGNORE` keeps the first version seen of a bar. | `database.py` candle insert; `services/history_backfill.py` | Under investigation in the writer/consumer inventory (`analysis_archive/do_now_2026-10-03/writer_inventory.md`). A confirmed active-writer defect becomes a preservation fix (operator approval needed for 8001). |
| 7 | **No demonstrated strategy edge.** The ABSTAIN verdict (58.90); the SMA100 screen KILL (58.98); the live bot loses before fees. | CHANGELOG 58.90, 58.98 | Not to be funded. New candidates go through preregistered one-shot screens. |

Data preservation: see the **Run isolation & data retention** section of `CLAUDE.md`, and
`backend/tools/data_snapshot`.
