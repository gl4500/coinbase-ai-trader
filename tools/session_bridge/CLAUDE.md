# CLAUDE.md

Claude Code loads this file and ignores `AGENTS.md`. The import below pulls in the
shared CrewAI guidance so every coding assistant works from the same instructions.
Keep shared conventions in `AGENTS.md`; add Claude-specific notes under the import.

@AGENTS.md

---

## Reading this mailbox (Claude-specific)

Established by failure on 2026-09-27. `Store.inbox()` selected `ORDER BY created LIMIT 50`
-- the OLDEST fifty unacknowledged. A 65-message backlog therefore froze the reader on old
mail, and 15 newer peer messages were unreadable for hours while I reported the peer as
silent. `wait()` reads through `inbox()`, so the long-poll could not wake on a message it
could not see: a backlog silently disabled it.

- **`inbox()` is a WINDOW, not the mailbox.** It returns the newest `Store.INBOX_WINDOW`
  (50) unacknowledged messages, presented oldest-first so reading order still carries the
  thread. `inbox(limit=N)` widens it deliberately.
- **Ask for the total before concluding anything about volume.** `pending_count()` gives the
  true count; `wait()` returns `pending_total` and `truncated` alongside its messages.
- **Ack what you read.** An unacknowledged backlog is not harmless -- it is the only thing
  that pushes messages out of the window. `ack` means RECEIVED, not actioned; reply
  separately for content. Draining to zero at task boundaries keeps the window inert.
- **Never attribute silence to the peer until you have checked your own reader.** Order of
  suspicion: my reader, then the transport, then the peer. Going in reverse is what produced
  the 2026-09-27 misreport.
- **A bound must report itself.** Same rule as `wait()` reporting a timeout AS a timeout
  rather than as an empty list: a partial answer has to be distinguishable from a complete
  one. Any new windowed read here carries its own count.

### Running the bridge suite

The bridge needs `crewai` + `mcp`, which live in `.coordination-runtime`, NOT in `.venv`
(`.venv` has pytest but neither dependency; the runtime needed `pytest` installed before
`test_mcp` could be collected at all). From the repo root:

```bash
PYTHONPATH=tools/session_bridge/src .coordination-runtime/Scripts/python.exe -m pytest tools/session_bridge/tests -q
```

35 tests as of 2026-09-27. Running it under `.venv` silently skips the CrewAI-route and MCP
coverage -- which is exactly the coverage that caught the MCP event-loop hang.

