# Session link implementation

- [x] Inspect current sessions, worktrees, CLI capabilities, and official docs.
- [x] Install isolated Python 3.11 CrewAI runtime and scaffold a CrewAI Flow.
- [x] Test persistent delivery, acknowledgments, task ownership, and hook notices.
- [x] Implement Flow routing, SQLite store, MCP server, CLI, and Claude hooks.
- [x] Test actual two-process MCP exchange with a temporary database.
- [x] Register both clients and send a real handshake to the existing Claude session.
- [x] Record observed connection status and operating instructions.

Seven tests pass. Real Claude-session reply received through the CLI and acknowledged;
Codex confirmation sent back. Bidirectional live-session communication is verified.
Hook auto-notification remains unverified; existing sessions use the CLI without restart.

Scope: tools/session_bridge, docs/handoffs/session-link.md, ignored local runtime/database,
and narrowly scoped CLI configuration. Do not change trading code or Claude's task files.
