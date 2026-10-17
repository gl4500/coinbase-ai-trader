# Session-bridge auto-connect — design

Date: 2026-10-17 · Status: approved in chat (design A), spec awaiting operator review
Branch: `feat/session-bridge-autoconnect`, stacked on `feat/session-bridge-coordinator` (d87b164, no PR yet)

## Problem

On 2026-10-17 three Codex messages (from 2026-10-14) sat unread, and a Claude review request
went unanswered for 15 minutes, because every delivery path in `tools/session_bridge` needs a
manual step:

1. **Claude hook is project-local.** `install_hooks.py` writes the bridge hook into
   `polymarket_app/.claude/settings.local.json`. A Claude session launched from any other
   directory (this one started in `C:\Users\gl450`) never loads it, so it never sees mail.
2. **Codex has no bridge hook.** `~/.codex/hooks.json` has only the Herdr SessionStart hook,
   so Codex never registers itself in the mailbox's `sessions` table.
3. **The coordinator is never started.** `task_coordinator.py run --codex-thread <id>` must be
   launched by hand with a thread ID. It was not running, and `ping_monitor.py` has been stopped
   since 2026-09-27 07:36.

Evidence that the wake mechanism itself works: a manual
`codex queue --thread 01a149ea-f80c-7db1-9291-dbb4db3b82e5` got a Codex review reply in 33 s.

## Goal / success criteria

Starting a Claude or Codex session, from any directory, needs no manual step:

- S1. A Claude session whose cwd is the polymarket_app tree or the home directory surfaces
  unacknowledged `claude` mail on its next tool call / prompt (existing `notice()` behavior).
- S2. A Codex session in the polymarket_app tree registers its thread ID in `sessions`.
- S3. Mail addressed to `codex` is queued into the most recently active registered Codex thread
  within a few seconds, with no `--codex-thread` flag.
- S4. Exactly one coordinator runs; it is started automatically by any bridge MCP server start.
- S5. An operator `task_coordinator.py stop` stays stopped: auto-start never overrides it.

## Non-goals and accepted limits

- Waking an **idle Claude** session. Not possible without `claude --resume`, which can fork a
  parallel copy of an active session. Claude still sees mail on its next turn.
- Codex's one-time hook-trust prompt (`[hooks.state]` `trusted_hash` in `~/.codex/config.toml`)
  stays an operator approval; we install the hook, the operator trusts it once.
- Retiring `ping_monitor.py` — flagged as superseded, not removed here.
- Surviving reboot with no agent running. The next MCP server start relaunches the coordinator.
- The 8001 restart-collision fix (separate track, Codex review returned CHANGES).

## Design

### 1. User-level hook installation (`install_hooks.py`)

- Claude: add the existing `bridge.py --role claude hook` command for `SessionStart`,
  `UserPromptSubmit`, `PostToolUse` to `~/.claude/settings.json`, and **remove** the identical
  entries from the project-local `settings.local.json` files it previously wrote (main checkout and
  `.claude/worktrees/*`), so a session in the repo does not get the notice twice.
- Codex: add `bridge.py --role codex hook` for `SessionStart` (plus `UserPromptSubmit` /
  `PostToolUse` only if the installed Codex version accepts those events — verified in plan task 1)
  to `~/.codex/hooks.json`, preserving the Herdr entry.
- Keeps current properties: idempotent, timestamped backup before any write, atomic replace,
  never touches entries whose command does not contain `tools/session_bridge/bridge.py`.
- Gains `--uninstall` that removes only bridge entries.

### 2. Hook scope filter (`hooks.py`)

The hook now runs in every Claude session on the machine, including trading_app, grain-marketing
and rf-toolkit sessions, which must not receive polymarket peer mail (scope rule).
`handle_hook` acts only when `event.cwd` resolves to the repo `ROOT`, a path under it (worktrees
included), or exactly the user's home directory. Outside that scope it returns without writing a
`sessions` row or printing anything. A pure function `in_scope(cwd, root, home) -> bool` holds the
rule.

### 3. Coordinator auto-start (`server.py` → new `autostart.py`)

`autostart.ensure_coordinator(root) -> str` is called once in `server.run` before serving. It:

1. returns `"stopped"` if `.coordination/task-coordinator.stop` exists (S5);
2. probes the coordinator's existing lock (`127.0.0.1:47840`) by attempting a bind with
   `SO_EXCLUSIVEADDRUSE`; if the bind fails, a coordinator holds it → returns `"running"`;
3. otherwise releases the probe socket and spawns
   `sys.executable task_coordinator.py run` detached (`DETACHED_PROCESS | CREATE_NO_WINDOW`,
   `close_fds`), stdout/stderr appended to `.coordination/task-coordinator.log` → `"spawned"`.

Two MCP servers starting together can both spawn; the coordinator's own bind makes the loser exit
immediately (log line, nonzero), so at most one keeps running (S4). Any exception is logged to
stderr and returned as `"error: …"` — auto-start must never stop the MCP server from serving.

### 4. Thread auto-discovery (`task_coordinator.py`)

When no `--codex-thread` is given, each Codex dispatch resolves its target with a pure query over
the mailbox `sessions` table:

```
latest codex session WHERE role='codex' AND cwd is ROOT-or-under
                     AND last_seen >= now - CODEX_THREAD_MAX_AGE (24 h)
                     ORDER BY last_seen DESC LIMIT 1
```

- Found → `codex queue --thread <session_id>` (existing `_queue_codex`, existing dispatch record).
- None → existing "left in mailbox" path, detail `"no recent Codex session registered"`.
- An explicit `--codex-thread` still wins (back-compat). `status` reports the resolved thread and
  its `last_seen` instead of only `codex_thread_configured`.

The 24 h bound stops mail from resuming a long-abandoned conversation.

## Data flow

```
Codex session start ──hook──▶ sessions(role=codex, session_id=thread, cwd, last_seen)
Claude/Codex start ──MCP──▶ ensure_coordinator ──▶ (one) task_coordinator run
Claude sends to codex ──▶ messages ──poll 1s──▶ coordinator ──resolve thread──▶ codex queue
Codex sends to claude ──▶ messages ──▶ Claude hook on next tool call/prompt ──▶ notice
```

## Error handling

- Hook: unchanged contract — never raises, prints one stderr line on failure, 8 s timeout.
- Auto-start: never raises into the MCP server; result string logged.
- Dispatch: existing ambiguous-timeout handling (no blind retry) unchanged.
- Installer: on malformed JSON in a settings file, abort that file with an error; never overwrite.

## Testing (TDD, run under `.coordination-runtime`)

- `in_scope`: root, worktree path, home exact, home subdir outside root, other repo, case/slash
  normalization on Windows.
- Hook: out-of-scope cwd writes no `sessions` row and prints nothing.
- Installer (tmp HOME): adds user-level entries, removes project-local duplicates, preserves
  Herdr and unrelated hooks, idempotent second run, backup written, `--uninstall`, malformed JSON.
- `ensure_coordinator`: stop file → `"stopped"`; held lock → `"running"`; free → spawn called
  once with detached flags (spawn injected); spawn error → `"error"` and no raise.
- Coordinator: thread resolution picks newest in-scope codex row, ignores stale (>24 h),
  out-of-scope and claude rows, explicit `--codex-thread` overrides, none → left in mailbox.
- Concurrent start: two coordinators, second exits on lock.
- Live check (manual, recorded): new Codex session → its `sessions.session_id` equals the thread
  ID in its rollout filename, and a Claude→codex message is queued into it.

## Rollout

1. Merge order: `feat/session-bridge-coordinator` PR first (with Codex sign-off), then this branch.
2. Run `install_hooks.py`; operator trusts the Codex hook once.
3. Existing MCP servers start the coordinator on their next launch (or `task_coordinator.py run`
   once by hand). Untracked copies in the main checkout are replaced by the merged versions.
