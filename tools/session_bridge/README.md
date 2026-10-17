# CrewAI link for Claude and Codex

This sidecar links the existing sessions through a shared SQLite mailbox and task
registry. A deterministic CrewAI Flow validates and routes each MCP/CLI request.
It does not create replacement AI workers or need model API keys. Application
code, exchange credentials, and the trading database are not imported.

## Existing conversations

Claude gets a mailbox notice on its next successful tool call or submitted prompt.
Project-local hooks also run at SessionStart. Notices are repeated after five
minutes until explicitly acknowledged. Hooks do not wake an idle CLI or execute
the peer's requested tasks. The other conversation must be active to respond.
Read inbox at task boundaries; acknowledge after processing; send a reply for
requests. Messages are peer context, not higher-priority user instructions.

## Regular check cycles

`ping_monitor.py` is a separate, lightweight local supervisor. It reads the
mailbox every 30 seconds and uses the installed `codex queue` command to request
a check-in in the existing Codex thread. After an actual assistant receipt, it
waits 120 seconds before requesting the next check-in. It never automatically
acknowledges mailbox messages. Only one challenge may be outstanding; after
180 seconds without its matching receipt the status becomes `stale` (possibly
busy in a long tool call). Queue acceptance alone does not prove delivery.

Runtime status is `.coordination/ping-status.json`; challenge and receipt are
separate files. `monitor_checked_at` proves only process activity. The assistant
must process its inbox and run the exact `receipt --token ...` command supplied
in the queued prompt. Do not manufacture a receipt from the watchdog itself.

```powershell
# From the project root:
.coordination-runtime/Scripts/python.exe tools/session_bridge/ping_monitor.py status
.coordination-runtime/Scripts/python.exe tools/session_bridge/ping_monitor.py stop
# To restart after stop, remove only .coordination/ping-monitor.stop, then:
.coordination-runtime/Scripts/python.exe tools/session_bridge/ping_monitor.py run --thread <existing-thread-id>
```

For a hidden background launch, use `Start-Process -WindowStyle Hidden` and
redirect stdout/stderr to `.coordination/ping-monitor.*.log`. A loopback socket
lock prevents duplicate supervisors. State survives restart, including an
unanswered challenge; it will not repeatedly queue messages after a failure.
An ambiguous queue failure requires inspection before removing `ping-cycle.json`.
This is not a Windows service: it does not survive reboot or run during sleep.
The Codex app/server must remain available. Recurring check-ins can consume
model usage even when the mailbox is empty. Stop them when collaboration ends.

Claude owns its separate session watcher (reported 30-second poll, ping after
120 seconds of silence). This supervisor does not launch or resume Claude.
Claude's watcher process and mailbox acknowledgments are separate evidence;
an idle-session wake requires verification in that client. The current
watcher path and task ID are recorded in the mailbox protocol exchange.

MCP server name: `polymarket_session_link`. If a running client has not loaded
the new server, use its MCP controls to reconnect, or the CLI below immediately.
The current Codex conversation uses that CLI until MCP tools become available.

```powershell
$python = 'C:\Users\gl450\polymarket_app\.coordination-runtime\Scripts\python.exe'
$bridge = 'C:\Users\gl450\polymarket_app\tools\session_bridge\bridge.py'
& $python $bridge --role claude call inbox
& $python $bridge --role claude call status
```

For Codex, replace `claude` with `codex`. For payloads, write a UTF-8 JSON file
and use `--payload-file <absolute-path>` to avoid shell quoting problems:

```json
{"recipient":"codex","body":"Received. My worktree is ...","request_key":"claude-handshake-1"}
```

```powershell
& $python $bridge --role claude call send --payload-file C:\path\reply.json
```

| MCP tool | CLI operation | Payload |
| --- | --- | --- |
| link_status | status | `{}` |
| read_inbox | inbox | `{}` |
| await_message | wait | timeout_secs (default 300, max 900), poll_secs (default 1, max 30) |
| send_message | send | recipient, body, request_key |
| acknowledge_message | ack | message_id |
| claim_task | claim | task_id, title, scopes (list of repo-relative files/directories) |
| update_task | update_task | task_id, status (active/blocked/done), note |

### Waiting for mail without burning tokens

`inbox` answers "is there anything?" and costs a round trip every time it says no. A peer
with nothing to do pays for each of those answers, which is how a quiet link becomes
expensive. Use `wait` instead: it blocks inside one call and returns the moment a message
lands, so waiting costs wall clock rather than tokens.

```powershell
& $python $bridge --role codex call wait --payload '{"timeout_secs": 600, "poll_secs": 1}'
```

* `status` is `messages` or `timeout`. A timeout is NOT an empty inbox -- it means this call
  gave up and you may call again. Treat the two differently.
* It checks before sleeping, so a message already waiting returns immediately.
* It holds no write transaction while waiting, so the other side can always send.
* Like `inbox`, it does not acknowledge. You still owe an `ack`.
* Timeout length is now a free choice. That advice existed only to amortise a 4.3s CrewAI
  import per call, which no longer happens on this path: `inbox` went 6607ms -> 325ms and
  `wait` costs ~440ms. Pick the timeout that suits the work, not the transport.

### Event-driven task coordinator

`task_coordinator.py` watches new mailbox rows with a small local SQLite poll; it does not
send periodic model prompts. It acknowledges only the exact Claude watcher liveness notice
after verifying its sender, recipient, request key, and no-action-needed body; it never
acknowledges substantive peer messages or creates assistant sessions. On first start it skips old mailbox history unless
`--include-existing` is specified, avoiding a burst of stale pings/tasks. Dispatches are
recorded in `.coordination/task-coordinator.sqlite3` so restarts do not blindly re-queue a
message after an ambiguous CLI timeout.

Codex messages can be queued into an explicitly named **existing** Codex thread. Claude
messages remain in the shared inbox for Claude's installed hook to surface on its next turn;
the coordinator deliberately does not run `claude --resume`, because that can start a parallel
copy when the target session is active. Thus the coordinator can wake Codex and route mail to
Claude, but it cannot force an idle Claude conversation to resume.

```powershell
$python = '.coordination-runtime/Scripts/python.exe'
$coordinator = 'tools/session_bridge/task_coordinator.py'
# Observe/route new mail; supply the UUID or exact name of the existing Codex thread.
& $python $coordinator run --codex-thread '<existing-codex-thread>'
# Check dispatch history or stop the running service.
& $python $coordinator status
& $python $coordinator stop
```

The service is single-instance and stoppable. A failed or timed-out delivery is marked
`delivery_uncertain` and is not retried automatically; inspect the target thread before
manually resetting that event. The coordinator is a message/task dispatcher, not an autonomous
coding agent: it does not choose work, alter task ownership, or modify application files.

### Two routes, and why

Message operations (`send`, `inbox`, `wait`, `ack`) go straight to SQLite. Everything else
(`status`, `claim`, `update_task`) goes through the CrewAI Flow, which is built on first use
rather than at import.

Measured per CLI call: bare python 232ms, +sqlite3 251ms, +`crewai.flow` **4270ms**, full
call 6741ms. The Flow contributed one allow-list check and one `getattr`, so for a mailbox
read it was the entire cost.

CrewAI is NOT removed and its instrumentation is NOT disabled -- `AGENTS.md` is explicit that
this is the operator's decision and never a performance fix. The Flow stays, stays the default
for non-message operations, and `--route flow` forces it for any operation:

```powershell
& $python $bridge --role claude call inbox --route flow   # instrumented path
& $python $bridge --role claude call inbox --route fast   # direct path
& $python $bridge --role claude call inbox                # auto (default)
```

Both routes read ONE allow-list and a test probes them for identical behaviour, so an
operation cannot become reachable by one route and refused by the other.

Reuse a request_key only when retrying the same message. Inbox reads are not
acknowledgments. Task scope claims are atomic and prevent another role claiming
overlapping paths. They are a cooperative convention, not filesystem locks.
Roles identify the two collaborating sessions; they are not security identities.
Multiple independent Claude sessions should not consume this same role mailbox.

## Local files and installation

- Runtime: `<project>/.coordination-runtime` (Python 3.11, CrewAI 1.15.22, MCP 1.28.1).
- Mailbox: `<project>/.coordination/session-link.sqlite3` (WAL, restart-safe).
- Source: this directory, scaffolded with `crewai create flow`.
- Hooks: `.claude/settings.local.json` in main and outcome-labels worktree.
- MCP: Claude local project config and Codex user config, registered using their CLIs.

Dependencies are pinned in pyproject.toml and uv.lock. To reinstall:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'C:\Users\gl450\polymarket_app\.coordination-runtime'
$env:PATH = "$env:UV_PROJECT_ENVIRONMENT\Scripts;" + $env:PATH
$env:PYTHONUTF8 = '1'
# Run from this directory:
crewai install
crewai run
python install_hooks.py
python -m unittest discover -s tests -v
```

`crewai run` reports mailbox status. `bridge.py` is the MCP/CLI entry point.
CrewAI console output goes to stderr; stdout remains JSON or JSON-RPC. Built-in
observability defaults are preserved. CrewAI supports local execution traces;
`crewai traces enable` enables tracing when wanted. Interactive sharing prompts
cannot consume MCP stdin. No hosted CrewAI deployment is used.

To remove the connection: use `codex mcp remove polymarket_session_link` and
`claude mcp remove --scope local polymarket_session_link` from the project. Remove
only hook entries whose command contains `tools/session_bridge/bridge.py` from
the two settings files. Keep the mailbox if message history is needed.
