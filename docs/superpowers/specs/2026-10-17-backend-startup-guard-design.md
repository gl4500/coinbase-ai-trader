# Backend startup guard — design (v3: v2 + Codex initial notes 93fe5451 + live listener evidence)

Date: 2026-10-17. Branch `fix/backend-startup-guard` (off `main` 3b5fe0d). Status: awaiting Codex review. No change to the running 8001 service.

## Incident

On 2026-10-17 a manual start of the backend while 8001 was already running killed the live instance: every entry point runs `taskkill /F` on whatever listens on 8001 before starting. The new process then hit `disk I/O error` opening `coinbase.db`. The service recovered by itself later. The handle-release explanation for the I/O error (the killed process's SQLite handles not yet released) is a **hypothesis**, not established.

## Codex review v1 (CHANGES) — what this design answers

1. `main.py` alone is insufficient: `launcher.py:693`, `start_app.py:196` and `start_backend.ps1:42-47` each force-kill the port owner before `main.py` runs.
2. Default to refusing an occupied port. Never kill an arbitrary owner. Automatic recovery only with verified ownership and repeated failures over a grace period. Keep `/api/status` (readiness) distinct from `/api/health` (scan-loop health) so the watchdog still works.
3. DB retry: open/configure only, fresh connection per attempt, async backoff, logged with the SQLite error, a total deadline that accounts for the 30 s busy timeout, never retry corruption, and fail startup on a non-ok integrity check after an I/O recovery.
4. Replacement: check the kill result, wait for the process to exit AND the port to be free, abort on timeout, parse the exact port, dedupe IPv4/IPv6, revalidate before signaling, and never kill a process that won a startup race. Don't call `/F` graceful.
5. Single-instance startup lock.
6. Focused tests for each.

## Design

### One policy point: `backend/startup_guard.py` (new, stdlib only)

All entry points stop killing anything themselves. `main.py` calls `startup_guard.guard_startup(port, mode)` before `uvicorn.run`; it returns a held lock or exits.

**Modes** (CLI flag on `main.py`, or env `BACKEND_START_MODE`):
- `refuse` (default, including Task Scheduler logon autostart).
- `replace`: explicit operator request to replace a running instance.
- `replace-unhealthy`: used only by the launcher watchdog.

**Steps:**
1. **Single-instance lock.** Open `backend/.locks/backend-<port>.lock` and take an exclusive non-blocking `msvcrt.locking` lock, held for the process lifetime and released by the OS when the process exits, including after a kill. Write `{pid, started_utc, argv}` into it for diagnostics. If the lock is held:
   - `refuse`: exit code 3, "another backend instance is starting/running (PID n)".
   - `replace*`: go to step 3 targeting the holder, then retry the lock once after the old process exits.
   This closes the simultaneous-start race: two concurrent starts cannot both pass.
2. **Exact listener check.** Parse `netstat -ano -p TCP` and `-p TCPv6`. Keep only rows whose local endpoint port equals `port` exactly (split on the last `:`, handle `[::]:8001`, reject `:80010`) and whose state is LISTENING. Dedupe PIDs. None found: start. Found:
   - `refuse`: log the PID and command line, exit code 2. Kill nothing.
   - `replace` / `replace-unhealthy`: step 3.
3. **Verified replacement.**
   - **Ownership comes from the lock file, not the command line.** Live evidence (2026-10-17, read-only): the 8001 listener is PID 27796, `C:\ProgramData\spyder-6\python.exe` with command line `"...python.exe" main.py`. That is the base interpreter, a child of the venv launcher stub PID 19816 (`.venv\Scripts\python.exe main.py`). The listener's command line is relative and its executable lies outside the repo, so CIM data cannot prove which `main.py` it runs. Rule: the guard writes `{pid: os.getpid(), creation_time, port, repo_backend_dir}` into the lock file after acquiring it. `os.getpid()` in `main.py` is the real interpreter, i.e. the future listener. A listener counts as owned only if the lock file records that exact PID AND its `Win32_Process.CreationDate` matches the recorded creation time AND the recorded `repo_backend_dir` equals this backend directory. Otherwise (no lock file, a legacy instance started before this change, a PID mismatch, or an unreadable record) ownership is UNPROVEN: refuse, exit code 4, "port owner not verifiable; stop it manually". Fail safe, never guess.
   - **Transition note:** the backend running today predates the guard and has no lock file, so neither `replace` mode will touch it. The operator stops it once by hand at the next planned restart; after that, instances are verifiable.
   - **`replace-unhealthy` only:** health is proven failing only by `/api/health` returning 503 with JSON `status == "unhealthy"`, a connection refusal, or a 5 s timeout. Each failure is timestamped. Replacement requires at least 3 failures AND at least 30 s elapsed between the first and last failure, with no success in between (probe every 10 s, so the minimum span is 30 s). A 200 with `status == "ok"` resets the count and refuses (exit 5, "backend healthy"). Any other reply (401, other 4xx/5xx, non-JSON or malformed body) is AMBIGUOUS: refuse (exit 5, "health reply ambiguous; not replacing"). Only positively identified unhealthiness counts.
   - **Revalidate immediately before signaling:** same PID, same CreationDate, still the listener, lock record unchanged. Otherwise abort with exit 6. This **narrows but does not eliminate** the check-to-signal race: with this PID/`taskkill` design a tiny window remains between the last check and the signal (a limitation of the chosen mechanism, not of every Windows process-handle API). That residual is documented, not claimed away.
   - `taskkill /F /T /PID n`: check the return code; non-zero aborts with exit 6. Documented as a forced kill, not graceful. A cooperative shutdown endpoint is out of scope (noted as future work).
   - Wait until the PID has exited AND no LISTENING row remains on `port`, polling every 0.5 s with a 30 s deadline; on timeout, exit 7. Then retry the lock (step 1) once; failure exits 3.
4. Return the held lock; `main.py` keeps a reference until exit.

Exit codes are distinct so the launcher and logs can say why a start did not happen.

### Entry points

| Entry point | Change |
|---|---|
| `backend/main.py` `__main__` | Remove `_free_port`; call `guard_startup(port, mode)`. Delete the `_free_port` function (no other callers; checked by grep). |
| Task Scheduler `CoinbaseTraderBackend` (pythonw main.py) | No change needed: inherits `refuse`. If 8001 is held at logon, the start is refused and logged; `RestartCount=3` re-runs are harmless. |
| `launcher.py` `_start_backend` | Remove the pre-spawn `_free_port(8001)`. Spawn `main.py` (refuse). If it exits 2/3 within a few seconds, log "backend already running/starting — not replaced". Delete launcher `_free_port`. |
| `launcher.py` watchdog (pure `_Watchdog` state machine; PR #105 review P105-01/02) | Own child: any non-ok probe counts, 3 strikes -> kill own child by PID, wait, restart. Not owned: only a positively `unhealthy` reply counts, 3 strikes -> spawn `main.py --mode replace-unhealthy`. Any launcher spawn is *in flight* (no restarts) until healthy, exited, or 180 s; a replacement that exits without a healthy backend starts a 600 s cooldown so a refusing guard is not respawned every few seconds. |
| `start_app.py` | Remove `_free_port(8001)`; launch `main.py` and pass through `--replace` if the user gives it. |
| `start_backend.ps1` | Remove the netstat/taskkill block; add a `-Replace` switch that passes `--mode replace`. |

### Database open retry (`database.init_db`)

- Retry scope: only the open plus `PRAGMA busy_timeout` / `journal_mode` configuration step. Each attempt uses a fresh `aiosqlite.connect`, closed on failure.
- Retryable: `sqlite3.OperationalError` whose message is `disk I/O error`, `database is locked` or `unable to open database file`. Never retried: `malformed`, `not a database`, `corrupt`, or any `DatabaseError` that is not operational. These propagate immediately.
- Backoff: async 0.5, 1, 2, 4 s. During init the per-attempt busy timeout is 5 s (normal operation keeps 30 s). Total deadline 30 s on monotonic time: a retry starts only if a full 5 s attempt still fits, and a connection that completes after the deadline (OS I/O blocked past the busy timeout) is closed and refused (`DatabaseOpenTimeout`); otherwise the last error propagates and startup fails. Each attempt logs attempt n, elapsed time, exception class, message and SQLite error code/name where available (Python 3.11 `sqlite_errorname`).
- After a successful open that followed a `disk I/O error` attempt, run a bounded `PRAGMA quick_check`, capped at 60 s by a progress handler (the DB is 553 MB). Result `ok`: continue. Any other result: fail startup (CRITICAL log). If the 60 s cap interrupts the check (inconclusive), startup also fails: fail closed, as agreed in Codex's initial notes. A check interrupted by the cap and a check that failed are logged with different reasons.

## Tests (TDD, `backend/tests/test_startup_guard.py`, `test_database_open_retry.py`, launcher/start_app tests)

- Exact-port parsing: `0.0.0.0:8001`, `[::]:8001`, `127.0.0.1:80010` rejected, IPv4/IPv6 duplicate PID deduped, non-LISTENING rows ignored.
- Refuse: listener present gives exit 2 with no kill call; lock held gives exit 3.
- Lock: a second process (subprocess helper) cannot acquire it while the first holds it; the lock is released after the holder is killed.
- Unrelated listener: `replace` refuses (4), no kill. The same applies to a legacy listener with no lock file, a lock PID differing from the listener PID, a CreationDate mismatch, or a different `repo_backend_dir`.
- Lock record: the guard writes `os.getpid()`, the creation time and the backend dir; a base-interpreter listener whose parent is the venv stub is matched by the recorded PID, not by command line.
- Transient probe: `replace-unhealthy` with fail/ok/fail refuses (5). Three failures within less than 30 s elapsed do not proceed; three failures spanning at least 30 s proceed. 401, a non-JSON body and a 500 without `status == "unhealthy"` each refuse as ambiguous.
- Revalidation: a PID or creation-time change before signaling aborts (6) with no kill.
- Kill failure: non-zero `taskkill` aborts (6). Exit timeout: port still listening after the deadline gives 7.
- Entry points: `launcher._start_backend` and `start_app` never call taskkill/netstat-kill (patched subprocess asserts); `start_backend.ps1` contains no `taskkill` (static test); `main.py` has no `_free_port`.
- DB: retry opens a fresh connection each attempt; locked then success returns; exhaustion within the deadline propagates the last error; `malformed` is not retried; quick_check non-ok fails; quick_check after a non-I/O retry is not run.

Tests use fakes for netstat/CIM/taskkill/HTTP; nothing touches the live 8001, its process or `coinbase.db` (temp DB files only).

## Rollout

Code and tests on the branch, then Codex review on the exact head SHA, then merge. The running 8001 is NOT restarted by this work. The new guard takes effect at the operator's next restart.

## Out of scope

A cooperative graceful shutdown endpoint; changing the Task Scheduler task.
