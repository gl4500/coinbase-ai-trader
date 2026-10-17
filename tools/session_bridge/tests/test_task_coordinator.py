import socket
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from session_bridge.store import Store
from task_coordinator import TaskCoordinator, resolve_codex_thread, run


class TaskCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.mailbox = root / "mailbox.sqlite3"
        self.state = root / "coordinator.sqlite3"
        self.codex = Store(self.mailbox, "codex")
        self.claude = Store(self.mailbox, "claude")

    def tearDown(self):
        self.temp.cleanup()

    def test_routes_new_codex_message_once_without_acknowledging_it(self):
        message = self.claude.send("codex", "Review the new result", "task-review-1")
        queued = []
        events = []
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda thread, prompt: queued.append((thread, prompt)) or (0, "queued"),
            on_event=events.append,
            include_existing=True,
        )

        self.assertEqual(len(coordinator.process_once()), 1)
        self.assertEqual(coordinator.process_once(), [])
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0][0], "existing-thread")
        self.assertIn(message["id"], queued[0][1])
        self.assertIsNone(self.codex.inbox()[0]["acknowledged"])
        self.assertEqual(events[0]["state"], "queued")

    def test_claude_watcher_liveness_notice_is_acked_but_not_forwarded(self):
        message = self.claude.send(
            "codex",
            "PING from the Claude watcher: no traffic. Silence-triggered liveness "
            "check, not a scheduled heartbeat. No action needed beyond an ack if you are alive.",
            "claude-watcher-ping-test",
        )
        queued = []
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda *args: queued.append(args) or (0, "queued"),
            include_existing=True,
        )

        self.assertEqual(coordinator.process_once(), [])
        self.assertEqual(queued, [])
        with self.codex.connect() as db:
            acknowledged = db.execute(
                "SELECT acknowledged FROM messages WHERE id=?", (message["id"],)
            ).fetchone()[0]
        self.assertIsNotNone(acknowledged)

    def test_claude_mail_stays_for_existing_hook_and_is_never_auto_resumed(self):
        message = self.codex.send("claude", "Please review this finding", "review-claude-1")
        queued = []
        events = []
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda *args: queued.append(args) or (0, "queued"),
            on_event=events.append,
            include_existing=True,
        )

        self.assertEqual(coordinator.process_once()[0]["state"], "mailbox_notice")
        self.assertEqual(queued, [])
        self.assertEqual(events[0]["message_id"], message["id"])
        self.assertEqual(len(self.claude.inbox()), 1)

    def test_first_start_skips_old_backlog_but_handles_later_mail(self):
        old = self.claude.send("codex", "Old item", "old-item")
        queued = []
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda *args: queued.append(args) or (0, "queued"),
        )
        self.assertEqual(coordinator.process_once(), [])
        self.assertEqual(queued, [])

        new = self.claude.send("codex", "New item", "new-item")
        events = coordinator.process_once()
        self.assertEqual([event["message_id"] for event in events], [new["id"]])
        self.assertNotEqual(old["id"], new["id"])

    def test_ambiguous_queue_failure_is_not_retried_automatically(self):
        self.claude.send("codex", "Work item", "uncertain-delivery")
        calls = []
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda *args: calls.append(args) or (124, "timeout; uncertain"),
            include_existing=True,
        )
        self.assertEqual(coordinator.process_once()[0]["state"], "delivery_uncertain")
        self.assertEqual(coordinator.process_once(), [])
        self.assertEqual(len(calls), 1)

    def test_restart_marks_inflight_delivery_uncertain_without_retry(self):
        message = self.claude.send("codex", "Work item", "crash-during-delivery")
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            include_existing=True,
        )
        row = coordinator._new_messages()[0]
        self.assertTrue(coordinator._claim_dispatch(row))

        second = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda *args: self.fail("uncertain delivery must not be retried"),
            include_existing=True,
        )
        second.process_once()
        status = second.status()
        record = next(
            item for item in status["recent_dispatches"] if item["message_id"] == message["id"]
        )
        self.assertEqual(record["state"], "delivery_uncertain")


    def _route_codex(self):
        queued = []
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda thread, prompt: queued.append((thread, prompt)) or (0, "queued"),
            include_existing=True,
        )
        return coordinator, queued

    def test_real_message_with_ping_like_body_is_still_routed(self):
        message = self.claude.send(
            "codex", "PING from the review: P1 stop-loss bug, please look", "review-ping-like"
        )
        coordinator, queued = self._route_codex()
        coordinator.process_once()
        self.assertEqual(len(queued), 1)
        self.assertIn(message["id"], queued[0][1])

    def test_watcher_key_on_substantive_body_is_still_routed(self):
        message = self.claude.send(
            "codex", "Real finding: the exit ladder skips MAX_HOLD", "claude-watcher-ping-misused"
        )
        coordinator, queued = self._route_codex()
        coordinator.process_once()
        self.assertEqual(len(queued), 1)
        self.assertIn(message["id"], queued[0][1])

    def test_watcher_shaped_message_in_wrong_direction_is_not_swallowed(self):
        self.codex.send(
            "claude",
            "PING from the Claude watcher: no traffic. Silence-triggered liveness "
            "check, not a scheduled heartbeat. No action needed beyond an ack if you are alive.",
            "claude-watcher-ping-reversed",
        )
        events = []
        coordinator = TaskCoordinator(
            self.mailbox,
            self.state,
            codex_thread="existing-thread",
            queue=lambda *args: (0, "queued"),
            on_event=events.append,
            include_existing=True,
        )
        result = coordinator.process_once()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["state"], "mailbox_notice")


class CodexThreadDiscoveryTests(unittest.TestCase):
    REPO = Path(r"C:\Users\gl450\polymarket_app")
    HOME = Path(r"C:\Users\gl450")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.mailbox = root / "mailbox.sqlite3"
        self.state = root / "coordinator.sqlite3"
        self.codex = Store(self.mailbox, "codex")
        self.claude = Store(self.mailbox, "claude")

    def tearDown(self):
        self.temp.cleanup()

    def register(self, role, session_id, cwd, last_seen):
        with self.codex.connect(True) as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (role, session_id, cwd, last_seen))

    def test_resolves_newest_in_scope_recent_codex_session(self):
        now = 1_000_000.0
        self.register("codex", "old", str(self.REPO), now - 3600)
        self.register("codex", "new", str(self.REPO / "backend"), now - 60)
        self.register("codex", "other-repo", str(self.HOME / "trading_app"), now - 1)
        self.register("claude", "claude-1", str(self.REPO), now)
        found = resolve_codex_thread(self.mailbox, self.REPO, self.HOME, now)
        self.assertEqual(found["session_id"], "new")

    def test_ignores_codex_sessions_older_than_24h(self):
        now = 1_000_000.0
        self.register("codex", "stale", str(self.REPO), now - 24 * 3600 - 1)
        self.assertIsNone(resolve_codex_thread(self.mailbox, self.REPO, self.HOME, now))

    def test_dispatch_uses_discovered_thread_without_flag(self):
        now = time.time()
        self.register("codex", "discovered-thread", str(self.REPO), now - 5)
        self.claude.send("codex", "please review", "auto-1")
        queued = []
        coordinator = TaskCoordinator(
            self.mailbox, self.state, root=self.REPO, home=self.HOME,
            queue=lambda thread, prompt: queued.append(thread) or (0, "queued"),
            on_event=lambda e: None, include_existing=True)
        coordinator.process_once()
        self.assertEqual(queued, ["discovered-thread"])

    def test_explicit_thread_flag_still_wins(self):
        self.register("codex", "discovered-thread", str(self.REPO), time.time())
        self.claude.send("codex", "please review", "auto-2")
        queued = []
        coordinator = TaskCoordinator(
            self.mailbox, self.state, codex_thread="pinned", root=self.REPO, home=self.HOME,
            queue=lambda thread, prompt: queued.append(thread) or (0, "queued"),
            on_event=lambda e: None, include_existing=True)
        coordinator.process_once()
        self.assertEqual(queued, ["pinned"])

    def test_no_registered_codex_session_leaves_mail_in_mailbox(self):
        self.claude.send("codex", "please review", "auto-3")
        events = []
        coordinator = TaskCoordinator(
            self.mailbox, self.state, root=self.REPO, home=self.HOME,
            queue=lambda *a: self.fail("must not queue"), on_event=events.append,
            include_existing=True)
        coordinator.process_once()
        self.assertEqual(events[0]["state"], "mailbox_only")
        self.assertIn("no recent Codex session registered", events[0]["detail"])

    def test_home_dir_codex_session_is_not_a_target(self):
        now = 1_000_000.0
        self.register("codex", "repo-thread", str(self.REPO), now - 60)
        self.register("codex", "home-thread", str(self.HOME), now - 1)
        found = resolve_codex_thread(self.mailbox, self.REPO, self.HOME, now)
        self.assertEqual(found["session_id"], "repo-thread")

    def test_dispatch_detail_names_the_thread(self):
        self.register("codex", "discovered-thread", str(self.REPO), time.time())
        self.claude.send("codex", "please review", "auto-4")
        events = []
        coordinator = TaskCoordinator(
            self.mailbox, self.state, root=self.REPO, home=self.HOME,
            queue=lambda thread, prompt: (0, "queued"), on_event=events.append,
            include_existing=True)
        coordinator.process_once()
        self.assertIn("discovered-thread", events[0]["detail"])

    def test_target_lookup_failure_leaves_message_deliverable(self):
        self.register("codex", "discovered-thread", str(self.REPO), time.time())
        message = self.claude.send("codex", "please review", "auto-5")
        broken = TaskCoordinator(
            self.mailbox, self.state, root=self.REPO, home=self.HOME,
            queue=lambda *a: self.fail("must not queue"), on_event=lambda e: None,
            include_existing=True)

        def boom():
            raise sqlite3.OperationalError("database is locked")

        broken._codex_target = boom
        with self.assertRaises(sqlite3.OperationalError):
            broken.process_once()
        queued = []
        healthy = TaskCoordinator(
            self.mailbox, self.state, root=self.REPO, home=self.HOME,
            queue=lambda thread, prompt: queued.append(thread) or (0, "queued"),
            on_event=lambda e: None)
        healthy.process_once()
        self.assertEqual(queued, ["discovered-thread"], message["id"])

    def test_run_survives_a_failing_cycle(self):
        stop_file = Path(self.temp.name) / "stop"
        calls = []

        class Flaky:
            def process_once(self):
                calls.append(1)
                if len(calls) == 1:
                    raise sqlite3.OperationalError("disk I/O error")
                stop_file.touch()
                return []

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        run(Flaky(), 0.1, port=port, stop_file=stop_file)
        self.assertEqual(len(calls), 2)

    def test_status_reports_missing_codex_session(self):
        coordinator = TaskCoordinator(self.mailbox, self.state, root=self.REPO, home=self.HOME,
                                      on_event=lambda e: None)
        status = coordinator.status()
        self.assertIsNone(status["codex_thread"])
        self.assertIn("trust", status["codex_thread_hint"])


if __name__ == "__main__":
    unittest.main()
