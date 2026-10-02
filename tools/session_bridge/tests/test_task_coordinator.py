import tempfile
import unittest
from pathlib import Path

from session_bridge.store import Store
from task_coordinator import TaskCoordinator


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

if __name__ == "__main__":
    unittest.main()
