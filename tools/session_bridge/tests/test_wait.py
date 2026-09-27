"""Long-poll `wait`: one blocking call instead of a token-burning poll loop.

The problem this solves is economic, not functional. `inbox()` already works, but a peer
with nothing to do called it repeatedly and paid for every empty answer. `wait()` spends
WALL CLOCK inside one process instead, and returns the moment a message lands.

So the properties that matter are about WHEN it returns and what it says about silence:
it must not sleep when the answer is already known, it must actually wake on arrival
rather than on its own timer, it must never let the other side's write block behind it,
and a timeout must be reported as a timeout rather than as an empty inbox.
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path

from session_bridge.store import Store


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'bridge.sqlite3'
        self.codex = Store(self.db, 'codex')
        self.claude = Store(self.db, 'claude')

    def tearDown(self):
        self.temp.cleanup()

    def test_returns_at_once_when_the_answer_is_already_known(self):
        """A pending message must not be delayed by the poll interval."""
        self.codex.send('claude', 'already here', 'k1')
        start = time.monotonic()
        result = self.claude.wait(timeout_secs=30.0, poll_secs=5.0)
        elapsed = time.monotonic() - start

        self.assertEqual(result['status'], 'messages')
        self.assertEqual([m['body'] for m in result['messages']], ['already here'])
        self.assertLess(elapsed, 1.0, 'waited on a message that was already deliverable')

    def test_wakes_on_arrival_not_on_its_own_timer(self):
        """The whole point. A message sent DURING the wait must return early.

        Falsification: if wait() ignored arrivals and simply slept out its timeout, elapsed
        would be ~4s rather than ~0.3s, and this assertion is what catches that.
        """
        def send_soon():
            time.sleep(0.3)
            Store(self.db, 'codex').send('claude', 'arrived late', 'k2')

        sender = threading.Thread(target=send_soon)
        start = time.monotonic()
        sender.start()
        result = self.claude.wait(timeout_secs=4.0, poll_secs=0.05)
        elapsed = time.monotonic() - start
        sender.join()

        self.assertEqual(result['status'], 'messages')
        self.assertEqual([m['body'] for m in result['messages']], ['arrived late'])
        self.assertLess(elapsed, 3.0, 'wait slept through an arrival instead of waking on it')
        self.assertGreater(result['waited_secs'], 0.0)

    def test_a_waiting_reader_never_blocks_the_senders_write(self):
        """If wait held a write transaction while sleeping, the peer could not send at all.

        This is the failure that would make the feature worse than polling: the sender
        would block or time out against a reader that is doing nothing but waiting.
        """
        errors = []

        def send_during_wait():
            time.sleep(0.2)
            try:
                Store(self.db, 'codex').send('claude', 'not blocked', 'k3')
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(exc)

        sender = threading.Thread(target=send_during_wait)
        sender.start()
        result = self.claude.wait(timeout_secs=5.0, poll_secs=0.05)
        sender.join()

        self.assertEqual(errors, [], f'sender blocked behind a waiting reader: {errors}')
        self.assertEqual(result['status'], 'messages')

    def test_timeout_is_reported_as_a_timeout_not_as_an_empty_inbox(self):
        """Silence must be self-describing. An empty list alone cannot say whether the
        link is quiet or the call gave up, and the caller acts differently on each."""
        start = time.monotonic()
        result = self.claude.wait(timeout_secs=0.4, poll_secs=0.05)
        elapsed = time.monotonic() - start

        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['messages'], [])
        self.assertGreaterEqual(elapsed, 0.4)
        self.assertEqual(result['timeout_secs'], 0.4)

    def test_waiting_does_not_acknowledge_what_it_returns(self):
        """Reading is not acking, same contract as notice(). A crash after wait() must not
        have silently consumed the message."""
        self.codex.send('claude', 'still mine', 'k4')
        self.claude.wait(timeout_secs=1.0, poll_secs=0.05)
        self.assertEqual(len(self.claude.inbox()), 1)
        self.assertIsNone(self.claude.inbox()[0]['acknowledged'])

    def test_only_the_recipients_own_mail_wakes_them(self):
        """A message addressed to the other role must not satisfy this role's wait."""
        self.claude.send('codex', 'for codex only', 'k5')
        result = self.claude.wait(timeout_secs=0.3, poll_secs=0.05)
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['messages'], [])

    def test_timeout_and_poll_are_validated_without_coercion(self):
        """A blocking call in an agent's tool loop must not be able to hang forever, and
        bool is not a number: True would otherwise sail through as 1 second.
        """
        for bad in (0, -1, True, False, '5', None, float('nan'), float('inf')):
            with self.assertRaises(ValueError, msg=f'timeout_secs={bad!r} was accepted'):
                self.claude.wait(timeout_secs=bad, poll_secs=0.05)
        for bad in (0, -1, True, '1', None, float('inf')):
            with self.assertRaises(ValueError, msg=f'poll_secs={bad!r} was accepted'):
                self.claude.wait(timeout_secs=1.0, poll_secs=bad)
        with self.assertRaises(ValueError):
            self.claude.wait(timeout_secs=Store.MAX_WAIT_SECS + 1, poll_secs=0.05)
        with self.assertRaises(ValueError):
            self.claude.wait(timeout_secs=1.0, poll_secs=2.0)

    def test_poll_never_outlasts_the_timeout_it_serves(self):
        """A poll interval longer than the remaining budget would overshoot the deadline,
        turning a 1s timeout into a multi-second stall."""
        start = time.monotonic()
        result = self.claude.wait(timeout_secs=0.5, poll_secs=0.4)
        elapsed = time.monotonic() - start
        self.assertEqual(result['status'], 'timeout')
        self.assertLess(elapsed, 1.5, 'overshot its own deadline by a whole poll interval')


class WaitDispatchTests(unittest.TestCase):
    """The store method is useless to a peer unless the CLI/flow layer admits it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'bridge.sqlite3'

    def tearDown(self):
        self.temp.cleanup()

    def test_wait_is_reachable_through_dispatch(self):
        from session_bridge.main import dispatch

        Store(self.db, 'codex').send('claude', 'via dispatch', 'd1')
        result = dispatch(self.db, 'claude', 'wait', {'timeout_secs': 5.0, 'poll_secs': 0.05})
        self.assertEqual(result['status'], 'messages')
        self.assertEqual([m['body'] for m in result['messages']], ['via dispatch'])

    def test_dispatch_still_rejects_operations_that_do_not_exist(self):
        """Widening the allow-list must not turn it into a passthrough to any attribute."""
        from session_bridge.main import dispatch

        for bogus in ('connect', 'normalize_scopes', 'nonsense', '__init__'):
            with self.assertRaises(Exception, msg=f'dispatch accepted {bogus!r}'):
                dispatch(self.db, 'claude', bogus, {})


if __name__ == '__main__':
    unittest.main()
