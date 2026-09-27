"""New mail must never be invisible, and a truncated window must say it truncated.

Found by execution on 2026-09-27, from the operator's report that the link "isn't seeing
anything new". It was not the peer. `inbox()` read

    ... WHERE recipient=? AND acknowledged IS NULL ORDER BY created LIMIT 50

-- the OLDEST fifty. Once fifty unacknowledged messages accumulate (mine reached 65), every
subsequent message lands outside the window, so the reader sees a frozen snapshot of old
mail and concludes the peer has gone quiet. The peer had answered; I could not see it, and
I reported "no reply yet" to the operator on a question that had been answered.

`wait()` inherits the same query, so the long-poll built earlier the same day could not
wake on a message it was unable to see. A backlog silently disabled the feature.

Two properties, and the second is the one that generalises: a bounded window is fine, but
a bound that hides its own effect is not. This is the same defect `wait()` already avoids
by reporting a timeout AS a timeout rather than as an empty list -- an empty answer and a
truncated answer both have to be distinguishable from a complete one.
"""
import tempfile
import unittest
from pathlib import Path

from session_bridge.store import Store


class InboxVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'bridge.sqlite3'
        self.codex = Store(self.db, 'codex')
        self.claude = Store(self.db, 'claude')

    def tearDown(self):
        self.temp.cleanup()

    def _flood(self, count):
        """Send `count` messages oldest-first, returning their bodies in send order."""
        bodies = []
        for i in range(count):
            body = 'msg-%03d' % i
            self.codex.send('claude', body, 'k%03d' % i)
            bodies.append(body)
        return bodies

    def test_the_newest_message_is_visible_behind_a_backlog(self):
        """The actual defect. 65 unacknowledged is the state that produced it live."""
        bodies = self._flood(65)
        seen = [m['body'] for m in self.claude.inbox()]
        self.assertIn(bodies[-1], seen,
                      'newest message invisible behind a backlog -- the reported bug')

    def test_a_message_arriving_into_a_full_window_becomes_visible(self):
        """Arrival, not just accumulation: the window was already full when this was sent."""
        self._flood(Store.INBOX_WINDOW)
        self.codex.send('claude', 'the-one-that-matters', 'late')
        seen = [m['body'] for m in self.claude.inbox()]
        self.assertIn('the-one-that-matters', seen)

    def test_wait_wakes_on_a_message_it_could_not_previously_see(self):
        """`wait()` calls `inbox()`, so the truncation disabled the long-poll too. Without
        the fix this returns 50 stale messages and never surfaces the new one."""
        self._flood(Store.INBOX_WINDOW + 10)
        self.codex.send('claude', 'wake-for-this', 'wake')
        result = self.claude.wait(timeout_secs=5.0, poll_secs=0.1)
        self.assertEqual(result['status'], 'messages')
        self.assertIn('wake-for-this', [m['body'] for m in result['messages']])

    def test_truncation_is_reported_rather_than_silent(self):
        """A bounded window is acceptable; a window that hides its own effect is not. Had
        this existed, the live 65-vs-50 gap would have been visible instead of inferred."""
        self._flood(Store.INBOX_WINDOW + 15)
        self.assertEqual(self.claude.pending_count(), Store.INBOX_WINDOW + 15)
        self.assertEqual(len(self.claude.inbox()), Store.INBOX_WINDOW)

        result = self.claude.wait(timeout_secs=5.0, poll_secs=0.1)
        self.assertEqual(result['pending_total'], Store.INBOX_WINDOW + 15)
        self.assertEqual(result['truncated'], 15)

    def test_an_untruncated_window_reports_no_truncation(self):
        """Non-vacuity for the field above: it must not read 'truncated' unconditionally."""
        self._flood(3)
        self.assertEqual(self.claude.pending_count(), 3)
        result = self.claude.wait(timeout_secs=5.0, poll_secs=0.1)
        self.assertEqual(result['truncated'], 0)
        self.assertEqual(result['pending_total'], 3)

    def test_the_window_is_still_chronological(self):
        """Newest-biased selection, oldest-first presentation. Reading order carries the
        thread of a conversation, and reversing it would be a different regression."""
        self._flood(Store.INBOX_WINDOW + 5)
        created = [m['created'] for m in self.claude.inbox()]
        self.assertEqual(created, sorted(created))

    def test_acknowledging_the_backlog_reveals_the_older_mail(self):
        """The cost of newest-bias is that old mail leaves the window. It must come back as
        the backlog drains, or the fix would merely move the blind spot to the other end."""
        bodies = self._flood(Store.INBOX_WINDOW + 5)
        self.assertNotIn(bodies[0], [m['body'] for m in self.claude.inbox()])
        for message in self.claude.inbox():
            self.claude.ack(message['id'])
        self.assertIn(bodies[0], [m['body'] for m in self.claude.inbox()])

    # ---- the window's SIZE, not just its direction -------------------------------------
    # 50 was inherited from the original query and bounds the wrong quantity. Measured over
    # 535 real messages: median 614 bytes but p90 3,577, so fifty messages is ~7.7k tokens
    # typically and 57k in the worst case observed. What a reader can afford is bytes; the
    # count cap only stops pathological row counts.

    def test_the_window_is_bounded_by_bytes_not_only_by_count(self):
        """Ten 5KB messages must not all arrive at once just because ten is under 50."""
        big = 'x' * 5000
        for i in range(10):
            self.codex.send('claude', '%s-%03d' % (big, i), 'big%03d' % i)
        got = self.claude.inbox()
        self.assertLess(len(got), 10, 'byte budget did not bind on large messages')
        billed = sum(len(m['body'].encode('utf-8')) for m in got)
        self.assertLessEqual(billed, Store.INBOX_MAX_BYTES + 5000,
                             'returned more than the budget plus one message of slack')

    def test_small_messages_still_fill_the_window_to_the_count_cap(self):
        """Non-vacuity for the test above: the byte budget must not throttle normal traffic.
        At the observed median of ~614 bytes the count cap is what binds, not the budget."""
        for i in range(Store.INBOX_WINDOW + 5):
            self.codex.send('claude', 'small-%03d' % i, 's%03d' % i)
        self.assertEqual(len(self.claude.inbox()), Store.INBOX_WINDOW)

    def test_a_message_larger_than_the_whole_budget_is_still_delivered(self):
        """Otherwise the budget makes a legitimate message permanently unreadable and the
        link wedges -- a worse failure than the one being fixed. The cap on send() is 20000
        characters, so such a message is constructible by design."""
        huge = 'y' * (Store.INBOX_MAX_BYTES + 1000)
        self.codex.send('claude', huge, 'huge')
        got = self.claude.inbox()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['body'], huge)

    def test_the_newest_survives_the_byte_budget_not_the_oldest(self):
        """Trimming has to come off the old end. Trimming the new end would reinstate the
        original defect through a different mechanism."""
        big = 'z' * 6000
        for i in range(8):
            self.codex.send('claude', '%s-%03d' % (big, i), 'z%03d' % i)
        got = self.claude.inbox()
        self.assertTrue(got[-1]['body'].endswith('-007'), 'newest trimmed by the byte budget')

    def test_truncation_counts_bytes_trimmed_as_truncated_too(self):
        """A byte-trimmed read is a partial read and must say so, the same as a count-trimmed
        one. Silent partiality is the whole defect this file exists for."""
        big = 'w' * 6000
        for i in range(8):
            self.codex.send('claude', '%s-%03d' % (big, i), 'w%03d' % i)
        result = self.claude.wait(timeout_secs=5.0, poll_secs=0.1)
        self.assertEqual(result['pending_total'], 8)
        self.assertEqual(result['truncated'], 8 - len(result['messages']))
        self.assertGreater(result['truncated'], 0)

    def test_an_explicit_budget_overrides_the_default(self):
        """Callers draining a backlog deliberately need to widen both bounds."""
        for i in range(6):
            self.codex.send('claude', 'q' * 4000, 'q%03d' % i)
        self.assertEqual(len(self.claude.inbox(limit=6, max_bytes=10 ** 7)), 6)

    def test_pending_count_is_scoped_to_the_reader(self):
        """A count that leaked the peer's backlog would misreport the link's state."""
        self._flood(4)
        self.claude.send('codex', 'one for them', 'mine')
        self.assertEqual(self.claude.pending_count(), 4)
        self.assertEqual(self.codex.pending_count(), 1)
