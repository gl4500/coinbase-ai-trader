import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from session_bridge.store import Store


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'bridge.sqlite3'
        self.codex = Store(self.db, 'codex')
        self.claude = Store(self.db, 'claude')

    def tearDown(self):
        self.temp.cleanup()

    def test_delivery_survives_restart_and_requires_recipient_ack(self):
        msg = self.codex.send('claude', 'hello', 'key1')
        self.assertEqual(Store(self.db, 'claude').inbox()[0]['body'], 'hello')
        with self.assertRaises(ValueError):
            self.codex.ack(msg['id'])
        self.assertEqual(len(self.claude.inbox()), 1)
        self.claude.ack(msg['id'])
        self.assertEqual(self.claude.inbox(), [])
        self.assertEqual(self.claude.ack(msg['id'])['id'], msg['id'])

    def test_idempotency_cannot_silently_change_message(self):
        first = self.codex.send('claude', 'hello', 'key1')
        self.assertEqual(first, self.codex.send('claude', 'hello', 'key1'))
        with self.assertRaises(ValueError):
            self.codex.send('claude', 'different', 'key1')

    def test_atomic_task_claim_and_owner_only_updates(self):
        def claim(role):
            try:
                return Store(self.db, role).claim('one', 'Task one', ['backend'])['owner']
            except ValueError:
                return None
        with ThreadPoolExecutor(2) as pool:
            winners = list(pool.map(claim, ['codex', 'claude']))
        owner = next(x for x in winners if x)
        self.assertEqual(sum(x is not None for x in winners), 1)
        loser = Store(self.db, 'claude' if owner == 'codex' else 'codex')
        with self.assertRaises(ValueError):
            loser.update_task('one', 'done', 'stolen')

    def test_overlapping_paths_block_other_owner_until_done(self):
        self.claude.claim('labels', 'labels', ['backend/models'])
        with self.assertRaises(ValueError):
            self.codex.claim('other', 'other', ['BACKEND/models/file.py'])
        self.claude.update_task('labels', 'done', 'ready')
        self.codex.claim('other', 'other', ['backend/models/file.py'])

    def test_invalid_scopes_and_roles_rejected(self):
        for scope in ['../outside', 'C:/outside', '/root', 'backend/*']:
            with self.assertRaises(ValueError):
                self.codex.claim('bad', 'bad', [scope])
        with self.assertRaises(ValueError):
            Store(self.db, 'unknown')

    def test_hook_notices_deduplicated_and_do_not_ack(self):
        self.codex.send('claude', 'please reply', 'key1')
        self.assertEqual(len(self.claude.notice('session1', 'worktree')), 1)
        self.assertEqual(self.claude.notice('session1', 'worktree'), [])
        self.assertEqual(len(self.claude.inbox()), 1)
        self.assertEqual(len(self.claude.notice('session2', 'worktree')), 1)


if __name__ == '__main__':
    unittest.main()
