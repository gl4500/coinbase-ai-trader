import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('ping_monitor', Path(__file__).parents[1] / 'ping_monitor.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class PingTests(unittest.TestCase):
    def test_first_cycle_is_due(self):
        self.assertEqual(monitor.decision({}, None, 1000), 'queue')

    def test_unanswered_cycle_never_floods_queue(self):
        state = {'token': 'one', 'queued_at': 1000}
        self.assertEqual(monitor.decision(state, None, 1100), 'waiting')
        self.assertEqual(monitor.decision(state, None, 1181), 'stale')
        self.assertEqual(monitor.decision(state, None, 100000), 'stale')

    def test_receipt_must_match_challenge(self):
        state = {'token': 'one', 'queued_at': 1000}
        self.assertEqual(monitor.decision(state, {'token': 'old', 'at': 1190}, 1200), 'stale')

    def test_acknowledged_cycle_waits_before_next(self):
        state = {'token': 'one', 'queued_at': 1000}
        receipt = {'token': 'one', 'at': 1010}
        self.assertEqual(monitor.decision(state, receipt, 1100), 'responded')
        self.assertEqual(monitor.decision(state, receipt, 1130), 'queue')


if __name__ == '__main__':
    unittest.main()
