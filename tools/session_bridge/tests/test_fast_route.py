"""Two routes to the same mailbox, and the guarantee that they cannot diverge.

Importing crewai.flow costs ~4.3s measured, to run an allow-list check and one getattr.
For message operations that is the whole cost of a call, so those get a direct route.

Nothing is removed: the CrewAI Flow stays and stays reachable, because AGENTS.md is
explicit that silencing its instrumentation is the operator's decision and not a
performance fix. This makes the fast route the default and the Flow route selectable.

The risk in having two routes is the defect class this project keeps hitting -- one fact
in two places with only one of them checked. So the allow-list has a single home and an
equivalence test pins the two routes to the same answers.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from session_bridge.store import Store

LAUNCHER = Path(__file__).resolve().parents[1] / 'bridge.py'


class FastRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'bridge.sqlite3'

    def tearDown(self):
        self.temp.cleanup()

    def test_both_routes_admit_exactly_the_same_operations(self):
        """One allow-list, not two. If the sets ever differ, an operation is reachable by
        one route and refused by the other, which is worse than either alone.

        This PROBES both routes rather than comparing two declarations. Asserting that two
        helpers return the same constant would pass no matter how the routes behaved --
        coverage derived from the thing under test proves nothing.
        """
        from session_bridge import main

        candidates = set(main.ALLOWED_OPERATIONS) | {
            'connect', 'normalize_scopes', 'task_dict', 'notice', '__init__', 'nope'}

        # 'wait' BLOCKS for timeout_secs, whose default is 300. Probing it with an empty
        # payload made this test sleep for five minutes per route before I caught it, so the
        # probe supplies a minimal non-blocking payload for anything that would block.
        payloads = {'wait': {'timeout_secs': 0.01, 'poll_secs': 0.01}}

        def reachable(route):
            found = set()
            for op in candidates:
                try:
                    route(self.db, 'claude', op, payloads.get(op, {}))
                except Exception as exc:  # noqa: BLE001 - classifying, not handling
                    if 'unknown bridge operation' not in str(exc):
                        found.add(op)  # got past the gate, failed later on its own terms
                else:
                    found.add(op)
            return found

        self.assertEqual(reachable(main.dispatch), reachable(main.dispatch_direct))
        # Non-vacuity: the probe must actually be discriminating, not accept everything.
        admitted = reachable(main.dispatch_direct)
        self.assertIn('inbox', admitted)
        self.assertNotIn('nope', admitted)
        self.assertNotIn('connect', admitted)

    def test_both_routes_return_identical_results(self):
        """Equivalence on real calls, not just on the allow-list."""
        from session_bridge.main import dispatch, dispatch_direct

        Store(self.db, 'codex').send('claude', 'same both ways', 'k1')
        via_flow = dispatch(self.db, 'claude', 'inbox', {})
        direct = dispatch_direct(self.db, 'claude', 'inbox', {})
        self.assertEqual(via_flow, direct)
        self.assertEqual([m['body'] for m in direct], ['same both ways'])

    def test_both_routes_reject_the_same_unknown_operations(self):
        from session_bridge.main import dispatch, dispatch_direct

        for bogus in ('connect', 'normalize_scopes', '__init__', 'nonsense'):
            with self.assertRaises(Exception, msg=f'flow accepted {bogus!r}'):
                dispatch(self.db, 'claude', bogus, {})
            with self.assertRaises(Exception, msg=f'direct accepted {bogus!r}'):
                dispatch_direct(self.db, 'claude', bogus, {})

    def test_the_fast_route_really_does_not_import_crewai(self):
        """The entire point, and the only assertion that can falsify the speed claim.

        Run in a subprocess so the check is about what the CLI actually loads, not about
        what this test process happens to have imported already.
        """
        script = (
            'import sys, json;'
            'sys.path.insert(0, %r);'
            'from session_bridge.main import dispatch_direct;'
            'dispatch_direct(%r, "claude", "inbox", {});'
            'print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] == "crewai")))'
            % (str(LAUNCHER.parent / 'src'), str(self.db))
        )
        out = subprocess.run([sys.executable, '-c', script], capture_output=True,
                             text=True, timeout=120,
                             encoding='utf-8', errors='replace')
        self.assertEqual(out.returncode, 0, out.stderr)
        last = [ln for ln in out.stdout.splitlines() if ln.strip()][-1]
        self.assertEqual(json.loads(last.strip()), [],
                         'the fast route pulled crewai in after all')

    def test_the_flow_route_is_still_reachable_and_still_uses_crewai(self):
        """AGENTS.md: silencing CrewAI instrumentation is the operator's call, so the
        instrumented path must remain available rather than deleted."""
        script = (
            'import sys, json;'
            'sys.path.insert(0, %r);'
            'from session_bridge.main import dispatch;'
            'dispatch(%r, "claude", "inbox", {});'
            'print(json.dumps(any(m.split(".")[0] == "crewai" for m in sys.modules)))'
            % (str(LAUNCHER.parent / 'src'), str(self.db))
        )
        out = subprocess.run([sys.executable, '-c', script], capture_output=True,
                             text=True, timeout=180,
                             encoding='utf-8', errors='replace')
        self.assertEqual(out.returncode, 0, out.stderr)
        # CrewAI prints a Flow panel to stdout, so only the final line is our JSON.
        last = [ln for ln in out.stdout.splitlines() if ln.strip()][-1]
        self.assertTrue(json.loads(last.strip()),
                        'the Flow route no longer goes through crewai')

    def test_cli_uses_the_fast_route_for_messages_but_honours_an_explicit_override(self):
        """Default fast for message ops; `--route flow` forces the instrumented path."""
        Store(self.db, 'codex').send('claude', 'cli route', 'k2')

        def run(extra):
            proc = subprocess.run(
                [sys.executable, str(LAUNCHER), '--role', 'claude', '--db', str(self.db),
                 *extra, 'call', 'inbox'],
                capture_output=True, text=True, timeout=180,
                encoding='utf-8', errors='replace')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            text = proc.stdout[proc.stdout.index('['):]
            return json.loads(text)

        self.assertEqual([m['body'] for m in run([])], ['cli route'])
        self.assertEqual([m['body'] for m in run(['--route', 'flow'])], ['cli route'])


if __name__ == '__main__':
    unittest.main()
