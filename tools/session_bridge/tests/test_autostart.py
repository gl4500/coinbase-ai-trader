import socket
import tempfile
import unittest
from pathlib import Path

from session_bridge.autostart import ensure_coordinator


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class EnsureCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / ".coordination" / "session-link.sqlite3"
        self.calls = []
        self.port = free_port()

    def tearDown(self):
        self.temp.cleanup()

    def spawn(self, argv, cwd, log):
        self.calls.append((argv, cwd, log))

    def ensure(self, db=None):
        return ensure_coordinator(self.root, db or self.db, spawn=self.spawn, port=self.port)

    def test_free_lock_spawns_coordinator_run_once(self):
        self.assertEqual(self.ensure(), "spawned")
        argv, cwd, log = self.calls[0]
        self.assertEqual(argv[-2:], [str(self.root / "tools/session_bridge/task_coordinator.py"), "run"])
        self.assertEqual(cwd, self.root)
        self.assertEqual(log, self.root / ".coordination" / "task-coordinator.log")

    def test_held_lock_means_running(self):
        holder = socket.socket()
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        holder.bind(("127.0.0.1", self.port))
        try:
            self.assertEqual(self.ensure(), "running")
        finally:
            holder.close()
        self.assertEqual(self.calls, [])

    def test_operator_stop_file_is_respected(self):
        (self.root / ".coordination").mkdir()
        (self.root / ".coordination" / "task-coordinator.stop").touch()
        self.assertEqual(self.ensure(), "stopped")
        self.assertEqual(self.calls, [])

    def test_non_default_mailbox_never_spawns(self):
        self.assertEqual(self.ensure(self.root / "elsewhere.sqlite3"), "skipped: non-default mailbox")
        self.assertEqual(self.calls, [])

    def test_spawn_failure_is_returned_not_raised(self):
        def boom(*_):
            raise OSError("no python")
        result = ensure_coordinator(self.root, self.db, spawn=boom, port=self.port)
        self.assertTrue(result.startswith("error: OSError"))
