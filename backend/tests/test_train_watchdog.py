"""
Tests for _is_training_stale helper in main.py.

Rationale: train_worker.py only writes cnn_train_progress.json at start and
at end. Between those two points, the file's mtime stays at startup time,
so it cannot be used to detect a hung worker. Instead, staleness is based
on two signals:

  1. The run has been in 'running' state for at least a grace period
     (training takes a while to get past phase 1 data loading).
  2. backend/logs/cnn_training.log has not been written to for at least
     the log-staleness window.
"""
import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _fn():
    import main
    return main._is_training_stale


class TestTrainingStaleness:
    def test_not_running_is_never_stale(self):
        data = {"status": "idle"}
        assert _fn()(data, log_mtime=0.0, now=1_000_000.0) is False

    def test_completed_is_not_stale(self):
        data = {"status": "completed", "started_at": 0.0}
        assert _fn()(data, log_mtime=0.0, now=1_000_000.0) is False

    def test_running_without_started_at_is_not_stale(self):
        data = {"status": "running"}
        assert _fn()(data, log_mtime=100.0, now=999_999.0) is False

    def test_running_within_startup_grace_is_not_stale(self):
        # Started 5 minutes ago, grace is 30 min → not stale.
        now = 1_000_000.0
        data = {"status": "running", "started_at": now - 5 * 60}
        assert _fn()(data, log_mtime=now - 20 * 60, now=now) is False

    def test_running_with_recent_log_is_not_stale(self):
        # Ran for 2 hours but log was written 1 minute ago → healthy.
        now = 1_000_000.0
        data = {"status": "running", "started_at": now - 2 * 3600}
        assert _fn()(data, log_mtime=now - 60, now=now) is False

    def test_running_with_stale_log_after_grace_is_stale(self):
        # Ran for 2 hours, last log 45 minutes ago → past the 30-min window.
        now = 1_000_000.0
        data = {"status": "running", "started_at": now - 2 * 3600}
        assert _fn()(data, log_mtime=now - 45 * 60, now=now) is True

    def test_missing_log_mtime_is_not_stale(self):
        # Can't determine → don't kill.
        now = 1_000_000.0
        data = {"status": "running", "started_at": now - 2 * 3600}
        assert _fn()(data, log_mtime=None, now=now) is False

    def test_log_stale_threshold_is_30_min(self):
        import main
        assert main._TRAIN_STALE_LOG_SECS == 1800

    def test_running_log_idle_20m_is_not_stale(self):
        # Phase-2 dataset build logs every ~10-13 min; 20 min idle must NOT trip
        # the watchdog under the new 30-min threshold.
        now = 1_000_000.0
        data = {"status": "running", "started_at": now - 2 * 3600}
        assert _fn()(data, log_mtime=now - 20 * 60, now=now) is False
