import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
import pytest  # noqa: E402

from services import diagnostics as d  # noqa: E402

_NOW = 1_700_000_000.0  # fixed epoch for determinism


class TestWindowCutoff:
    def test_all_is_none(self):
        assert d.window_cutoff("all", _NOW) is None

    def test_30d_is_iso_30_days_back(self):
        cut = d.window_cutoff("30d", _NOW)
        assert cut is not None and cut.endswith("+00:00") and "T" in cut

    def test_90d_older_than_30d(self):
        assert d.window_cutoff("90d", _NOW) < d.window_cutoff("30d", _NOW)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            d.window_cutoff("7d", _NOW)
