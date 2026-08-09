import inspect
import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
import main  # noqa: E402


def test_endpoint_exists_and_is_async():
    assert hasattr(main, "get_diagnostics")
    assert inspect.iscoroutinefunction(main.get_diagnostics)
