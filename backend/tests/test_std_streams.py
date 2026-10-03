"""pythonw (the logon autostart task) starts with sys.stdout/sys.stderr = None. uvicorn's default
log formatter calls sys.stdout.isatty(), so the backend exited 1 before logging anything."""

import subprocess
import sys
from pathlib import Path

import pytest

from services.std_streams import ensure_std_streams


def test_missing_streams_are_pointed_at_an_append_only_log(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    log = tmp_path / "logs" / "console.log"
    stream = ensure_std_streams(log)
    try:
        assert sys.stdout is stream and sys.stderr is stream
        print("hello from pythonw")
        assert not sys.stdout.isatty()
    finally:
        stream.close()
    assert "hello from pythonw" in log.read_text(encoding="utf-8")


def test_present_streams_are_left_alone(tmp_path):
    before = (sys.stdout, sys.stderr)
    assert ensure_std_streams(tmp_path / "console.log") is None
    assert (sys.stdout, sys.stderr) == before
    assert not (tmp_path / "console.log").exists()


def test_uvicorn_logging_configures_once_streams_are_guarded(tmp_path, monkeypatch):
    from uvicorn.config import Config

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    with pytest.raises(ValueError):  # the original failure
        Config(app="x:y")
    stream = ensure_std_streams(tmp_path / "console.log")
    try:
        Config(app="x:y")  # configure_logging runs in __init__
    finally:
        stream.close()


@pytest.mark.skipif(sys.platform != "win32", reason="pythonw is Windows-only")
def test_real_pythonw_without_handles_reaches_uvicorn_logging(tmp_path):
    """Exactly what Task Scheduler does: pythonw with no inherited console handles."""
    pyw = Path(sys.executable).with_name("pythonw.exe")
    if not pyw.exists():
        pytest.skip("no pythonw next to this interpreter")
    backend = Path(__file__).resolve().parents[1]
    result = tmp_path / "result.txt"
    script = tmp_path / "probe.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(backend)!r})\n"
        "from services.std_streams import ensure_std_streams\n"
        f"ensure_std_streams({str(tmp_path / 'console.log')!r})\n"
        "from uvicorn.config import Config\n"
        "Config(app='x:y')\n"
        f"open({str(result)!r}, 'w').write('ok')\n"
    )
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.run(
        [str(pyw), str(script)],
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        close_fds=True,
        creationflags=flags,
        timeout=60,
    )
    assert proc.returncode == 0 and result.read_text() == "ok"
