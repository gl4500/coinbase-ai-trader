"""Every supported entry point funnels through startup_guard; none kills a port owner itself."""

import ast
import importlib
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _src(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def _func_names(tree):
    return {
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _func_src(src, name):
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)


def test_main_has_no_free_port_and_calls_guard():
    src = _src("backend/main.py")
    tree = ast.parse(src)
    assert "_free_port" not in _func_names(tree) and "_free_port(" not in src
    main_block = next(
        n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)
    )
    block = ast.unparse(main_block)
    assert "startup_guard.run_or_exit(" in block
    assert block.index("startup_guard.run_or_exit(") < block.index("uvicorn.run(")


def test_start_backend_ps1_has_no_kill_and_supports_replace():
    src = _src("start_backend.ps1")
    assert "taskkill" not in src.lower() and "stop-process" not in src.lower()
    assert "[switch]$Replace" in src and "--mode" in src and "replace" in src


def test_start_app_has_no_kill_path():
    src = _src("start_app.py")
    assert "taskkill" not in src and "_free_port" not in src


def test_start_app_passes_replace_through():
    import start_app

    importlib.reload(start_app)
    assert start_app._backend_args(["start_app.py"]) == ["main.py"]
    assert start_app._backend_args(["start_app.py", "--replace"]) == [
        "main.py",
        "--mode",
        "replace",
    ]


def _launcher():
    try:
        import launcher
    except Exception as e:  # tkinter missing on some runners
        pytest.skip(f"launcher import unavailable: {e}")
    return importlib.reload(launcher)


def test_launcher_has_no_port_killer():
    src = _src("launcher.py")
    assert "_free_port" not in src
    assert "taskkill" not in _func_src(src, "_start_backend")


def test_launcher_backend_command():
    launcher = _launcher()
    assert launcher._backend_cmd("py.exe") == ["py.exe", "main.py"]
    assert launcher._backend_cmd("py.exe", "replace-unhealthy") == [
        "py.exe",
        "main.py",
        "--mode",
        "replace-unhealthy",
    ]


def test_launcher_restart_plan_kills_only_its_own_child():
    launcher = _launcher()
    assert launcher._restart_plan(own_child_running=True) == "kill_own_child_then_start"
    assert launcher._restart_plan(own_child_running=False) == "spawn_replace_unhealthy"


def test_launcher_explains_guard_exit_codes():
    launcher = _launcher()
    assert "already running" in launcher._guard_exit_message(2)
    assert "lock" in launcher._guard_exit_message(3)
    assert "verif" in launcher._guard_exit_message(4)
    assert launcher._guard_exit_message(0) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell parser")
def test_start_backend_ps1_parses_in_windows_powershell():
    # Without a UTF-8 BOM, Windows PowerShell 5.1 reads the em dashes as ANSI and
    # one byte becomes a closing quote, breaking the script (4 parse errors before).
    import subprocess

    path = os.path.join(REPO, "start_backend.ps1")
    with open(path, "rb") as f:
        assert f.read(3) == b"\xef\xbb\xbf"
    cmd = (
        "$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{path}',[ref]$null,[ref]$e); $e.Count"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60
    )
    assert out.stdout.strip() == "0", out.stdout + out.stderr
