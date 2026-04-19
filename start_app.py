"""
Coinbase Trader — CLI Launcher
===================================
Starts the backend and frontend in separate console windows, then opens
the browser once both services are ready.

Usage:
    python start_app.py                # normal start
    python start_app.py --no-browser   # skip auto-open
    python start_app.py --startup      # enable start-on-login (Windows registry)
    python start_app.py --no-startup   # disable start-on-login

Mirrors the AI Trading Competition launcher.py pattern exactly:
  - CREATE_NEW_CONSOLE so each service has its own window (killable independently)
  - Polls /api/status every 0.5 s, 60 s timeout (fast fail detection)
  - Kills any stale process on port 8001 before starting
  - No shell=True anywhere (no command injection surface)
"""

import os
import sys
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

try:
    import winreg as _winreg
    _HAS_WINREG = True
except ImportError:
    _winreg = None
    _HAS_WINREG = False

_STARTUP_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
_STARTUP_APP_NAME = "CoinbaseAITrader"

# ── Resolve paths relative to this file (works frozen or not) ─────────────────
ROOT         = Path(sys.executable if getattr(sys, "frozen", False) else __file__).parent.resolve()
VENV         = ROOT / ".venv"
PYTHON       = VENV / "Scripts" / "python.exe"
BACKEND_DIR  = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

BACKEND_STATUS_URL = "http://localhost:8001/api/status"
FRONTEND_URL       = "http://localhost:5174"

NO_BROWSER   = "--no-browser"  in sys.argv
SET_STARTUP  = "--startup"     in sys.argv
CLR_STARTUP  = "--no-startup"  in sys.argv


def _startup_cmd() -> str:
    """Registry value: pythonw.exe + path to launcher.py (no console window on login)."""
    python_dir  = Path(sys.executable).parent
    pythonw     = python_dir / "pythonw.exe"
    interpreter = str(pythonw) if pythonw.exists() else sys.executable
    script      = str(ROOT / "launcher.py")
    return f'"{interpreter}" "{script}"'


def _set_startup(enabled: bool) -> None:
    if not _HAS_WINREG:
        print("[WARN] winreg not available — start-on-login not supported.")
        return
    try:
        key = _winreg.OpenKey(
            _winreg.HKEY_CURRENT_USER,
            _STARTUP_REG_PATH, 0, _winreg.KEY_SET_VALUE,
        )
        if enabled:
            _winreg.SetValueEx(key, _STARTUP_APP_NAME, 0,
                               _winreg.REG_SZ, _startup_cmd())
            print("[INFO] Start-on-login ENABLED.")
        else:
            try:
                _winreg.DeleteValue(key, _STARTUP_APP_NAME)
            except FileNotFoundError:
                pass
            print("[INFO] Start-on-login DISABLED.")
        _winreg.CloseKey(key)
    except Exception as e:
        print(f"[WARN] Registry write failed: {e}")

# ── Port ownership map ────────────────────────────────────────────────────────
# Explicit declaration of which ports belong to which app.
# _free_port() will NEVER touch ports that belong to another known app.
COINBASE_APP_PORTS = {8001, 5174}   # ports this app owns
TRADING_APP_PORTS = {8000, 5173}    # trading_app ports — never kill these


# ── Helpers ───────────────────────────────────────────────────────────────────

def _python_exe() -> str:
    """Prefer venv Python; fall back to the Python running this script."""
    return str(PYTHON) if PYTHON.exists() else sys.executable


def _check_port_occupied(port: int) -> bool:
    """Return True if something is already listening on `port`."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        return any(f":{port}" in l and ("LISTENING" in l or "LISTEN" in l)
                   for l in out.splitlines())
    except Exception:
        return False


def _free_port(port: int) -> None:
    """
    Kill the process bound to `port`.
    Safety guard: raises if `port` is in TRADING_APP_PORTS so we never
    accidentally kill the AI trading app.
    """
    if port in TRADING_APP_PORTS:
        raise RuntimeError(
            f"Port {port} belongs to the AI Trading App — refusing to free it."
        )
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        for line in out.splitlines():
            if f":{port}" in line and ("LISTENING" in line or "LISTEN" in line):
                pid = line.split()[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    print(f"[INFO] Freed port {port} (killed PID {pid})")
    except RuntimeError:
        raise
    except Exception:
        pass


def _warn_if_trading_app_running() -> None:
    """Print a friendly notice if the AI trading app appears to be running."""
    if _check_port_occupied(8000):
        print("[NOTE] AI Trading App backend detected on port 8000 — no conflict.")
        print("       Both apps use separate ports and can run simultaneously.")
        print()


def _wait_for_backend(timeout: int = 60) -> bool:
    """Poll /api/status every 0.5 s until the backend responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BACKEND_STATUS_URL, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _wait_for_frontend(timeout: int = 15) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(FRONTEND_URL, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ── Launch sequence ───────────────────────────────────────────────────────────

def main():
    print()
    print("  Coinbase Trader")
    print("  ==================")
    print()

    # Handle startup-toggle flags before doing anything else
    if SET_STARTUP:
        _set_startup(True)
        print("       Re-run without --startup to just launch normally.")
        print()
    if CLR_STARTUP:
        _set_startup(False)
        print()

    python = _python_exe()
    print(f"[INFO] Python: {python}")

    # 0. Detect other apps — friendly notice, never interfere
    _warn_if_trading_app_running()

    # 1. Free port 8001 if stale process is occupying it
    print("[INFO] Checking port 8001...")
    _free_port(8001)

    # 2. Start backend with no console window
    print("[INFO] Starting backend...")
    subprocess.Popen(
        [python, "main.py"],
        cwd=str(BACKEND_DIR),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    # 3. Poll until backend is serving (mirrors trading_app exactly)
    print("[INFO] Waiting for backend (up to 60 s)...")
    if _wait_for_backend(timeout=60):
        print("[INFO] Backend ready.")
    else:
        print("[WARN] Backend did not respond in 60 s — starting frontend anyway.")

    # 4. Start frontend in its own console window
    print("[INFO] Starting frontend...")
    env = os.environ.copy()
    env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"

    # Resolve npm.cmd explicitly — stripped PATH in GUI processes won't find it
    import shutil
    npm = None
    for candidate in [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npm.cmd",
        Path(os.environ.get("APPDATA", "")) / "npm" / "npm.cmd",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs" / "npm.cmd",
    ]:
        if candidate.exists():
            npm = str(candidate)
            break
    if not npm:
        npm = shutil.which("npm") or shutil.which("npm.cmd") or "npm"

    print(f"[INFO] npm: {npm}")
    subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        creationflags=subprocess.CREATE_NO_WINDOW,
        env=env,
    )

    # 5. Wait for Vite, then open browser
    if not NO_BROWSER:
        print("[INFO] Waiting for frontend...")
        if _wait_for_frontend(timeout=15):
            print(f"[INFO] Opening browser at {FRONTEND_URL}")
            webbrowser.open(FRONTEND_URL)
        else:
            print(f"[WARN] Frontend not ready — open {FRONTEND_URL} manually.")

    print()
    print("[INFO] Both services running in the background (no console windows).")
    print("[INFO] To stop them, end 'python.exe' and 'node.exe' in Task Manager.")
    print(f"[INFO] App: {FRONTEND_URL}")
    print(f"[INFO] API: http://localhost:8001/docs")
    print()


if __name__ == "__main__":
    main()
