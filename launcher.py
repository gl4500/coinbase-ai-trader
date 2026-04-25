"""
Coinbase Trader — GUI Launcher
==================================
Dark-themed tkinter control panel that:
  - Starts / stops backend (FastAPI on :8001) and frontend (Vite on :5174)
  - Shows live status with colour-coded indicators
  - Streams log output from both processes
  - Opens the app in the default browser when ready
  - Handles clean shutdown on close
  - "Start on login" toggle (Windows registry HKCU — no admin required)

Compile to .exe:  run build_exe.ps1
"""

import os
import sys
import subprocess
import threading
import queue
import time
import webbrowser
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import scrolledtext, ttk
from pathlib import Path
from logging.handlers import RotatingFileHandler
import logging

# winreg is Windows-only — guard so unit tests run on Linux/Mac CI too
try:
    import winreg as _winreg
    _HAS_WINREG = True
except ImportError:
    _winreg = None
    _HAS_WINREG = False

_STARTUP_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
_STARTUP_APP_NAME = "CoinbaseAITrader"

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(sys.executable if getattr(sys, "frozen", False) else __file__).parent.resolve()
VENV         = ROOT / ".venv"
PYTHON       = VENV / "Scripts" / "python.exe"
BACKEND_DIR  = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

BACKEND_URL        = "http://localhost:8001"
BACKEND_STATUS_URL = f"{BACKEND_URL}/api/status"
FRONTEND_URL       = "http://localhost:5174"

# ── Launcher file log ─────────────────────────────────────────────────────────
_LOG_DIR = ROOT / "backend" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_file_handler = RotatingFileHandler(
    str(_LOG_DIR / "launcher.log"), maxBytes=2_000_000, backupCount=5, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
_launcher_log = logging.getLogger("launcher")
_launcher_log.addHandler(_file_handler)
_launcher_log.setLevel(logging.DEBUG)
_launcher_log.propagate = False

# ── Colours ───────────────────────────────────────────────────────────────────

BG        = "#0f0f1a"
BG_CARD   = "#1a1a2e"
BG_PANEL  = "#16213e"
ACCENT    = "#6366f1"   # indigo
GREEN     = "#4ade80"
RED       = "#f87171"
AMBER     = "#fbbf24"
TEXT      = "#e2e8f0"
TEXT_DIM  = "#64748b"
BORDER    = "#1e293b"


# ── Subprocess helper — always hides console windows ─────────────────────────

def _no_window() -> dict:
    """Return kwargs that suppress any console window on Windows."""
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": subprocess.CREATE_NO_WINDOW, "startupinfo": si}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _python_exe() -> str:
    """Return path to python to use: venv if exists, else sys.executable."""
    if PYTHON.exists():
        return str(PYTHON)
    return sys.executable


def _npm_cmd() -> str:
    """
    Resolve npm.cmd explicitly — windowed EXEs run with a stripped PATH so
    plain 'npm' raises WinError 2 even when npm is installed.
    Search order:
      1. Common Node.js install locations
      2. PATH entries inherited from the parent environment
    """
    import shutil

    # Common Windows Node.js install paths
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npm.cmd",
        Path(os.environ.get("APPDATA", "")) / "npm" / "npm.cmd",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs" / "npm.cmd",
        Path(os.environ.get("USERPROFILE", "")) / "AppData" / "Roaming" / "nvm" / "npm.cmd",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Fall back to PATH search (works in console sessions)
    found = shutil.which("npm") or shutil.which("npm.cmd")
    if found:
        return found

    raise FileNotFoundError(
        "npm not found. Install Node.js from https://nodejs.org/ "
        "then restart the launcher."
    )


def _is_url_up(url: str, timeout: float = 5.0) -> bool:
    try:
        urllib.request.urlopen(BACKEND_STATUS_URL, timeout=timeout)
        return True
    except Exception:
        return False


def _is_frontend_up(timeout: float = 1.5) -> bool:
    try:
        urllib.request.urlopen(FRONTEND_URL, timeout=timeout)
        return True
    except Exception:
        return False


# Ports owned by the AI trading app — the launcher must never kill these.
_TRADING_APP_PORTS = {8000, 5173}


def _free_port(port: int) -> None:
    """
    Kill the process bound to `port`.
    Guard: refuses to touch trading_app ports so both apps can coexist.
    """
    if port in _TRADING_APP_PORTS:
        _launcher_log.warning("Skipping _free_port(%d) — belongs to AI trading app", port)
        return
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL, timeout=5,
            **_no_window(),
        )
        for line in out.splitlines():
            if f":{port}" in line and ("LISTENING" in line or "LISTEN" in line):
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   **_no_window())
                    _launcher_log.info("Freed port %d (killed PID %s)", port, pid)
    except Exception:
        pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Terminate a process and all its children (Windows)."""
    if proc is None:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


# ── Start-on-login registry helpers ──────────────────────────────────────────

def _startup_exe_path() -> str:
    """
    Return the path that goes into the registry Run key.
    - When frozen (.exe): use the executable directly.
    - When running as a Python script: use 'pythonw.exe launcher.py'
      so no console window appears on login.
    """
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    python_dir  = Path(sys.executable).parent
    pythonw     = python_dir / "pythonw.exe"
    interpreter = str(pythonw) if pythonw.exists() else sys.executable
    script      = str(ROOT / "launcher.py")
    return f'"{interpreter}" "{script}"'


def _is_startup_enabled() -> bool:
    """Return True if the registry Run key for this app exists."""
    if not _HAS_WINREG:
        return False
    try:
        key = _winreg.OpenKey(
            _winreg.HKEY_CURRENT_USER,
            _STARTUP_REG_PATH,
            0, _winreg.KEY_READ,
        )
        _winreg.QueryValueEx(key, _STARTUP_APP_NAME)
        _winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _set_startup(enabled: bool) -> bool:
    """
    Add or remove the Windows registry Run key.
    Returns True on success, False on failure.
    No admin required — writes to HKCU (current user only).
    """
    if not _HAS_WINREG:
        return False
    try:
        key = _winreg.OpenKey(
            _winreg.HKEY_CURRENT_USER,
            _STARTUP_REG_PATH,
            0, _winreg.KEY_SET_VALUE,
        )
        if enabled:
            _winreg.SetValueEx(key, _STARTUP_APP_NAME, 0,
                               _winreg.REG_SZ, _startup_exe_path())
        else:
            try:
                _winreg.DeleteValue(key, _STARTUP_APP_NAME)
            except FileNotFoundError:
                pass   # already gone — not an error
        _winreg.CloseKey(key)
        return True
    except Exception as e:
        _launcher_log.warning("Startup registry write failed: %s", e)
        return False


# ── Main Application Window ───────────────────────────────────────────────────

class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coinbase Trader")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("780x620")
        self.minsize(620, 480)

        # Set window icon (crystal ball emoji fallback if no .ico)
        self._set_icon()

        self._backend_proc:  subprocess.Popen | None = None
        self._frontend_proc: subprocess.Popen | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self._running = True
        self._startup_var: tk.BooleanVar | None = None   # set in _build_ui

        self._build_ui()
        self._start_status_loop()
        self._start_log_drain()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-start all services 1 s after the window opens (only if not already running)
        self.after(1000, self._auto_start)

    # ── Icon ──────────────────────────────────────────────────────────────────

    def _set_icon(self):
        """Try to load launcher.ico next to this file; silently skip if missing."""
        ico = ROOT / "launcher.ico"
        if ico.exists():
            try:
                self.iconbitmap(str(ico))
                return
            except Exception:
                pass
        # Draw a simple canvas icon via PhotoImage (16×16 indigo square)
        try:
            img = tk.PhotoImage(width=64, height=64)
            img.put(ACCENT, to=(0, 0, 64, 64))
            self.iconphoto(True, img)
        except Exception:
            pass

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_PANEL, pady=12)
        hdr.pack(fill="x")

        tk.Label(hdr, text="🪙", font=("Segoe UI Emoji", 26), bg=BG_PANEL, fg=TEXT).pack(side="left", padx=(18, 8))
        title_frame = tk.Frame(hdr, bg=BG_PANEL)
        title_frame.pack(side="left")
        tk.Label(title_frame, text="Coinbase Trader",
                 font=("Segoe UI", 16, "bold"), bg=BG_PANEL, fg=TEXT).pack(anchor="w")
        tk.Label(title_frame, text="Advanced Trade · RSI · MACD · CNN",
                 font=("Segoe UI", 9), bg=BG_PANEL, fg=TEXT_DIM).pack(anchor="w")

        # Browser button in header
        tk.Button(
            hdr, text="🌐  Open App", command=self._open_browser,
            font=("Segoe UI", 9, "bold"),
            bg=ACCENT, fg="white", relief="flat",
            activebackground="#4f46e5", activeforeground="white",
            padx=14, pady=6, cursor="hand2",
        ).pack(side="right", padx=18)

        # ── Status cards ──────────────────────────────────────────────────────
        cards_frame = tk.Frame(self, bg=BG, pady=10)
        cards_frame.pack(fill="x", padx=16)
        cards_frame.columnconfigure(0, weight=1)
        cards_frame.columnconfigure(1, weight=1)

        self._backend_dot  = tk.Label(None)
        self._frontend_dot = tk.Label(None)
        self._backend_lbl  = tk.Label(None)
        self._frontend_lbl = tk.Label(None)

        self._backend_card  = self._make_service_card(
            cards_frame, "Backend", BACKEND_URL, 0,
            "dot_ref_b", "lbl_ref_b",
            self._start_backend, self._stop_backend,
        )
        self._frontend_card = self._make_service_card(
            cards_frame, "Frontend", FRONTEND_URL, 1,
            "dot_ref_f", "lbl_ref_f",
            self._start_frontend, self._stop_frontend,
        )

        # ── Master controls ───────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG, pady=4)
        ctrl.pack(fill="x", padx=16)

        btn_cfg = dict(font=("Segoe UI", 10, "bold"), relief="flat",
                       padx=18, pady=8, cursor="hand2")

        self._start_all_btn = tk.Button(
            ctrl, text="▶  Start All", command=self._start_all,
            bg=GREEN, fg="#0f0f0f",
            activebackground="#22c55e", activeforeground="#0f0f0f",
            **btn_cfg,
        )
        self._start_all_btn.pack(side="left", padx=(0, 8))

        self._stop_all_btn = tk.Button(
            ctrl, text="■  Stop All", command=self._stop_all,
            bg=RED, fg="white",
            activebackground="#ef4444", activeforeground="white",
            **btn_cfg,
        )
        self._stop_all_btn.pack(side="left", padx=(0, 8))

        tk.Button(
            ctrl, text="⟳  Clear Log", command=self._clear_log,
            bg=BG_CARD, fg=TEXT_DIM,
            activebackground=BG_PANEL, activeforeground=TEXT,
            font=("Segoe UI", 9), relief="flat", padx=12, pady=8, cursor="hand2",
        ).pack(side="right")

        # "Start on login" checkbox — Windows registry HKCU, no admin needed
        self._startup_var = tk.BooleanVar(value=_is_startup_enabled())
        startup_cb = tk.Checkbutton(
            ctrl,
            text="Start on login",
            variable=self._startup_var,
            command=self._on_startup_toggle,
            bg=BG, fg=TEXT_DIM,
            activebackground=BG, activeforeground=TEXT,
            selectcolor=BG_CARD,
            font=("Segoe UI", 9),
            relief="flat", cursor="hand2",
            bd=0,
        )
        startup_cb.pack(side="right", padx=(0, 12))

        # ── Separator ─────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(8, 0))

        # ── Log panel ─────────────────────────────────────────────────────────
        log_hdr = tk.Frame(self, bg=BG, pady=6)
        log_hdr.pack(fill="x", padx=16)
        tk.Label(log_hdr, text="LOG OUTPUT", font=("Segoe UI", 8, "bold"),
                 bg=BG, fg=TEXT_DIM).pack(side="left")

        self._log = scrolledtext.ScrolledText(
            self, bg="#0d0d1a", fg="#94a3b8",
            font=("Cascadia Code", 8) if self._font_exists("Cascadia Code") else ("Consolas", 8),
            relief="flat", bd=0, padx=10, pady=8,
            wrap="word", state="disabled",
            insertbackground=TEXT,
        )
        self._log.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # Tag colours for log lines
        self._log.tag_config("info",    foreground="#94a3b8")
        self._log.tag_config("backend", foreground="#818cf8")
        self._log.tag_config("frontend",foreground="#34d399")
        self._log.tag_config("warn",    foreground=AMBER)
        self._log.tag_config("error",   foreground=RED)
        self._log.tag_config("system",  foreground=ACCENT)

        self._log_line("system", "Coinbase Trader ready — auto-starting services…")
        self._log_line("system", f"Backend  → {BACKEND_URL}")
        self._log_line("system", f"Frontend → {FRONTEND_URL}")
        if _is_startup_enabled():
            self._log_line("system", "Start on login: ON  (toggle the checkbox to disable)")

    @staticmethod
    def _font_exists(name: str) -> bool:
        import tkinter.font as tkfont
        try:
            return name in tkfont.families()
        except Exception:
            return False

    def _make_service_card(self, parent, title, url, col,
                            dot_attr, lbl_attr, start_cb, stop_cb):
        card = tk.Frame(parent, bg=BG_CARD, padx=16, pady=12,
                        highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col else 0, 8 if col == 0 else 0), pady=4)
        if col == 0:
            card.grid(padx=(0, 6))
        else:
            card.grid(padx=(6, 0))

        top = tk.Frame(card, bg=BG_CARD)
        top.pack(fill="x")

        dot = tk.Label(top, text="●", font=("Segoe UI", 12), bg=BG_CARD, fg=TEXT_DIM)
        dot.pack(side="left")
        tk.Label(top, text=f"  {title}", font=("Segoe UI", 11, "bold"),
                 bg=BG_CARD, fg=TEXT).pack(side="left")

        btn_frame = tk.Frame(card, bg=BG_CARD)
        btn_frame.pack(fill="x", pady=(8, 0))

        b_cfg = dict(font=("Segoe UI", 8), relief="flat", padx=10, pady=4, cursor="hand2")
        tk.Button(btn_frame, text="Start", command=start_cb,
                  bg="#1e3a5f", fg="#93c5fd",
                  activebackground="#1d4ed8", activeforeground="white",
                  **b_cfg).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Stop", command=stop_cb,
                  bg="#3b1f2b", fg="#fca5a5",
                  activebackground="#7f1d1d", activeforeground="white",
                  **b_cfg).pack(side="left")

        lbl = tk.Label(card, text="Stopped", font=("Segoe UI", 8),
                       bg=BG_CARD, fg=TEXT_DIM)
        lbl.pack(anchor="w", pady=(4, 0))

        setattr(self, dot_attr, dot)
        setattr(self, lbl_attr, lbl)
        return card

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_line(self, tag: str, text: str):
        self._log.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._log.insert("end", f"[{ts}] {text}\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")
        # Mirror to rotating log file
        _launcher_log.info(f"[{tag.upper()}] {text}")

    def _queue_log(self, tag: str, text: str):
        self._log_queue.put((tag, text))

    def _start_log_drain(self):
        def drain():
            while self._running:
                try:
                    tag, text = self._log_queue.get(timeout=0.1)
                    self._log_line(tag, text)
                except queue.Empty:
                    pass
        threading.Thread(target=drain, daemon=True).start()

    def _stream_proc(self, proc: subprocess.Popen, tag: str):
        """Stream stdout+stderr of a process to the log panel."""
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lo = line.lower()
                t = "error" if "error" in lo or "traceback" in lo or "exception" in lo \
                    else "warn" if "warn" in lo \
                    else tag
                self._queue_log(t, line)

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _on_startup_toggle(self):
        """Called when the 'Start on login' checkbox is clicked."""
        enabled = self._startup_var.get()
        ok = _set_startup(enabled)
        if ok:
            state = "enabled" if enabled else "disabled"
            self._log_line("system", f"Start on login {state}.")
        else:
            # Roll back the checkbox if the write failed
            self._startup_var.set(not enabled)
            self._log_line("warn",
                "Could not write to registry — start-on-login unchanged."
                "  (winreg unavailable or access denied)"
            )

    # ── Status loop ───────────────────────────────────────────────────────────

    def _start_status_loop(self):
        _BE_FAIL_THRESH = 3
        be_fail_count = 0

        def loop():
            nonlocal be_fail_count
            while self._running:
                be_up = _is_url_up(BACKEND_URL)
                fe_up = _is_frontend_up()
                self.after(0, self._update_status, be_up, fe_up)
                if be_up:
                    be_fail_count = 0
                elif self._backend_proc and self._backend_proc.poll() is None:
                    # Process alive but not accepting connections — WinError 64
                    be_fail_count += 1
                    if be_fail_count >= _BE_FAIL_THRESH:
                        be_fail_count = 0
                        self._queue_log(
                            "warn",
                            "Backend not responding (3 checks) — auto-restarting (WinError 64 recovery)…",
                        )
                        self.after(0, self._restart_backend)
                else:
                    be_fail_count = 0
                time.sleep(1)

        threading.Thread(target=loop, daemon=True).start()

    def _restart_backend(self):
        self._log_line("warn", "Restarting backend…")
        if self._backend_proc:
            _kill_tree(self._backend_proc)
            self._backend_proc = None
        self._start_backend()

    def _update_status(self, be_up: bool, fe_up: bool):
        self.dot_ref_b.config(fg=GREEN if be_up else RED)
        self.dot_ref_f.config(fg=GREEN if fe_up else RED)
        self.lbl_ref_b.config(
            text=f"Running — {BACKEND_URL}" if be_up else "Stopped",
            fg=GREEN if be_up else TEXT_DIM,
        )
        self.lbl_ref_f.config(
            text=f"Running — {FRONTEND_URL}" if fe_up else "Stopped",
            fg=GREEN if fe_up else TEXT_DIM,
        )

    # ── Venv setup ────────────────────────────────────────────────────────────

    def _ensure_venv(self, on_done):
        """
        Ensure the venv exists AND all backend requirements are installed.
        Checks for fastapi in site-packages (not just python.exe) so a
        build-only venv (pyinstaller/pillow only) is correctly detected.
        Runs in a background thread; calls on_done in the main thread when ready.
        """
        def _run():
            req      = ROOT / "backend" / "requirements.txt"
            pip      = VENV / "Scripts" / "pip.exe"
            fastapi  = VENV / "Lib" / "site-packages" / "fastapi"

            # 1. Create venv if python.exe is missing
            if not PYTHON.exists():
                self._log_line("system", "Virtual environment not found — creating…")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "venv", str(VENV)],
                        capture_output=True, text=True,
                        **_no_window(),
                    )
                    if result.returncode != 0:
                        self._log_line("error", f"venv creation failed: {result.stderr.strip()}")
                        return
                    self._log_line("system", "Virtual environment created.")
                except Exception as e:
                    self._log_line("error", f"venv creation failed: {e}")
                    return

            # 2. Install requirements if fastapi is not yet in the venv
            #    (covers: fresh venv, build-only venv, or partial install)
            if not fastapi.exists():
                self._log_line("system", "Backend packages not installed — running pip install…")
                self._log_line("system", "This takes ~1 minute on first run.")
                try:
                    proc = subprocess.Popen(
                        [str(pip), "install", "-r", str(req)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, encoding="utf-8", errors="replace",
                        **_no_window(),
                    )
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line:
                            self._queue_log("system", line)
                    proc.wait()
                    if proc.returncode != 0:
                        self._log_line("error", "pip install failed — check the log above.")
                        return
                    self._log_line("system", "Backend packages installed.")
                except Exception as e:
                    self._log_line("error", f"pip install failed: {e}")
                    return
            else:
                self._log_line("system", "Backend packages OK.")

            self.after(0, on_done)

        threading.Thread(target=_run, daemon=True).start()

    # ── Process control ───────────────────────────────────────────────────────

    def _start_backend(self):
        if _is_url_up(BACKEND_URL):
            self._log_line("system", "Backend already responding — skipping start.")
            return
        if self._backend_proc and self._backend_proc.poll() is None:
            self._log_line("warn", "Backend process exists but not responding.")
            return

        python = _python_exe()
        self._log_line("system", f"Starting backend with {Path(python).name}…")
        self._log_line("system", "Freeing port 8001 if occupied…")
        _free_port(8001)

        try:
            # Run via `python main.py` so main.py's port-cleanup and
            # RotatingFileHandler logging initialise correctly — same pattern
            # as the AI trading app.
            self._backend_proc = subprocess.Popen(
                [python, "main.py"],
                cwd=str(BACKEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                **_no_window(),
            )
            threading.Thread(
                target=self._stream_proc,
                args=(self._backend_proc, "backend"),
                daemon=True,
            ).start()
            self._log_line("backend", f"Backend PID {self._backend_proc.pid} started.")
        except Exception as e:
            self._log_line("error", f"Backend start failed: {e}")

    def _start_frontend(self):
        if _is_frontend_up():
            self._log_line("system", "Frontend already responding — skipping start.")
            return
        if self._frontend_proc and self._frontend_proc.poll() is None:
            self._log_line("warn", "Frontend process exists but not responding.")
            return

        try:
            npm = _npm_cmd()
        except FileNotFoundError as e:
            self._log_line("error", str(e))
            return

        self._log_line("system", f"npm: {npm}")

        def _run():
            # Install node_modules if vite binary is missing (first run or incomplete install)
            vite_bin = FRONTEND_DIR / "node_modules" / ".bin" / "vite.cmd"
            if not vite_bin.exists():
                self._log_line("system", "node_modules not found — running npm install (~1 min)…")
                try:
                    proc = subprocess.Popen(
                        [npm, "install"],
                        cwd=str(FRONTEND_DIR),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        encoding="utf-8", errors="replace",
                        **_no_window(),
                    )
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line:
                            self._queue_log("frontend", line)
                    proc.wait()
                    if proc.returncode != 0:
                        self._log_line("error", "npm install failed — check log above.")
                        return
                    self._log_line("system", "npm install complete.")
                except Exception as e:
                    self._log_line("error", f"npm install failed: {e}")
                    return

            # Start Vite dev server
            self._log_line("system", "Starting frontend (npm run dev)…")
            try:
                self._frontend_proc = subprocess.Popen(
                    [npm, "run", "dev"],
                    cwd=str(FRONTEND_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    encoding="utf-8", errors="replace",
                    **_no_window(),
                )
                threading.Thread(
                    target=self._stream_proc,
                    args=(self._frontend_proc, "frontend"),
                    daemon=True,
                ).start()
                self._log_line("frontend", f"Frontend PID {self._frontend_proc.pid} started.")
            except Exception as e:
                self._log_line("error", f"Frontend start failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _stop_backend(self):
        if self._backend_proc:
            _kill_tree(self._backend_proc)
            self._backend_proc = None
            self._log_line("system", "Backend stopped.")

    def _stop_frontend(self):
        if self._frontend_proc:
            _kill_tree(self._frontend_proc)
            self._frontend_proc = None
            self._log_line("system", "Frontend stopped.")

    def _auto_start(self):
        """Called once on launch — only starts if services aren't already up."""
        be_up = _is_url_up(BACKEND_URL)
        fe_up = _is_frontend_up()
        if be_up and fe_up:
            self._log_line("system", "Services already running — skipping auto-start.")
            return
        self._start_all()

    def _start_all(self):
        if _is_url_up(BACKEND_URL) and _is_frontend_up():
            self._log_line("warn", "Services already running — use Stop All first.")
            return
        self._log_line("system", "═══ Starting all services… ═══")
        # Ensure venv + packages exist before launching — handles first-run
        self._ensure_venv(on_done=self._start_all_after_venv)

    def _start_all_after_venv(self):
        self._start_backend()
        # Poll backend every 0.5 s for up to 60 s (mirrors trading_app pattern)
        def _wait_and_start_fe():
            self._log_line("system", "Waiting for backend to be ready (up to 60 s)…")
            deadline = time.time() + 60
            while time.time() < deadline:
                if _is_url_up(BACKEND_URL):
                    self._log_line("system", "Backend ready — starting frontend.")
                    self.after(0, self._start_frontend)
                    self.after(5000, self._open_browser_if_up)
                    return
                time.sleep(0.5)
            self._log_line("warn", "Backend did not respond in 60 s — starting frontend anyway.")
            self.after(0, self._start_frontend)
        threading.Thread(target=_wait_and_start_fe, daemon=True).start()

    def _stop_all(self):
        self._log_line("system", "═══ Stopping all services… ═══")
        self._stop_backend()
        self._stop_frontend()

    def _open_browser(self):
        """Open App button — wait up to 15 s for Vite before opening."""
        def _wait_and_open():
            if _is_frontend_up(timeout=0.5):
                webbrowser.open(FRONTEND_URL)
                return
            self._log_line("system", "Waiting for frontend to be ready…")
            deadline = time.time() + 15
            while time.time() < deadline:
                if _is_frontend_up():
                    self._log_line("system", f"Opening browser → {FRONTEND_URL}")
                    webbrowser.open(FRONTEND_URL)
                    return
                time.sleep(0.5)
            self._log_line("warn", f"Frontend not responding — opening anyway: {FRONTEND_URL}")
            webbrowser.open(FRONTEND_URL)
        threading.Thread(target=_wait_and_open, daemon=True).start()

    def _open_browser_if_up(self):
        if _is_frontend_up():
            self._log_line("system", f"Opening browser → {FRONTEND_URL}")
            webbrowser.open(FRONTEND_URL)
        else:
            self._log_line("warn", "Frontend not yet ready — open the browser manually.")

    def _on_close(self):
        self._running = False
        self._stop_all()
        self.after(500, self.destroy)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
