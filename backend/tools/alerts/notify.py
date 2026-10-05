"""Notification de-duplication and a Windows balloon notification."""

from __future__ import annotations

import subprocess
import sys


def should_notify(key: str, now: float, state: dict, cooldown_s: float) -> bool:
    last = state.get(key)
    if last is not None and now - last < cooldown_s:
        return False
    state[key] = now
    return True


def windows_toast(title: str, message: str) -> None:
    """Best effort, non-blocking. A failure to notify never fails the check (the log is the record)."""
    if sys.platform != "win32":
        return
    t = title.replace("'", "''")[:60]
    m = message.replace("'", "''")[:240]
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; $n.Icon = [System.Drawing.SystemIcons]::Information; "
        f"$n.BalloonTipTitle = '{t}'; $n.BalloonTipText = '{m}'; $n.Visible = $true; "
        "$n.ShowBalloonTip(15000); Start-Sleep -Seconds 16; $n.Dispose()"
    )
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass
