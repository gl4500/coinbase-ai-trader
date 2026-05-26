"""Verify MONOLITH_SCAN_DISABLED env gates the scan-loop spawn in main.lifespan.

Phase 3 cutover (per docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md)
sets MONOLITH_SCAN_DISABLED=true to silence the monolith's scan loop once
model_service is driving v3 inference. Default (unset) preserves today's behavior.
"""
import os

import pytest


def test_scan_loop_helper_respects_env(monkeypatch):
    from main import _should_run_monolith_scan
    monkeypatch.delenv("MONOLITH_SCAN_DISABLED", raising=False)
    assert _should_run_monolith_scan() is True
    monkeypatch.setenv("MONOLITH_SCAN_DISABLED", "true")
    assert _should_run_monolith_scan() is False
    monkeypatch.setenv("MONOLITH_SCAN_DISABLED", "false")
    assert _should_run_monolith_scan() is True
    monkeypatch.setenv("MONOLITH_SCAN_DISABLED", "1")
    assert _should_run_monolith_scan() is False
