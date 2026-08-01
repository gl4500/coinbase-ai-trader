import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def test_scorecard_cli_help():
    """--help should exit 0 and mention --track / --cache."""
    result = subprocess.run(
        [str(PYTHON), "-m", "tools.scorecard", "--help"],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--track" in result.stdout
    assert "--cache" in result.stdout


def test_scorecard_cli_missing_track_arg():
    """Running without --track should fail with a usage error."""
    result = subprocess.run(
        [str(PYTHON), "-m", "tools.scorecard"],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


def test_scorecard_cli_v4_not_implemented():
    """--track v4 should fail loudly — v1 implements v3 only."""
    result = subprocess.run(
        [str(PYTHON), "-m", "tools.scorecard", "--track", "v4"],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "not implemented" in result.stderr.lower()


@pytest.mark.slow
def test_scorecard_cli_v3_smoke():
    """Smoke run on the real cache — should print a report and exit 0.

    Marked slow: loads the ~167k-sample cache and retrains 5 XGB folds.
    Skip in fast CI; run manually with: pytest -m slow.
    """
    cache_path = BACKEND / "cnn_dataset_cache.pt"
    if not cache_path.exists():
        pytest.skip(f"cache not present at {cache_path}")
    result = subprocess.run(
        [str(PYTHON), "-m", "tools.scorecard", "--track", "v3"],
        cwd=str(BACKEND),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0
    assert "precision" in result.stdout.lower()
    assert "ece" in result.stdout.lower()
    assert "recommended" in result.stdout.lower()
