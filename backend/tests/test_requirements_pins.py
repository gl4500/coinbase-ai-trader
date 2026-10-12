"""Requirement files must be installable together.

requirements-dev.txt includes requirements.txt via ``-r``; a package pinned in both files at
different versions makes ``pip install -r requirements-dev.txt`` fail (ResolutionImpossible).
Dependabot bumps only one file at a time, so this guards against the two drifting apart.
"""

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==\s*([^\s#;]+)")


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN.match(line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def test_dev_requirements_include_runtime_requirements():
    dev = (BACKEND / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert "-r requirements.txt" in (line.strip() for line in dev)


def test_no_package_pinned_to_two_versions():
    runtime = _pins(BACKEND / "requirements.txt")
    dev = _pins(BACKEND / "requirements-dev.txt")
    conflicts = {
        name: (runtime[name], dev[name])
        for name in runtime.keys() & dev.keys()
        if runtime[name] != dev[name]
    }
    assert conflicts == {}, (
        f"pinned differently in requirements.txt vs requirements-dev.txt: {conflicts}"
    )
