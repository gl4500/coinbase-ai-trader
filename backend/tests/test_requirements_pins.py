"""Requirement files must be installable together.

requirements-dev.txt includes requirements.txt via ``-r``; a package pinned in both files at
different versions makes ``pip install -r requirements-dev.txt`` fail (ResolutionImpossible).
Dependabot bumps only one file at a time, so this guards against the two drifting apart.

Scope: exact ``==`` pins only (the format both files use); ranges and markers are not parsed.
Names are normalized per PyPA (https://packaging.python.org/en/latest/specifications/name-normalization/).
"""

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==\s*([^\s#;]+)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict[str, set[str]]:
    """Every exact pin per normalized name; conflicting duplicates are kept, not overwritten."""
    pins: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN.match(line.strip())
        if m:
            pins.setdefault(_normalize(m.group(1)), set()).add(m.group(2))
    return pins


def test_dev_requirements_include_runtime_requirements():
    dev = (BACKEND / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    assert "-r requirements.txt" in (line.strip() for line in dev)


def test_no_package_pinned_to_two_versions():
    runtime = _pins(BACKEND / "requirements.txt")
    dev = _pins(BACKEND / "requirements-dev.txt")
    versions = {
        name: runtime.get(name, set()) | dev.get(name, set())
        for name in runtime.keys() | dev.keys()
    }
    conflicts = {name: sorted(v) for name, v in versions.items() if len(v) > 1}
    assert conflicts == {}, (
        f"pinned differently in requirements.txt vs requirements-dev.txt: {conflicts}"
    )


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_names_are_normalized_per_pypa(tmp_path):
    pins = _pins(_write(tmp_path, "r.txt", "Pytest.Asyncio==0.24.0\npytest__cov==6.0.0\n"))
    assert pins == {"pytest-asyncio": {"0.24.0"}, "pytest-cov": {"6.0.0"}}


def test_conflicting_pins_within_one_file_are_all_kept(tmp_path):
    pins = _pins(_write(tmp_path, "r.txt", "pytest-asyncio==0.24.0\npytest-asyncio==1.3.0\n"))
    assert pins == {"pytest-asyncio": {"0.24.0", "1.3.0"}}


def test_identical_duplicates_and_extras_are_not_conflicts(tmp_path):
    pins = _pins(
        _write(tmp_path, "r.txt", "bandit[toml]==1.8.3\nbandit==1.8.3  # again\nruff>=0.9\n")
    )
    assert pins == {"bandit": {"1.8.3"}}
