"""The recorder must never import the trading app: it runs beside it, not inside it."""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[3] / "tools" / "recorder"
FORBIDDEN = {"agents", "services", "database", "config", "clients", "main"}


def test_recorder_imports_nothing_from_the_app():
    files = sorted(PKG.glob("*.py"))
    assert len(files) >= 6
    for py in files:
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in FORBIDDEN, f"{py.name} imports {n}"
