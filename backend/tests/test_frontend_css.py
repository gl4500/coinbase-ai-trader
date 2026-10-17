"""Compile the actual frontend stylesheet to catch build and badge regressions.

Run with the frontend npm dependencies installed; no backend or network is used.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
pytestmark = pytest.mark.skipif(
    not (FRONTEND / "node_modules/postcss/package.json").exists(),
    reason="CSS integration checks require npm install in frontend",
)


@pytest.fixture(scope="module")
def compiled_css():
    script = r"""
import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import postcss from 'postcss';
const config = (await import(pathToFileURL(path.resolve('postcss.config.js')))).default;
const plugins = await Promise.all(Object.entries(config.plugins).map(async ([name, options]) => {
  const plugin = (await import(name)).default;
  return plugin(options);
}));
const from = path.resolve('src/index.css');
const input = await fs.readFile(from, 'utf8');
const result = await postcss(plugins).process(input, { from });
const rules = {};
result.root.walkRules(rule => {
  const declarations = {};
  rule.nodes.filter(n => n.type === 'decl').forEach(n => declarations[n.prop] = n.value);
  for (const selector of rule.selector.split(',')) {
    rules[selector.trim()] = { ...rules[selector.trim()], ...declarations };
  }
});
const unresolved = [];
result.root.walkAtRules(rule => {
  if (['apply', 'tailwind', 'config', 'utility'].includes(rule.name)) unresolved.push(rule.name);
});
console.log(JSON.stringify({rules, unresolved}));
"""
    node = shutil.which("node")
    assert node, "Node.js is required for the frontend CSS regression checks"
    result = subprocess.run(
        [node, "--input-type=module", "-"],
        input=script,
        text=True,
        capture_output=True,
        cwd=FRONTEND,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_stylesheet_compiles_without_unresolved_tailwind_directives(compiled_css):
    assert compiled_css["unresolved"] == []
    assert compiled_css["rules"][".card"]["padding"]
    assert compiled_css["rules"]["body"]["background-color"]


@pytest.mark.parametrize("color", ["green", "red", "blue", "purple", "amber", "gray"])
def test_colored_badges_retain_base_layout_and_color(compiled_css, color):
    rule = compiled_css["rules"][f".badge-{color}"]
    assert rule["display"] == "inline-flex"
    assert rule["align-items"] == "center"
    assert rule["border-radius"]
    assert rule["font-weight"]
    assert "padding-inline" in rule or "padding-left" in rule
    assert "background-color" in rule
    assert "color" in rule
