"""Snyk runs periodically, not on every push: the free plan's monthly test quota is small.

Operator request 2026-11-20. Every push to every branch used to run two `snyk test` calls,
which exhausted the quota. Snyk now lives in its own workflow (weekly, manual, and on PRs
that change dependency files), and the per-push pipeline no longer runs or waits for it.
"""

import re
import unittest
from pathlib import Path
from typing import Any, Dict

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_BOOL = "tag:yaml.org,2002:bool"


class _WorkflowLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans, as GitHub reads workflows.

    Plain YAML 1.1 turns the bare key `on` into the boolean True, so a lookup of "on" would
    miss the triggers entirely. Only true/false are booleans here.
    """


_WorkflowLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_WorkflowLoader.add_implicit_resolver(
    _BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def _load(name: str) -> Dict[str, Any]:
    """The parsed workflow ``.github/workflows/<name>``, read with YAML 1.2 booleans."""
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=_WorkflowLoader)


def _triggers(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The workflow's ``on:`` mapping. Raises KeyError if the workflow declares none."""
    return doc["on"]


def _run_text(job: Dict[str, Any]) -> str:
    """Every ``run:`` script of ``job``'s steps, newline-joined; steps without one add ""."""
    return "\n".join(step.get("run", "") for step in job["steps"])


class PerPushPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ci = _load("ci.yml")

    def test_no_ci_job_runs_snyk(self) -> None:
        for name, job in self.ci["jobs"].items():
            uses = [step.get("uses", "") for step in job.get("steps", [])]
            self.assertFalse(any("snyk" in u for u in uses), name)
            self.assertNotIn("snyk test", _run_text(job), name)

    def test_the_security_gate_does_not_wait_for_snyk(self) -> None:
        self.assertEqual(
            self.ci["jobs"]["security-gate"]["needs"],
            ["secret-scan", "sast-python", "test-backend", "frontend-build", "lint-python"],
        )


class SnykWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = _load("snyk.yml")
        self.on = _triggers(self.doc)

    def test_triggers_are_weekly_manual_and_dependency_prs_only(self) -> None:
        self.assertEqual(set(self.on), {"schedule", "workflow_dispatch", "pull_request"})
        self.assertEqual(self.on["schedule"], [{"cron": "0 6 * * 1"}])
        pull_request = self.on["pull_request"]
        self.assertEqual(pull_request["branches"], ["main"])
        self.assertEqual(
            set(pull_request["paths"]),
            {
                "backend/requirements*.txt",
                "backend/.snyk",
                "frontend/package.json",
                "frontend/package-lock.json",
                "frontend/.snyk",
                ".github/workflows/snyk.yml",
            },
        )

    def test_the_scan_is_unchanged_high_severity_sca_for_both_ecosystems(self) -> None:
        (job,) = self.doc["jobs"].values()
        steps = job["steps"]
        names = [step.get("name", "") for step in steps]
        self.assertIn("Check SNYK_TOKEN present", names)
        guarded = [s for s in steps if "snyk" in (s.get("uses", "") + s.get("run", ""))]
        self.assertTrue(guarded)
        for step in guarded:
            self.assertEqual(step.get("if"), "steps.tok.outputs.present == 'true'")

        backend = next(s for s in steps if s.get("working-directory") == "backend")
        frontend = next(s for s in steps if s.get("working-directory") == "frontend")
        run = backend["run"]
        self.assertLess(run.index("pip install -r requirements.txt"), run.index("snyk test"))
        self.assertIn("--package-manager=pip", run)
        self.assertIn("--severity-threshold=high", run)
        run = frontend["run"]
        self.assertLess(run.index("npm install"), run.index("snyk test"))
        self.assertIn("--severity-threshold=high", run)
        for step in (backend, frontend):
            self.assertEqual(step["env"]["SNYK_TOKEN"], "${{ secrets.SNYK_TOKEN }}")

    def test_a_missing_token_is_reported_as_not_scanned(self) -> None:
        """A token-less run concludes success, so its summary must say nothing was scanned
        rather than read as coverage."""
        (job,) = self.doc["jobs"].values()
        check = next(s for s in job["steps"] if s.get("name") == "Check SNYK_TOKEN present")
        self.assertIn("GITHUB_STEP_SUMMARY", check["run"])
        self.assertIn("not scanned", check["run"])

    def test_the_loader_keeps_on_as_a_key(self) -> None:
        self.assertIn("on", self.doc)
        self.assertNotIn(True, self.doc)

    def test_it_requests_read_only_repository_access(self) -> None:
        self.assertEqual(self.doc["permissions"], {"contents": "read"})

    def test_every_action_is_pinned_to_a_full_commit_sha(self) -> None:
        """A branch (``@master``) or tag can move under us; only a full SHA is immutable."""
        (job,) = self.doc["jobs"].values()
        uses = [step["uses"] for step in job["steps"] if "uses" in step]
        self.assertTrue(uses)
        for ref in uses:
            self.assertRegex(ref, r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$", ref)


if __name__ == "__main__":
    unittest.main()
