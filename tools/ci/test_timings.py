"""Exercise timing artifacts at process boundaries and reject incomplete merges."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from timings import merge_reports


class TimingMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reports = [
            {
                "schema": 1,
                "shard": i,
                "splits": 2,
                "source": "same-sha",
                "exitstatus": 0,
                "selected": [f"test_sample.py::test_{i}"],
                "durations": {f"test_sample.py::test_{i}": float(i)},
            }
            for i in (1, 2)
        ]

    def merge(self, reports):
        paths = []
        for i, report in enumerate(reports):
            path = self.root / f"{i}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path)
        return merge_reports(paths, expected_shards=2)

    def test_merge_retains_every_selected_test_duration(self):
        durations, metadata = self.merge(self.reports)
        self.assertEqual(durations, {"test_sample.py::test_1": 1.0, "test_sample.py::test_2": 2.0})
        self.assertEqual(metadata["selected_tests"], 2)

    def test_missing_duplicate_failed_or_different_source_shards_block(self):
        cases = [self.reports[:1]]
        for field, value in (("shard", 1), ("exitstatus", 1), ("source", "other-sha")):
            reports = copy.deepcopy(self.reports)
            reports[1][field] = value
            cases.append(reports)
        for reports in cases:
            with self.subTest(reports=reports), self.assertRaises(ValueError):
                self.merge(reports)

    def test_overlapping_or_unmeasured_tests_block(self):
        cases = []
        reports = copy.deepcopy(self.reports)
        reports[1]["selected"] = reports[0]["selected"]
        reports[1]["durations"] = reports[0]["durations"]
        cases.append(reports)
        reports = copy.deepcopy(self.reports)
        reports[1]["durations"] = {}
        cases.append(reports)
        for reports in cases:
            with self.subTest(reports=reports), self.assertRaises(ValueError):
                self.merge(reports)

    def test_nonfinite_negative_and_boolean_durations_block(self):
        for duration in (float("nan"), float("inf"), -1, True):
            reports = copy.deepcopy(self.reports)
            reports[0]["durations"]["test_sample.py::test_1"] = duration
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.merge(reports)


class TimingCaptureTests(unittest.TestCase):
    def test_real_pytest_shards_emit_disjoint_complete_timing_artifacts(self):
        repo = Path(__file__).resolve().parents[2]
        python = os.environ.get("CI_TEST_PYTHON", sys.executable)
        env = {
            **os.environ,
            "PYTHONPATH": str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "PYTEST_ADDOPTS": "",
            "GITHUB_SHA": "fixture-sha",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_sample.py").write_text(
                "import pytest\n@pytest.mark.parametrize('n', range(8))\n"
                "def test_value(n):\n    assert n >= 0\n",
                encoding="utf-8",
            )
            paths = []
            for group in (1, 2):
                path = root / f"timings-{group}.json"
                result = subprocess.run(
                    [
                        python,
                        "-m",
                        "pytest",
                        "test_sample.py",
                        "-p",
                        "tools.ci.timings",
                        "--splits",
                        "2",
                        "--group",
                        str(group),
                        "--ci-timings",
                        str(path),
                        "-q",
                    ],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                paths.append(path)
            durations, metadata = merge_reports(paths, expected_shards=2)
            self.assertEqual(len(durations), 8)
            self.assertEqual(metadata["selected_tests"], 8)


if __name__ == "__main__":
    unittest.main()
