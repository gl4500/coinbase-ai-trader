"""Check CI prerequisite statuses through the gate's process boundary."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("check_results.py")


class CheckResultsTests(unittest.TestCase):
    def run_gate(self, payload):
        return subprocess.run(
            [sys.executable, str(SCRIPT)],
            env={**os.environ, "NEEDS_JSON": payload},
            capture_output=True,
            text=True,
        )

    def test_all_prerequisites_succeed(self):
        result = self.run_gate(json.dumps({"tests": {"result": "success"}}))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_any_unsuccessful_prerequisite_blocks(self):
        for status in ("failure", "skipped", "cancelled", "pending", ""):
            with self.subTest(status=status):
                result = self.run_gate(
                    json.dumps({"lint": {"result": "success"}, "tests": {"result": status}})
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("tests", result.stdout)

    def test_missing_or_malformed_results_block(self):
        for payload in ("", "{", "{}", "[]", "null", '{"tests": {}}'):
            with self.subTest(payload=payload):
                self.assertNotEqual(self.run_gate(payload).returncode, 0)


if __name__ == "__main__":
    unittest.main()
