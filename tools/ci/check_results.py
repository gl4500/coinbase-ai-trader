"""Fail CI unless every declared prerequisite completed successfully."""
import json
import os
import sys


def main():
    try:
        results = json.loads(os.environ["NEEDS_JSON"])
        if not isinstance(results, dict) or not results:
            raise ValueError("expected a non-empty prerequisite object")
        passed = True
        for name, job in results.items():
            status = job.get("result") if isinstance(job, dict) else None
            print(f"{name}: {status}")
            passed = passed and status == "success"
        return 0 if passed else 1
    except (KeyError, ValueError, TypeError) as exc:
        print(f"Invalid CI prerequisites: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
