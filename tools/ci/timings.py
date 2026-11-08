"""Record actual shard timings and merge validated candidates for pytest-split."""

import argparse
import json
import math
import os
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--ci-timings", help="Write selected node IDs and measured durations")


def pytest_configure(config):
    if config.getoption("--ci-timings"):
        config.pluginmanager.register(TimingRecorder(config), "ci-timing-recorder")


class TimingRecorder:
    def __init__(self, config):
        self.config = config
        self.selected = []
        self.durations = {}

    def pytest_collection_finish(self, session):
        self.selected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report):
        self.durations[report.nodeid] = self.durations.get(report.nodeid, 0.0) + report.duration

    def pytest_sessionfinish(self, session, exitstatus):
        report = {
            "schema": 1,
            "shard": getattr(self.config.option, "group", None) or 1,
            "splits": getattr(self.config.option, "splits", None) or 1,
            "source": os.environ.get("GITHUB_SHA", "local"),
            "exitstatus": int(exitstatus),
            "selected": self.selected,
            "durations": self.durations,
        }
        path = Path(self.config.getoption("--ci-timings"))
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )


def _validate_report(report, expected_shards):
    if report.get("schema") != 1 or report.get("exitstatus") != 0:
        raise ValueError("Invalid or unsuccessful timing report")
    if report.get("splits") != expected_shards or report.get("shard") not in range(
        1, expected_shards + 1
    ):
        raise ValueError("Unexpected shard configuration")
    selected, durations = report.get("selected"), report.get("durations")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ValueError("Missing or repeated selected node IDs")
    if not all(isinstance(node, str) for node in selected) or not isinstance(durations, dict):
        raise ValueError("Invalid node IDs or duration map")
    if set(selected) != set(durations):
        raise ValueError("Timing measurements do not cover the selected tests exactly")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in durations.values()
    ):
        raise ValueError("Durations must be finite non-negative numbers")


def merge_reports(paths, expected_shards=4):
    if len(paths) != expected_shards:
        raise ValueError("Expected one report from every shard")
    merged, groups, sources, shards = {}, set(), set(), []
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        _validate_report(report, expected_shards)
        if report["shard"] in groups or set(merged).intersection(report["selected"]):
            raise ValueError("Duplicate shard or overlapping test selection")
        groups.add(report["shard"])
        sources.add(report["source"])
        merged.update(report["durations"])
        shards.append(
            {
                "shard": report["shard"],
                "tests": len(report["selected"]),
                "seconds": sum(report["durations"].values()),
            }
        )
    if len(sources) != 1:
        raise ValueError("Shards came from different source revisions")
    metadata = {
        "source": sources.pop(),
        "selected_tests": len(merged),
        "shards": sorted(shards, key=lambda item: item["shard"]),
    }
    return dict(sorted(merged.items())), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["merge"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    durations, metadata = merge_reports(args.reports, args.expected_shards)
    args.output.write_text(
        json.dumps(durations, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = args.output.with_name(args.output.stem + ".meta.json")
    summary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
