"""The report layer: read-only on inputs, isolated output, explicit no-data.

The three properties worth testing are not the arithmetic -- `test_atr_causality_probe.py` anchors
that against production -- but the ones that make the run safe to point at real Phase 2 data:
it must not modify its inputs, must refuse to write next to them, and must say plainly when there
was nothing to scan rather than emitting a clean-looking empty report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.atr_causality_report import (
    DEFAULT_CONFIG,
    run_probe,
    scan_frame,
)

_BAR = 3_600_000


def _frame(rows: int = 400, *, seed: int = 11, atr: float = 0.05):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=rows))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.015, size=rows)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.015, size=rows)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    labels = [0.05 if row + 24 < rows else float("nan") for row in range(rows)]
    return pd.DataFrame(
        {
            "ts": np.arange(rows, dtype="int64") * _BAR,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr14_pct": np.full(rows, atr),
            "label_h24": labels,
        }
    )


def _write_frames(directory, pids=("AAA-USD", "BBB-USD")):
    directory.mkdir(parents=True, exist_ok=True)
    for index, pid in enumerate(pids):
        frame = _frame(seed=11 + index)
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False), directory / f"{pid}.parquet"
        )
    return directory


def test_legacy_differs_from_itself_in_nothing():
    """The self-check. If legacy shows changes against legacy the probe is measuring noise and
    every other number in the report is worthless."""
    report = scan_frame(_frame(), product_id="AAA-USD", horizon=24)
    legacy = next(v for v in report.variants if v.variant == "legacy")
    assert report.comparable_records > 100, "too few records to conclude anything"
    assert legacy.pnl_changed == 0
    assert legacy.result_changed == 0
    assert legacy.kind_migration == {}
    assert legacy.sign_flips == 0
    assert legacy.pnl_delta_abs_mean_over_changed is None


def test_every_variant_is_reported_separately():
    report = scan_frame(_frame(), product_id="AAA-USD", horizon=24)
    names = [v.variant for v in report.variants]
    assert names == ["legacy", "lag_only", "ordering_only", "gap_only", "combined"], (
        "attribution needs each change reported on its own, and in a stable order"
    )
    assert (
        report.ordering_ambiguous + report.enumerated_policy_agreement == report.comparable_records
    ), "every comparable record falls in exactly one of the two enumerated-policy buckets"


def test_a_frame_missing_columns_is_skipped_with_a_reason_not_crashed():
    frame = _frame().drop(columns=["atr14_pct"])
    report = scan_frame(frame, product_id="BAD-USD", horizon=24)
    assert report.skipped_reason is not None and "atr14_pct" in report.skipped_reason
    assert report.comparable_records == 0
    assert report.variants == []


def test_a_missing_label_column_is_skipped_with_a_reason():
    report = scan_frame(_frame(), product_id="AAA-USD", horizon=999)
    assert report.skipped_reason == "no label_h999 column"


def test_no_data_is_reported_explicitly_rather_than_as_a_clean_empty_run(tmp_path):
    """An empty result must be legible as empty. A report that merely contains no rows looks
    like a successful scan that found nothing interesting."""
    frames = tmp_path / "frames"
    frames.mkdir()
    payload = run_probe(frames, tmp_path / "out", horizons=(24,))

    assert payload["status"] == "no_data"
    assert "no scannable Phase 2 frames" in payload["no_data_reason"]
    assert payload["frames_found"] == 0 and payload["frames_scanned"] == 0
    written = json.loads(
        (tmp_path / "out" / "atr_causality_report.json").read_text(encoding="utf-8")
    )
    assert written["status"] == "no_data", (
        "the no-data verdict must be in the file, not only returned"
    )


def test_the_probe_refuses_to_write_next_to_its_read_only_inputs(tmp_path):
    frames = _write_frames(tmp_path / "frames")
    with pytest.raises(ValueError, match="must not be the frames directory"):
        run_probe(frames, frames, horizons=(24,))


def test_a_run_leaves_every_input_file_byte_identical(tmp_path):
    """The property that makes it safe to point at real Phase 2 output."""
    frames = _write_frames(tmp_path / "frames")
    before = {
        path.name: (path.stat().st_size, path.read_bytes())
        for path in sorted(frames.glob("*.parquet"))
    }
    run_probe(frames, tmp_path / "out", horizons=(24,), max_entries=120)
    after = {
        path.name: (path.stat().st_size, path.read_bytes())
        for path in sorted(frames.glob("*.parquet"))
    }
    assert set(after) == set(before), "the run added or removed an input file"
    for name in before:
        assert after[name][1] == before[name][1], f"{name} was modified by a read-only probe"


def test_an_end_to_end_run_reports_per_product_per_variant(tmp_path):
    frames = _write_frames(tmp_path / "frames", pids=("AAA-USD", "BBB-USD"))
    payload = run_probe(frames, tmp_path / "out", horizons=(24,), max_entries=200)

    assert payload["status"] == "ok"
    assert payload["frames_found"] == 2 and payload["frames_scanned"] == 2
    assert payload["frames_skipped"] == []
    products = {r["product_id"] for r in payload["reports"]}
    assert products == {"AAA-USD", "BBB-USD"}
    for report in payload["reports"]:
        assert report["comparable_records"] > 0, "a scanned frame reporting nothing is vacuous"
        legacy = next(v for v in report["variants"] if v["variant"] == "legacy")
        assert legacy["pnl_changed"] == 0 and legacy["result_changed"] == 0
    assert payload["caveats"], "the report must carry its own limitations"
    assert any("not a profitability" in c.lower() for c in payload["caveats"])
    assert payload["config"] == dict(DEFAULT_CONFIG)


def test_the_output_is_written_atomically(tmp_path, monkeypatch):
    """A crashed run must not leave a half-written report that a reader would trust."""
    import tools.strategy_discovery.atr_causality_report as module

    frames = _write_frames(tmp_path / "frames", pids=("AAA-USD",))
    destination = tmp_path / "out" / "atr_causality_report.json"

    real_replace = module.os.replace

    def fail_on_report(src, dst):
        if str(dst).endswith("atr_causality_report.json"):
            raise OSError("interrupted")
        return real_replace(src, dst)

    monkeypatch.setattr(module.os, "replace", fail_on_report)
    with pytest.raises(OSError, match="interrupted"):
        run_probe(frames, tmp_path / "out", horizons=(24,), max_entries=50)

    assert not destination.exists(), "no report should exist after an interrupted write"
    assert not list((tmp_path / "out").glob("*.partial")), "the temp file must be cleaned up"


def test_the_stored_label_decides_who_is_a_candidate(tmp_path):
    """Codex c6f81f65 item 4. The label column was checked and never read, so the population was
    whatever the recomputation produced rather than what the producer published.

    Here rows 0-9 are blanked in the STORED label while remaining perfectly recomputable. They
    must be excluded, because a row the producer left unlabelled is not part of the published
    population however well it would have scored.
    """
    frame = _frame()
    frame.loc[0:9, "label_h24"] = float("nan")
    report = scan_frame(frame, product_id="HOLE-USD", horizon=24)
    baseline = scan_frame(_frame(), product_id="FULL-USD", horizon=24)

    assert report.stored_labels_finite == baseline.stored_labels_finite - 10
    assert report.comparable_records == baseline.comparable_records - 10, (
        "the blanked rows must leave the population, not merely be recomputed anyway"
    )


def test_a_gapped_or_reordered_clock_is_skipped_not_silently_mislabelled():
    """Codex c6f81f65 item 3. Row-count horizons are false on a broken clock, and `ts` was
    required but never validated."""
    gapped = _frame()
    ts = gapped["ts"].to_numpy(dtype="int64").copy()
    ts[50:] += _BAR  # a one-bar hole
    gapped["ts"] = ts
    assert "contiguous hourly" in (
        scan_frame(gapped, product_id="GAP-USD", horizon=24).skipped_reason or ""
    )

    reversed_frame = _frame()
    reversed_frame["ts"] = reversed_frame["ts"].to_numpy(dtype="int64")[::-1]
    assert scan_frame(reversed_frame, product_id="REV-USD", horizon=24).skipped_reason is not None


def test_a_non_positive_price_is_skipped():
    frame = _frame()
    frame.loc[7, "low"] = 0.0
    assert "finite and positive" in (
        scan_frame(frame, product_id="ZERO-USD", horizon=24).skipped_reason or ""
    )


def test_a_frame_with_no_comparable_records_is_no_data_not_a_successful_scan(tmp_path):
    """Codex c6f81f65 item 2. Zero evidence reported as `ok` would read as a finding of no
    effect."""
    frame = _frame()
    frame["label_h24"] = float("nan")  # readable, valid, and entirely unlabelled
    frames = tmp_path / "frames"
    frames.mkdir()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), frames / "EMPTY-USD.parquet")

    payload = run_probe(frames, tmp_path / "out", horizons=(24,))
    assert payload["status"] == "no_data"
    assert "no comparable records" in payload["no_data_reason"]
    assert payload["frames_without_evidence"][0]["product_id"] == "EMPTY-USD"
    assert payload["frames_scanned"] == 0


def test_the_output_may_not_be_written_inside_the_frames_directory(tmp_path):
    """Codex c6f81f65 item 5. Equality alone permitted a subdirectory, contradicting the promise."""
    frames = _write_frames(tmp_path / "frames", pids=("AAA-USD",))
    with pytest.raises(ValueError, match="or inside it"):
        run_probe(frames, frames / "out", horizons=(24,))


def test_result_changed_catches_an_exit_move_that_leaves_pnl_identical():
    """Codex c6f81f65 item 7. Counting only PnL understates the change.

    `result_changed` is the union of pnl, exit kind and bars held, so a variant that moves the
    holding period at equal PnL is still counted.
    """
    report = scan_frame(_frame(), product_id="AAA-USD", horizon=24)
    for summary in report.variants:
        assert summary.result_changed >= summary.pnl_changed, (
            f"{summary.variant}: result_changed must be a superset of pnl_changed"
        )


def test_a_clean_frame_exposes_no_entries():
    from tools.strategy_discovery.atr_causality_report import audit_frame_clock

    audit = audit_frame_clock(_frame(), product_id="CLEAN-USD", horizon=24)
    assert audit.bad_steps == 0
    assert audit.stored_labels_finite > 0, "vacuous otherwise"
    assert audit.exposed_entries == 0
    assert audit.exposed_fraction == 0.0


def test_only_entries_whose_own_window_spans_the_gap_are_exposed():
    """Codex c6e41c6a. A frame failing the guard is NOT every entry being affected.

    One hole after row 200 in a 400-row frame at horizon 24 exposes exactly the entries whose
    [entry, entry+24) window contains that step -- rows 177..200 inclusive, 24 of them -- not the
    whole frame. Asserting the exact count is what makes the distinction usable.
    """
    from tools.strategy_discovery.atr_causality_report import audit_frame_clock

    frame = _frame()
    ts = frame["ts"].to_numpy(dtype="int64").copy()
    ts[201:] += _BAR  # a single one-bar hole between rows 200 and 201
    frame["ts"] = ts

    audit = audit_frame_clock(frame, product_id="ONEGAP-USD", horizon=24)
    assert audit.bad_steps == 1
    assert audit.exposed_entries == 24, (
        f"expected the 24 windows covering the hole, got {audit.exposed_entries}"
    )
    assert audit.exposed_fraction < 0.07, (
        "one hole must not be reported as contaminating the whole frame"
    )


def test_the_clock_audit_persists_and_refuses_to_write_into_the_frames_dir(tmp_path):
    from tools.strategy_discovery.atr_causality_report import run_clock_audit

    frames = _write_frames(tmp_path / "frames", pids=("AAA-USD", "BBB-USD"))
    with pytest.raises(ValueError, match="or inside it"):
        run_clock_audit(frames, frames / "out")

    payload = run_clock_audit(frames, tmp_path / "out", horizons=(24,))
    assert payload["frames_audited"] == 2
    assert payload["frames_with_gaps"] == 0
    assert payload["entries_whose_window_spans_a_gap"] == 0
    written = json.loads((tmp_path / "out" / "clock_audit.json").read_text(encoding="utf-8"))
    assert written["frames_audited"] == 2
    assert any("NOT the same as every entry" in c for c in written["caveats"])
