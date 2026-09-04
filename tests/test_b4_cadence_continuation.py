from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts import build_b4_cadence_continuation_plan as builder
from scripts import run_b4_cadence_continuation as wrapper
from scripts import run_b4_cadence_pair_hardware as frozen_runner


NOW = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)


def _plan() -> dict:
    return {
        "schema": "b4_cadence_pair_hardware_supplement_plan_v1",
        "backend_id": "tianyan176",
        "source_hashes": {
            "runner_sha256": builder.digest_file(Path(frozen_runner.__file__)),
        },
        "unchanged_design": {"shots": 123, "virtual_seed": 17},
        "sessions": [
            {
                "session_index": 0,
                "operational_start_utc": "2026-08-24T10:00:00+00:00",
                "operational_deadline_utc": "2026-08-24T14:00:00+00:00",
                "operational_completion_deadline_utc": "2026-08-24T14:30:00+00:00",
                "virtual_start_seconds": 0.0,
                "baseline_measurements": [
                    {
                        "measurement_id": "session00-baseline-session_start",
                        "position": "session_start",
                        "target_utc": "2026-08-24T10:00:00+00:00",
                    },
                    {
                        "measurement_id": "session00-baseline-session_end",
                        "position": "session_end",
                        "target_utc": "2026-08-24T12:00:00+00:00",
                    },
                ],
                "cycles": [
                    {
                        "cycle_id": "session00-block00-cycle00",
                        "sense_target_utc": "2026-08-24T10:20:00+00:00",
                        "mirror_target_utc": "2026-08-24T10:21:30+00:00",
                        "shots": 123,
                    },
                    {
                        "cycle_id": "session00-block01-cycle00",
                        "sense_target_utc": "2026-08-24T11:00:00+00:00",
                        "mirror_target_utc": "2026-08-24T11:06:00+00:00",
                        "shots": 123,
                    },
                ],
            },
            {
                "session_index": 1,
                "operational_start_utc": "2026-08-25T10:00:00+00:00",
                "operational_deadline_utc": "2026-08-25T14:00:00+00:00",
                "operational_completion_deadline_utc": "2026-08-25T14:30:00+00:00",
                "virtual_start_seconds": 86400.0,
                "ou_state": [0.1, -0.2],
                "baseline_measurements": [
                    {
                        "measurement_id": "session01-baseline-session_start",
                        "position": "session_start",
                        "target_utc": "2026-08-25T10:00:00+00:00",
                    },
                    {
                        "measurement_id": "session01-baseline-session_end",
                        "position": "session_end",
                        "target_utc": "2026-08-25T12:00:00+00:00",
                    },
                ],
                "cycles": [
                    {
                        "cycle_id": "session01-block00-cycle00",
                        "sense_target_utc": "2026-08-25T10:20:00+00:00",
                        "mirror_target_utc": "2026-08-25T10:26:00+00:00",
                        "shots": 123,
                    },
                    {
                        "cycle_id": "session01-block01-cycle00",
                        "sense_target_utc": "2026-08-25T11:00:00+00:00",
                        "mirror_target_utc": "2026-08-25T11:01:30+00:00",
                        "shots": 123,
                    },
                ],
            },
        ],
    }


def _journal_rows(*, include_session1: bool = False) -> list[dict]:
    rows = [
        {"event": "cadence_cycle_completed", "session_index": 0, "cycle_id": "session00-block00-cycle00"},
        {"event": "cadence_cycle_completed", "session_index": 0, "cycle_id": "session00-block01-cycle00"},
        {
            "event": "session_baseline_measured",
            "session_index": 0,
            "measurement_id": "session00-baseline-session_start",
            "position": "session_start",
        },
        {
            "event": "session_baseline_measured",
            "session_index": 0,
            "measurement_id": "session00-baseline-session_end",
            "position": "session_end",
        },
        {"event": "session_gate_test", "session_index": 0},
    ]
    if include_session1:
        rows.append(
            {
                "event": "submission_started",
                "cycle_id": "session01-baseline-session_start",
            }
        )
    previous = None
    for sequence, row in enumerate(rows):
        row.update(
            {
                "sequence": sequence,
                "previous_record_sha256": previous,
                "record_sha256": f"journal-hash-{sequence}",
                "counts": {"scientific": 999},
            }
        )
        previous = row["record_sha256"]
    return rows


def _write_fixture(tmp_path: Path, *, include_session1: bool = False) -> tuple[Path, Path, bytes]:
    base = tmp_path / builder.BASE_PLAN_NAME
    base.write_text(json.dumps(_plan(), indent=2) + "\n", encoding="utf-8")
    original = base.read_bytes()
    journal = tmp_path / "snapshots.jsonl"
    journal.write_text(
        "".join(json.dumps(row) + "\n" for row in _journal_rows(include_session1=include_session1)),
        encoding="utf-8",
    )
    return base, journal, original


def test_builder_writes_separate_constant_shift_plan_and_wrapper_validates(tmp_path: Path) -> None:
    base, journal, original = _write_fixture(tmp_path)
    override = tmp_path / "supplement_plan_session1_continuation.json"
    plan = builder.build_continuation_plan(
        base,
        journal,
        override,
        new_session_start_utc="2026-08-26T10:10:00+00:00",
        delay_reason="backend calibration and host restart delayed Session 1",
        now_utc=NOW,
    )

    assert base.read_bytes() == original
    assert override.is_file()
    builder.validate_continuation_shape(json.loads(base.read_text()), plan)
    assert plan["sessions"][1]["operational_start_utc"] == "2026-08-26T10:10:00+00:00"
    assert plan["sessions"][1]["virtual_start_seconds"] == 86400.0
    assert plan["sessions"][1]["ou_state"] == [0.1, -0.2]
    assert plan["sessions"][1]["cycles"][0]["shots"] == 123
    assert plan["sessions"][0]["operational_deadline_utc"] == plan["sessions"][1][
        "operational_completion_deadline_utc"
    ]
    assert plan["continuation"]["registered_inter_session_gap_seconds"] == 86400.0
    assert plan["continuation"]["inter_session_gap_deviation_seconds"] == 87000.0
    assert wrapper.validate_resume_contract(override, tmp_path, now_utc=NOW) == plan


def test_builder_rejects_any_session1_lifecycle_event(tmp_path: Path) -> None:
    base, journal, _ = _write_fixture(tmp_path, include_session1=True)
    with pytest.raises(RuntimeError, match="session 1 already has lifecycle events"):
        builder.build_continuation_plan(
            base,
            journal,
            tmp_path / "continuation.json",
            new_session_start_utc="2026-08-26T10:10:00+00:00",
            delay_reason="calibration delay",
            now_utc=NOW,
        )


def test_safe_journal_projection_drops_scientific_fields(tmp_path: Path) -> None:
    _, journal, _ = _write_fixture(tmp_path)
    rows, tail = builder.load_safe_journal_metadata(journal)
    assert tail == "journal-hash-4"
    assert all("counts" not in row for row in rows)
    assert all(set(row) <= set(builder.SAFE_JOURNAL_KEYS) | {"query_id_count"} for row in rows)


def test_selected_plan_name_is_scoped_and_restored(tmp_path: Path) -> None:
    previous = frozen_runner.PLAN_NAME
    override = tmp_path / "override.json"
    with wrapper.selected_plan_name(override):
        assert frozen_runner.PLAN_NAME == "override.json"
    assert frozen_runner.PLAN_NAME == previous


def test_only_cqlib_504_is_classified_as_a_transient_gateway_error() -> None:
    GatewayError = type("CqlibRequestError", (Exception,), {})
    assert wrapper.is_transient_gateway_error(GatewayError("status code 504 Gateway Time-out"))
    assert not wrapper.is_transient_gateway_error(GatewayError("status code 401"))
    assert not wrapper.is_transient_gateway_error(RuntimeError("status code 504"))
