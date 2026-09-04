#!/usr/bin/env python3
"""Build a non-overwriting wall-clock continuation plan for B-4 session 1.

The frozen plan remains byte-for-byte evidence.  This builder permits one narrowly
scoped operational amendment after a delayed second session:

* session 1 wall-clock targets and deadlines move by one constant offset;
* session 0's submission deadline is extended only so the frozen runner can traverse
  its already-complete, idempotent records before reaching session 1;
* every virtual-time, OU, design, identifier, cadence, seed, and shot field is unchanged.

The journal is inspected only for lifecycle metadata.  Scientific result fields are
neither retained nor returned.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_b4_cadence_pair_hardware as frozen_runner


BASE_PLAN_NAME = "supplement_plan.json"
DEFAULT_OUTPUT_NAME = "supplement_plan_session1_continuation.json"
WRAPPER_PATH = ROOT / "scripts" / "run_b4_cadence_continuation.py"
BAD_EVENTS = frozenset({
    "submission_rejected",
    "partial",
    "session_pair_block_discarded",
    "session_baseline_measurement_lost",
    "submission_limit_reached",
})
SAFE_JOURNAL_KEYS = (
    "sequence",
    "recorded_at_utc",
    "event",
    "previous_record_sha256",
    "record_sha256",
    "snapshot_id",
    "job_role",
    "role",
    "cycle_id",
    "registered_pair_id",
    "session_index",
    "measurement_id",
    "position",
    "planned_target_utc",
    "target_utc",
    "settings",
    "shots_per_setting",
)


def digest_file(path: Path) -> str:
    return frozen_runner.digest_file(path.resolve())


def parse_utc(value: str) -> datetime:
    return frozen_runner.parse_utc(value)


def iso(value: datetime) -> str:
    return frozen_runner.iso(value)


def load_safe_journal_metadata(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Retain only operational metadata and verify sequence/link continuity.

    The scientific payload is deliberately excluded from the retained object.  The
    journal's previously frozen record hashes are linked here, but are not recomputed:
    recomputation would inspect count-derived payload fields outside this tool's scope.
    """
    path = path.resolve()
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for expected_sequence, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"blank journal line at sequence {expected_sequence}")
            source_row = json.loads(line)
            if not isinstance(source_row, dict):
                raise ValueError(f"journal sequence {expected_sequence} is not an object")
            if source_row.get("sequence") != expected_sequence:
                raise ValueError("journal sequence is not contiguous")
            if source_row.get("previous_record_sha256") != previous:
                raise ValueError("journal previous-record hash mismatch")
            recorded_hash = source_row.get("record_sha256")
            if not isinstance(recorded_hash, str) or not recorded_hash:
                raise ValueError("journal record SHA-256 is absent")
            metadata = {
                key: source_row[key] for key in SAFE_JOURNAL_KEYS if key in source_row
            }
            tasks = source_row.get("tasks")
            metadata["query_id_count"] = len(tasks) if isinstance(tasks, list) else 0
            rows.append(metadata)
            previous = str(recorded_hash)
    return rows, previous


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def event_belongs_to_session(row: Mapping[str, Any], session_index: int) -> bool:
    if _int_or_none(row.get("session_index")) == session_index:
        return True
    prefix = f"session{session_index:02d}-"
    return any(
        str(row.get(key, "")).startswith(prefix)
        for key in ("cycle_id", "measurement_id", "registered_pair_id")
    )


def _session(plan: Mapping[str, Any], session_index: int) -> Mapping[str, Any]:
    matches = [
        row for row in plan["sessions"] if int(row["session_index"]) == session_index
    ]
    if len(matches) != 1:
        raise ValueError(f"plan must contain exactly one session {session_index}")
    return matches[0]


def validate_initial_journal(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    continuation_session_index: int,
) -> dict[str, Any]:
    """Require complete session 0, untouched session 1, and no failure marker."""
    if continuation_session_index != 1:
        raise ValueError("this prospective amendment is restricted to session 1")
    bad = [
        {
            "sequence": row.get("sequence"),
            "event": row.get("event"),
        }
        for row in rows
        if row.get("event") in BAD_EVENTS
    ]
    if bad:
        raise RuntimeError(f"journal contains failure markers: {bad}")

    first = _session(plan, 0)
    expected_cycles = [str(row["cycle_id"]) for row in first["cycles"]]
    completed_cycles = [
        str(row.get("cycle_id"))
        for row in rows
        if row.get("event") == "cadence_cycle_completed"
        and event_belongs_to_session(row, 0)
    ]
    if Counter(completed_cycles) != Counter(expected_cycles):
        missing = sorted(set(expected_cycles) - set(completed_cycles))
        unexpected = sorted(set(completed_cycles) - set(expected_cycles))
        raise RuntimeError(
            "session 0 cycle completion is not exact: "
            f"completed={len(completed_cycles)}, expected={len(expected_cycles)}, "
            f"missing={missing}, unexpected={unexpected}"
        )

    planned_baselines = {
        str(row["measurement_id"]): str(row["position"])
        for row in first.get("baseline_measurements", [])
    }
    if set(planned_baselines.values()) != {"session_start", "session_end"}:
        raise RuntimeError("session 0 plan must contain start and end baselines")
    measured_baselines = [
        (str(row.get("measurement_id")), str(row.get("position")))
        for row in rows
        if row.get("event") == "session_baseline_measured"
        and event_belongs_to_session(row, 0)
    ]
    if Counter(measured_baselines) != Counter(planned_baselines.items()):
        raise RuntimeError("session 0 baseline measurements are not present exactly once")

    gate_rows = [
        row
        for row in rows
        if row.get("event") == "session_gate_test"
        and _int_or_none(row.get("session_index")) == 0
    ]
    if len(gate_rows) != 1:
        raise RuntimeError("session 0 gate record is not present exactly once")

    session1_rows = [
        {
            "sequence": row.get("sequence"),
            "event": row.get("event"),
            "cycle_id": row.get("cycle_id"),
            "measurement_id": row.get("measurement_id"),
        }
        for row in rows
        if event_belongs_to_session(row, continuation_session_index)
    ]
    if session1_rows:
        raise RuntimeError(
            "session 1 already has lifecycle events; refusing to invent a new wall-clock grid: "
            f"{session1_rows}"
        )
    return {
        "session0_completed_cycles": len(completed_cycles),
        "session0_baseline_measurements": len(measured_baselines),
        "session0_gate_present": True,
        "session1_lifecycle_events": 0,
        "bad_events": 0,
    }


def _shift(value: str, delta: timedelta) -> str:
    return iso(parse_utc(value) + delta)


def _normalise_allowed_changes(
    plan: Mapping[str, Any],
    base_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace the amendment's allowed fields with their base values for equality proof."""
    normalised = deepcopy(dict(plan))
    normalised.pop("continuation", None)
    base_sessions = {
        int(row["session_index"]): row for row in base_plan["sessions"]
    }
    for session in normalised["sessions"]:
        index = int(session["session_index"])
        base = base_sessions[index]
        if index == 0:
            session["operational_deadline_utc"] = base["operational_deadline_utc"]
        elif index == 1:
            for key in (
                "operational_start_utc",
                "operational_deadline_utc",
                "operational_completion_deadline_utc",
            ):
                session[key] = base[key]
            base_cycles = {str(row["cycle_id"]): row for row in base["cycles"]}
            for cycle in session["cycles"]:
                original = base_cycles[str(cycle["cycle_id"])]
                cycle["sense_target_utc"] = original["sense_target_utc"]
                cycle["mirror_target_utc"] = original["mirror_target_utc"]
            base_measurements = {
                str(row["measurement_id"]): row
                for row in base.get("baseline_measurements", [])
            }
            for measurement in session.get("baseline_measurements", []):
                original = base_measurements[str(measurement["measurement_id"])]
                measurement["target_utc"] = original["target_utc"]
    return normalised


def validate_continuation_shape(
    base_plan: Mapping[str, Any],
    continuation_plan: Mapping[str, Any],
) -> None:
    """Prove that the continuation changes no registered/design field."""
    continuation = continuation_plan.get("continuation")
    if not isinstance(continuation, Mapping):
        raise ValueError("override plan lacks continuation provenance")
    if int(continuation.get("session_index", -1)) != 1:
        raise ValueError("override continuation session must be 1")
    if not str(continuation.get("delay_reason", "")).strip():
        raise ValueError("override continuation must state a delay reason")
    if _normalise_allowed_changes(continuation_plan, base_plan) != dict(base_plan):
        raise ValueError(
            "continuation changed fields outside the allowed operational UTC amendment"
        )

    delta_seconds = float(continuation["wallclock_shift_seconds"])
    if delta_seconds <= 0.0:
        raise ValueError("session 1 continuation must move forward in wall clock")
    delta = timedelta(seconds=delta_seconds)
    base_second = _session(base_plan, 1)
    amended_second = _session(continuation_plan, 1)
    for key in (
        "operational_start_utc",
        "operational_deadline_utc",
        "operational_completion_deadline_utc",
    ):
        if amended_second[key] != _shift(str(base_second[key]), delta):
            raise ValueError(f"session 1 {key} was not shifted by the registered constant")
    base_cycles = {str(row["cycle_id"]): row for row in base_second["cycles"]}
    for cycle in amended_second["cycles"]:
        original = base_cycles[str(cycle["cycle_id"])]
        for key in ("sense_target_utc", "mirror_target_utc"):
            if cycle[key] != _shift(str(original[key]), delta):
                raise ValueError(f"cycle {cycle['cycle_id']} {key} has a non-constant shift")
    base_measurements = {
        str(row["measurement_id"]): row
        for row in base_second.get("baseline_measurements", [])
    }
    for measurement in amended_second.get("baseline_measurements", []):
        original = base_measurements[str(measurement["measurement_id"])]
        if measurement["target_utc"] != _shift(str(original["target_utc"]), delta):
            raise ValueError(
                f"baseline {measurement['measurement_id']} has a non-constant shift"
            )
    first = _session(continuation_plan, 0)
    if first["operational_deadline_utc"] != amended_second["operational_completion_deadline_utc"]:
        raise ValueError(
            "session 0 idempotent-traversal deadline must equal session 1 completion deadline"
        )


def build_continuation_plan(
    base_plan_path: Path,
    journal_path: Path,
    output_path: Path,
    *,
    new_session_start_utc: str,
    delay_reason: str,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    base_plan_path = base_plan_path.resolve()
    journal_path = journal_path.resolve()
    output_path = output_path.resolve()
    if base_plan_path.name != BASE_PLAN_NAME:
        raise ValueError(f"base plan must be named {BASE_PLAN_NAME}")
    if output_path.parent != base_plan_path.parent:
        raise ValueError("continuation plan must be written directly beside the base plan")
    if output_path == base_plan_path:
        raise ValueError("continuation plan cannot overwrite the frozen base plan")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {output_path}")
    if not delay_reason.strip():
        raise ValueError("a non-empty delay reason is required")

    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    actual_runner_hash = digest_file(Path(frozen_runner.__file__))
    if base_plan.get("source_hashes", {}).get("runner_sha256") != actual_runner_hash:
        raise RuntimeError("base plan runner hash does not match the frozen runner bytes")
    rows, journal_tail = load_safe_journal_metadata(journal_path)
    preconditions = validate_initial_journal(
        base_plan, rows, continuation_session_index=1
    )

    original_second = _session(base_plan, 1)
    original_start = parse_utc(str(original_second["operational_start_utc"]))
    new_start = parse_utc(new_session_start_utc)
    now = datetime.now(timezone.utc) if now_utc is None else now_utc.astimezone(timezone.utc)
    minimum_lead = timedelta(seconds=120)
    if new_start < now + minimum_lead:
        raise ValueError("new session 1 start must be at least 120 seconds prospective")
    delta = new_start - original_start
    if delta.total_seconds() <= 0.0:
        raise ValueError("new session 1 start must be later than the frozen start")

    amended = deepcopy(base_plan)
    amended_second = _session(amended, 1)
    for key in (
        "operational_start_utc",
        "operational_deadline_utc",
        "operational_completion_deadline_utc",
    ):
        amended_second[key] = _shift(str(amended_second[key]), delta)
    for cycle in amended_second["cycles"]:
        cycle["sense_target_utc"] = _shift(str(cycle["sense_target_utc"]), delta)
        cycle["mirror_target_utc"] = _shift(str(cycle["mirror_target_utc"]), delta)
    for measurement in amended_second.get("baseline_measurements", []):
        measurement["target_utc"] = _shift(str(measurement["target_utc"]), delta)

    amended_first = _session(amended, 0)
    original_first_deadline = str(amended_first["operational_deadline_utc"])
    amended_first["operational_deadline_utc"] = amended_second[
        "operational_completion_deadline_utc"
    ]
    amended["continuation"] = {
        "schema": "b4_cadence_pair_hardware_continuation_v1",
        "created_at_utc": iso(now),
        "session_index": 1,
        "delay_reason": delay_reason.strip(),
        "base_plan_path": str(base_plan_path),
        "base_plan_sha256": digest_file(base_plan_path),
        "journal_path": str(journal_path),
        "journal_records_at_freeze": len(rows),
        "journal_tail_sha256_at_freeze": journal_tail,
        "journal_preconditions": preconditions,
        "original_session_start_utc": str(original_second["operational_start_utc"]),
        "continued_session_start_utc": str(amended_second["operational_start_utc"]),
        "wallclock_shift_seconds": delta.total_seconds(),
        "registered_inter_session_gap_seconds": (
            original_start - parse_utc(str(_session(base_plan, 0)["operational_start_utc"]))
        ).total_seconds(),
        "continued_inter_session_gap_seconds": (
            new_start - parse_utc(str(_session(base_plan, 0)["operational_start_utc"]))
        ).total_seconds(),
        "inter_session_gap_deviation_seconds": delta.total_seconds(),
        "session0_idempotent_traversal_deadline": {
            "original_utc": original_first_deadline,
            "continued_utc": amended_first["operational_deadline_utc"],
            "role": (
                "permits the frozen runner to traverse already-complete session 0 records; "
                "it authorizes no session 0 submission"
            ),
        },
        "frozen_runner_path": str(Path(frozen_runner.__file__).resolve()),
        "frozen_runner_sha256": actual_runner_hash,
        "wrapper_path": str(WRAPPER_PATH.resolve()),
        "wrapper_sha256": digest_file(WRAPPER_PATH),
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": digest_file(Path(__file__)),
        "invariants": {
            "base_plan_not_overwritten": True,
            "session1_constant_wallclock_shift_only": True,
            "session0_complete_records_are_idempotent_only": True,
            "virtual_ou_design_identifiers_shots_cadence_unchanged": True,
            "scientific_results_read": False,
        },
    }
    validate_continuation_shape(base_plan, amended)
    frozen_runner.write_new_json(output_path, amended)
    return amended


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-session-start-utc", required=True)
    parser.add_argument("--delay-reason", required=True)
    arguments = parser.parse_args()
    plan = build_continuation_plan(
        arguments.base_plan,
        arguments.journal,
        arguments.output,
        new_session_start_utc=arguments.new_session_start_utc,
        delay_reason=arguments.delay_reason,
    )
    continuation = plan["continuation"]
    print(json.dumps({
        "status": "CONTINUATION_PLAN_FROZEN",
        "output": str(arguments.output.resolve()),
        "output_sha256": digest_file(arguments.output),
        "base_plan_sha256": continuation["base_plan_sha256"],
        "session_index": continuation["session_index"],
        "continued_session_start_utc": continuation["continued_session_start_utc"],
        "wallclock_shift_seconds": continuation["wallclock_shift_seconds"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
