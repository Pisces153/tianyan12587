#!/usr/bin/env python3
"""Run the audited B-4 Session-1 continuation without changing frozen inputs.

This wrapper selects a separately frozen continuation plan by temporarily routing the
original runner's plan filename constant.  The original runner, base plan, configs,
manifest, journal, raw data, and registered scientific design remain byte-for-byte
unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_b4_cadence_continuation_plan as builder
from scripts import drift_campaign_v4
from scripts import run_b4_cadence_pair_hardware as frozen_runner


def _session(plan: Mapping[str, Any], session_index: int) -> Mapping[str, Any]:
    matches = [
        row for row in plan["sessions"] if int(row["session_index"]) == session_index
    ]
    if len(matches) != 1:
        raise ValueError(f"plan must contain exactly one session {session_index}")
    return matches[0]


def is_transient_gateway_error(error: Exception) -> bool:
    message = str(error)
    return type(error).__name__ == "CqlibRequestError" and (
        "504" in message or "Gateway Time-out" in message
    )


def platform_with_gateway_retry(
    config: Mapping[str, Any],
    *,
    max_errors: int,
    retry_seconds: float,
) -> Any:
    errors = 0
    while True:
        try:
            return drift_campaign_v4.platform_from_config(config)
        except Exception as error:
            if not is_transient_gateway_error(error) or errors >= max_errors:
                raise
            errors += 1
            print(
                frozen_runner.canonical_json(
                    {
                        "event": "transient_gateway_retry",
                        "stage": "login",
                        "failure_class": type(error).__name__,
                        "attempt": errors,
                        "maximum": max_errors,
                    }
                ),
                flush=True,
            )
            time.sleep(retry_seconds)


def validate_base_state(output: Path) -> dict[str, Any]:
    """Validate the frozen base plan/journal before deferred authentication begins."""
    output = output.resolve()
    base_plan_path = output / builder.BASE_PLAN_NAME
    journal_path = output / "snapshots.jsonl"
    if (output / frozen_runner.REPORT_NAME).exists():
        raise RuntimeError("completion report already exists; continuation is no longer allowed")
    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    current_runner_hash = builder.digest_file(Path(frozen_runner.__file__))
    if base_plan.get("source_hashes", {}).get("runner_sha256") != current_runner_hash:
        raise RuntimeError("base plan no longer pins the current frozen runner")
    rows, _journal_tail = builder.load_safe_journal_metadata(journal_path)
    builder.validate_initial_journal(base_plan, rows, continuation_session_index=1)
    return base_plan


def wait_for_running_resilient(
    platform: Any,
    *,
    backend_id: str,
    poll_seconds: int,
    stable_polls: int,
    max_gateway_errors: int,
    max_wait_seconds: int,
) -> None:
    consecutive = 0
    gateway_errors = 0
    started = time.monotonic()
    while consecutive < stable_polls:
        if time.monotonic() - started > max_wait_seconds:
            raise TimeoutError("running-status gate exceeded its maximum wait")
        try:
            machines = platform.query_quantum_computer_list()
        except Exception as error:
            if not is_transient_gateway_error(error) or gateway_errors >= max_gateway_errors:
                raise
            gateway_errors += 1
            consecutive = 0
            print(
                frozen_runner.canonical_json(
                    {
                        "event": "transient_gateway_retry",
                        "stage": "machine_status",
                        "failure_class": type(error).__name__,
                        "attempt": gateway_errors,
                        "maximum": max_gateway_errors,
                    }
                ),
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        row = next(
            (
                item
                for item in machines
                if isinstance(item, list) and len(item) >= 4 and str(item[3]) == backend_id
            ),
            None,
        )
        status = None if row is None else str(row[2])
        consecutive = consecutive + 1 if status == "running" else 0
        print(
            frozen_runner.canonical_json(
                {
                    "event": "machine_status",
                    "recorded_at_utc": frozen_runner.iso(frozen_runner.utc_now()),
                    "backend_id": backend_id,
                    "status": status,
                    "consecutive_running_polls": consecutive,
                    "required_running_polls": stable_polls,
                }
            ),
            flush=True,
        )
        if consecutive < stable_polls:
            time.sleep(poll_seconds)


def validate_resume_contract(
    plan_override_path: Path,
    output: Path,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Fail closed before authentication or any platform submission path is reached."""
    output = output.resolve()
    plan_override_path = plan_override_path.resolve()
    base_plan_path = output / builder.BASE_PLAN_NAME
    journal_path = output / "snapshots.jsonl"
    report_path = output / frozen_runner.REPORT_NAME
    if plan_override_path.parent != output:
        raise ValueError("continuation plan must be directly inside the sealed output directory")
    if plan_override_path == base_plan_path:
        raise ValueError("continuation wrapper refuses to replace or select the frozen base plan")
    if report_path.exists():
        raise RuntimeError("completion report already exists; continuation is no longer allowed")

    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    continuation_plan = json.loads(plan_override_path.read_text(encoding="utf-8"))
    builder.validate_continuation_shape(base_plan, continuation_plan)
    continuation = continuation_plan["continuation"]

    exact_paths = {
        "base_plan_path": base_plan_path,
        "journal_path": journal_path,
        "frozen_runner_path": Path(frozen_runner.__file__).resolve(),
        "wrapper_path": Path(__file__).resolve(),
        "builder_path": Path(builder.__file__).resolve(),
    }
    for key, expected in exact_paths.items():
        if Path(str(continuation.get(key, ""))).resolve() != expected:
            raise RuntimeError(f"continuation {key} changed")
    expected_hashes = {
        "base_plan_sha256": builder.digest_file(base_plan_path),
        "frozen_runner_sha256": builder.digest_file(Path(frozen_runner.__file__)),
        "wrapper_sha256": builder.digest_file(Path(__file__)),
        "builder_sha256": builder.digest_file(Path(builder.__file__)),
    }
    for key, expected in expected_hashes.items():
        if continuation.get(key) != expected:
            raise RuntimeError(f"continuation {key} does not match current bytes")
    if base_plan.get("source_hashes", {}).get("runner_sha256") != expected_hashes[
        "frozen_runner_sha256"
    ]:
        raise RuntimeError("base plan no longer pins the current frozen runner")

    rows, journal_tail = builder.load_safe_journal_metadata(journal_path)
    preconditions = builder.validate_initial_journal(
        base_plan, rows, continuation_session_index=1
    )
    if len(rows) != int(continuation["journal_records_at_freeze"]):
        raise RuntimeError("journal changed after continuation plan freeze")
    if journal_tail != continuation["journal_tail_sha256_at_freeze"]:
        raise RuntimeError("journal tail changed after continuation plan freeze")
    if preconditions != continuation["journal_preconditions"]:
        raise RuntimeError("journal preconditions changed after continuation plan freeze")

    now = datetime.now(timezone.utc) if now_utc is None else now_utc.astimezone(timezone.utc)
    second = _session(continuation_plan, 1)
    if now > frozen_runner.parse_utc(str(second["operational_deadline_utc"])):
        raise RuntimeError("continuation Session 1 submission deadline has expired")
    first = _session(continuation_plan, 0)
    if now > frozen_runner.parse_utc(str(first["operational_deadline_utc"])):
        raise RuntimeError("Session 0 idempotent-traversal allowance has expired")
    return continuation_plan


@contextmanager
def selected_plan_name(plan_override_path: Path) -> Iterator[None]:
    previous = frozen_runner.PLAN_NAME
    frozen_runner.PLAN_NAME = plan_override_path.name
    try:
        yield
    finally:
        frozen_runner.PLAN_NAME = previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    plan_mode = parser.add_mutually_exclusive_group(required=True)
    plan_mode.add_argument("--plan-override", type=Path)
    plan_mode.add_argument(
        "--deferred-plan-name",
        help="Create this non-existing continuation plan inside --output after the running gate passes.",
    )
    parser.add_argument("--delay-reason")
    parser.add_argument("--start-lead-seconds", type=float, default=300.0)
    parser.add_argument("--loop-config", type=Path, default=frozen_runner.DEFAULT_LOOP_CONFIG)
    parser.add_argument("--backend-config", type=Path, default=frozen_runner.DEFAULT_BACKEND_CONFIG)
    parser.add_argument("--peer-config", type=Path, default=frozen_runner.DEFAULT_PEER_CONFIG)
    parser.add_argument("--stage1-manifest", type=Path, default=frozen_runner.DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--output", type=Path, default=frozen_runner.DEFAULT_OUTPUT)
    parser.add_argument("--max-wait-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--status-poll-seconds", type=int, default=30)
    parser.add_argument("--gateway-retry-seconds", type=float, default=300.0)
    parser.add_argument("--stable-running-polls", type=int, default=3)
    parser.add_argument("--max-status-wait-seconds", type=int, default=900)
    parser.add_argument("--max-transient-gateway-errors", type=int, default=6)
    parser.add_argument("--max-opening-lateness-seconds", type=float, default=60.0)
    parser.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    override = None if arguments.plan_override is None else arguments.plan_override.resolve()
    if override is None:
        validate_base_state(output)
        if not arguments.delay_reason or not arguments.delay_reason.strip():
            raise ValueError("--delay-reason is required with --deferred-plan-name")
        if float(arguments.start_lead_seconds) < 120.0:
            raise ValueError("--start-lead-seconds must be at least 120")
        deferred_name = Path(str(arguments.deferred_plan_name))
        if deferred_name.name != str(arguments.deferred_plan_name):
            raise ValueError("--deferred-plan-name must be a filename, not a path")
        if deferred_name.name == builder.BASE_PLAN_NAME:
            raise ValueError("deferred continuation cannot overwrite the frozen base plan")
        override = output / deferred_name.name
        if override.exists():
            raise FileExistsError(f"refusing to overwrite evidence artifact: {override}")
    else:
        validate_resume_contract(override, output)
    config = drift_campaign_v4.load_config(arguments.backend_config.resolve())
    platform = platform_with_gateway_retry(
        config,
        max_errors=arguments.max_transient_gateway_errors,
        retry_seconds=float(arguments.gateway_retry_seconds),
    )
    wait_for_running_resilient(
        platform,
        backend_id=str(config["backend"]["backend_id"]),
        poll_seconds=arguments.status_poll_seconds,
        stable_polls=arguments.stable_running_polls,
        max_gateway_errors=arguments.max_transient_gateway_errors,
        max_wait_seconds=arguments.max_status_wait_seconds,
    )

    if arguments.plan_override is None:
        # The absolute wall-clock grid is frozen only after the backend is actually
        # reachable and stable, preventing a gateway outage from aging the amendment.
        new_start = datetime.now(timezone.utc) + timedelta(
            seconds=float(arguments.start_lead_seconds)
        )
        plan = builder.build_continuation_plan(
            output / builder.BASE_PLAN_NAME,
            output / "snapshots.jsonl",
            override,
            new_session_start_utc=frozen_runner.iso(new_start),
            delay_reason=str(arguments.delay_reason),
        )
        print(
            frozen_runner.canonical_json(
                {
                    "event": "continuation_plan_frozen",
                    "plan_path": str(override),
                    "plan_sha256": builder.digest_file(override),
                    "session_index": 1,
                    "session_start_utc": plan["continuation"]["continued_session_start_utc"],
                }
            ),
            flush=True,
        )

    # Recheck the frozen journal after status polling, before the original runner can
    # traverse its first submission path.
    plan = validate_resume_contract(override, output)
    session_start = frozen_runner.parse_utc(str(_session(plan, 1)["operational_start_utc"]))
    opening_lateness = (datetime.now(timezone.utc) - session_start).total_seconds()
    if opening_lateness > float(arguments.max_opening_lateness_seconds):
        raise RuntimeError(
            "Session 1 opening target became too late during the running-status gate: "
            f"{opening_lateness:.3f} seconds"
        )

    with selected_plan_name(override):
        report = frozen_runner.execute(
            arguments.loop_config.resolve(),
            arguments.backend_config.resolve(),
            arguments.peer_config.resolve(),
            arguments.stage1_manifest.resolve(),
            output,
            confirm_hardware=arguments.confirm_hardware,
            max_wait_seconds=arguments.max_wait_seconds,
            poll_seconds=arguments.poll_seconds,
            platform_factory=lambda _config: platform,
        )
    print(
        json.dumps(
            {"completed": report["completed"], **report["observed"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
