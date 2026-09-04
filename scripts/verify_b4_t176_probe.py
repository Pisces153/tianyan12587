#!/usr/bin/env python3
"""T-B6.3: first and only T176 terminal-verification probe branch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import b4_dry_run_common as common
from scripts import drift_campaign_v4


DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\hardware\B4_TB6\t176_terminal_check.json")
DEFAULT_REVALIDATED_OUTPUT = DEFAULT_OUTPUT.with_name("t176_terminal_check_revalidated.json")


def validate_returned_results(config: Mapping[str, Any], results: list[Mapping[str, Any]]) -> dict[str, Any]:
    identifiers_valid = bool(results) and all(
        isinstance(result, Mapping) and result.get("experimentTaskId") is not None
        for result in results
    )
    raw_counts_valid = bool(results)
    if raw_counts_valid:
        for result in results:
            common.raw_counts(
                result,
                config["backend"]["physical_qubits"],
                int(config["measurement"]["probe_job"]["shots_per_setting"]),
            )
    timestamp_rows = [drift_campaign_v4.platform_timestamp(result) for result in results]
    return {
        "result_identifiers_valid": identifiers_valid,
        "raw_counts_valid": raw_counts_valid,
        "result_fields_and_raw_counts_valid": identifiers_valid and raw_counts_valid,
        "execution_timestamp_available": bool(timestamp_rows) and all(
            bool(row["execution_timestamp_available"]) for row in timestamp_rows
        ),
        "execution_timestamps": timestamp_rows,
        "execution_timestamp_required_for_terminal_pass": False,
        "missing_execution_timestamp_policy": config["timestamp_policy"]["missing_platform_timestamp"],
    }


def execute(config_path: Path, output: Path, *, confirm_hardware: bool,
            platform_factory: Callable[[Mapping[str, Any]], Any] = common.platform_from_config) -> dict[str, Any]:
    if not confirm_hardware:
        raise RuntimeError("T176 terminal check requires --confirm-hardware")
    config = common.load_config(config_path)
    if str(config["backend"]["backend_id"]) != "tianyan176":
        raise ValueError("terminal-check script is frozen to tianyan176")
    programs = drift_campaign_v4.build_probe_programs(config)
    platform = platform_factory(config)
    failure: str | None = None
    record: dict[str, Any] | None = None
    results: list[Mapping[str, Any]] = []
    calibration_payload: Any = None
    validation = {
        "result_identifiers_valid": False,
        "raw_counts_valid": False,
        "result_fields_and_raw_counts_valid": False,
        "execution_timestamp_available": False,
        "execution_timestamps": [],
        "execution_timestamp_required_for_terminal_pass": False,
        "missing_execution_timestamp_policy": config["timestamp_policy"]["missing_platform_timestamp"],
    }
    try:
        calibration_payload = platform.download_config(machine="tianyan176")
        record, results = common.run_job(
            platform=platform,
            config=config,
            circuits=[str(row["qcis"]) for row in programs],
            shots_per_setting=int(config["measurement"]["probe_job"]["shots_per_setting"]),
            name="XA202609_B4_TB6_T176_TERMINAL_CHECK",
            max_wait_seconds=900,
            poll_seconds=5,
        )
        validation = validate_returned_results(config, results)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    passed = (
        failure is None
        and record is not None
        and bool(validation["result_fields_and_raw_counts_valid"])
        and common.calibration_time_raw(calibration_payload) is not None
    )
    report = {
        "schema": "b4_tb6_t176_terminal_check_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend_id": "tianyan176",
        "first_job_only": True,
        "job": record,
        "calibrationTime_raw": common.calibration_time_raw(calibration_payload),
        "account_quota_job_acceptance": bool(record),
        **validation,
        "frozen_physical_qubits": config["backend"]["physical_qubits"],
        "passed": passed,
        "failure": failure,
        "result4_downgrade_required": not passed,
        "failure_branch": "freeze method on one device plus simulation-to-hardware transfer; report immediately; do not seek a substitute backend",
    }
    common.write_new(output, report)
    return report


def revalidate_existing(
    config_path: Path,
    source_report_path: Path,
    output: Path,
    *,
    confirm_hardware: bool,
    platform_factory: Callable[[Mapping[str, Any]], Any] = common.platform_from_config,
) -> dict[str, Any]:
    if not confirm_hardware:
        raise RuntimeError("T176 existing-job revalidation requires --confirm-hardware")
    config = common.load_config(config_path)
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source.get("backend_id") != "tianyan176" or not source.get("job"):
        raise ValueError("source report is not a submitted T176 terminal check")
    query_ids = [str(value) for value in source["job"].get("query_ids", [])]
    if not query_ids:
        raise ValueError("source report has no query IDs to revalidate")
    platform = platform_factory(config)
    failure: str | None = None
    calibration_payload: Any = None
    validation: dict[str, Any]
    try:
        calibration_payload = platform.download_config(machine="tianyan176")
        results = platform.query_experiment(query_ids, max_wait_time=300, sleep_time=5)
        if not isinstance(results, list):
            raise RuntimeError("platform query did not return a result list")
        validation = validate_returned_results(config, results)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        validation = {
            "result_identifiers_valid": False,
            "raw_counts_valid": False,
            "result_fields_and_raw_counts_valid": False,
            "execution_timestamp_available": False,
            "execution_timestamps": [],
            "execution_timestamp_required_for_terminal_pass": False,
            "missing_execution_timestamp_policy": config["timestamp_policy"]["missing_platform_timestamp"],
        }
    passed = (
        failure is None
        and bool(validation["result_fields_and_raw_counts_valid"])
        and common.calibration_time_raw(calibration_payload) is not None
    )
    report = {
        "schema": "b4_tb6_t176_terminal_check_revalidated_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend_id": "tianyan176",
        "verification_mode": "existing_query_ids_no_resubmission",
        "source_report": str(source_report_path.resolve()),
        "source_created_at_utc": source.get("created_at_utc"),
        "job": source["job"],
        "new_hardware_job_submitted": False,
        "calibrationTime_raw": common.calibration_time_raw(calibration_payload),
        "account_quota_job_acceptance": bool(source.get("account_quota_job_acceptance")),
        **validation,
        "frozen_physical_qubits": config["backend"]["physical_qubits"],
        "passed": passed,
        "failure": failure,
        "result4_downgrade_required": not passed,
        "correction_reason": "execution timestamp is optional under the frozen timestamp policy; terminal pass requires returned IDs, parseable raw counts, and calibrationTime",
    }
    common.write_new(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revalidate-from", type=Path)
    parser.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()
    output = arguments.output
    if arguments.revalidate_from is not None and output == DEFAULT_OUTPUT:
        output = DEFAULT_REVALIDATED_OUTPUT
    report = (
        revalidate_existing(
            arguments.config,
            arguments.revalidate_from,
            output,
            confirm_hardware=arguments.confirm_hardware,
        )
        if arguments.revalidate_from is not None
        else execute(arguments.config, output, confirm_hardware=arguments.confirm_hardware)
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
