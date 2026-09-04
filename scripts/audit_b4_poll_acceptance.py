#!/usr/bin/env python3
"""Audit B-4 poll coverage and, only after 24 h, test supervisor recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import psutil


DEFAULT_ROOT = Path(r"E:\TianYan\XA-202609\artifacts\telemetry\platform_config_poll")
BACKENDS = ("tianyan-287", "tianyan176")
REQUIRED_SPAN_SECONDS = 24.0 * 3600.0
MAX_RECOVERY_SECONDS = 5.0 * 60.0


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def load_rows(root: Path, backend: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / backend).glob("platform_config_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def summarize_backend(rows: Sequence[Mapping[str, Any]], backend: str) -> dict[str, Any]:
    successes = sorted(
        (row for row in rows if row.get("status") == "ok"),
        key=lambda row: parse_utc(str(row["query_finished_utc"])),
    )
    if not successes:
        return {
            "backend_id": backend,
            "success_count": 0,
            "error_count": sum(row.get("status") != "ok" for row in rows),
            "historical_span_seconds": 0.0,
            "current_continuous_segment_start_utc": None,
            "continuous_span_seconds": 0.0,
            "maximum_success_gap_seconds": None,
            "gap_count": 0,
            "current_segment_gap_count": 0,
            "hardware_job_count": sum(bool(row.get("hardware_job_submitted")) for row in rows),
            "coverage_24h_passed": False,
        }
    stamps = [parse_utc(str(row["query_finished_utc"])) for row in successes]
    gaps = [(right - left).total_seconds() for left, right in zip(stamps[:-1], stamps[1:])]
    flagged_indices = [
        index for index, row in enumerate(successes)
        if bool(row.get("polling_gap_detected"))
    ]
    segment_start_index = flagged_indices[-1] if flagged_indices else 0
    historical_span = (stamps[-1] - stamps[0]).total_seconds()
    continuous_span = (stamps[-1] - stamps[segment_start_index]).total_seconds()
    return {
        "backend_id": backend,
        "success_count": len(successes),
        "error_count": sum(row.get("status") != "ok" for row in rows),
        "first_success_utc": successes[0]["query_finished_utc"],
        "last_success_utc": successes[-1]["query_finished_utc"],
        "historical_span_seconds": historical_span,
        "current_continuous_segment_start_utc": successes[segment_start_index]["query_finished_utc"],
        "continuous_span_seconds": continuous_span,
        "maximum_success_gap_seconds": max(gaps) if gaps else None,
        "gap_count": len(flagged_indices),
        "current_segment_gap_count": 0,
        "hardware_job_count": sum(bool(row.get("hardware_job_submitted")) for row in rows),
        "coverage_24h_passed": bool(continuous_span >= REQUIRED_SPAN_SECONDS),
    }


def snapshot(root: Path) -> dict[str, Any]:
    summaries = [summarize_backend(load_rows(root, backend), backend) for backend in BACKENDS]
    return {
        "backends": summaries,
        "coverage_24h_all_backends_passed": all(row["coverage_24h_passed"] for row in summaries),
        "zero_hardware_jobs": all(row["hardware_job_count"] == 0 for row in summaries),
    }


def _matching_poller(backend: str) -> psutil.Process:
    matches: list[psutil.Process] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = [str(value) for value in (process.info["cmdline"] or [])]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        joined = " ".join(command)
        if "poll_platform_config.py" in joined and backend in joined:
            matches.append(process)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one poller for {backend}; found {len(matches)}")
    process = matches[0]
    parent = process.parent()
    if parent is None:
        raise RuntimeError("poller has no supervisor parent")
    parent_command = " ".join(parent.cmdline())
    if "run_platform_config_poll_supervisor.bat" not in parent_command or backend not in parent_command:
        raise RuntimeError("poller parent is not the expected backend-specific supervisor")
    return process


def kill_and_measure_recovery(root: Path, backend: str, timeout_seconds: float = MAX_RECOVERY_SECONDS) -> dict[str, Any]:
    before_successes = sum(row.get("status") == "ok" for row in load_rows(root, backend))
    process = _matching_poller(backend)
    old_pid = process.pid
    parent_pid = process.ppid()
    killed_at = datetime.now(timezone.utc)
    process.terminate()
    process.wait(timeout=30.0)
    deadline = time.monotonic() + timeout_seconds
    new_pid: int | None = None
    process_recovery_seconds: float | None = None
    telemetry_recovery_seconds: float | None = None
    while time.monotonic() < deadline:
        try:
            candidate = _matching_poller(backend)
        except RuntimeError:
            candidate = None
        if candidate is not None and candidate.pid != old_pid and candidate.ppid() == parent_pid:
            if new_pid is None:
                new_pid = candidate.pid
                process_recovery_seconds = (datetime.now(timezone.utc) - killed_at).total_seconds()
            current_successes = sum(row.get("status") == "ok" for row in load_rows(root, backend))
            if current_successes > before_successes:
                telemetry_recovery_seconds = (datetime.now(timezone.utc) - killed_at).total_seconds()
                break
        time.sleep(2.0)
    passed = bool(new_pid is not None and telemetry_recovery_seconds is not None and telemetry_recovery_seconds <= timeout_seconds)
    return {
        "backend_id": backend,
        "old_pid": old_pid,
        "supervisor_pid": parent_pid,
        "new_pid": new_pid,
        "killed_at_utc": killed_at.replace(microsecond=0).isoformat(),
        "process_recovery_seconds": process_recovery_seconds,
        "telemetry_recovery_seconds": telemetry_recovery_seconds,
        "required_recovery_seconds": timeout_seconds,
        "passed": passed,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(root: Path, output: Path, *, wait_until_eligible: bool, kill_backend: str) -> dict[str, Any]:
    while True:
        coverage = snapshot(root)
        if coverage["coverage_24h_all_backends_passed"] or not wait_until_eligible:
            break
        write_report(output, {
            "schema": "b4_platform_poll_acceptance_v1",
            "status": "waiting_for_24h_coverage",
            "checked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            **coverage,
        })
        time.sleep(60.0)
    recovery = None
    status = "not_eligible"
    if coverage["coverage_24h_all_backends_passed"]:
        recovery = kill_and_measure_recovery(root, kill_backend)
        status = "passed" if recovery["passed"] else "failed"
    report = {
        "schema": "b4_platform_poll_acceptance_v1",
        "status": status,
        "checked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **coverage,
        "kill_recovery_test": recovery,
        "acceptance_passed": bool(
            coverage["coverage_24h_all_backends_passed"]
            and coverage["zero_hardware_jobs"]
            and recovery
            and recovery["passed"]
        ),
    }
    write_report(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "B0_ACCEPTANCE_20260805.json")
    parser.add_argument("--wait-until-eligible", action="store_true")
    parser.add_argument("--kill-backend", choices=BACKENDS, default="tianyan-287")
    arguments = parser.parse_args()
    report = run(arguments.root, arguments.output, wait_until_eligible=arguments.wait_until_eligible, kill_backend=arguments.kill_backend)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["acceptance_passed"] or report["status"] == "not_eligible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
