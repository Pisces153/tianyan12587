#!/usr/bin/env python3
"""Poll TianYan platform calibration metadata without submitting jobs.

One process owns one backend and one credential environment variable.  The only
SDK method called by the polling path is ``download_config``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping


DEFAULT_OUTPUT_ROOT = Path(r"E:\TianYan\XA-202609\artifacts\telemetry\platform_config_poll")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include UTC offset")
    return parsed.astimezone(timezone.utc)


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, datetime):
        return iso(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def calibration_time_raw(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        if "calibrationTime" in payload:
            return json_ready(payload["calibrationTime"])
        for value in payload.values():
            found = calibration_time_raw(value)
            if found is not None:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = calibration_time_raw(value)
            if found is not None:
                return found
    return None


def error_fields_raw(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Return every raw field whose key contains ``error`` (case-insensitive)."""
    found: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "error" in str(key).lower():
                found[path] = json_ready(value)
            found.update(error_fields_raw(value, path))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            found.update(error_fields_raw(value, f"{prefix}[{index}]"))
    return found


def default_platform_factory(login_key: str, backend_id: str) -> Any:
    from cqlib.quantum_platform import TianYanPlatform

    return TianYanPlatform(login_key=login_key, auto_login=True, machine_name=backend_id)


def output_path(root: Path, backend_id: str, timestamp: datetime) -> Path:
    return root / backend_id / f"platform_config_{timestamp:%Y%m%d}.jsonl"


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_success_utc(root: Path, backend_id: str) -> datetime | None:
    directory = root / backend_id
    if not directory.exists():
        return None
    for path in sorted(directory.glob("platform_config_*.jsonl"), reverse=True):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and isinstance(row.get("query_finished_utc"), str):
                return parse_utc(row["query_finished_utc"])
    return None


def poll_once(
    *,
    backend_id: str,
    login_key: str,
    account_label: str | None,
    output_root: Path,
    previous_success_utc: datetime | None,
    interval_seconds: float,
    jitter_seconds: float,
    platform_factory: Callable[[str, str], Any] = default_platform_factory,
    clock: Callable[[], datetime] = utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], datetime | None]:
    started = clock()
    started_tick = timer()
    try:
        platform = platform_factory(login_key, backend_id)
        # Deliberately no submit/experiment API call in this module.
        payload = platform.download_config(machine=backend_id)
        finished = clock()
        row: dict[str, Any] = {
            "schema": "b4_platform_config_poll_v1",
            "status": "ok",
            "backend_id": backend_id,
            "account_label": account_label,
            "query_started_utc": iso(started),
            "query_finished_utc": iso(finished),
            "query_elapsed_seconds": max(0.0, float(timer() - started_tick)),
            "calibrationTime_raw": calibration_time_raw(payload),
            "platform_error_fields_raw": error_fields_raw(payload),
            "download_config_raw": json_ready(payload),
            "hardware_job_submitted": False,
        }
        if previous_success_utc is None:
            row["gap_since_previous_success_seconds"] = None
            row["polling_gap_detected"] = False
            row["polling_gap_window"] = None
        else:
            gap = max(0.0, (finished - previous_success_utc).total_seconds())
            expected_max = interval_seconds + jitter_seconds + 60.0
            row["gap_since_previous_success_seconds"] = gap
            row["polling_gap_detected"] = bool(gap > expected_max)
            row["polling_gap_window"] = (
                {"start_utc": iso(previous_success_utc), "end_utc": iso(finished), "duration_seconds": gap}
                if gap > expected_max
                else None
            )
        append_jsonl(output_path(output_root, backend_id, finished), row)
        return row, finished
    except Exception as error:
        finished = clock()
        row = {
            "schema": "b4_platform_config_poll_v1",
            "status": "error",
            "backend_id": backend_id,
            "account_label": account_label,
            "query_started_utc": iso(started),
            "query_finished_utc": iso(finished),
            "query_elapsed_seconds": max(0.0, float(timer() - started_tick)),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "previous_success_utc": iso(previous_success_utc) if previous_success_utc else None,
            "hardware_job_submitted": False,
        }
        append_jsonl(output_path(output_root, backend_id, finished), row)
        return row, previous_success_utc


def next_delay(interval_seconds: float, jitter_seconds: float, rng: random.Random) -> float:
    return max(0.0, interval_seconds + rng.uniform(-jitter_seconds, jitter_seconds))


def run_forever(
    *,
    backend_id: str,
    credential_env: str,
    account_label: str | None,
    output_root: Path,
    interval_seconds: float,
    jitter_seconds: float,
    retry_attempts: int,
    retry_base_seconds: float,
    seed: int | None,
    once: bool,
) -> int:
    login_key = os.environ.get(credential_env)
    if not login_key:
        raise RuntimeError(f"{credential_env} is not set in this process")
    rng = random.Random(seed)
    previous_success = latest_success_utc(output_root, backend_id)
    while True:
        row: dict[str, Any] | None = None
        for attempt in range(1, retry_attempts + 1):
            row, previous_success = poll_once(
                backend_id=backend_id,
                login_key=login_key,
                account_label=account_label,
                output_root=output_root,
                previous_success_utc=previous_success,
                interval_seconds=interval_seconds,
                jitter_seconds=jitter_seconds,
            )
            if row["status"] == "ok":
                break
            if attempt < retry_attempts:
                time.sleep(retry_base_seconds * (2 ** (attempt - 1)))
        if once:
            return 0 if row and row["status"] == "ok" else 1
        time.sleep(next_delay(interval_seconds, jitter_seconds, rng))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--credential-env", default="TIANYAN_LOGIN_KEY")
    parser.add_argument("--account-label")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--interval-seconds", type=float, default=3600.0)
    parser.add_argument("--jitter-seconds", type=float, default=300.0)
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.jitter_seconds < 0:
        raise ValueError("interval must be positive and jitter nonnegative")
    return run_forever(
        backend_id=args.backend,
        credential_env=args.credential_env,
        account_label=args.account_label,
        output_root=args.output_root,
        interval_seconds=args.interval_seconds,
        jitter_seconds=args.jitter_seconds,
        retry_attempts=max(1, args.retry_attempts),
        retry_base_seconds=max(0.0, args.retry_base_seconds),
        seed=args.seed,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
