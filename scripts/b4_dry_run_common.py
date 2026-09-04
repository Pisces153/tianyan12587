"""Shared execution primitives for the four T-B6 hardware measurements."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts import drift_campaign as campaign
from scripts import drift_campaign_v4


def load_config(path: Path) -> dict[str, Any]:
    return drift_campaign_v4.load_config(path.resolve())


def platform_from_config(config: Mapping[str, Any]) -> Any:
    return drift_campaign_v4.platform_from_config(config)


def write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite T-B6 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_job(
    *,
    platform: Any,
    config: Mapping[str, Any],
    circuits: Sequence[str],
    shots_per_setting: int,
    name: str,
    max_wait_seconds: int,
    poll_seconds: int,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if not circuits or shots_per_setting <= 0:
        raise ValueError("job requires circuits and positive shots")
    backend = str(config["backend"]["backend_id"])
    started_utc = clock()
    started = timer()
    query_ids = platform.submit_experiment(
        circuit=list(circuits),
        name=name,
        num_shots=int(shots_per_setting),
        machine_name=backend,
    )
    submitted_utc = clock()
    if not isinstance(query_ids, list) or len(query_ids) != len(circuits):
        raise RuntimeError("platform returned incomplete task-ID list")
    results = platform.query_experiment(query_ids, max_wait_time=max_wait_seconds, sleep_time=poll_seconds)
    finished = timer()
    finished_utc = clock()
    if not isinstance(results, list):
        raise RuntimeError("platform query did not return a result list")
    returned_ids = {
        str(row["experimentTaskId"])
        for row in results
        if isinstance(row, Mapping) and row.get("experimentTaskId") is not None
    }
    missing = [str(value) for value in query_ids if str(value) not in returned_ids]
    if missing:
        raise RuntimeError(f"platform result missing {len(missing)} task IDs")
    record = {
        "backend_id": backend,
        "job_name": name,
        "settings": len(circuits),
        "shots_per_setting": int(shots_per_setting),
        "total_shots": len(circuits) * int(shots_per_setting),
        "started_utc": started_utc.replace(microsecond=0).isoformat(),
        "submitted_utc": submitted_utc.replace(microsecond=0).isoformat(),
        "finished_utc": finished_utc.replace(microsecond=0).isoformat(),
        "roundtrip_seconds": float(finished - started),
        "query_ids": [str(value) for value in query_ids],
        "result_count": len(results),
        "hardware_job_submitted": True,
    }
    return record, results


def estimate_rate_and_overhead(measurements: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    groups: dict[int, list[float]] = {}
    total_shots: dict[int, int] = {}
    settings_per_job: set[int] = set()
    for row in measurements:
        shots = int(row["shots_per_setting"])
        groups.setdefault(shots, []).append(float(row["roundtrip_seconds"]))
        total_shots[shots] = int(row["total_shots"])
        settings_per_job.add(int(row["settings"]))
    if len(groups) != 2 or any(len(values) < 2 for values in groups.values()):
        raise ValueError("throughput fit requires exactly two shot levels with at least two repeats each")
    if len(settings_per_job) != 1:
        raise ValueError("throughput fit requires one fixed setting count across shot levels")
    low, high = sorted(groups)
    low_time = statistics.median(groups[low])
    high_time = statistics.median(groups[high])
    shot_delta = total_shots[high] - total_shots[low]
    time_delta = high_time - low_time
    if shot_delta <= 0 or time_delta <= 0.0:
        raise ValueError("non-positive throughput contrast")
    rate = shot_delta / time_delta
    overhead_per_job = low_time - total_shots[low] / rate
    setting_count = settings_per_job.pop()
    overhead_per_setting = overhead_per_job / setting_count
    return {
        "low_shots_per_setting": low,
        "high_shots_per_setting": high,
        "repeats_per_level": min(len(groups[low]), len(groups[high])),
        "median_low_roundtrip_seconds": low_time,
        "median_high_roundtrip_seconds": high_time,
        "effective_shots_per_second": rate,
        "settings_per_job": setting_count,
        "fixed_overhead_seconds_per_job": overhead_per_job,
        "fixed_overhead_seconds_per_setting": overhead_per_setting,
        "wallclock_model": "roundtrip_seconds = total_shots / effective_shots_per_second + settings * fixed_overhead_seconds_per_setting",
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("invalid percentile inputs")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def calibration_time_raw(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() == "calibrationtime":
                return value
            nested = calibration_time_raw(value)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for value in payload:
            nested = calibration_time_raw(value)
            if nested is not None:
                return nested
    return None


def raw_counts(result: Mapping[str, Any], physical_qubits: Sequence[int], shots: int) -> dict[str, int]:
    counts = campaign.result_counts(result, physical_qubits, shots)
    width = len(physical_qubits)
    return {format(index, f"0{width}b"): int(count) for index, count in enumerate(counts) if int(count) > 0}
