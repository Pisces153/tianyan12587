#!/usr/bin/env python3
"""Freeze auditable platform task times for B4 T287 or grouped T176 tasks."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import drift_campaign


LEDGER_SCHEMA = "b4_platform_task_time_ledger_v1"
FREEZE_SCHEMA = "b4_platform_task_time_freeze_v1"
T176_ANALYSIS_SCHEMA = "b4_t176_platform_timing_analysis_v1"
QUERY_MANIFEST_SCHEMA = "b4_t176_timing_query_manifest_v1"
LEDGER_FILE_NAME = "platform_task_time_ledger.json"
FREEZE_FILE_NAME = "freeze_manifest.json"
FREEZE_HASH_FILE_NAME = "freeze_manifest.sha256"
QUERY_MANIFEST_FILE_NAME = "query_id_manifest.json"
T176_SOURCE_RECORD_ALLOWLIST = (
    "id",
    "backend",
    "runStartTime",
    "finishTime",
    "startTime",
    "status",
    "shots",
    "difference",
    "role",
)
PRIMARY_SF_ROLES = {
    "primary_sf_short_lag_reference",
    "primary_sf_only_when_non_event_and_same_regime",
}


def parse_platform_time(value: Any, timezone_name: str) -> datetime:
    if value in (None, ""):
        raise ValueError("platform time is missing")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def primary_sf_query_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    query_ids: set[str] = set()
    for row in records:
        if row.get("event") != "collected":
            continue
        for observation in row.get("observations", []):
            if (
                str(observation.get("analysis_role", "")) in PRIMARY_SF_ROLES
                and bool(observation.get("primary_sf_eligible"))
                and not bool(observation.get("burst_flag"))
            ):
                query_id = str(observation.get("query_id", ""))
                if not query_id:
                    raise ValueError("eligible primary-SF observation is missing query_id")
                query_ids.add(query_id)
    return sorted(query_ids)


def cadence_collected_query_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    query_ids: set[str] = set()
    for row in records:
        if row.get("event") != "collected":
            continue
        for value in row.get("query_ids", []):
            query_id = str(value)
            if not query_id:
                raise ValueError("collected cadence job contains an empty query_id")
            query_ids.add(query_id)
    return sorted(query_ids)


def target_query_ids(records: Sequence[Mapping[str, Any]], target_set: str) -> list[str]:
    if target_set == "primary-sf":
        return primary_sf_query_ids(records)
    if target_set == "cadence-collected":
        return cadence_collected_query_ids(records)
    raise ValueError(f"unknown target set: {target_set}")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def load_query_id_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate a grouped query-ID manifest.

    Query order is evidence: it identifies each setting's position inside its
    submitted job, so it is deliberately not sorted.
    """

    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"query-ID manifest must be an object: {path}")
    if payload.get("schema") not in (None, QUERY_MANIFEST_SCHEMA):
        raise ValueError(f"unexpected query-ID manifest schema: {payload.get('schema')}")
    jobs_raw = payload.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError(f"query-ID manifest has no jobs list: {path}")

    jobs: list[dict[str, Any]] = []
    flattened: list[str] = []
    seen: set[str] = set()
    for job_index, raw_job in enumerate(jobs_raw):
        if not isinstance(raw_job, Mapping):
            raise ValueError(f"query-ID manifest job {job_index} is not an object")
        name = str(raw_job.get("name", "")).strip()
        if not name:
            raise ValueError(f"query-ID manifest job {job_index} has no name")
        settings = _positive_int(raw_job.get("settings"), f"job {name} settings")
        shots = _positive_int(raw_job.get("shots_per_setting"), f"job {name} shots_per_setting")
        declared_role = raw_job.get("role")
        if declared_role is None:
            lowered_name = name.casefold()
            matching_roles = [
                role for role in ("baseline", "sense", "mirror") if role in lowered_name
            ]
            role = matching_roles[0] if len(matching_roles) == 1 else "other"
        else:
            role = str(declared_role).strip().casefold()
            if role not in {"baseline", "sense", "mirror"}:
                raise ValueError(
                    f"query-ID manifest job {name} has unsupported role: {declared_role}"
                )
        raw_query_ids = raw_job.get("query_ids")
        if not isinstance(raw_query_ids, list):
            raise ValueError(f"query-ID manifest job {name} has no query_ids list")
        query_ids = [str(value).strip() for value in raw_query_ids]
        if any(not value for value in query_ids):
            raise ValueError(f"query-ID manifest job {name} contains an empty query_id")
        if len(query_ids) != settings:
            raise ValueError(
                f"query-ID manifest job {name} declares {settings} settings "
                f"but contains {len(query_ids)} query IDs"
            )
        duplicates = sorted(value for value in query_ids if value in seen)
        if duplicates:
            raise ValueError(f"duplicate query IDs in manifest: {duplicates}")
        seen.update(query_ids)
        flattened.extend(query_ids)
        job = {
            "job_index": job_index,
            "name": name,
            "role": role,
            "settings": settings,
            "shots_per_setting": shots,
            "query_ids": query_ids,
        }
        if raw_job.get("roundtrip_seconds") is not None:
            roundtrip = float(raw_job["roundtrip_seconds"])
            if not math.isfinite(roundtrip) or roundtrip <= 0:
                raise ValueError(f"job {name} roundtrip_seconds must be positive and finite")
            job["declared_roundtrip_seconds"] = roundtrip
        jobs.append(job)

    declared_all = payload.get("all_query_ids")
    if declared_all is not None:
        if not isinstance(declared_all, list):
            raise ValueError("query-ID manifest all_query_ids is not a list")
        normalized_all = [str(value).strip() for value in declared_all]
        if normalized_all != flattened:
            raise ValueError("query-ID manifest all_query_ids does not match grouped job order")
    if payload.get("n") is not None and _positive_int(payload["n"], "manifest n") != len(flattened):
        raise ValueError("query-ID manifest n does not match query-ID count")

    return {
        "input_path": path,
        "raw": raw,
        "sha256": drift_campaign.digest_bytes(raw),
        "backend": str(payload.get("backend", "")),
        "schema": payload.get("schema"),
        "date_utc": payload.get("date_utc"),
        "jobs": jobs,
        "all_query_ids": flattened,
    }


def _is_platform_page(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and isinstance(payload.get("data"), Mapping)
        and isinstance(payload["data"].get("records"), list)
    )


def _safe_request_url(value: Any) -> str | None:
    """Keep endpoint provenance without retaining query strings or fragments."""

    if not value:
        return None
    parts = urlsplit(str(value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _har_platform_responses(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    log = payload.get("log")
    if not isinstance(log, Mapping) or not isinstance(log.get("entries"), list):
        return []
    candidates: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(log["entries"]):
        if not isinstance(entry, Mapping):
            continue
        response = entry.get("response")
        content = response.get("content") if isinstance(response, Mapping) else None
        if not isinstance(content, Mapping) or not isinstance(content.get("text"), str):
            continue
        encoded = content["text"]
        try:
            if content.get("encoding") == "base64":
                raw = base64.b64decode(encoded, validate=True)
            else:
                raw = encoded.encode("utf-8")
            candidate_payload = json.loads(raw.decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError):
            continue
        if not _is_platform_page(candidate_payload):
            continue
        request = entry.get("request")
        request_url = request.get("url") if isinstance(request, Mapping) else None
        candidates.append({
            "payload": candidate_payload,
            "raw": raw,
            "source_format": "har-response",
            "har_entry_index": entry_index,
            "request_url": _safe_request_url(request_url),
            "input_path": path,
        })
    return candidates


def _platform_response_candidates(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw_input = path.read_bytes()
    payload = json.loads(raw_input.decode("utf-8-sig"))
    if _is_platform_page(payload):
        return raw_input, [{
            "payload": payload,
            "raw": raw_input,
            "source_format": "task-list-json",
            "har_entry_index": None,
            "request_url": None,
            "input_path": path,
        }]
    if isinstance(payload, Mapping):
        candidates = _har_platform_responses(payload, path)
        if candidates:
            return raw_input, candidates
    raise ValueError(f"platform input has no task-list data.records response: {path}")


def load_platform_pages(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    record_index: dict[str, dict[str, Any]] = {}
    seen_page_hashes: set[str] = set()
    for path in paths:
        raw_input, candidates = _platform_response_candidates(path)
        input_sha256 = drift_campaign.digest_bytes(raw_input)
        for candidate in candidates:
            raw = candidate["raw"]
            page_sha256 = drift_campaign.digest_bytes(raw)
            if page_sha256 in seen_page_hashes:
                continue
            seen_page_hashes.add(page_sha256)
            payload = candidate["payload"]
            if payload.get("code") != 200:
                raise ValueError(f"platform page failed: {path}")
            data = payload["data"]
            records = data["records"]
            current = int(data.get("current", len(pages) + 1))
            page = {
                "input_path": path,
                "input_sha256": input_sha256,
                "raw": raw,
                "sha256": page_sha256,
                "current": current,
                "size": int(data.get("size", len(records))),
                "total": int(data.get("total", len(records))),
                "pages": int(data.get("pages", 1)),
                "records": records,
                "source_format": candidate["source_format"],
                "har_entry_index": candidate["har_entry_index"],
                "request_url": candidate["request_url"],
            }
            pages.append(page)
            for record in page["records"]:
                if not isinstance(record, Mapping):
                    raise ValueError(f"platform page contains a non-object record: {path}")
                query_id = str(record.get("id", ""))
                if not query_id:
                    raise ValueError(f"platform record has no id: {path}")
                record_sha256 = drift_campaign.digest_payload(record)
                existing = record_index.get(query_id)
                if existing is not None and existing["record_sha256"] != record_sha256:
                    raise ValueError(f"conflicting platform records for task {query_id}")
                record_index[query_id] = {
                    "record": record,
                    "record_sha256": record_sha256,
                    "page": page,
                }
    pages.sort(key=lambda page: page["current"])
    return pages, record_index


def build_entry(query_id: str, indexed: Mapping[str, Any], timezone_name: str) -> dict[str, Any]:
    record = indexed["record"]
    created = parse_platform_time(record.get("startTime"), timezone_name)
    started = parse_platform_time(record.get("runStartTime"), timezone_name)
    finished = parse_platform_time(record.get("finishTime"), timezone_name)
    runtime_seconds = (finished - started).total_seconds()
    if runtime_seconds < 0:
        raise ValueError(f"negative platform runtime for task {query_id}")
    return {
        "query_id": query_id,
        "platform_task_id": str(record.get("id")),
        "creation_time_raw": str(record.get("startTime")),
        "creation_time_utc": utc_text(created),
        "execution_start_time_raw": str(record.get("runStartTime")),
        "execution_start_time_utc": utc_text(started),
        "execution_end_time_raw": str(record.get("finishTime")),
        "execution_end_time_utc": utc_text(finished),
        "runtime_seconds": round(runtime_seconds, 3),
        "runtime_source": "finishTime-runStartTime",
        "platform_status": record.get("status"),
        "source_page": int(indexed["page"]["current"]),
        "source_page_sha256": str(indexed["page"]["sha256"]),
        "source_record_sha256": str(indexed["record_sha256"]),
    }


def manifest_query_metadata(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for job in manifest["jobs"]:
        for setting_index, query_id in enumerate(job["query_ids"]):
            metadata[query_id] = {
                "job_index": int(job["job_index"]),
                "job_name": str(job["name"]),
                "role": str(job["role"]),
                "setting_index": setting_index,
                "settings_in_job": int(job["settings"]),
                "shots_per_setting": int(job["shots_per_setting"]),
            }
    return metadata


def build_manifest_entries(
    manifest: Mapping[str, Any],
    record_index: Mapping[str, Mapping[str, Any]],
    timezone_name: str,
) -> list[dict[str, Any]]:
    metadata = manifest_query_metadata(manifest)
    entries: list[dict[str, Any]] = []
    for query_id in manifest["all_query_ids"]:
        entry = build_entry(query_id, record_index[query_id], timezone_name)
        entry.update(metadata[query_id])
        record = record_index[query_id]["record"]
        difference = record.get("difference")
        entry.update({
            "backend": record.get("backend"),
            "shots": record.get("shots"),
            "role": record.get("role"),
            "platform_difference_seconds": float(difference) if difference is not None else None,
            "platform_difference_matches_runtime": (
                abs(float(difference) - float(entry["runtime_seconds"])) <= 0.001
                if difference is not None
                else None
            ),
        })
        entries.append(entry)
    return entries


def _enrich_expected_manifest_entry(
    entry: dict[str, Any],
    query_id: str,
    record: Mapping[str, Any],
    query_metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    entry.update(query_metadata[query_id])
    difference = record.get("difference")
    entry.update({
        "backend": record.get("backend"),
        "shots": record.get("shots"),
        "role": record.get("role"),
        "platform_difference_seconds": float(difference) if difference is not None else None,
        "platform_difference_matches_runtime": (
            abs(float(difference) - float(entry["runtime_seconds"])) <= 0.001
            if difference is not None
            else None
        ),
    })


def build_allowlisted_manifest_pages(
    manifest: Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
    record_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Create target-only source extracts; never copy result/count payloads."""

    metadata = manifest_query_metadata(manifest)
    query_ids_by_page: dict[str, list[str]] = {}
    for query_id in manifest["all_query_ids"]:
        page_sha256 = str(record_index[query_id]["page"]["sha256"])
        query_ids_by_page.setdefault(page_sha256, []).append(query_id)

    sanitized_pages: list[dict[str, Any]] = []
    sanitized_index: dict[str, dict[str, Any]] = {}
    for source_page in pages:
        query_ids = query_ids_by_page.get(str(source_page["sha256"]), [])
        if not query_ids:
            continue
        sanitized_records: list[dict[str, Any]] = []
        for query_id in query_ids:
            source_record = record_index[query_id]["record"]
            expected_shots = int(metadata[query_id]["shots_per_setting"])
            source_shots = source_record.get("shotsNumber")
            if source_shots is not None and int(source_shots) != expected_shots:
                raise ValueError(
                    f"platform task {query_id} shotsNumber={source_shots} "
                    f"does not match manifest shots_per_setting={expected_shots}"
                )
            sanitized_records.append({
                "id": query_id,
                "backend": str(manifest.get("backend", "")),
                "runStartTime": source_record.get("runStartTime"),
                "finishTime": source_record.get("finishTime"),
                "startTime": source_record.get("startTime"),
                "status": source_record.get("status"),
                "shots": expected_shots,
                "difference": source_record.get("difference"),
                "role": metadata[query_id]["role"],
            })
        sanitized_payload = {
            "code": 200,
            "data": {
                "records": sanitized_records,
                "current": int(source_page["current"]),
                "size": int(source_page["size"]),
                "total": int(source_page["total"]),
                "pages": int(source_page["pages"]),
            },
        }
        raw = (
            json.dumps(sanitized_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        sanitized_page = {
            "input_path": source_page["input_path"],
            "input_sha256": source_page["input_sha256"],
            "raw_response_sha256": source_page["sha256"],
            "raw": raw,
            "sha256": drift_campaign.digest_bytes(raw),
            "current": int(source_page["current"]),
            "size": int(source_page["size"]),
            "total": int(source_page["total"]),
            "pages": int(source_page["pages"]),
            "records": sanitized_records,
            "source_format": "allowlisted-metadata-extract",
            "har_entry_index": source_page.get("har_entry_index"),
            "request_url": source_page.get("request_url"),
        }
        sanitized_pages.append(sanitized_page)
        for record in sanitized_records:
            query_id = str(record["id"])
            sanitized_index[query_id] = {
                "record": record,
                "record_sha256": drift_campaign.digest_payload(record),
                "page": sanitized_page,
            }
    sanitized_pages.sort(key=lambda page: page["current"])
    return sanitized_pages, sanitized_index


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _entry_time(entry: Mapping[str, Any], field: str) -> datetime:
    return datetime.fromisoformat(str(entry[field]).replace("Z", "+00:00"))


def _runtime_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty runtime sequence")
    return {
        "count": len(values),
        "minimum_seconds": _rounded(min(values)),
        "median_seconds": _rounded(statistics.median(values)),
        "mean_seconds": _rounded(statistics.fmean(values)),
        "maximum_seconds": _rounded(max(values)),
        "population_sd_seconds": _rounded(statistics.pstdev(values)),
        "sum_seconds": _rounded(sum(values)),
    }


def ordinary_least_squares(observations: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Fit runtime = intercept + slope * shots using only stdlib math."""

    n = len(observations)
    unique_shot_counts = sorted({float(shots) for shots, _ in observations})
    result: dict[str, Any] = {
        "observation_count": n,
        "unique_shot_count": len(unique_shot_counts),
        "shot_counts": [int(value) if value.is_integer() else value for value in unique_shot_counts],
        "model": "runtime_seconds = intercept_seconds + seconds_per_shot * shots_per_setting",
    }
    if n < 2 or len(unique_shot_counts) < 2:
        result.update({
            "status": "insufficient_distinct_shot_counts",
            "physically_valid": False,
            "intercept_seconds": None,
            "seconds_per_shot": None,
            "shot_rate_per_second": None,
        })
        return result

    xs = [float(pair[0]) for pair in observations]
    ys = [float(pair[1]) for pair in observations]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    sxx = sum((value - x_mean) ** 2 for value in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    if sxx == 0:
        raise AssertionError("distinct shot counts produced zero variance")
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * value for value in xs]
    residuals = [actual - estimate for actual, estimate in zip(ys, predicted, strict=True)]
    residual_sum_squares = sum(value * value for value in residuals)
    total_sum_squares = sum((value - y_mean) ** 2 for value in ys)
    r_squared = None if total_sum_squares == 0 else 1.0 - residual_sum_squares / total_sum_squares
    residual_degrees_of_freedom = n - 2
    residual_standard_error = None
    slope_standard_error = None
    intercept_standard_error = None
    if residual_degrees_of_freedom > 0:
        residual_variance = residual_sum_squares / residual_degrees_of_freedom
        residual_standard_error = math.sqrt(residual_variance)
        slope_standard_error = math.sqrt(residual_variance / sxx)
        intercept_standard_error = math.sqrt(residual_variance * (1.0 / n + x_mean * x_mean / sxx))
    physically_valid = slope > 0 and intercept >= 0
    result.update({
        "status": "fit_complete" if physically_valid else "fit_complete_nonphysical",
        "physically_valid": physically_valid,
        "intercept_seconds": _rounded(intercept),
        "seconds_per_shot": _rounded(slope),
        "shot_rate_per_second": _rounded(1.0 / slope) if slope > 0 else None,
        "r_squared": _rounded(r_squared) if r_squared is not None else None,
        "root_mean_squared_error_seconds": _rounded(math.sqrt(residual_sum_squares / n)),
        "residual_sum_squares": _rounded(residual_sum_squares),
        "residual_degrees_of_freedom": residual_degrees_of_freedom,
        "residual_standard_error_seconds": (
            _rounded(residual_standard_error) if residual_standard_error is not None else None
        ),
        "intercept_standard_error_seconds": (
            _rounded(intercept_standard_error) if intercept_standard_error is not None else None
        ),
        "seconds_per_shot_standard_error": (
            _rounded(slope_standard_error) if slope_standard_error is not None else None
        ),
    })
    return result


def _interval_diagnostics(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervals = sorted(
        (
            _entry_time(entry, "execution_start_time_utc"),
            _entry_time(entry, "execution_end_time_utc"),
        )
        for entry in entries
    )
    if not intervals:
        raise ValueError("cannot diagnose an empty job")

    union_seconds = 0.0
    union_start, union_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= union_end:
            union_end = max(union_end, end)
        else:
            union_seconds += (union_end - union_start).total_seconds()
            union_start, union_end = start, end
    union_seconds += (union_end - union_start).total_seconds()

    events: list[tuple[datetime, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    active = 0
    max_concurrent = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        active += change
        max_concurrent = max(max_concurrent, active)

    execution_start = min(start for start, _ in intervals)
    execution_end = max(end for _, end in intervals)
    execution_envelope = (execution_end - execution_start).total_seconds()
    runtime_sum = sum(float(entry["runtime_seconds"]) for entry in entries)
    creation_start = min(_entry_time(entry, "creation_time_utc") for entry in entries)
    platform_roundtrip = (execution_end - creation_start).total_seconds()
    return {
        "execution_start_utc": utc_text(execution_start),
        "execution_end_utc": utc_text(execution_end),
        "execution_envelope_seconds": _rounded(execution_envelope),
        "execution_union_seconds": _rounded(union_seconds),
        "idle_gap_inside_execution_envelope_seconds": _rounded(max(0.0, execution_envelope - union_seconds)),
        "platform_roundtrip_start_utc": utc_text(creation_start),
        "platform_roundtrip_seconds": _rounded(platform_roundtrip),
        "queue_before_first_execution_seconds": _rounded(max(0.0, (execution_start - creation_start).total_seconds())),
        "runtime_sum_seconds": _rounded(runtime_sum),
        "runtime_sum_to_execution_envelope_ratio": (
            _rounded(runtime_sum / execution_envelope) if execution_envelope > 0 else None
        ),
        "runtime_sum_to_platform_roundtrip_ratio": (
            _rounded(runtime_sum / platform_roundtrip) if platform_roundtrip > 0 else None
        ),
        "execution_parallelism_ratio": _rounded(runtime_sum / union_seconds) if union_seconds > 0 else None,
        "max_concurrent_settings": max_concurrent,
        "has_overlapping_execution_intervals": max_concurrent > 1,
        "distinct_execution_interval_count": len(set(intervals)),
        "all_settings_share_execution_interval": len(set(intervals)) == 1,
        "runtime_sum_exceeds_platform_roundtrip": runtime_sum > platform_roundtrip + 0.001,
    }


def build_job_diagnostic(job: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda entry: int(entry["setting_index"]))
    runtimes = [float(entry["runtime_seconds"]) for entry in ordered]
    interval = _interval_diagnostics(ordered)
    if len(ordered) == 1:
        classification = "not_applicable_single_setting"
    elif interval["has_overlapping_execution_intervals"]:
        classification = "parallel_or_batched"
    else:
        classification = "serial"
    diagnostic: dict[str, Any] = {
        "job_index": int(job["job_index"]),
        "job_name": str(job["name"]),
        "role": str(job["role"]),
        "settings": int(job["settings"]),
        "shots_per_setting": int(job["shots_per_setting"]),
        "query_ids": [str(entry["query_id"]) for entry in ordered],
        "setting_runtime_seconds": [_rounded(value) for value in runtimes],
        "runtime_summary": _runtime_summary(runtimes),
        **interval,
        "serialization_diagnosis": classification,
        "diagnosis_basis": (
            "overlap of runStartTime-finishTime intervals"
            if interval["has_overlapping_execution_intervals"]
            else "non-overlap of runStartTime-finishTime intervals"
        ),
    }
    declared_roundtrip = job.get("declared_roundtrip_seconds")
    if declared_roundtrip is not None:
        diagnostic["declared_roundtrip_seconds"] = _rounded(float(declared_roundtrip))
        diagnostic["runtime_sum_to_declared_roundtrip_ratio"] = _rounded(
            sum(runtimes) / float(declared_roundtrip)
        )
        diagnostic["runtime_sum_exceeds_declared_roundtrip"] = (
            sum(runtimes) > float(declared_roundtrip) + 0.001
        )
    return diagnostic


def build_sensing_family_rate_model(
    job_diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit only the two-setting phase-sensing family; exclude mirror circuits."""

    eligible = [
        diagnostic
        for diagnostic in job_diagnostics
        if int(diagnostic["settings"]) == 2
        and str(diagnostic["role"]) in {"baseline", "sense"}
    ]
    excluded = [
        {
            "job_name": str(diagnostic["job_name"]),
            "reason": (
                "mirror circuit class excluded from sensing rate; retained only for "
                "role envelope and parallelism"
                if str(diagnostic["role"]) == "mirror"
                else "not a two-setting BASELINE/SENSE phase-sensing role"
            ),
            "role": str(diagnostic["role"]),
        }
        for diagnostic in job_diagnostics
        if diagnostic not in eligible
    ]
    observations = sorted(
        (
            float(diagnostic["shots_per_setting"]),
            float(diagnostic["runtime_summary"]["mean_seconds"]),
            str(diagnostic["job_name"]),
            str(diagnostic["role"]),
        )
        for diagnostic in eligible
    )
    result: dict[str, Any] = {
        "model_scope": "two-setting BASELINE+SENSE phase-sensing circuit family only",
        "observation_unit": "job mean per-setting finishTime-runStartTime",
        "included_jobs": [item[2] for item in observations],
        "included_roles": [item[3] for item in observations],
        "excluded_roles": excluded,
        "shot_level_count": len({item[0] for item in observations}),
        "confidence_interval": None,
        "confidence_interval_status": "not_estimable_with_two_shot_levels",
    }
    if len(observations) != 2 or observations[0][0] == observations[1][0]:
        result.update({
            "status": "requires_exactly_two_distinct_sensing_shot_levels",
            "planning_rate_per_second": None,
            "planning_overhead_seconds": None,
        })
        return result

    low_shots, low_runtime, low_job, low_role = observations[0]
    high_shots, high_runtime, high_job, high_role = observations[1]
    shot_difference = high_shots - low_shots
    runtime_difference = high_runtime - low_runtime
    slope = runtime_difference / shot_difference
    intercept = low_runtime - slope * low_shots
    rate = 1.0 / slope if slope > 0 else None
    unconstrained = ordinary_least_squares([
        (low_shots, low_runtime),
        (high_shots, high_runtime),
    ])
    unconstrained.update({
        "intercept_seconds": round(intercept, 11),
        "seconds_per_shot": round(slope, 15),
        "shot_rate_per_second": round(rate, 11) if rate is not None else None,
    })

    constrained_overhead = max(0.0, intercept)
    predictions = []
    for shots, observed, job_name, role in observations:
        predicted = constrained_overhead + slope * shots
        predictions.append({
            "role": role,
            "job_name": job_name,
            "shots_per_setting": int(shots),
            "observed_runtime_seconds": _rounded(observed),
            "constrained_prediction_seconds": round(predicted, 11),
            "prediction_minus_observed_seconds": round(predicted - observed, 11),
        })
    conservative = all(
        row["prediction_minus_observed_seconds"] >= -1e-9
        for row in predictions
    )
    result.update({
        "status": "conservative_two_level_planning_model" if conservative else "not_conservative",
        "shot_difference": int(shot_difference),
        "runtime_difference_seconds": _rounded(runtime_difference),
        "unconstrained_two_level_ols": unconstrained,
        "physical_constraint": {
            "method": "clamp negative intercept to zero without refitting slope",
            "unconstrained_overhead_seconds": round(intercept, 11),
            "planning_overhead_seconds": round(constrained_overhead, 11),
            "planning_rate_per_second": round(rate, 11) if rate is not None else None,
            "planning_seconds_per_shot": round(slope, 15),
            "predictions": predictions,
            "conservative_upper_bound_on_both_observed_levels": conservative,
            "maximum_overprediction_seconds": round(
                max(row["prediction_minus_observed_seconds"] for row in predictions), 11
            ),
        },
        "planning_rate_per_second": round(rate, 11) if rate is not None else None,
        "planning_overhead_seconds": round(constrained_overhead, 11),
        "planning_use": (
            "conservative for same phase-sensing circuit family only; mirror timing excluded"
        ),
        "limitations": (
            "Only two shot levels are available. Residual degrees of freedom are zero; "
            "no variance estimate or confidence interval is identifiable."
        ),
        "low_shot_role": low_role,
        "high_shot_role": high_role,
        "low_shot_job": low_job,
        "high_shot_job": high_job,
    })
    return result


def build_t176_timing_analysis(
    manifest: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    entry_by_query_id = {str(entry["query_id"]): entry for entry in entries}
    job_diagnostics: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        job_entries = [entry_by_query_id[query_id] for query_id in job["query_ids"]]
        job_diagnostics.append(build_job_diagnostic(job, job_entries))

    grouped: dict[int, list[float]] = {}
    task_observations: list[tuple[float, float]] = []
    for entry in entries:
        shots = int(entry["shots_per_setting"])
        runtime = float(entry["runtime_seconds"])
        grouped.setdefault(shots, []).append(runtime)
        task_observations.append((float(shots), runtime))
    shot_level_summaries = [
        {
            "shots_per_setting": shots,
            **_runtime_summary(values),
        }
        for shots, values in sorted(grouped.items())
    ]
    level_observations = [
        (float(row["shots_per_setting"]), float(row["mean_seconds"]))
        for row in shot_level_summaries
    ]
    task_fit = ordinary_least_squares(task_observations)
    level_fit = ordinary_least_squares(level_observations)
    sensing_family_rate_model = build_sensing_family_rate_model(job_diagnostics)

    per_job_execution_wall = sum(
        float(diagnostic["execution_envelope_seconds"])
        for diagnostic in job_diagnostics
    )
    sum_per_task_runtime = sum(float(entry["runtime_seconds"]) for entry in entries)
    timing_basis_scenarios = {
        "scope": "three submitted jobs in query-ID manifest",
        "queue_wait_included": False,
        "per_job_execution_wall": {
            "definition": "sum(max(finishTime)-min(runStartTime)) once per submitted job",
            "seconds": _rounded(per_job_execution_wall),
            "job_seconds": [
                {
                    "job_name": diagnostic["job_name"],
                    "seconds": diagnostic["execution_envelope_seconds"],
                }
                for diagnostic in job_diagnostics
            ],
        },
        "sum_per_task_runtime": {
            "definition": "sum(finishTime-runStartTime) over every platform task",
            "seconds": _rounded(sum_per_task_runtime),
            "job_seconds": [
                {
                    "job_name": diagnostic["job_name"],
                    "seconds": diagnostic["runtime_sum_seconds"],
                }
                for diagnostic in job_diagnostics
            ],
        },
        "sum_per_task_to_per_job_wall_ratio": (
            _rounded(sum_per_task_runtime / per_job_execution_wall)
            if per_job_execution_wall > 0
            else None
        ),
        "billing_basis_status": "unresolved_without_observed_quota_decrement",
    }
    difference_entries = [
        entry for entry in entries if entry.get("platform_difference_seconds") is not None
    ]

    mirror_jobs = [
        diagnostic
        for diagnostic in job_diagnostics
        if diagnostic["role"] == "mirror"
    ]
    if not mirror_jobs and job_diagnostics:
        largest = max(job_diagnostics, key=lambda row: int(row["settings"]))
        if int(largest["settings"]) > 2:
            mirror_jobs = [largest]

    return {
        "schema": T176_ANALYSIS_SCHEMA,
        "backend": manifest.get("backend"),
        "run_date_utc": manifest.get("date_utc"),
        "runtime_source": "finishTime-runStartTime",
        "queue_wait_excluded_from_setting_runtime": True,
        "known_shot_counts_source": QUERY_MANIFEST_FILE_NAME,
        "setting_count": len(entries),
        "per_setting_runtime_summary": _runtime_summary(
            [float(entry["runtime_seconds"]) for entry in entries]
        ),
        "platform_difference_validation": {
            "available_count": len(difference_entries),
            "match_count": sum(
                entry.get("platform_difference_matches_runtime") is True
                for entry in difference_entries
            ),
            "all_available_match_finish_minus_run_start": bool(difference_entries)
            and all(
                entry.get("platform_difference_matches_runtime") is True
                for entry in difference_entries
            ),
            "absolute_tolerance_seconds": 0.001,
        },
        "shot_level_summaries": shot_level_summaries,
        "runtime_regression": {
            "planning_fit": "sensing_family_two_level_conservative_model",
            "sensing_family_two_level_conservative_model": sensing_family_rate_model,
            "cross_role_fits_are_diagnostic_only": True,
            "cross_role_diagnostic_reason": (
                "MIRROR is a different circuit class. Mixed-role fits diagnose confounding; "
                "they are not sensing-rate estimates."
            ),
            "equal_weight_shot_level_mean_ols": level_fit,
            "all_setting_ols_sensitivity": task_fit,
            "queue_free_interpretation": (
                "Both fits use finishTime-runStartTime; startTime queue delay is excluded."
            ),
            "shot_count_only_model_usable_for_planning": (
                bool(level_fit.get("physically_valid"))
                and level_fit.get("r_squared") is not None
                and float(level_fit["r_squared"]) >= 0.9
            ),
            "planning_minimum_r_squared": 0.9,
            "model_limit": (
                "Job classes use different circuits; a poor or nonphysical fit means shot count "
                "alone does not identify a portable rate/overhead model."
            ),
        },
        "timing_basis_scenarios": timing_basis_scenarios,
        "job_diagnostics": job_diagnostics,
        "mirror_serialization_diagnostics": mirror_jobs,
    }


def verify_ledger_artifact(ledger_path: Path) -> dict[str, Any]:
    issues: list[str] = []
    root = ledger_path.parent.resolve()
    freeze_path = root / FREEZE_FILE_NAME
    freeze_hash_path = root / FREEZE_HASH_FILE_NAME
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if ledger.get("schema") != LEDGER_SCHEMA:
        issues.append("unexpected ledger schema")
    if freeze.get("schema") != FREEZE_SCHEMA:
        issues.append("unexpected freeze schema")
    actual_ledger_hash = drift_campaign.digest_file(ledger_path)
    if freeze.get("ledger_sha256") != actual_ledger_hash:
        issues.append("ledger SHA-256 mismatch")
    actual_freeze_hash = drift_campaign.digest_file(freeze_path)
    expected_freeze_hash = freeze_hash_path.read_text(encoding="utf-8").strip().upper()
    if actual_freeze_hash != expected_freeze_hash:
        issues.append("freeze manifest SHA-256 mismatch")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        issues.append("ledger entries is not a list")
        entries = []
    if ledger.get("entries_sha256") != drift_campaign.digest_payload(entries):
        issues.append("entries SHA-256 mismatch")
    target_query_ids = ledger.get("target_query_ids")
    if not isinstance(target_query_ids, list):
        issues.append("target_query_ids is not a list")
        target_query_ids = []
    if ledger.get("target_query_ids_sha256") != drift_campaign.digest_payload(target_query_ids):
        issues.append("target query-id SHA-256 mismatch")
    if sorted(str(entry.get("query_id")) for entry in entries) != sorted(str(value) for value in target_query_ids):
        issues.append("ledger entries do not exactly match target query IDs")

    manifest: dict[str, Any] | None = None
    manifest_descriptor = ledger.get("query_id_manifest")
    if manifest_descriptor is not None:
        if not isinstance(manifest_descriptor, Mapping):
            issues.append("query_id_manifest descriptor is not an object")
        else:
            manifest_relative_path = Path(str(manifest_descriptor.get("file", "")))
            manifest_path = (root / manifest_relative_path).resolve()
            if manifest_path.parent != root or manifest_path.name != QUERY_MANIFEST_FILE_NAME:
                issues.append(f"invalid query-ID manifest path: {manifest_relative_path}")
            elif not manifest_path.is_file():
                issues.append(f"missing query-ID manifest: {manifest_relative_path}")
            elif drift_campaign.digest_file(manifest_path) != manifest_descriptor.get("sha256"):
                issues.append("query-ID manifest SHA-256 mismatch")
            else:
                try:
                    manifest = load_query_id_manifest(manifest_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(f"invalid query-ID manifest: {exc}")
                else:
                    if list(manifest["all_query_ids"]) != [str(value) for value in target_query_ids]:
                        issues.append("target query IDs do not match query-ID manifest order")
                    if freeze.get("query_id_manifest_sha256") != manifest["sha256"]:
                        issues.append("freeze query-ID manifest SHA-256 mismatch")

    source_records: dict[str, Mapping[str, Any]] = {}
    for source in ledger.get("source_pages", []):
        relative_path = Path(str(source.get("file", "")))
        source_path = (root / relative_path).resolve()
        if source_path.parent != (root / "source_pages").resolve():
            issues.append(f"invalid source path: {relative_path}")
            continue
        if not source_path.is_file():
            issues.append(f"missing source page: {relative_path}")
            continue
        if drift_campaign.digest_file(source_path) != source.get("sha256"):
            issues.append(f"source page SHA-256 mismatch: {relative_path}")
            continue
        payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
        for record in payload.get("data", {}).get("records", []):
            source_records[str(record.get("id"))] = record

    timezone_name = str(ledger.get("platform_timezone"))
    query_metadata = manifest_query_metadata(manifest) if manifest is not None else {}
    for entry in entries:
        query_id = str(entry.get("query_id"))
        record = source_records.get(query_id)
        if record is None:
            issues.append(f"source record missing for task {query_id}")
            continue
        if drift_campaign.digest_payload(record) != entry.get("source_record_sha256"):
            issues.append(f"source record SHA-256 mismatch for task {query_id}")
        expected = build_entry(
            query_id,
            {
                "record": record,
                "record_sha256": drift_campaign.digest_payload(record),
                "page": {
                    "current": entry.get("source_page"),
                    "sha256": entry.get("source_page_sha256"),
                },
            },
            timezone_name,
        )
        if manifest is not None and query_id in query_metadata:
            _enrich_expected_manifest_entry(expected, query_id, record, query_metadata)
        if expected != entry:
            issues.append(f"derived ledger fields mismatch for task {query_id}")

    if manifest is not None and len(entries) == len(manifest["all_query_ids"]):
        try:
            expected_analysis = build_t176_timing_analysis(manifest, entries)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"could not reproduce T176 timing analysis: {exc}")
        else:
            actual_analysis = ledger.get("t176_timing_analysis")
            if actual_analysis != expected_analysis:
                issues.append("T176 timing analysis mismatch")
            actual_analysis_hash = drift_campaign.digest_payload(actual_analysis)
            if ledger.get("t176_timing_analysis_sha256") != actual_analysis_hash:
                issues.append("T176 timing analysis SHA-256 mismatch")
            if freeze.get("t176_timing_analysis_sha256") != actual_analysis_hash:
                issues.append("freeze T176 timing analysis SHA-256 mismatch")
    return {
        "valid": not issues,
        "issues": issues,
        "ledger": ledger,
        "freeze": freeze,
        "ledger_sha256": actual_ledger_hash,
        "freeze_sha256": actual_freeze_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    target_source = parser.add_mutually_exclusive_group(required=True)
    target_source.add_argument("--campaign-root", type=Path)
    target_source.add_argument(
        "--query-id-manifest",
        "--t176-query-ids",
        dest="query_id_manifest",
        type=Path,
        help="grouped jobs/query IDs with known shots_per_setting",
    )
    parser.add_argument("--platform-response", type=Path, action="append", required=True)
    parser.add_argument("--platform-timezone", default="Asia/Shanghai")
    parser.add_argument(
        "--target-set",
        choices=("primary-sf", "cadence-collected"),
        default="primary-sf",
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.out.exists():
        raise FileExistsError(f"Refusing to overwrite time-ledger output: {arguments.out}")

    manifest: dict[str, Any] | None = None
    journal_path: Path | None = None
    if arguments.query_id_manifest is not None:
        manifest = load_query_id_manifest(arguments.query_id_manifest)
        selected_query_ids = list(manifest["all_query_ids"])
        selected_target_set = "query-id-manifest"
    else:
        journal_path = arguments.campaign_root / "snapshots.jsonl"
        records = drift_campaign.CampaignStore(arguments.campaign_root).records
        selected_query_ids = target_query_ids(records, arguments.target_set)
        selected_target_set = arguments.target_set
    raw_pages, raw_record_index = load_platform_pages(arguments.platform_response)
    missing = [query_id for query_id in selected_query_ids if query_id not in raw_record_index]
    if missing:
        raise ValueError(f"platform responses missing {len(missing)} target tasks: {missing}")
    if manifest is not None:
        selected_raw_page_hashes = {
            str(raw_record_index[query_id]["page"]["sha256"])
            for query_id in selected_query_ids
        }
        pages, record_index = build_allowlisted_manifest_pages(
            manifest, raw_pages, raw_record_index
        )
        entries = build_manifest_entries(manifest, record_index, arguments.platform_timezone)
    else:
        selected_raw_page_hashes = set()
        pages = raw_pages
        record_index = raw_record_index
        entries = [
            build_entry(query_id, record_index[query_id], arguments.platform_timezone)
            for query_id in selected_query_ids
        ]
    selected_page_hashes = {entry["source_page_sha256"] for entry in entries}
    selected_pages = [page for page in pages if page["sha256"] in selected_page_hashes]

    arguments.out.mkdir(parents=True)
    source_root = arguments.out / "source_pages"
    source_root.mkdir()
    source_pages: list[dict[str, Any]] = []
    for page in selected_pages:
        relative_path = Path("source_pages") / f"page_{page['current']:03d}_{page['sha256'][:12]}.json"
        destination = arguments.out / relative_path
        destination.write_bytes(page["raw"])
        source_descriptor = {
            "file": relative_path.as_posix(),
            "sha256": page["sha256"],
            "current": page["current"],
            "size": page["size"],
            "total": page["total"],
            "pages": page["pages"],
        }
        if manifest is not None:
            source_descriptor.update({
                "source_format": "allowlisted-metadata-extract",
                "allowlisted_record_fields": list(T176_SOURCE_RECORD_ALLOWLIST),
                "raw_source_copied": False,
                "raw_input_path": str(page["input_path"].resolve()),
                "raw_input_sha256": page["input_sha256"],
                "raw_response_sha256": page["raw_response_sha256"],
            })
            if page["har_entry_index"] is not None:
                source_descriptor.update({
                    "har_entry_index": page["har_entry_index"],
                    "request_url_without_query": page["request_url"],
                })
        elif page["source_format"] == "har-response":
            source_descriptor.update({
                "source_format": "har-response-extracted-json",
                "input_file_name": page["input_path"].name,
                "input_sha256": page["input_sha256"],
                "har_entry_index": page["har_entry_index"],
                "request_url_without_query": page["request_url"],
            })
        source_pages.append(source_descriptor)

    query_manifest_descriptor: dict[str, Any] | None = None
    if manifest is not None:
        manifest_destination = arguments.out / QUERY_MANIFEST_FILE_NAME
        manifest_destination.write_bytes(manifest["raw"])
        query_manifest_descriptor = {
            "file": QUERY_MANIFEST_FILE_NAME,
            "sha256": manifest["sha256"],
            "backend": manifest["backend"],
            "date_utc": manifest["date_utc"],
            "job_count": len(manifest["jobs"]),
            "query_id_count": len(manifest["all_query_ids"]),
        }

    input_pages: list[dict[str, Any]] = []
    for page in raw_pages:
        input_page = {
            "current": page["current"],
            "sha256": page["sha256"],
            "selected_for_target_evidence": (
                page["sha256"] in selected_raw_page_hashes
                if manifest is not None
                else page["sha256"] in selected_page_hashes
            ),
        }
        if manifest is not None:
            input_page.update({
                "input_file_name": page["input_path"].name,
                "input_sha256": page["input_sha256"],
                "raw_input_copied": False,
            })
        elif page["source_format"] == "har-response":
            input_page.update({
                "source_format": "har-response",
                "input_file_name": page["input_path"].name,
                "input_sha256": page["input_sha256"],
                "har_entry_index": page["har_entry_index"],
                "request_url_without_query": page["request_url"],
            })
        input_pages.append(input_page)

    ledger = {
        "schema": LEDGER_SCHEMA,
        "retrieval_method": "manual browser DevTools platform-response export",
        "retrieval_note": "Client audit timestamps are not used as analysis timestamps.",
        "platform_timezone": arguments.platform_timezone,
        "analysis_timestamp_field": "runStartTime",
        "creation_timestamp_field": "startTime",
        "execution_end_field": "finishTime",
        "runtime_derivation": "finishTime-runStartTime",
        "target_set": selected_target_set,
        "hardware_submission_performed": False,
        "t176_quarantine_read": False,
        "target_query_ids": selected_query_ids,
        "target_query_ids_sha256": drift_campaign.digest_payload(selected_query_ids),
        "source_pages": source_pages,
        "input_pages": input_pages,
        "entries": entries,
        "entries_sha256": drift_campaign.digest_payload(entries),
        "recovery_summary": {
            "recovered": len(entries),
            "target": len(selected_query_ids),
            "missing": 0,
            "missing_query_ids": [],
        },
    }
    if journal_path is not None:
        ledger["campaign_journal_sha256"] = drift_campaign.digest_file(journal_path)
    if manifest is not None and query_manifest_descriptor is not None:
        timing_analysis = build_t176_timing_analysis(manifest, entries)
        ledger.update({
            "target_source": "grouped query-ID manifest",
            "backend": manifest["backend"],
            "endpoint_contribution": "none",
            "pooling_permitted": False,
            "not_main_endpoint": True,
            "query_id_manifest": query_manifest_descriptor,
            "t176_timing_analysis": timing_analysis,
            "t176_timing_analysis_sha256": drift_campaign.digest_payload(timing_analysis),
        })
    ledger_path = arguments.out / LEDGER_FILE_NAME
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "schema": FREEZE_SCHEMA,
        "ledger_file": LEDGER_FILE_NAME,
        "ledger_sha256": drift_campaign.digest_file(ledger_path),
        "source_files": source_pages,
        "target_count": len(selected_query_ids),
        "recovered_count": len(entries),
        "missing_count": 0,
        "analysis_performed_before_freeze": manifest is not None,
    }
    if manifest is not None:
        freeze.update({
            "query_id_manifest_file": QUERY_MANIFEST_FILE_NAME,
            "query_id_manifest_sha256": manifest["sha256"],
            "t176_timing_analysis_sha256": ledger["t176_timing_analysis_sha256"],
        })
    freeze_path = arguments.out / FREEZE_FILE_NAME
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_hash = drift_campaign.digest_file(freeze_path)
    (arguments.out / FREEZE_HASH_FILE_NAME).write_text(freeze_hash + "\n", encoding="utf-8")
    verification = verify_ledger_artifact(ledger_path)
    if not verification["valid"]:
        raise RuntimeError(f"time-ledger self-verification failed: {verification['issues']}")
    print(json.dumps({
        "status": "verified_complete",
        "recovered": len(entries),
        "target": len(selected_query_ids),
        "ledger": str(ledger_path),
        "ledger_sha256": verification["ledger_sha256"],
        "freeze_sha256": verification["freeze_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
