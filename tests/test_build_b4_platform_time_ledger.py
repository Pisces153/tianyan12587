from __future__ import annotations

import base64
import json
from pathlib import Path
import sys

import pytest

from scripts import build_b4_platform_time_ledger as module


def test_platform_local_time_is_normalized_to_utc_with_milliseconds() -> None:
    parsed = module.parse_platform_time("2026-08-10 21:30:58.811", "Asia/Shanghai")
    assert module.utc_text(parsed) == "2026-08-10T13:30:58.811Z"


def test_entry_uses_run_start_and_derives_runtime() -> None:
    record = {
        "id": "task-1",
        "startTime": "2026-08-10 21:30:43",
        "runStartTime": "2026-08-10 21:30:58.811",
        "finishTime": "2026-08-10 21:31:56.375",
        "status": 2,
    }
    entry = module.build_entry(
        "task-1",
        {
            "record": record,
            "record_sha256": "A" * 64,
            "page": {"current": 14, "sha256": "B" * 64},
        },
        "Asia/Shanghai",
    )
    assert entry["execution_start_time_utc"] == "2026-08-10T13:30:58.811Z"
    assert entry["execution_end_time_utc"] == "2026-08-10T13:31:56.375Z"
    assert entry["runtime_seconds"] == 57.564
    assert entry["runtime_source"] == "finishTime-runStartTime"


def test_cadence_collected_query_ids_are_unique_and_sorted() -> None:
    records = [
        {"event": "submitted", "query_ids": ["ignored"]},
        {"event": "collected", "query_ids": ["task-2", "task-1"]},
        {"event": "collected", "query_ids": ["task-2", "task-3"]},
    ]
    assert module.target_query_ids(records, "cadence-collected") == ["task-1", "task-2", "task-3"]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _grouped_manifest() -> dict[str, object]:
    jobs = [
        {
            "name": "BASELINE",
            "role": "baseline",
            "settings": 2,
            "shots_per_setting": 100,
            "query_ids": ["b1", "b2"],
        },
        {
            "name": "SENSE",
            "role": "sense",
            "settings": 2,
            "shots_per_setting": 10,
            "query_ids": ["s1", "s2"],
        },
        {
            "name": "MIRROR",
            "role": "mirror",
            "settings": 3,
            "shots_per_setting": 50,
            "query_ids": ["m1", "m2", "m3"],
        },
    ]
    all_query_ids = [value for job in jobs for value in job["query_ids"]]
    return {
        "backend": "tianyan176",
        "date_utc": "2026-08-17",
        "n": len(all_query_ids),
        "jobs": jobs,
        "all_query_ids": all_query_ids,
    }


def test_grouped_query_manifest_preserves_job_and_setting_order(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    _write_json(path, _grouped_manifest())

    manifest = module.load_query_id_manifest(path)

    assert manifest["all_query_ids"] == ["b1", "b2", "s1", "s2", "m1", "m2", "m3"]
    assert module.manifest_query_metadata(manifest)["m2"] == {
        "job_index": 2,
        "job_name": "MIRROR",
        "role": "mirror",
        "setting_index": 1,
        "settings_in_job": 3,
        "shots_per_setting": 50,
    }


def test_grouped_query_manifest_rejects_count_or_order_mismatch(tmp_path: Path) -> None:
    payload = _grouped_manifest()
    payload["all_query_ids"] = list(reversed(payload["all_query_ids"]))
    path = tmp_path / "ids.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="does not match grouped job order"):
        module.load_query_id_manifest(path)


def test_grouped_query_manifest_rejects_unknown_explicit_role(tmp_path: Path) -> None:
    payload = _grouped_manifest()
    payload["jobs"][0]["role"] = "calibration-ish"
    path = tmp_path / "ids.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="unsupported role"):
        module.load_query_id_manifest(path)


def test_load_platform_pages_accepts_raw_task_list_json(tmp_path: Path) -> None:
    page_path = tmp_path / "page.json"
    page = {
        "code": 200,
        "data": {
            "records": [{"id": "task-1"}],
            "current": 27,
            "size": 100,
            "total": 10685,
            "pages": 107,
        },
    }
    _write_json(page_path, page)

    pages, record_index = module.load_platform_pages([page_path])

    assert len(pages) == 1
    assert pages[0]["source_format"] == "task-list-json"
    assert pages[0]["current"] == 27
    assert record_index["task-1"]["record"] == {"id": "task-1"}


def test_load_platform_pages_accepts_multiple_raw_page_files(tmp_path: Path) -> None:
    paths: list[Path] = []
    for current, query_id in ((26, "task-1"), (27, "task-2")):
        path = tmp_path / f"page_{current}.json"
        _write_json(path, {
            "code": 200,
            "data": {
                "records": [{"id": query_id}],
                "current": current,
                "size": 1,
                "total": 2,
                "pages": 2,
            },
        })
        paths.append(path)

    pages, record_index = module.load_platform_pages(paths)

    assert [page["current"] for page in pages] == [26, 27]
    assert set(record_index) == {"task-1", "task-2"}


def test_load_platform_pages_extracts_plain_and_base64_har_responses(tmp_path: Path) -> None:
    page_1 = {
        "code": 200,
        "data": {"records": [{"id": "task-1"}], "current": 1, "size": 1, "total": 2, "pages": 2},
    }
    page_2 = {
        "code": 200,
        "data": {"records": [{"id": "task-2"}], "current": 2, "size": 1, "total": 2, "pages": 2},
    }
    page_2_text = json.dumps(page_2)
    har = {
        "log": {
            "entries": [
                {
                    "request": {"url": "https://qc.example/task/list?page=1&token=secret"},
                    "response": {"content": {"text": json.dumps(page_1)}},
                },
                {"request": {"url": "https://qc.example/unrelated"}, "response": {"content": {"text": "{}"}}},
                {
                    "request": {"url": "https://qc.example/task/list?page=2"},
                    "response": {
                        "content": {
                            "text": base64.b64encode(page_2_text.encode()).decode(),
                            "encoding": "base64",
                        }
                    },
                },
            ]
        }
    }
    har_path = tmp_path / "capture.har"
    _write_json(har_path, har)

    pages, record_index = module.load_platform_pages([har_path])

    assert [page["current"] for page in pages] == [1, 2]
    assert set(record_index) == {"task-1", "task-2"}
    assert pages[0]["request_url"] == "https://qc.example/task/list"
    assert pages[0]["har_entry_index"] == 0
    assert pages[1]["har_entry_index"] == 2


def test_ordinary_least_squares_reports_rate_and_overhead() -> None:
    fit = module.ordinary_least_squares([(100.0, 6.0), (200.0, 7.0), (300.0, 8.0)])

    assert fit["status"] == "fit_complete"
    assert fit["intercept_seconds"] == 5.0
    assert fit["seconds_per_shot"] == 0.01
    assert fit["shot_rate_per_second"] == 100.0
    assert fit["r_squared"] == 1.0


def _analysis_entry(
    query_id: str,
    *,
    job_index: int,
    job_name: str,
    setting_index: int,
    settings: int,
    shots: int,
    creation: str,
    start: str,
    finish: str,
    runtime: float,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "job_index": job_index,
        "job_name": job_name,
        "setting_index": setting_index,
        "settings_in_job": settings,
        "shots_per_setting": shots,
        "creation_time_utc": creation,
        "execution_start_time_utc": start,
        "execution_end_time_utc": finish,
        "runtime_seconds": runtime,
    }


def _synthetic_analysis_entries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specifications = [
        ("BASELINE", ["b1", "b2"], 100, "00:00:00", "00:00:01", "00:00:07", 6.0),
        ("SENSE", ["s1", "s2"], 10, "00:00:59", "00:01:00", "00:01:02", 2.0),
        ("MIRROR", ["m1", "m2", "m3"], 50, "00:01:59", "00:02:00", "00:02:10", 10.0),
    ]
    for job_index, (name, query_ids, shots, creation, start, finish, runtime) in enumerate(specifications):
        for setting_index, query_id in enumerate(query_ids):
            rows.append(_analysis_entry(
                query_id,
                job_index=job_index,
                job_name=name,
                setting_index=setting_index,
                settings=len(query_ids),
                shots=shots,
                creation=f"2026-08-17T{creation}.000Z",
                start=f"2026-08-17T{start}.000Z",
                finish=f"2026-08-17T{finish}.000Z",
                runtime=runtime,
            ))
    return rows


def test_t176_analysis_proves_parallel_batch_and_reports_two_timing_bases(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ids.json"
    _write_json(manifest_path, _grouped_manifest())
    manifest = module.load_query_id_manifest(manifest_path)

    analysis = module.build_t176_timing_analysis(manifest, _synthetic_analysis_entries())

    mirror = analysis["mirror_serialization_diagnostics"][0]
    assert mirror["serialization_diagnosis"] == "parallel_or_batched"
    assert mirror["max_concurrent_settings"] == 3
    assert mirror["all_settings_share_execution_interval"] is True
    assert mirror["runtime_sum_seconds"] == 30.0
    assert mirror["execution_envelope_seconds"] == 10.0
    scenarios = analysis["timing_basis_scenarios"]
    assert scenarios["per_job_execution_wall"]["seconds"] == 18.0
    assert scenarios["sum_per_task_runtime"]["seconds"] == 46.0
    assert scenarios["billing_basis_status"] == "unresolved_without_observed_quota_decrement"


def test_observed_t176_timings_lock_wall_and_per_task_scenarios(tmp_path: Path) -> None:
    specifications = [
        ("BASELINE", 2, 27664, "19:16:49.211", "19:17:07.946", 18.735),
        ("SENSE", 2, 6186, "19:17:45.920", "19:17:49.476", 3.556),
        ("MIRROR", 20, 16384, "19:18:32.337", "19:19:44.394", 72.057),
    ]
    jobs: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    for job_index, (name, count, shots, start, finish, runtime) in enumerate(specifications):
        query_ids = [f"{name.lower()}-{index}" for index in range(count)]
        jobs.append({
            "name": name,
            "settings": count,
            "shots_per_setting": shots,
            "query_ids": query_ids,
        })
        for setting_index, query_id in enumerate(query_ids):
            entries.append(_analysis_entry(
                query_id,
                job_index=job_index,
                job_name=name,
                setting_index=setting_index,
                settings=count,
                shots=shots,
                creation=f"2026-08-17T{start}Z",
                start=f"2026-08-17T{start}Z",
                finish=f"2026-08-17T{finish}Z",
                runtime=runtime,
            ))
    manifest_path = tmp_path / "observed_ids.json"
    _write_json(manifest_path, {
        "backend": "tianyan176",
        "date_utc": "2026-08-17",
        "n": len(entries),
        "jobs": jobs,
        "all_query_ids": [entry["query_id"] for entry in entries],
    })
    manifest = module.load_query_id_manifest(manifest_path)

    analysis = module.build_t176_timing_analysis(manifest, entries)

    scenarios = analysis["timing_basis_scenarios"]
    assert scenarios["per_job_execution_wall"]["seconds"] == 94.348
    assert scenarios["sum_per_task_runtime"]["seconds"] == 1485.722
    mirror = analysis["mirror_serialization_diagnostics"][0]
    assert mirror["max_concurrent_settings"] == 20
    assert mirror["runtime_sum_seconds"] == 1441.14
    assert mirror["execution_envelope_seconds"] == 72.057
    regression = analysis["runtime_regression"]
    assert regression["shot_count_only_model_usable_for_planning"] is False
    assert regression["equal_weight_shot_level_mean_ols"]["r_squared"] == 0.033286883
    sensing = regression["sensing_family_two_level_conservative_model"]
    assert sensing["included_roles"] == ["sense", "baseline"]
    assert sensing["unconstrained_two_level_ols"]["shot_rate_per_second"] == 1414.98122405956
    assert sensing["unconstrained_two_level_ols"]["intercept_seconds"] == -0.81578945898
    assert sensing["planning_rate_per_second"] == 1414.98122405956
    assert sensing["planning_overhead_seconds"] == 0.0
    assert sensing["confidence_interval"] is None
    assert sensing["confidence_interval_status"] == "not_estimable_with_two_shot_levels"
    assert sensing["physical_constraint"]["conservative_upper_bound_on_both_observed_levels"] is True
    assert sensing["physical_constraint"]["maximum_overprediction_seconds"] == 0.81578945898
    assert sensing["excluded_roles"] == [{
        "job_name": "MIRROR",
        "reason": "mirror circuit class excluded from sensing rate; retained only for role envelope and parallelism",
        "role": "mirror",
    }]


def _platform_record(query_id: str, start: str, run_start: str, finish: str) -> dict[str, object]:
    return {
        "id": query_id,
        "startTime": start,
        "runStartTime": run_start,
        "finishTime": finish,
        "status": 2,
        "isOver": 1,
        "graphResult": {"must": "not be copied"},
    }


def test_manifest_mode_writes_self_verifying_auditable_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "ids.json"
    _write_json(manifest_path, _grouped_manifest())
    times = {
        "b1": ("2026-08-17 00:00:00", "2026-08-17 00:00:01", "2026-08-17 00:00:07"),
        "b2": ("2026-08-17 00:00:00", "2026-08-17 00:00:01", "2026-08-17 00:00:07"),
        "s1": ("2026-08-17 00:00:59", "2026-08-17 00:01:00", "2026-08-17 00:01:02"),
        "s2": ("2026-08-17 00:00:59", "2026-08-17 00:01:00", "2026-08-17 00:01:02"),
        "m1": ("2026-08-17 00:01:59", "2026-08-17 00:02:00", "2026-08-17 00:02:10"),
        "m2": ("2026-08-17 00:01:59", "2026-08-17 00:02:00", "2026-08-17 00:02:10"),
        "m3": ("2026-08-17 00:01:59", "2026-08-17 00:02:00", "2026-08-17 00:02:10"),
    }
    page_path = tmp_path / "page.json"
    _write_json(page_path, {
        "code": 200,
        "data": {
            "records": [_platform_record(query_id, *time_values) for query_id, time_values in times.items()],
            "current": 27,
            "size": 100,
            "total": 10685,
            "pages": 107,
        },
    })
    out = tmp_path / "ledger"
    monkeypatch.setattr(sys, "argv", [
        "build_b4_platform_time_ledger.py",
        "--query-id-manifest",
        str(manifest_path),
        "--platform-response",
        str(page_path),
        "--out",
        str(out),
    ])

    assert module.main() == 0
    verification = module.verify_ledger_artifact(out / module.LEDGER_FILE_NAME)

    assert verification["valid"] is True
    ledger = verification["ledger"]
    assert ledger["target_set"] == "query-id-manifest"
    assert ledger["query_id_manifest"]["query_id_count"] == 7
    assert ledger["t176_timing_analysis_sha256"]
    assert ledger["endpoint_contribution"] == "none"
    assert ledger["pooling_permitted"] is False
    source = ledger["source_pages"][0]
    assert source["raw_source_copied"] is False
    extracted = json.loads((out / source["file"]).read_text(encoding="utf-8"))
    assert set(extracted["data"]["records"][0]) == set(module.T176_SOURCE_RECORD_ALLOWLIST)
    assert "graphResult" not in extracted["data"]["records"][0]
