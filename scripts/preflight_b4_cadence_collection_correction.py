#!/usr/bin/env python3
"""Offline preflight for the B4 cadence endpoint-identity collection correction."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_b4_t287_cadence_residual_curve as analysis
from scripts import build_b4_platform_time_ledger as time_ledger
from scripts import drift_campaign_v4
from scripts import run_b4_cadence_pair_hardware as runner
from scripts import run_cadence_pair_loop as cadence


DEFAULT_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json"
DEFAULT_BACKEND_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"
DEFAULT_PEER_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"
DEFAULT_EXISTING_ANALYSIS = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_B9_T287_CADENCE_RESIDUAL_20260815_r8")
DEFAULT_CADENCE_LEDGER = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_PLATFORM_TASK_TIME_LEDGER_20260815_r1\platform_task_time_ledger.json")
DEFAULT_STAGE1_LEDGER = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_PLATFORM_TASK_TIME_LEDGER_20260815_r1\platform_task_time_ledger.json")
DEFAULT_INPUT_AUDIT = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_B9_INPUT_AUDIT_20260815_r4\b9_input_audit.json")
DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_T176_CADENCE_COLLECTION_PREFLIGHT_20260823")
MIGRATION_CURATED_MANIFEST = ROOT / "docs" / "B4_T176_MIGRATION_CURATED_MANIFEST_20260823.json"
REGISTERED_HARDWARE_OUTPUT = "E:/TianYan/XA-202609/quarantine/tianyan176/B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823"
FROZEN_SENSING_ECONOMICS_SHA256 = "65246859977CC8542CDB84B155E8649F13B7C1D7AD54A28AA6BB76A216AD158C"
FROZEN_STATISTICAL_FIELDS = (
    "primary_adjudication",
    "endpoint_shot_floor_total",
    "expected_ratio",
    "preregistered_ratio_prediction",
    "preregistered_ratio_interval",
    "minimum_power",
    "expected_power",
    "measured_boundary_size",
    "maximum_boundary_size",
)


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    try:
        value = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def timing_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    budget = plan["measured_collection_budget"]
    correction = plan["collection_correction"]
    expected = plan["expected"]
    model = str(budget.get("timing_budget_model", ""))
    if model != "role_envelope_sum_task_runtime":
        raise ValueError(f"T176 migration preflight requires role envelope timing, got {model!r}")
    if budget.get("daily_budget_metric") != "quota_seconds":
        raise ValueError("T176 migration daily gate must use summed task-runtime quota")

    session_rows = list(budget["session_rows"])
    if len(session_rows) != int(expected["sessions"]):
        raise ValueError("timing budget session count disagrees with plan")
    quota_values = [float(row["quota_seconds"]) for row in session_rows]
    wall_values = [float(row["execution_wall_seconds"]) for row in session_rows]
    if len(set(quota_values)) != 1 or len(set(wall_values)) != 1:
        raise ValueError("registered T176 design requires one common timing envelope per session")
    quota_per_session = quota_values[0]
    wall_per_session = wall_values[0]
    declared_quota = float(correction["quota_seconds_per_session"])
    declared_wall = float(correction["execution_wall_seconds_per_session"])
    if not math.isclose(quota_per_session, declared_quota, rel_tol=0.0, abs_tol=1e-9) or not math.isclose(
        wall_per_session,
        declared_wall,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("plan timing envelope disagrees with frozen correction")
    quota_total = float(budget["quota_seconds_total"])
    wall_total = float(budget["execution_wall_seconds_total"])
    declared_quota_total = float(correction["quota_seconds_total"])
    declared_wall_total = float(correction["execution_wall_seconds_total"])
    if not all([
        math.isclose(quota_total, sum(quota_values), rel_tol=0.0, abs_tol=1e-9),
        math.isclose(wall_total, sum(wall_values), rel_tol=0.0, abs_tol=1e-9),
        math.isclose(quota_total, declared_quota_total, rel_tol=0.0, abs_tol=1e-9),
        math.isclose(wall_total, declared_wall_total, rel_tol=0.0, abs_tol=1e-9),
        math.isclose(
            float(budget["estimated_busy_seconds_total"]),
            quota_total,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    ]):
        raise ValueError("plan timing totals disagree with per-session or frozen role envelope")

    registered_pairs = int(expected["complete_cadence_pairs"])
    if registered_pairs != int(correction["registered_cycle_pairs_total"]):
        raise ValueError("plan pair count disagrees with frozen correction")
    return {
        "timing_budget_model": model,
        "daily_budget_metric": str(budget["daily_budget_metric"]),
        "daily_window_seconds": float(budget["daily_window_seconds"]),
        "daily_budget_passed": bool(budget["daily_budget_passed"]),
        "sessions": len(session_rows),
        "registered_cycle_pairs": registered_pairs,
        "quota_seconds_per_session": quota_per_session,
        "quota_seconds_total": quota_total,
        "execution_wall_seconds_per_session": wall_per_session,
        "execution_wall_seconds_total": wall_total,
        "role_task_runtime_seconds": dict(budget["role_task_runtime_seconds"]),
        "role_settings_per_job": dict(budget["role_settings_per_job"]),
        "role_jobs_per_session": dict(budget["role_jobs_per_session"]),
    }


def frozen_statistical_snapshot(correction: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in FROZEN_STATISTICAL_FIELDS if field not in correction]
    if missing:
        raise ValueError(f"collection correction is missing frozen statistical fields: {missing}")
    return {
        "source": "plan.collection_correction",
        "recomputed_by_preflight": False,
        **{field: correction[field] for field in FROZEN_STATISTICAL_FIELDS},
    }


def verify_reachability_evidence(correction: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(correction["reachability_evidence"]))
    expected_hash = str(correction["reachability_evidence_sha256"]).upper()
    raw = path.read_bytes()
    actual_hash = sha256(raw).hexdigest().upper()
    canonical_lf = raw.replace(b"\r\n", b"\n")
    canonical_lf_hash = sha256(canonical_lf).hexdigest().upper()
    bare_carriage_return_present = b"\r" in raw.replace(b"\r\n", b"")
    if actual_hash == expected_hash:
        match_mode = "byte_exact"
    elif (
        b"\r\n" in raw
        and not bare_carriage_return_present
        and canonical_lf_hash == expected_hash
    ):
        match_mode = "crlf_storage_canonical_lf_match"
    else:
        raise RuntimeError(
            "registered reachability evidence changed beyond accepted newline storage: "
            f"expected={expected_hash}, actual={actual_hash}, canonical_lf={canonical_lf_hash}"
        )
    return {
        "passed": True,
        "path": str(path.resolve()),
        "registered_sha256": expected_hash,
        "actual_storage_sha256": actual_hash,
        "canonical_lf_sha256": canonical_lf_hash,
        "match_mode": match_mode,
        "crlf_sequence_count": raw.count(b"\r\n"),
        "bare_carriage_return_present": bare_carriage_return_present,
        "statistical_content_drift_detected": False,
        "artifact_modified_by_preflight": False,
    }


def verify_migration_curated_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path.resolve()),
            "exists": False,
            "verified": False,
            "status": "pending_creation_and_freeze",
        }
    audit = runner.verify_stage1_manifest(path)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "verified": True,
        "status": "verified",
        "sha256": digest_file(path),
        "runner_audit": audit,
    }


def readiness_status(
    *,
    daily_budget_passed: bool,
    quarantine_read: bool,
    curated_manifest_verified: bool,
) -> str:
    if quarantine_read:
        return "BLOCKED_INPUT_ISOLATION_VIOLATION"
    if not daily_budget_passed:
        return "BLOCKED_DAILY_QUOTA"
    if not curated_manifest_verified:
        return "PENDING_MIGRATION_CURATED_MANIFEST"
    return "READY_FOR_REGISTERED_COLLECTION"


def recommended_hardware_command(
    *,
    config_path: Path,
    backend_config_path: Path,
    peer_config_path: Path,
    stage1_manifest_path: Path = MIGRATION_CURATED_MANIFEST,
) -> str:
    return (
        "python scripts/run_b4_cadence_pair_hardware.py "
        f"--loop-config {display_path(config_path)} "
        f"--backend-config {display_path(backend_config_path)} "
        f"--peer-config {display_path(peer_config_path)} "
        f"--stage1-manifest {display_path(stage1_manifest_path)} "
        f"--output {REGISTERED_HARDWARE_OUTPUT} "
        "--wait-for-running --confirm-hardware"
    )


def source_page_numbers(ledger: Mapping[str, Any]) -> list[int]:
    return [int(row["current"]) for row in ledger["source_pages"]]


def exact_existing_residual_audit(existing_analysis: Path) -> dict[str, Any]:
    with (existing_analysis / "tracking_residual_source_data.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return analysis.exact_slow_assignment_sensitivity(rows)


def run_offline_tests() -> dict[str, Any]:
    targets = [
        "tests/test_run_b4_cadence_pair_hardware.py",
        "tests/test_analyze_b4_t287_cadence_residual_curve.py",
        "tests/test_gate_reachability.py",
        "tests/test_build_b4_platform_time_ledger.py",
        "tests/test_preflight_b4_cadence_collection_correction.py",
    ]
    command = [sys.executable, "-m", "pytest", *targets, "-q"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"offline correction tests failed: {completed.stdout}\n{completed.stderr}")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": True,
    }


def build_report(
    *,
    config_path: Path,
    backend_config_path: Path,
    peer_config_path: Path,
    stage1_manifest_path: Path,
    existing_analysis: Path,
    cadence_ledger_path: Path,
    stage1_ledger_path: Path,
    input_audit_path: Path,
) -> dict[str, Any]:
    config = cadence.load_config(config_path)
    backend = drift_campaign_v4.load_config(backend_config_path)
    plan = runner.build_plan_payload(
        config,
        backend,
        operational_start_utc=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
        validate_registered_reachability=False,
    )

    frozen_module = ROOT / "src" / "adaptive" / "sensing_economics.py"
    frozen_hash = digest_file(frozen_module)
    if frozen_hash != FROZEN_SENSING_ECONOMICS_SHA256:
        raise RuntimeError("frozen sensing_economics.py hash changed")

    cadence_ledger_verification = time_ledger.verify_ledger_artifact(cadence_ledger_path)
    stage1_ledger_verification = time_ledger.verify_ledger_artifact(stage1_ledger_path)
    if not cadence_ledger_verification["valid"] or not stage1_ledger_verification["valid"]:
        raise RuntimeError("one or more platform time ledgers failed verification")
    cadence_ledger = cadence_ledger_verification["ledger"]
    stage1_ledger = stage1_ledger_verification["ledger"]
    input_audit = load_json(input_audit_path)
    primary = input_audit["t287_campaign"]["primary_sf_readiness"]

    existing_report = load_json(existing_analysis / "t287_cadence_residual_report.json")
    backend_id = str(backend["backend"]["backend_id"])
    correction = plan["collection_correction"]
    quarantine_read = any(
        bool(source.get("t176_quarantine_read", False))
        for source in (cadence_ledger, stage1_ledger, input_audit, existing_report)
    )
    assignment = exact_existing_residual_audit(existing_analysis)
    budget = plan["measured_collection_budget"]
    timing = timing_summary(plan)
    statistical_snapshot = frozen_statistical_snapshot(correction)
    reachability_integrity = verify_reachability_evidence(correction)
    curated_manifest = verify_migration_curated_manifest(stage1_manifest_path)
    status = readiness_status(
        daily_budget_passed=timing["daily_budget_passed"],
        quarantine_read=quarantine_read,
        curated_manifest_verified=bool(curated_manifest["verified"]),
    )
    if not all(
        session.get("operational_deadline_utc") and session.get("operational_completion_deadline_utc")
        for session in plan["sessions"]
    ):
        raise RuntimeError("corrected plan is missing hard same-session deadlines")
    offline_tests = run_offline_tests()
    return {
        "schema": "b4_cadence_collection_correction_preflight_v2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "backend_id": backend_id,
        "hardware_submission_performed": False,
        "t176_quarantine_read": quarantine_read,
        "migration_curated_manifest": {
            **curated_manifest,
            "required_before_hardware_submission": True,
            "command_role": (
                "verified runner integrity input"
                if bool(curated_manifest["verified"])
                else "pending runner integrity input"
            ),
        },
        "frozen_gate": {
            "module": "src/adaptive/sensing_economics.py",
            "sha256": frozen_hash,
            "cadence_ratio_gate_changed": False,
        },
        "implementation_hashes": {
            "corrected_config": digest_file(config_path),
            "backend_config": digest_file(backend_config_path),
            "peer_config": digest_file(peer_config_path),
            "hardware_runner": digest_file(ROOT / "scripts" / "run_b4_cadence_pair_hardware.py"),
            "cadence_analysis": digest_file(ROOT / "scripts" / "analyze_b4_t287_cadence_residual_curve.py"),
            "correction_preflight": digest_file(Path(__file__).resolve()),
            "platform_time_ledger_builder": digest_file(ROOT / "scripts" / "build_b4_platform_time_ledger.py"),
        },
        "offline_tests": offline_tests,
        "existing_collection": {
            "execution_integrity": {
                "tasks": "420/420",
                "cycles_reproduced": "21/21",
                "shield_ladder": "3/3",
                "data_corruption_found": False,
            },
            "analysis_path": str(existing_analysis.resolve()),
            "headline_verdict": existing_report["decision"]["headline_verdict"],
            "registered_endpoint_available": existing_report["registered_cadence_endpoint"]["available"],
            "raw_mirror_reachability": existing_report["mirror_endpoint_reachability"],
            "residual_assignment_sensitivity": assignment,
            "salvage_route_available": False,
            "interpretation": "clean execution, invalid endpoint identity and starved slow arm; no analysis change can confirm Tier 4 from existing data",
        },
        "corrected_collection": {
            "backend_id": backend_id,
            "config": str(config_path.resolve()),
            "config_sha256": digest_file(config_path),
            "backend_config": str(backend_config_path.resolve()),
            "backend_config_sha256": digest_file(backend_config_path),
            "peer_config": str(peer_config_path.resolve()),
            "peer_config_sha256": digest_file(peer_config_path),
            "expected": plan["expected"],
            "pairing_rule": correction["pairing_rule"],
            "incomplete_block_policy": correction["incomplete_session_pair_block_policy"],
            "same_session_deadline_policy": (
                "no new sensing-loop cycle may start after "
                f"{float(correction['operational_session_wallclock_seconds']):g} seconds; all loop, "
                "mirror QC, and shared-baseline job starts must fit inside the frozen "
                f"{float(correction['operational_session_completion_window_seconds']):g}-second "
                "completion window; an expired incomplete unit remains non-adjudicative"
            ),
            "measured_budget": budget,
            "timing_budget": timing,
            "budget_interpretation": (
                "daily and total gates use conservative summed queue-free task-runtime quota; "
                "execution wall time separately sums one parallel task envelope per job"
            ),
            "shot_budget_fraction_of_250m": int(budget["total_shots"]) / 250_000_000,
        },
        "registered_statistics": statistical_snapshot,
        "registered_statistics_integrity": reachability_integrity,
        "platform_time_replay": {
            "retrieval_method": cadence_ledger["retrieval_method"],
            "timestamp_fields": {
                "creation": cadence_ledger["creation_timestamp_field"],
                "analysis": cadence_ledger["analysis_timestamp_field"],
                "finish": "finishTime",
            },
            "cadence_collection": {
                "entries": len(cadence_ledger["entries"]),
                "target_query_ids": len(cadence_ledger["target_query_ids"]),
                "source_pages": source_page_numbers(cadence_ledger),
                "verified": cadence_ledger_verification["valid"],
            },
            "stage1_replay": {
                "entries": len(stage1_ledger["entries"]),
                "target_query_ids": len(stage1_ledger["target_query_ids"]),
                "source_pages": source_page_numbers(stage1_ledger),
                "primary_sf_tasks": int(primary["task_analysis_role_counts"]["primary_sf_only_when_non_event_and_same_regime"]),
                "missing_query_ids": input_audit["t287_campaign"]["platform_task_time_ledger"]["missing_query_ids"],
                "status": input_audit["t287_campaign"]["platform_task_time_ledger"]["status"],
                "verified": stage1_ledger_verification["valid"],
            },
            "conclusion": "the same task-list export path already replayed Stage-1 and closed the 36-primary-task timestamp gap",
        },
        "next_action": (
            "curate and freeze the migration manifest before any hardware submission"
            if status == "PENDING_MIGRATION_CURATED_MANIFEST"
            else (
                f"submit only the registered {backend_id} plan and retain every raw cycle"
                if status == "READY_FOR_REGISTERED_COLLECTION"
                else "resolve the reported preflight blocker; do not submit hardware"
            )
        ),
        "recommended_hardware_command": recommended_hardware_command(
            config_path=config_path,
            backend_config_path=backend_config_path,
            peer_config_path=peer_config_path,
            stage1_manifest_path=stage1_manifest_path,
        ),
        "recommended_hardware_command_executed": False,
    }


def review_markdown(report: Mapping[str, Any]) -> str:
    existing = report["existing_collection"]
    corrected = report["corrected_collection"]
    statistics = report["registered_statistics"]
    statistics_integrity = report["registered_statistics_integrity"]
    replay = report["platform_time_replay"]
    exact = existing["residual_assignment_sensitivity"]
    backend_id = str(report["backend_id"])
    expected = corrected["expected"]
    correction = report["corrected_collection"]
    timing = corrected["timing_budget"]
    sessions = int(expected["sessions"])
    cycles_per_cadence = int(expected["cycles_per_cadence_per_session"])
    registered_pairs = int(expected["complete_cadence_pairs"])
    return (
        "# B4 cadence 真机前 Claude 审计包\n\n"
        "## 硬停止状态\n\n"
        "- 真机提交：**未执行**。\n"
        "- poller：**未启动**。\n"
        "- 修正硬件目录：**未创建**。\n"
        f"- preflight 状态：**{report['status']}**。\n"
        f"- 迁移 curated manifest：`{report['migration_curated_manifest']['path']}`；exists={report['migration_curated_manifest']['exists']}。\n"
        f"- 离线测试：`{report['offline_tests']['stdout']}`\n\n"
        "## 旧数据裁决\n\n"
        f"- 执行完整性：420/420 tasks，21/21 cycles，shield 3/3；数据未坏。\n"
        f"- headline：**{existing['headline_verdict']}**。注册 residual² 逐 cycle 端点不可算。\n"
        f"- raw mirror ratio 替代端点：NO-GO；MDE/效应差 `{existing['raw_mirror_reachability']['mde_to_expected_effect_ratio']:.2f}×`，约需 `{existing['raw_mirror_reachability']['approximate_pairs_required']}` 对。\n"
        f"- residual 全枚举：C(21,3)={exact['assignment_count']}，one-sided p={exact['one_sided_exact_p_value']:.6f}；不含全局最大值而同等极端的组合={exact['assignments_at_least_as_extreme_without_global_maximum']}。仅 post-hoc sensitivity。\n\n"
        "## 修正采集\n\n"
        f"- backend=`{backend_id}`；{sessions} sessions；每 session fast {cycles_per_cadence} cycles + slow {cycles_per_cadence} cycles；总 {expected['cycles']} cycles、{registered_pairs} 注册对、{expected['total_tasks']} tasks。\n"
        "- 配对键：同 `session_index + cycle_index`。\n"
        f"- 冻结时限与丢弃规则：{correction['same_session_deadline_policy']}。\n"
        f"- quota={timing['quota_seconds_per_session']:.3f} s/session，total={timing['quota_seconds_total']:.3f} s；它是 {timing['daily_window_seconds']:.0f} s daily gate 的正式口径，passed={timing['daily_budget_passed']}。\n"
        f"- parallel execution wall={timing['execution_wall_seconds_per_session']:.3f} s/session，total={timing['execution_wall_seconds_total']:.3f} s；只报执行墙钟，不替代 quota gate。\n\n"
        "## 注册统计（只读）\n\n"
        f"- preflight 重算统计：`{statistics['recomputed_by_preflight']}`；来源=`{statistics['source']}`。\n"
        f"- reachability evidence integrity：`{statistics_integrity['match_mode']}`；registered SHA=`{statistics_integrity['registered_sha256']}`；storage SHA=`{statistics_integrity['actual_storage_sha256']}`；统计内容漂移=`{statistics_integrity['statistical_content_drift_detected']}`。\n"
        f"- endpoint={statistics['primary_adjudication']}；pairs={registered_pairs}；expected ratio={statistics['expected_ratio']:.9f}。\n"
        f"- expected power={statistics['expected_power']:.5f}；measured boundary size={statistics['measured_boundary_size']:.6f}。数值只转录，不改写。\n\n"
        "## 平台时间回放\n\n"
        f"- cadence：{replay['cadence_collection']['entries']}/{replay['cadence_collection']['target_query_ids']}，pages {replay['cadence_collection']['source_pages']}，已验 hash。\n"
        f"- Stage-1：{replay['stage1_replay']['entries']}/{replay['stage1_replay']['target_query_ids']}，其中 primary SF 36；missing={len(replay['stage1_replay']['missing_query_ids'])}。\n"
        "- 路径：manual browser DevTools task-list export；分析时间字段 `runStartTime`，不是 client audit timestamp。\n\n"
        "## Claude 必查\n\n"
        "1. `sensing_economics.py` hash 是否仍为 `652468...158C`，gate 零改。\n"
        "2. residual² endpoint 的冻结身份边界是否可接受；尤其 scalar normal² 与 two-field endpoint norm² 差异。\n"
        f"3. {correction['same_session_deadline_policy']}。不得看完数据再改。\n"
        f"4. {expected['total_tasks']} tasks、{corrected['measured_budget']['total_shots']:,} shots、quota {timing['quota_seconds_total']:.3f} s、wall {timing['execution_wall_seconds_total']:.3f} s 是否正确。\n"
        "5. 同 session `runStartTime` 整块规则、跨天不可续齐是否覆盖所有恢复路径。\n\n"
        "## 待审命令；curated manifest 冻结前禁止执行\n\n"
        f"```powershell\n{report['recommended_hardware_command']}\n```\n"
    )


def write_outputs(output: Path, report: Mapping[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite correction preflight: {output}")
    output.mkdir(parents=True)
    report_path = output / "correction_preflight_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_path = output / "CLAUDE_REVIEW_PACKET.md"
    review_path.write_text(review_markdown(report), encoding="utf-8")
    manifest = {
        "schema": "b4_cadence_collection_correction_preflight_manifest_v2",
        "analysis_script_sha256": digest_file(Path(__file__).resolve()),
        "files": [
            {"file": path.name, "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in (report_path, review_path)
        ],
        "hardware_submission_performed": False,
        "t176_quarantine_read": bool(report["t176_quarantine_read"]),
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend-config", type=Path, default=DEFAULT_BACKEND_CONFIG)
    parser.add_argument("--peer-config", type=Path, default=DEFAULT_PEER_CONFIG)
    parser.add_argument("--stage1-manifest", type=Path, default=MIGRATION_CURATED_MANIFEST)
    parser.add_argument("--existing-analysis", type=Path, default=DEFAULT_EXISTING_ANALYSIS)
    parser.add_argument("--cadence-ledger", type=Path, default=DEFAULT_CADENCE_LEDGER)
    parser.add_argument("--stage1-ledger", type=Path, default=DEFAULT_STAGE1_LEDGER)
    parser.add_argument("--input-audit", type=Path, default=DEFAULT_INPUT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(
        config_path=arguments.config.resolve(),
        backend_config_path=arguments.backend_config.resolve(),
        peer_config_path=arguments.peer_config.resolve(),
        stage1_manifest_path=arguments.stage1_manifest.resolve(),
        existing_analysis=arguments.existing_analysis.resolve(),
        cadence_ledger_path=arguments.cadence_ledger.resolve(),
        stage1_ledger_path=arguments.stage1_ledger.resolve(),
        input_audit_path=arguments.input_audit.resolve(),
    )
    write_outputs(arguments.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "expected_power": report["registered_statistics"]["expected_power"],
        "measured_boundary_size": report["registered_statistics"]["measured_boundary_size"],
        "cycles": report["corrected_collection"]["expected"]["cycles"],
        "registered_pairs": report["corrected_collection"]["expected"]["complete_cadence_pairs"],
        "quota_seconds_per_session": report["corrected_collection"]["timing_budget"]["quota_seconds_per_session"],
        "execution_wall_seconds_per_session": report["corrected_collection"]["timing_budget"]["execution_wall_seconds_per_session"],
        "output": str(arguments.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
