#!/usr/bin/env python3
"""Audit B-4 hardware inputs before starting B9 inference.

This is deliberately a gate, not an estimator.  It verifies immutable T287
campaign evidence, applies the frozen platform-timestamp rule, and records the
cadence supplement's actual independent-unit count.  It never reads T176
quarantine data and cannot submit hardware work.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import drift_campaign
from scripts import drift_campaign_v4
from scripts import build_b4_platform_time_ledger as platform_time_ledger


PRIMARY_SF_ROLES = {
    "primary_sf_short_lag_reference",
    "primary_sf_only_when_non_event_and_same_regime",
}


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def load_verified_journal(root: Path) -> list[dict[str, Any]]:
    """Read a campaign through its append-only SHA-256 chain verifier."""
    return drift_campaign.CampaignStore(root).records


def collected_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in records if row.get("event") == "collected"]


def audit_collected_file_hashes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        for path_field, hash_field in (
            ("raw_results_path", "raw_results_sha256"),
            ("counts_path", "counts_sha256"),
        ):
            path_text = row.get(path_field)
            expected = row.get(hash_field)
            if path_text is None or expected is None:
                mismatches.append({
                    "snapshot_id": row.get("snapshot_id"),
                    "field": path_field,
                    "reason": "missing registered path or SHA-256",
                })
                continue
            path = Path(str(path_text))
            actual = digest_file(path) if path.is_file() else None
            checked += 1
            if actual != str(expected).upper():
                mismatches.append({
                    "snapshot_id": row.get("snapshot_id"),
                    "path": str(path),
                    "expected_sha256": str(expected).upper(),
                    "actual_sha256": actual,
                })
    return {
        "collected_jobs": len(rows),
        "files_checked": checked,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def audit_platform_time_ledger(
    rows: Sequence[Mapping[str, Any]],
    ledger_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    expected_query_ids: set[str] = set()
    observations_missing_query_id = 0
    for row in rows:
        for observation in row.get("observations", []):
            if (
                str(observation.get("analysis_role", "")) in PRIMARY_SF_ROLES
                and bool(observation.get("primary_sf_eligible"))
                and not bool(observation.get("burst_flag"))
            ):
                query_id = str(observation.get("query_id", ""))
                if query_id:
                    expected_query_ids.add(query_id)
                else:
                    observations_missing_query_id += 1
    if ledger_path is None:
        return ({
            "provided": False,
            "status": "not_provided",
            "expected_query_ids": len(expected_query_ids),
            "matched_query_ids": 0,
            "missing_query_ids": sorted(expected_query_ids),
            "unexpected_query_ids": [],
            "observations_missing_query_id": observations_missing_query_id,
            "issues": [],
        }, {})
    try:
        verification = platform_time_ledger.verify_ledger_artifact(ledger_path)
        ledger = verification["ledger"]
        entries = ledger.get("entries", [])
        entry_index = {str(entry.get("query_id")): entry for entry in entries}
        ledger_query_ids = set(entry_index)
        missing = sorted(expected_query_ids - ledger_query_ids)
        unexpected = sorted(ledger_query_ids - expected_query_ids)
        complete = (
            bool(verification["valid"])
            and not missing
            and not unexpected
            and observations_missing_query_id == 0
        )
        return ({
            "provided": True,
            "status": "verified_complete" if complete else "invalid_or_incomplete",
            "expected_query_ids": len(expected_query_ids),
            "ledger_entries": len(entries),
            "matched_query_ids": len(expected_query_ids & ledger_query_ids),
            "missing_query_ids": missing,
            "unexpected_query_ids": unexpected,
            "observations_missing_query_id": observations_missing_query_id,
            "source_page_count": len(ledger.get("source_pages", [])),
            "analysis_timestamp_field": ledger.get("analysis_timestamp_field"),
            "platform_timezone": ledger.get("platform_timezone"),
            "ledger_sha256": verification["ledger_sha256"],
            "freeze_sha256": verification["freeze_sha256"],
            "issues": verification["issues"],
        }, entry_index if verification["valid"] else {})
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return ({
            "provided": True,
            "status": "invalid_or_incomplete",
            "expected_query_ids": len(expected_query_ids),
            "matched_query_ids": 0,
            "missing_query_ids": sorted(expected_query_ids),
            "unexpected_query_ids": [],
            "observations_missing_query_id": observations_missing_query_id,
            "issues": [str(error)],
        }, {})


def audit_primary_sf_observations(
    rows: Sequence[Mapping[str, Any]],
    *,
    platform_time_index: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply B4 primary-SF eligibility without a client-clock fallback."""
    platform_time_index = platform_time_index or {}
    observations: list[Mapping[str, Any]] = []
    role_counts: Counter[str] = Counter()
    for row in rows:
        for observation in row.get("observations", []):
            role = str(observation.get("analysis_role", ""))
            role_counts[role] += 1
            if role in PRIMARY_SF_ROLES and bool(observation.get("primary_sf_eligible")):
                observations.append(observation)
    non_burst = [row for row in observations if not bool(row.get("burst_flag"))]
    with_platform_time: list[Mapping[str, Any]] = []
    timestamp_sources: Counter[str] = Counter()
    for observation in non_burst:
        if observation.get("platform_timestamp_raw") not in (None, ""):
            with_platform_time.append(observation)
            timestamp_sources["campaign_journal"] += 1
            continue
        query_id = str(observation.get("query_id", ""))
        ledger_entry = platform_time_index.get(query_id)
        if ledger_entry is not None and ledger_entry.get("execution_start_time_utc") not in (None, ""):
            with_platform_time.append(observation)
            timestamp_sources["platform_task_time_ledger.runStartTime"] += 1
    missing = len(non_burst) - len(with_platform_time)
    return {
        "primary_roles": sorted(PRIMARY_SF_ROLES),
        "task_analysis_role_counts": dict(sorted(role_counts.items())),
        "eligible_non_burst_observations": len(non_burst),
        "observations_with_platform_timestamp": len(with_platform_time),
        "observations_missing_platform_timestamp": missing,
        "platform_timestamp_sources": dict(sorted(timestamp_sources.items())),
        "sidecar_ledger_observations": timestamp_sources["platform_task_time_ledger.runStartTime"],
        "client_timestamp_substitution_used": False,
        "status": (
            "ready_for_t287_sf"
            if len(non_burst) >= 3 and missing == 0
            else "blocked_missing_platform_timestamps"
        ),
        "reason": (
            None
            if len(non_burst) >= 3 and missing == 0
            else "Frozen B4 policy requires measured platform timestamps; client wall-clock cannot replace missing timestamps."
        ),
    }


def audit_raw_execution_timestamps(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check whether retained platform payloads expose collector-recognized times."""
    jobs_checked = 0
    result_records_checked = 0
    candidate_fields: Counter[str] = Counter()
    malformed_payloads: list[dict[str, Any]] = []
    for row in rows:
        path_text = row.get("raw_results_path")
        if path_text is None:
            malformed_payloads.append({"snapshot_id": row.get("snapshot_id"), "reason": "missing raw-results path"})
            continue
        path = Path(str(path_text))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            results = payload["results"]
            if not isinstance(results, list):
                raise ValueError("results is not a list")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            malformed_payloads.append({"snapshot_id": row.get("snapshot_id"), "reason": str(error)})
            continue
        jobs_checked += 1
        for result in results:
            if not isinstance(result, Mapping):
                malformed_payloads.append({"snapshot_id": row.get("snapshot_id"), "reason": "non-object result record"})
                continue
            result_records_checked += 1
            for field in drift_campaign_v4.PLATFORM_TIMESTAMP_FIELDS:
                if result.get(field) not in (None, ""):
                    candidate_fields[field] += 1
    return {
        "jobs_checked": jobs_checked,
        "result_records_checked": result_records_checked,
        "collector_recognized_platform_timestamp_fields": list(drift_campaign_v4.PLATFORM_TIMESTAMP_FIELDS),
        "nonempty_candidate_field_counts": dict(sorted(candidate_fields.items())),
        "recoverable_from_retained_raw_payload": bool(candidate_fields),
        "malformed_payloads": malformed_payloads,
    }


def audit_cadence_completion(rows: Sequence[Mapping[str, Any]], *, frozen_pair_count: int) -> dict[str, Any]:
    """Count actual controlled-injection cadence units without endpoint testing."""
    cycles = [row for row in rows if row.get("event") == "cadence_cycle_completed"]
    cycle_pair_strategies: dict[tuple[str, str], set[str]] = defaultdict(set)
    cadence_cycles: Counter[str] = Counter()
    score_rows = 0
    for cycle in cycles:
        cadence_cycles[str(cycle.get("cadence"))] += 1
        cycle_id = str(cycle.get("cycle_id"))
        for score in cycle.get("mirror_scores", []):
            score_rows += 1
            cycle_pair_strategies[(cycle_id, str(score.get("pair_id")))].add(str(score.get("strategy")))
    pair_cycles: dict[str, list[set[str]]] = defaultdict(list)
    for (_, pair_id), strategies in cycle_pair_strategies.items():
        pair_cycles[pair_id].append(strategies)
    complete_pairs = sum(
        all(strategies == {"adaptive", "fixed"} for strategies in cycle_strategies)
        for cycle_strategies in pair_cycles.values()
    )
    incomplete_pairs = sum(
        any(strategies != {"adaptive", "fixed"} for strategies in cycle_strategies)
        for cycle_strategies in pair_cycles.values()
    )
    return {
        "claim_scope": "controlled-injection cadence validation only; not natural-drift SF evidence",
        "completed_cycles": len(cycles),
        "cycles_by_cadence": dict(sorted(cadence_cycles.items())),
        "raw_mirror_score_rows": score_rows,
        "complete_registered_mirror_pairs": complete_pairs,
        "incomplete_registered_mirror_pairs": incomplete_pairs,
        "frozen_complete_pair_count": int(frozen_pair_count),
        "frozen_pair_count_matches_completed_cycles": len(cycles) == int(frozen_pair_count),
        "frozen_pair_count_matches_registered_mirror_pairs": complete_pairs == int(frozen_pair_count),
        "endpoint_status": (
            "not_tested_frozen_pair_count_mismatch"
            if len(cycles) != int(frozen_pair_count) and complete_pairs != int(frozen_pair_count)
            else "not_tested_pending_prior_t287_sf_and_map"
        ),
    }


def build_report(
    *,
    t287_records: Sequence[Mapping[str, Any]],
    cadence_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    platform_time_ledger_path: Path | None = None,
) -> dict[str, Any]:
    t287_collected = collected_rows(t287_records)
    cadence_collected = collected_rows(cadence_records)
    platform_time_audit, platform_time_index = audit_platform_time_ledger(
        t287_collected,
        platform_time_ledger_path,
    )
    sf = audit_primary_sf_observations(t287_collected, platform_time_index=platform_time_index)
    raw_timestamps = audit_raw_execution_timestamps(t287_collected)
    frozen_pair_count = int(config["stage1_hardware_freeze"]["complete_mirror_pair_count"])
    cadence = audit_cadence_completion(cadence_records, frozen_pair_count=frozen_pair_count)
    blockers: list[dict[str, str]] = []
    if sf["status"] != "ready_for_t287_sf":
        blockers.append({"step": "T287 SF", "reason": str(sf["reason"])})
    if platform_time_audit["provided"] and platform_time_audit["status"] != "verified_complete":
        blockers.append({
            "step": "platform task-time ledger",
            "reason": "Provided platform task-time ledger failed integrity or exact target-set validation.",
        })
    if (
        not cadence["frozen_pair_count_matches_completed_cycles"]
        and not cadence["frozen_pair_count_matches_registered_mirror_pairs"]
    ):
        blockers.append({
            "step": "cadence endpoint",
            "reason": "Frozen complete-pair count matches neither completed cycles nor registered raw mirror pairs; do not select a replacement post hoc.",
        })
    return {
        "schema": "b4_b9_input_audit_v2",
        "hardware_submission_performed": False,
        "t176_quarantine_read": False,
        "status": "blocked_before_t287_sf" if blockers else "ready_for_t287_sf",
        "b9_sequence": [
            "T287 SF",
            "sensing-economics map",
            "cadence residual curve",
            "Stage-2 T176 prediction",
            "T176 unseal and frozen replication",
            "report assembly",
        ],
        "t287_campaign": {
            "journal_records": len(t287_records),
            "file_hash_audit": audit_collected_file_hashes(t287_collected),
            "primary_sf_readiness": sf,
            "raw_execution_timestamp_audit": raw_timestamps,
            "platform_task_time_ledger": platform_time_audit,
        },
        "cadence_supplement": {
            "journal_records": len(cadence_records),
            "file_hash_audit": audit_collected_file_hashes(cadence_collected),
            "readiness": cadence,
        },
        "blocked_steps": blockers,
        "next_permitted_action": (
            "Obtain platform-provided execution timestamps from the platform record; do not substitute client logs or scheduled targets."
            if sf["status"] != "ready_for_t287_sf"
            else "Run the frozen T287 SF estimator."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t287-campaign-root", type=Path, required=True)
    parser.add_argument("--cadence-campaign-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--platform-time-ledger", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.out.exists():
        raise FileExistsError(f"Refusing to overwrite B9 audit output: {arguments.out}")
    report = build_report(
        t287_records=load_verified_journal(arguments.t287_campaign_root),
        cadence_records=load_verified_journal(arguments.cadence_campaign_root),
        config=json.loads(arguments.config.read_text(encoding="utf-8")),
        platform_time_ledger_path=arguments.platform_time_ledger,
    )
    arguments.out.mkdir(parents=True)
    report_path = arguments.out / "b9_input_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
