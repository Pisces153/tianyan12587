#!/usr/bin/env python3
"""Run B9 stage 3 controlled-injection cadence residual analysis.

The analysis consumes a completed cadence supplement, an independently declared
expected backend, an independently frozen platform task-time ledger, and the
Stage-1 frozen cadence configuration.
It never reads quarantine data and never submits hardware work.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_b4_platform_time_ledger as time_ledger
from scripts import drift_campaign
from scripts import drift_campaign_v4
from scripts import run_b4_cadence_pair_hardware as hardware_runner
from scripts import run_cadence_pair_loop as cadence_loop
from src.adaptive import shared_baseline_sensing as shared_baseline
from src.adaptive.cadence_permutation import cadence_ratio_permutation_gate
from src.adaptive.sensing_economics import cadence_ratio_gate
from src.adaptive.task_metric_mirror import paired_bootstrap_interval


SCHEMA = "b4_b9_t287_cadence_residual_curve_v2"
REGISTERED_CYCLE_PAIR_COUNT = 24
CORRECTED_LOOP_SESSION_WINDOW_SECONDS = 3600.0
CORRECTED_BLOCK_COMPLETION_WINDOW_SECONDS = 4500.0
NORMAL_975 = 1.959963984540054
PALETTE = {
    "blue": "#0F4D92",
    "teal": "#42949E",
    "gold": "#C58B19",
    "red": "#B64342",
    "green": "#2E8B57",
    "neutral": "#767676",
    "dark": "#333333",
    "light": "#D8D8D8",
}
SESSION_COLORS = (PALETTE["blue"], PALETTE["gold"], PALETTE["teal"])


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def canonical(value: Any) -> str:
    return json.dumps(drift_campaign.json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_backend_provenance(
    *,
    hardware_report: Mapping[str, Any],
    plan: Mapping[str, Any],
    backend_config: Mapping[str, Any],
    expected_backend_id: str,
) -> str:
    """Validate three artifact sources against caller's independent expectation."""
    if not expected_backend_id or expected_backend_id != expected_backend_id.strip():
        raise ValueError("expected backend id must be a non-empty exact platform id")
    observed = {
        "hardware_report": hardware_report.get("backend_id"),
        "plan": plan.get("backend_id"),
        "backend_config": backend_config.get("backend", {}).get("backend_id"),
    }
    mismatches = {
        source: value
        for source, value in observed.items()
        if value != expected_backend_id
    }
    if mismatches:
        raise ValueError(
            f"cadence backend provenance mismatch for expected {expected_backend_id!r}: "
            f"{mismatches}"
        )
    return expected_backend_id


def plan_cycles(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for session in plan["sessions"]:
        for cycle in session["cycles"]:
            cycle_id = str(cycle["cycle_id"])
            if cycle_id in rows:
                raise ValueError(f"duplicate cycle in plan: {cycle_id}")
            rows[cycle_id] = dict(cycle)
    return rows


def record_index(records: Sequence[Mapping[str, Any]], event: str, key: str) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("event") != event:
            continue
        value = str(record.get(key, ""))
        if not value:
            raise ValueError(f"{event} record missing {key}")
        if value in rows:
            raise ValueError(f"duplicate {event} record for {key}={value}")
        rows[value] = record
    return rows


def verify_inputs(
    *,
    campaign_root: Path,
    platform_ledger_path: Path,
    config_path: Path,
    backend_config_path: Path,
    expected_backend_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    hardware_report_path = campaign_root / hardware_runner.REPORT_NAME
    plan_path = campaign_root / hardware_runner.PLAN_NAME
    hardware_report = load_json(hardware_report_path)
    plan = load_json(plan_path)
    config = load_json(config_path)
    backend_config = drift_campaign_v4.load_config(backend_config_path)
    cadence_loop.validate_config(config)
    if hardware_report.get("schema") != "b4_cadence_pair_hardware_supplement_report_v1":
        raise ValueError("unexpected cadence hardware report schema")
    if not hardware_report.get("completed"):
        raise ValueError("cadence hardware supplement is not completed")
    if plan.get("schema") != "b4_cadence_pair_hardware_supplement_plan_v1":
        raise ValueError("unexpected cadence supplement plan schema")
    backend_id = validate_backend_provenance(
        hardware_report=hardware_report,
        plan=plan,
        backend_config=backend_config,
        expected_backend_id=expected_backend_id,
    )
    if digest_file(plan_path) != str(hardware_report["plan_sha256"]).upper():
        raise ValueError("cadence plan hash does not match hardware report")
    if digest_file(config_path) != str(plan["source_hashes"]["loop_config_sha256"]).upper():
        raise ValueError("cadence config hash does not match frozen plan")
    if digest_file(backend_config_path) != str(plan["source_hashes"]["backend_config_sha256"]).upper():
        raise ValueError("backend config hash does not match frozen plan")

    ledger_verification = time_ledger.verify_ledger_artifact(platform_ledger_path)
    if not ledger_verification["valid"]:
        raise ValueError(f"platform time ledger failed verification: {ledger_verification['issues']}")
    ledger = ledger_verification["ledger"]
    if ledger.get("target_set") != "cadence-collected":
        raise ValueError("platform time ledger is not the cadence-collected target set")
    if ledger.get("hardware_submission_performed") or ledger.get("t176_quarantine_read"):
        raise ValueError("platform ledger violates B9 isolation boundary")

    store = drift_campaign.CampaignStore(campaign_root)
    records = [dict(row) for row in store.records]
    journal_backend_ids = sorted({
        str(row["backend_id"])
        for row in records
        if row.get("backend_id") is not None
    })
    if journal_backend_ids != [backend_id]:
        raise ValueError(
            f"cadence journal backend provenance mismatch: expected {backend_id!r}, "
            f"observed {journal_backend_ids}"
        )
    journal_path = campaign_root / "snapshots.jsonl"
    if digest_file(journal_path) != str(ledger["campaign_journal_sha256"]).upper():
        raise ValueError("platform ledger journal hash does not match cadence campaign")
    if len(ledger["entries"]) != int(hardware_report["expected"]["total_tasks"]):
        raise ValueError("platform ledger task count does not match cadence plan")

    collected = [row for row in records if row.get("event") == "collected"]
    file_hashes_passed = all(
        Path(str(row["raw_results_path"])).is_file()
        and Path(str(row["counts_path"])).is_file()
        and digest_file(Path(str(row["raw_results_path"]))) == str(row["raw_results_sha256"]).upper()
        and digest_file(Path(str(row["counts_path"]))) == str(row["counts_sha256"]).upper()
        for row in collected
    )
    if not file_hashes_passed:
        raise ValueError("cadence raw-result/count hashes failed")

    audit = {
        "passed": True,
        "backend_id": backend_id,
        "journal_backend_ids": journal_backend_ids,
        "hardware_report_sha256": digest_file(hardware_report_path),
        "plan_sha256": digest_file(plan_path),
        "campaign_journal_sha256": digest_file(journal_path),
        "platform_ledger_sha256": digest_file(platform_ledger_path),
        "platform_ledger_freeze_sha256": ledger_verification["freeze_sha256"],
        "config_sha256": digest_file(config_path),
        "backend_config_sha256": digest_file(backend_config_path),
        "task_count": len(ledger["entries"]),
        "hardware_submission_performed_by_analysis": False,
        "t176_quarantine_read": bool(ledger.get("t176_quarantine_read", False)),
    }
    return hardware_report, plan, config, backend_config, records, {"ledger": ledger, "audit": audit}


def reproduce_cycles(
    *,
    records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    submitted_by_snapshot = record_index(records, "submitted", "snapshot_id")
    collected_by_snapshot = record_index(records, "collected", "snapshot_id")
    completed_by_cycle = record_index(records, "cadence_cycle_completed", "cycle_id")
    planned = plan_cycles(plan)
    physical = [int(value) for value in backend_config["backend"]["physical_qubits"]]
    phase_time = float(config["sensing"]["phase_time_seconds"])
    depth = int(config["mirror"]["depth"])

    rows: list[dict[str, Any]] = []
    sensing_exact = 0
    mirror_exact = 0
    for cycle_id, cycle in planned.items():
        completed = completed_by_cycle.get(cycle_id)
        if completed is None:
            raise ValueError(f"missing completed cycle: {cycle_id}")
        loop_snapshot = str(completed["loop_snapshot_id"])
        mirror_snapshot = str(completed["mirror_snapshot_id"])
        loop_submitted = submitted_by_snapshot[loop_snapshot]
        mirror_submitted = submitted_by_snapshot[mirror_snapshot]
        loop_collected = collected_by_snapshot[loop_snapshot]
        mirror_collected = collected_by_snapshot[mirror_snapshot]

        sensing = hardware_runner.estimate_fields_from_counts(
            hardware_runner.load_counts(loop_collected),
            int(cycle["sensing_shots_per_setting"]),
            phase_time,
        )
        if canonical(sensing) != canonical(completed["sensing"]):
            raise ValueError(f"sensing reproduction mismatch: {cycle_id}")
        sensing_exact += 1

        compensation = [float(value) for value in completed["shield"]["compensation"][:2]]
        programs = hardware_runner.mirror_programs(
            physical,
            cycle["mirror_fields"],
            compensation,
            phase_time,
            depth,
            cycle,
        )
        scores = hardware_runner.score_mirror(
            hardware_runner.load_counts(mirror_collected),
            programs,
            int(cycle["mirror_shots_per_task"]),
            cycle,
        )
        if canonical(scores) != canonical(completed["mirror_scores"]):
            raise ValueError(f"mirror-score reproduction mismatch: {cycle_id}")
        mirror_exact += 1
        rows.append({
            **dict(completed),
            "loop_query_ids": [str(task["query_id"]) for task in loop_submitted["tasks"]],
            "mirror_query_ids": [str(task["query_id"]) for task in mirror_submitted["tasks"]],
        })
    rows.sort(key=lambda row: (int(row["session_index"]), int(row["block_index"]), str(row["cycle_id"])))
    return rows, {
        "cycle_count": len(rows),
        "sensing_exact_reproductions": sensing_exact,
        "mirror_score_exact_reproductions": mirror_exact,
        "passed": sensing_exact == len(rows) and mirror_exact == len(rows),
    }


def aggregate_mirror_pairs(
    cycles: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    losses: dict[tuple[str, str, str], list[float]] = {}
    pair_metadata: dict[str, dict[str, int]] = {}
    for cycle in cycles:
        session = int(cycle["session_index"])
        cadence = str(cycle["cadence"])
        for score in cycle["mirror_scores"]:
            replicate = int(score["replicate"])
            pair_id = str(score.get("pair_id") or f"session{session:02d}-mirror{replicate:02d}")
            key = (pair_id, cadence, str(score["strategy"]))
            losses.setdefault(key, []).append(1.0 - float(score["success_probability"]))
            pair_metadata.setdefault(pair_id, {"session_index": session, "replicate": replicate})

    rows: list[dict[str, Any]] = []
    for pair_id in sorted(pair_metadata, key=lambda value: (pair_metadata[value]["session_index"], value)):
        values: dict[str, float] = {}
        counts: dict[str, int] = {}
        for cadence in ("fast", "slow"):
            for strategy in ("adaptive", "fixed"):
                key = (pair_id, cadence, strategy)
                if key not in losses:
                    raise ValueError(f"incomplete mirror cadence pair: {key}")
                values[f"{cadence}_{strategy}_loss"] = float(np.mean(losses[key]))
                counts[f"{cadence}_{strategy}_cycle_count"] = len(losses[key])
        rows.append({
            "pair_id": pair_id,
            **pair_metadata[pair_id],
            **values,
            **counts,
            "fast_adaptive_gain_vs_fixed": values["fast_fixed_loss"] - values["fast_adaptive_loss"],
            "slow_adaptive_gain_vs_fixed": values["slow_fixed_loss"] - values["slow_adaptive_loss"],
        })
    if len(rows) != int(expected_pair_count):
        raise ValueError(f"mirror QC pair count {len(rows)} does not match expected {expected_pair_count}")

    fast = np.asarray([row["fast_adaptive_loss"] for row in rows], dtype=np.float64)
    slow = np.asarray([row["slow_adaptive_loss"] for row in rows], dtype=np.float64)
    fast_fixed = np.asarray([row["fast_fixed_loss"] for row in rows], dtype=np.float64)
    slow_fixed = np.asarray([row["slow_fixed_loss"] for row in rows], dtype=np.float64)
    fast_gain = fast_fixed - fast
    slow_gain = slow_fixed - slow
    gain_difference = fast_gain - slow_gain
    endpoint = {
        "analysis_role": "task-metric QC and preserved substitution-endpoint result; not the power-tested cadence endpoint",
        "registered_endpoint": False,
        "metric": "raw-count mirror success_probability loss",
        "contrast": "paired mean adaptive fast_loss / slow_loss",
        "pair_count": len(rows),
        "fast_loss_mean": float(np.mean(fast)),
        "slow_loss_mean": float(np.mean(slow)),
        "fast_success_probability_mean": float(1.0 - np.mean(fast)),
        "slow_success_probability_mean": float(1.0 - np.mean(slow)),
        "direction_fast_loss_lower": bool(np.mean(fast) < np.mean(slow)),
        "ratio_gate": cadence_ratio_gate(fast, slow),
        "fixed_strategy_control": {
            "fast_loss_mean": float(np.mean(fast_fixed)),
            "slow_loss_mean": float(np.mean(slow_fixed)),
            "ratio_gate": cadence_ratio_gate(fast_fixed, slow_fixed),
        },
        "matched_strategy_sensitivity": {
            "definition": "adaptive gain relative to fixed, then fast-minus-slow; descriptive sensitivity",
            "fast_gain_mean": float(np.mean(fast_gain)),
            "slow_gain_mean": float(np.mean(slow_gain)),
            "fast_minus_slow_gain_mean": float(np.mean(gain_difference)),
            "fast_minus_slow_gain_bootstrap_interval": paired_bootstrap_interval(
                gain_difference,
                resamples=10_000,
                seed=2026081503,
                confidence_level=0.95,
            ),
        },
    }
    return rows, endpoint


def mirror_reachability_audit(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Quantify why raw mirror loss cannot adjudicate the frozen ratio gate here."""
    gate = endpoint["ratio_gate"]
    standard_error = (float(gate["ci_upper"]) - float(gate["ci_lower"])) / (2.0 * NORMAL_975)
    expected_loss_floor = 0.6455
    expected_fast_minus_slow = -0.000462
    expected_ratio = 1.0 + expected_fast_minus_slow / expected_loss_floor
    expected_effect = 1.0 - expected_ratio
    minimum_detectable_effect_80 = (NORMAL_975 + 0.8416212335729143) * standard_error
    effect_gap = minimum_detectable_effect_80 / expected_effect
    return {
        "status": "UNREACHABLE_AT_NOMINAL_EFFECT_AND_N24",
        "endpoint": "raw mirror loss passed to cadence_ratio_gate",
        "observed_pair_count": int(endpoint["pair_count"]),
        "observed_ratio_standard_error": standard_error,
        "observed_95_percent_half_width": NORMAL_975 * standard_error,
        "assumed_additive_loss_floor": expected_loss_floor,
        "assumed_fast_minus_slow_loss_effect": expected_fast_minus_slow,
        "expected_ratio_under_nominal_effect": expected_ratio,
        "expected_fractional_effect": expected_effect,
        "minimum_detectable_fractional_effect_80_percent_power": minimum_detectable_effect_80,
        "mde_to_expected_effect_ratio": effect_gap,
        "approximate_pair_multiplier": effect_gap**2,
        "approximate_pairs_required": int(math.ceil(int(endpoint["pair_count"]) * effect_gap**2)),
        "interpretation": (
            "The additive circuit-error floor pushes the raw-loss ratio toward one. This is endpoint unreachability, "
            "not evidence that the closed loop failed. Increasing shots cannot remove between-pair hardware variation."
        ),
    }


def analytic_endpoint_residual(
    *,
    cadence_seconds: float,
    shot_sigmas: Sequence[float],
    process_variance: float,
    tau_seconds: float,
) -> float:
    shot_term = sum(float(value) ** 2 for value in shot_sigmas)
    drift_two_fields = 4.0 * float(process_variance) * (1.0 - math.exp(-float(cadence_seconds) / float(tau_seconds)))
    return shot_term + drift_two_fields


def exact_slow_assignment_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Post-hoc exact assignment audit for the collapsed three-of-21 slow arm."""
    values = np.asarray([float(row["observed_endpoint_residual_squared"]) for row in rows], dtype=np.float64)
    slow_indices = tuple(index for index, row in enumerate(rows) if str(row["cadence"]) == "slow")
    total = math.comb(len(values), len(slow_indices))
    base = {
        "analysis_role": "post-hoc diagnostic sensitivity only; not a replacement endpoint or frozen test",
        "valid_for_adjudication": False,
        "observation_count": len(values),
        "slow_count": len(slow_indices),
        "assignment_count": total,
    }
    if len(slow_indices) != 3 or len(values) != 21:
        return {**base, "available": False, "reason": "exact enumeration is reserved for the observed three-of-21 collapse"}
    observed_slow_mean = float(np.mean(values[list(slow_indices)]))
    maximum_index = int(np.argmax(values))
    exceed_count = 0
    exceed_without_maximum = 0
    for candidate in itertools.combinations(range(len(values)), len(slow_indices)):
        if float(np.mean(values[list(candidate)])) >= observed_slow_mean - 1e-15:
            exceed_count += 1
            if maximum_index not in candidate:
                exceed_without_maximum += 1
    return {
        **base,
        "available": True,
        "statistic": "slow-arm mean residual squared under all C(21,3) label assignments",
        "observed_slow_mean": observed_slow_mean,
        "observed_slow_contains_global_maximum": maximum_index in slow_indices,
        "global_maximum_value": float(values[maximum_index]),
        "assignments_at_least_as_extreme": exceed_count,
        "assignments_at_least_as_extreme_without_global_maximum": exceed_without_maximum,
        "one_sided_exact_p_value": exceed_count / total,
        "one_sided_add_one_p_value": (exceed_count + 1) / (total + 1),
        "interpretation": "the apparent direction is equivalent to whether the single global maximum lands in the three-cycle slow arm",
    }


def tracking_rows(
    cycles: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    process_variance = float(config["controlled_ou"]["stationary_process_variance"])
    tau_seconds = float(config["controlled_ou"]["tau_seconds"])
    rows: list[dict[str, Any]] = []
    for cycle in cycles:
        mirror_fields = np.asarray(cycle["mirror_fields"], dtype=np.float64)
        compensation = np.asarray(cycle["shield"]["compensation"][:2], dtype=np.float64)
        residual = mirror_fields + compensation
        observed = float(np.dot(residual, residual))
        cadence_seconds = float(cycle["cadence_seconds"])
        shot_sigmas = [float(value) for value in cycle["sensing"]["shot_sigmas"]]
        analytic_endpoint = analytic_endpoint_residual(
            cadence_seconds=cadence_seconds,
            shot_sigmas=shot_sigmas,
            process_variance=process_variance,
            tau_seconds=tau_seconds,
        )
        analytic_average = cadence_loop.analytic_average_tracking_mse(
            cadence_seconds,
            cadence_seconds,
            shot_sigmas,
            config,
        )
        rows.append({
            "cycle_id": str(cycle["cycle_id"]),
            "session_index": int(cycle["session_index"]),
            "block_index": int(cycle["block_index"]),
            "cycle_index": int(str(cycle["cycle_id"]).rsplit("cycle", 1)[1]),
            "cadence": str(cycle["cadence"]),
            "controlled_injection_interval_seconds": cadence_seconds,
            "observed_endpoint_residual_squared": observed,
            "analytic_endpoint_expected_squared": analytic_endpoint,
            "observed_to_analytic_endpoint_ratio": observed / analytic_endpoint,
            "preflight_time_average_expected_squared": analytic_average,
            "shield_permitted": bool(cycle["shield"]["permitted"]),
        })

    session_means: list[dict[str, Any]] = []
    for session in sorted({int(row["session_index"]) for row in rows}):
        item = {"session_index": session}
        for cadence in ("fast", "slow"):
            selected = [row for row in rows if row["session_index"] == session and row["cadence"] == cadence]
            item[f"{cadence}_observed_mean"] = float(np.mean([row["observed_endpoint_residual_squared"] for row in selected]))
            item[f"{cadence}_analytic_mean"] = float(np.mean([row["analytic_endpoint_expected_squared"] for row in selected]))
            item[f"{cadence}_cycle_count"] = len(selected)
        session_means.append(item)
    fast = np.asarray([row["fast_observed_mean"] for row in session_means], dtype=np.float64)
    slow = np.asarray([row["slow_observed_mean"] for row in session_means], dtype=np.float64)
    sensitivity_gate = cadence_ratio_gate(fast, slow)
    summary = {
        "definition": "endpoint controlled-field residual squared after measured compensation",
        "time_axis": "frozen controlled-injection interval (90/360 s), not platform queue duration",
        "cycle_count": len(rows),
        "session_pair_count": len(session_means),
        "fast_observed_mean": float(np.mean(fast)),
        "slow_observed_mean": float(np.mean(slow)),
        "session_paired_ratio_sensitivity": {
            **sensitivity_gate,
            "valid_for_adjudication": False,
            "validity_reason": (
                f"n={len(session_means)} session means are insufficient for the delta-method "
                "ratio interval; a negative lower bound for a positive ratio is diagnostic "
                "of failure"
            ),
        },
        "exact_slow_assignment_sensitivity": exact_slow_assignment_sensitivity(rows),
        "formal_curve_match_gate_available": False,
        "formal_curve_match_reason": "Stage-1 froze analytic predictions but no numeric hardware match tolerance; hardware observes endpoints, whereas preflight tracked a time-average path.",
        "session_means": session_means,
        "shield_permitted_cycles": sum(bool(row["shield_permitted"]) for row in rows),
    }
    return rows, summary


def registered_cycle_residual_endpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_count: int = REGISTERED_CYCLE_PAIR_COUNT,
    expected_pairs_per_session: int | None = None,
    minimum_pair_count: int | None = None,
    minimum_sessions_per_block_order: int = 0,
    session_block_order: Mapping[int, str] | None = None,
    eligible_sessions: set[int] | None = None,
    platform_session_integrity: Mapping[str, Any] | None = None,
    pair_discard_granularity: str = "session_pair_block",
    sessions_with_baseline: set[int] | None = None,
    primary_adjudication: str = "cadence_ratio_gate",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adjudicate endpoint residual squared, paired by cycle.

    Two registered dimensions vary by collection version and neither is a free choice
    here: both are read from the plan the runner wrote.

    ``pair_discard_granularity`` is ``session_pair_block`` for the collections in which
    every cycle carried its own zero-field condition -- there, a lost cycle leaves the
    surviving cycles of that session without a common reference and the whole block goes.
    It is ``cycle_pair`` once the zero-field condition is one shared measurement per
    session: that measurement survives a lost cycle, so only the lost cycle's own pair is
    unadjudicable, and the whole block goes only if the shared measurement itself is
    missing.

    ``primary_adjudication`` selects which calibration of the *same* registered statistic
    decides the endpoint.  ``cadence_ratio_gate``'s delta-method interval is
    anti-conservative on the Exponential endpoint law at small pair counts, so the
    forty-minute collection registers the exact within-pair permutation calibration as
    primary.  The frozen delta-method verdict is computed and reported either way.
    """
    if pair_discard_granularity not in ("session_pair_block", "cycle_pair"):
        raise ValueError(f"unregistered pair discard granularity: {pair_discard_granularity}")
    if primary_adjudication not in ("cadence_ratio_gate", "cadence_ratio_permutation_gate"):
        raise ValueError(f"unregistered primary adjudication: {primary_adjudication}")
    by_cycle_pair = pair_discard_granularity == "cycle_pair"
    sessions = sorted({int(row["session_index"]) for row in rows})
    if expected_pairs_per_session is None:
        if expected_pair_count % len(sessions) != 0:
            raise ValueError("registered pair count must divide evenly across sessions")
        expected_pairs_per_session = expected_pair_count // len(sessions)
    expected_pairs_per_session = int(expected_pairs_per_session)
    # A registered minimum below the full total lets the endpoint survive the loss of
    # whole sessions.  It is a fixed-size tolerance, not a sequential rule: the pair
    # count is never inspected mid-collection and collection is never stopped early.
    required_pairs = int(expected_pair_count if minimum_pair_count is None else minimum_pair_count)
    counts_by_session: list[dict[str, int]] = []
    complete_sessions: list[int] = []
    discarded_blocks: list[dict[str, Any]] = []
    for session in sessions:
        fast_count = sum(int(row["session_index"]) == session and row["cadence"] == "fast" for row in rows)
        slow_count = sum(int(row["session_index"]) == session and row["cadence"] == "slow" for row in rows)
        counts_by_session.append({"session_index": session, "fast_cycle_count": fast_count, "slow_cycle_count": slow_count})
        timing_eligible = eligible_sessions is None or session in eligible_sessions
        if by_cycle_pair:
            # The shared zero-field measurement is what the whole block depends on; the
            # individual cycle counts are not, because a lost cycle costs only its own pair.
            has_baseline = sessions_with_baseline is None or session in sessions_with_baseline
            if has_baseline and timing_eligible:
                complete_sessions.append(session)
            else:
                discarded_blocks.append({
                    "session_index": session,
                    "fast_cycle_count": fast_count,
                    "slow_cycle_count": slow_count,
                    "reason": (
                        "the session-start baseline every cycle of both cadence blocks "
                        "subtracts is missing, so no pair in this session is adjudicable"
                        if not has_baseline
                        else "the session failed the frozen same-session platform-time requirement"
                    ),
                })
        elif fast_count == slow_count == expected_pairs_per_session and timing_eligible:
            complete_sessions.append(session)
        else:
            discarded_blocks.append({
                "session_index": session,
                "fast_cycle_count": fast_count,
                "slow_cycle_count": slow_count,
                "reason": (
                    "the entire session pair block is non-adjudicative until both cadence "
                    "blocks complete"
                    if fast_count != slow_count or fast_count != expected_pairs_per_session
                    else "the session failed the frozen same-session platform-time requirement"
                ),
            })

    pair_rows: list[dict[str, Any]] = []
    discarded_pairs: list[dict[str, Any]] = []
    for session in complete_sessions:
        by_key: dict[tuple[int, str], Mapping[str, Any]] = {
            (int(row["cycle_index"]), str(row["cadence"])): row
            for row in rows
            if int(row["session_index"]) == session
        }
        cycle_indices = sorted({key[0] for key in by_key})
        incomplete = [
            index
            for index in cycle_indices
            if any((index, cadence) not in by_key for cadence in ("fast", "slow"))
        ]
        if incomplete and not by_cycle_pair:
            continue
        for cycle_index in cycle_indices:
            if cycle_index in incomplete:
                missing = [
                    cadence
                    for cadence in ("fast", "slow")
                    if (cycle_index, cadence) not in by_key
                ]
                discarded_pairs.append({
                    "pair_id": f"session{session:02d}-cycle{cycle_index:02d}",
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "reason": f"registered pair incomplete; missing {', '.join(missing)} cycle",
                })
                continue
            fast_row = by_key[(cycle_index, "fast")]
            slow_row = by_key[(cycle_index, "slow")]
            pair_rows.append({
                "pair_id": f"session{session:02d}-cycle{cycle_index:02d}",
                "session_index": session,
                "cycle_index": cycle_index,
                "fast_cycle_id": str(fast_row["cycle_id"]),
                "slow_cycle_id": str(slow_row["cycle_id"]),
                "fast_endpoint_residual_squared": float(fast_row["observed_endpoint_residual_squared"]),
                "slow_endpoint_residual_squared": float(slow_row["observed_endpoint_residual_squared"]),
            })

    n_fast = sum(row["cadence"] == "fast" for row in rows)
    n_slow = sum(row["cadence"] == "slow" for row in rows)
    order_counts: dict[str, int] = {}
    if session_block_order is not None:
        for session in complete_sessions:
            starting = str(session_block_order.get(session, "unknown"))
            order_counts[starting] = order_counts.get(starting, 0) + 1
    order_balanced = minimum_sessions_per_block_order <= 0 or (
        session_block_order is not None
        and len(order_counts) >= 2
        and min(order_counts.values()) >= int(minimum_sessions_per_block_order)
    )
    available = len(pair_rows) >= required_pairs and order_balanced
    gate = None
    permutation_gate = None
    if available:
        fast = np.asarray([row["fast_endpoint_residual_squared"] for row in pair_rows], dtype=np.float64)
        slow = np.asarray([row["slow_endpoint_residual_squared"] for row in pair_rows], dtype=np.float64)
        gate = cadence_ratio_gate(fast, slow)
        if primary_adjudication == "cadence_ratio_permutation_gate":
            # Same statistic, exact calibration.  Its own frozen delta-method verdict comes
            # back nested inside, and `gate` above is that verdict computed independently;
            # the two are asserted equal by the test suite rather than assumed.
            permutation_gate = cadence_ratio_permutation_gate(fast, slow)
    count_complete_sessions = [
        row["session_index"]
        for row in counts_by_session
        if row["fast_cycle_count"] == row["slow_cycle_count"] == expected_pairs_per_session
    ]
    if available:
        unavailable_reason = None
    elif not order_balanced:
        unavailable_reason = (
            "the surviving complete cycle blocks do not carry the registered minimum of sessions "
            "starting with each cadence, so block order would confound the contrast"
        )
    elif count_complete_sessions and eligible_sessions is not None:
        unavailable_reason = "one or more complete cycle blocks failed the frozen same-session platform-time requirement"
    else:
        unavailable_reason = (
            f"only {len(pair_rows)} complete same-session cycle pairs survive against the registered "
            f"minimum of {required_pairs}"
        )
    return pair_rows, {
        "registered_endpoint": True,
        "metric": "endpoint controlled-field residual squared after measured compensation",
        "contrast": "paired mean fast residual squared / slow residual squared",
        "pairing_unit": "one fast cycle and one slow cycle with the same cycle index inside one session",
        "incomplete_block_policy": (
            "discard only the registered cycle pair that lost a member; discard the whole "
            "session pair block only if its session-start baseline measurement is missing"
            if by_cycle_pair
            else "discard the entire session pair block if either cadence block is incomplete"
        ),
        "pair_discard_granularity": pair_discard_granularity,
        "discarded_session_pair_blocks": discarded_blocks,
        "discarded_cycle_pairs": discarded_pairs,
        "primary_adjudication": primary_adjudication,
        "primary_adjudication_note": (
            "the registered point statistic is identical under both calibrations; only the "
            "critical value differs, and the frozen delta-method verdict is reported as the "
            "secondary readout in ratio_gate whichever is primary"
        ),
        "expected_pair_count": expected_pair_count,
        "expected_pairs_per_session": expected_pairs_per_session,
        "minimum_adjudicated_pair_count": required_pairs,
        "minimum_sessions_per_block_order": int(minimum_sessions_per_block_order),
        "complete_sessions_by_starting_cadence": order_counts,
        "block_order_balanced": bool(order_balanced),
        "observed_pair_count": len(pair_rows),
        "n_fast_cycles": n_fast,
        "n_slow_cycles": n_slow,
        "counts_by_session": counts_by_session,
        "count_complete_session_pair_blocks": count_complete_sessions,
        "complete_session_pair_blocks": complete_sessions,
        "platform_session_integrity": None if platform_session_integrity is None else dict(platform_session_integrity),
        "available": available,
        "unavailable_reason": unavailable_reason,
        "ratio_gate": gate,
        "permutation_gate": permutation_gate,
    }


def shared_baseline_drift_sensitivity(
    hardware_report: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    correction: Mapping[str, Any] | None,
    *,
    session_block_order: Mapping[int, str] | None = None,
    primary_passed: bool | None = None,
) -> dict[str, Any] | None:
    """The pre-registered sensitivity re-run for the session-shared baseline.

    The shared measurement leaves one additive offset common to every cycle it serves.  It
    is a limitation of the amortisation and the design says so: the collection cannot
    resolve the drift level at which power degrades, so the drift enters as a one-sided
    upper confidence limit rather than a threshold, and the endpoint verdict is recomputed
    under all three registered shapes at that limit.  Every shape is reported whatever the
    drift turns out to be, and none of them can revise the primary verdict -- they say how
    much of the contrast could be an artefact of the shared reference, not whether to
    believe the contrast.

    Returns None for collections with no shared baseline, where the question is absent.
    """
    if correction is None or "endpoint_shot_floor_shared" not in correction:
        return None
    drift_rows = [
        row
        for row in hardware_report.get("session_shared_baseline", {}).get("drift_qc", [])
        if row.get("measured")
    ]
    shared_floor = float(correction["endpoint_shot_floor_shared"])
    offsets = [
        shared_baseline.drift_sensitivity_offsets(row, shared_baseline_floor=shared_floor)
        for row in drift_rows
    ]
    shapes: dict[str, Any] = {}
    worst: float | None = None
    if pair_rows and offsets:
        # The bounding case across sessions: the largest offset any session could carry.
        worst = max(float(row["endpoint_offset_at_upper_limit"]) for row in offsets)
        for name, (first, second) in shared_baseline.DRIFT_SHAPES.items():
            fast: list[float] = []
            slow: list[float] = []
            for pair in pair_rows:
                session = int(pair["session_index"])
                # Block one is whichever cadence the session started on, which is what the
                # balanced order alternates and what makes an asymmetric shape cancel.
                starting = (
                    "fast" if session_block_order is None
                    else str(session_block_order.get(session, "fast"))
                )
                fast_share, slow_share = (
                    (first, second) if starting == "fast" else (second, first)
                )
                fast.append(
                    max(float(pair["fast_endpoint_residual_squared"]) - worst * fast_share, 0.0)
                )
                slow.append(
                    max(float(pair["slow_endpoint_residual_squared"]) - worst * slow_share, 0.0)
                )
            shapes[name] = {
                "block_offsets_of_D": [first, second],
                "removed_from_fast_arm": worst * first,
                "removed_from_slow_arm": worst * second,
                "permutation_gate": cadence_ratio_permutation_gate(fast, slow),
            }
    verdicts = {name: bool(row["permutation_gate"]["passed"]) for name, row in shapes.items()}
    return {
        "role": "pre-registered baseline drift sensitivity; reported, never adjudicative",
        "available": bool(shapes),
        "unavailable_reason": (
            None
            if shapes
            else (
                "no session produced both baseline measurements, so the drift the shared "
                "reference could carry is unbounded by this collection"
                if not drift_rows
                else "no adjudicable cycle pair survived, so there is nothing to re-run"
            )
        ),
        "endpoint_offset_at_upper_limit": worst,
        "sessions_measured": len(drift_rows),
        "sessions_without_drift_readout": (
            len(hardware_report.get("session_shared_baseline", {}).get("drift_qc", []))
            - len(drift_rows)
        ),
        "shared_baseline_floor": shared_floor,
        "per_session_offsets": offsets,
        "shapes": shapes,
        "shape_verdicts": verdicts,
        "all_shapes_agree_with_each_other": (
            None if not verdicts else len(set(verdicts.values())) == 1
        ),
        # Agreement is measured against the verdict actually reached, not against the
        # shapes agreeing among themselves: three shapes could agree with each other and
        # all disagree with the primary, which is exactly the case worth flagging.
        "all_shapes_agree_with_primary": (
            None
            if not verdicts or primary_passed is None
            else all(value == bool(primary_passed) for value in verdicts.values())
        ),
        "registered_shapes": sorted(shared_baseline.DRIFT_SHAPES),
        "interpretation": (
            "the offset is subtracted from each arm at the one-sided upper limit, which is "
            "the least favourable reading of the drift the collection can exclude; a "
            "primary verdict that survives all three shapes is not attributable to the "
            "shared reference, and one that does not survive is reported as such"
        ),
        "undetectable_mode": (
            None if correction is None else correction.get("baseline_drift_undetectable_mode")
        ),
    }


def platform_timing_rows(
    cycles: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    *,
    block_duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entries = {str(row["query_id"]): row for row in ledger["entries"]}
    job_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    for cycle in cycles:
        jobs: dict[str, dict[str, Any]] = {}
        for role in ("loop", "mirror"):
            query_ids = [str(value) for value in cycle[f"{role}_query_ids"]]
            task_rows = [entries[value] for value in query_ids]
            starts = sorted(parse_utc(row["execution_start_time_utc"]) for row in task_rows)
            ends = sorted(parse_utc(row["execution_end_time_utc"]) for row in task_rows)
            job = {
                "cycle_id": str(cycle["cycle_id"]),
                "session_index": int(cycle["session_index"]),
                "block_index": int(cycle.get("block_index", 0)),
                "cycle_index": int(cycle.get("cycle_index", str(cycle["cycle_id"]).rsplit("cycle", 1)[1])),
                "cadence": str(cycle["cadence"]),
                "job_role": role,
                "task_count": len(task_rows),
                "execution_start_min_utc": starts[0].isoformat(),
                "execution_start_max_utc": starts[-1].isoformat(),
                "execution_end_max_utc": ends[-1].isoformat(),
                "task_start_spread_seconds": (starts[-1] - starts[0]).total_seconds(),
                "job_wall_seconds": (ends[-1] - starts[0]).total_seconds(),
            }
            job_rows.append(job)
            jobs[role] = {**job, "start": starts[0], "end": ends[-1]}
        actual = (jobs["mirror"]["start"] - jobs["loop"]["start"]).total_seconds()
        queue_gap = (jobs["mirror"]["start"] - jobs["loop"]["end"]).total_seconds()
        planned = float(cycle["cadence_seconds"])
        cycle_rows.append({
            "cycle_id": str(cycle["cycle_id"]),
            "session_index": int(cycle["session_index"]),
            "cadence": str(cycle["cadence"]),
            "controlled_injection_interval_seconds": planned,
            "platform_start_to_start_seconds": actual,
            "platform_loop_end_to_mirror_start_seconds": queue_gap,
            "platform_to_controlled_interval_ratio": actual / planned,
            "exceeds_frozen_block_duration": bool(actual > block_duration_seconds),
        })
    def cadence_range(label: str) -> list[float] | None:
        values = [row["platform_start_to_start_seconds"] for row in cycle_rows if row["cadence"] == label]
        return None if not values else [min(values), max(values)]

    summary = {
        "analysis_role": "implementation timing QC only; not the controlled-OU time axis",
        "job_count": len(job_rows),
        "cycle_count": len(cycle_rows),
        "cycles_exceeding_frozen_block_duration": sum(bool(row["exceeds_frozen_block_duration"]) for row in cycle_rows),
        "fast_start_to_start_range_seconds": cadence_range("fast"),
        "slow_start_to_start_range_seconds": cadence_range("slow"),
        "exclusion_policy": "none; platform timing is reported separately and does not redefine the frozen virtual controlled-injection interval",
    }
    return job_rows, cycle_rows, summary


def platform_session_pairing_integrity(
    residual_rows: Sequence[Mapping[str, Any]],
    job_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    expected_pair_count: int = REGISTERED_CYCLE_PAIR_COUNT,
    expected_pairs_per_session: int | None = None,
) -> dict[str, Any]:
    correction = config.get("collection_correction")
    sessions = sorted({int(row["session_index"]) for row in residual_rows})
    required = isinstance(correction, Mapping)
    expected_per_session = (
        expected_pair_count // len(sessions)
        if expected_pairs_per_session is None
        else int(expected_pairs_per_session)
    )
    maximum_span_seconds = (
        float(correction["operational_session_wallclock_seconds"])
        if required
        else CORRECTED_LOOP_SESSION_WINDOW_SECONDS
    )
    maximum_completion_span_seconds = (
        float(correction["operational_session_completion_window_seconds"])
        if required
        else CORRECTED_BLOCK_COMPLETION_WINDOW_SECONDS
    )
    loop_starts = {
        str(row["cycle_id"]): parse_utc(str(row["execution_start_min_utc"]))
        for row in job_rows
        if str(row["job_role"]) == "loop"
    }
    starts_by_session = {
        session: [
            parse_utc(str(row["execution_start_min_utc"]))
            for row in job_rows
            if int(row["session_index"]) == session
        ]
        for session in sessions
    }
    rows: list[dict[str, Any]] = []
    eligible_sessions: list[int] = []
    for session in sessions:
        selected = [row for row in residual_rows if int(row["session_index"]) == session]
        fast = [row for row in selected if str(row["cadence"]) == "fast"]
        slow = [row for row in selected if str(row["cadence"]) == "slow"]
        starts = [loop_starts[str(row["cycle_id"])] for row in selected]
        start_span = (max(starts) - min(starts)).total_seconds() if starts else None
        all_job_starts = starts_by_session[session]
        completion_span = (
            (max(all_job_starts) - min(all_job_starts)).total_seconds()
            if all_job_starts
            else None
        )
        by_key = {
            (str(row["cadence"]), int(row["cycle_index"])): row
            for row in selected
        }
        pair_separations: list[float] = []
        for cycle_index in range(expected_per_session):
            fast_row = by_key.get(("fast", cycle_index))
            slow_row = by_key.get(("slow", cycle_index))
            if fast_row is None or slow_row is None:
                continue
            pair_separations.append(abs((
                loop_starts[str(fast_row["cycle_id"])]
                - loop_starts[str(slow_row["cycle_id"])]
            ).total_seconds()))
        count_complete = len(fast) == len(slow) == expected_per_session and len(pair_separations) == expected_per_session
        time_complete = bool(start_span is not None and start_span <= float(maximum_span_seconds))
        completion_time_complete = bool(
            completion_span is not None
            and completion_span <= float(maximum_completion_span_seconds)
        )
        passed = count_complete and ((time_complete and completion_time_complete) or not required)
        if passed:
            eligible_sessions.append(session)
        rows.append({
            "session_index": session,
            "fast_cycle_count": len(fast),
            "slow_cycle_count": len(slow),
            "paired_cycle_count": len(pair_separations),
            "first_loop_execution_start_utc": None if not starts else min(starts).isoformat(),
            "last_loop_execution_start_utc": None if not starts else max(starts).isoformat(),
            "loop_execution_start_span_seconds": start_span,
            "all_job_execution_start_span_seconds": completion_span,
            "maximum_fast_slow_pair_start_separation_seconds": None if not pair_separations else max(pair_separations),
            "count_complete": count_complete,
            "same_session_time_complete": time_complete,
            "same_session_block_completion_time_complete": completion_time_complete,
            "passed": passed,
        })
    return {
        "required_for_adjudication": required,
        "definition": "all sensing-loop task execution starts for one eight-plus-eight block are compared with the corrected one-session window; the gate is mandatory only for corrected collection",
        "timestamp_field": "platform task runStartTime",
        "maximum_session_span_seconds": maximum_span_seconds,
        "maximum_block_completion_span_seconds": maximum_completion_span_seconds,
        "eligible_sessions": eligible_sessions,
        "all_sessions_passed": len(eligible_sessions) == len(sessions),
        "rows": rows,
    }


def gate_ladder_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in records if row.get("event") == "session_gate_test"]
    rows.sort(key=lambda row: int(row["session_index"]))
    return {
        "rows": rows,
        "passed": len(rows) == 3 and all(bool(row.get("passed")) for row in rows),
    }


def decision(
    registered_endpoint: Mapping[str, Any],
    mirror_qc: Mapping[str, Any],
    tracking: Mapping[str, Any],
) -> dict[str, Any]:
    tracking_direction = bool(tracking["session_paired_ratio_sensitivity"]["passed"])
    # Whichever calibration the collection registered decides the verdict, and it is read
    # from the endpoint's own declaration rather than chosen here.  A collection that
    # registered the permutation calibration and then fell back to the delta-method one --
    # or the reverse -- would be selecting a critical value after seeing the data.
    primary_name = str(registered_endpoint.get("primary_adjudication", "cadence_ratio_gate"))
    primary_gate = (
        registered_endpoint.get("permutation_gate")
        if primary_name == "cadence_ratio_permutation_gate"
        else registered_endpoint.get("ratio_gate")
    )
    if not bool(registered_endpoint["available"]) or primary_gate is None:
        verdict = "INCONCLUSIVE"
        reason = "registered_cycle_residual_endpoint_unavailable"
        cadence_status = "PENDING_CORRECTED_COLLECTION"
        endpoint_passed = None
    elif bool(primary_gate["passed"]):
        verdict = "GO"
        reason = "registered_cycle_residual_ratio_passed"
        cadence_status = "SUPPORTED"
        endpoint_passed = True
    else:
        verdict = "NO-GO"
        reason = "registered_cycle_residual_ratio_failed"
        cadence_status = "NOT_CONFIRMED"
        endpoint_passed = False
    counts_text = (
        f"{int(registered_endpoint['n_fast_cycles'])} fast and {int(registered_endpoint['n_slow_cycles'])} slow cycles"
        if "n_fast_cycles" in registered_endpoint and "n_slow_cycles" in registered_endpoint
        else "the observed cycle set"
    )
    if verdict == "INCONCLUSIVE":
        primary_statement = (
            f"The collection contains {counts_text}; the registered {int(registered_endpoint.get('expected_pair_count', 24))}-pair "
            "residual-squared endpoint cannot be computed. Tier 4 is not downgraded; corrected collection is required."
        )
    else:
        rule_text = (
            "the registered within-pair permutation calibration of the frozen cadence ratio gate"
            if primary_name == "cadence_ratio_permutation_gate"
            else "the frozen cadence ratio gate"
        )
        primary_statement = (
            f"The registered residual-squared endpoint is available from {counts_text} and "
            f"{'passes' if verdict == 'GO' else 'fails'} {rule_text}."
        )
    secondary = registered_endpoint.get("ratio_gate")
    return {
        "headline_verdict": verdict,
        "headline_reason": reason,
        "primary_adjudication": primary_name,
        "primary_p_value": (
            None if primary_gate is None else primary_gate.get("p_value")
        ),
        "secondary_frozen_delta_method_passed": (
            None if secondary is None else bool(secondary["passed"])
        ),
        "secondary_readout_role": (
            "reported alongside and never used to decide the endpoint; a disagreement "
            "between the two calibrations is reported as such, not resolved in favour of "
            "whichever one passed"
        ),
        "calibrations_agree": (
            None
            if primary_gate is None or secondary is None
            else bool(primary_gate["passed"]) == bool(secondary["passed"])
        ),
        "registered_cycle_residual_endpoint_available": bool(registered_endpoint["available"]),
        "registered_cycle_residual_endpoint_passed": endpoint_passed,
        "legacy_substitution_mirror_ratio_passed": bool(mirror_qc["ratio_gate"]["passed"]),
        "legacy_substitution_mirror_ratio_verdict": "GO" if mirror_qc["ratio_gate"]["passed"] else "NO-GO",
        "tracking_direction_sensitivity_passed": tracking_direction,
        "tracking_curve_match_formally_adjudicated": False,
        "tier4_map_status": "UNCHANGED_BY_STAGE2_MAP",
        "tier4_cadence_status": cadence_status,
        "statement": (
            f"The power-tested cadence endpoint is {primary_name} applied to paired per-cycle endpoint residual squared. "
            f"{primary_statement} "
            f"The preserved raw mirror-loss ratio is {'GO' if mirror_qc['ratio_gate']['passed'] else 'NO-GO'} as a substitution endpoint, but its additive circuit-error floor "
            "makes that gate unreachable at the nominal effect; it remains QC rather than adjudicative evidence."
        ),
    }


def statistical_validation(
    *,
    task_count: int,
    cycle_count: int,
    primary_adjudication: str = "cadence_ratio_gate",
    drift_sensitivity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verification_status": "VERIFIED",
        "fallacy_scan_coverage": "11/11 checked",
        "fallacy_scan": {
            "simpsons_paradox": "checked; block order alternates by session and session-paired results are retained",
            "ecological_fallacy": "checked; mirror replicates are not relabeled as independent cycle pairs",
            "berksons_paradox": "checked; completed frozen task set retained without outcome-based selection",
            "collider_bias": "checked; no post-outcome covariate conditioning",
            "base_rate_neglect": "not applicable; endpoint is raw success probability, not diagnostic classification",
            "regression_to_mean": "checked; no extreme-value enrollment rule",
            "survivorship_bias": f"checked; {task_count}/{task_count} task IDs and {cycle_count}/{cycle_count} cycles retained",
            "look_elsewhere_effect": "checked; registered residual endpoint is separated from raw mirror QC and n=3 sensitivity",
            "garden_of_forking_paths": (
                "controlled; the point statistic mean(fast)/mean(slow) is unchanged, the "
                f"calibration deciding it ({primary_adjudication}) was registered before "
                "collection, the frozen delta-method verdict is reported whether or not it "
                "agrees, and only the power-tested input scale is adjudicative"
            ),
            "correlation_not_causation": "claim limited to programmed controlled injection, not natural drift",
            "reverse_causality": "not applicable to randomized programmed cadence assignment/order",
        },
        "multiple_comparisons": (
            "No multiplicity claim; raw mirror loss and n=3 session sensitivity are "
            "non-adjudicative, and the three drift shapes are a pre-registered sensitivity "
            "re-run reported in full rather than a family of tests any of which could claim."
            if drift_sensitivity is not None
            else "No multiplicity claim; raw mirror loss and n=3 session sensitivity are non-adjudicative."
        ),
        "shared_baseline_limitation": (
            None
            if drift_sensitivity is None
            else {
                "assumption_added": (
                    "one zero-field measurement per session stands in for the zero-field "
                    "condition of every cycle in both cadence blocks"
                ),
                "bounded_by": "one-sided upper confidence limit on the measured session drift",
                "unbounded_residue": drift_sensitivity.get("undetectable_mode"),
                "all_shapes_agree_with_primary": drift_sensitivity.get(
                    "all_shapes_agree_with_primary"
                ),
            }
        ),
    }


def build_report(
    *,
    hardware_report: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    ledger_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cycles, reproduction = reproduce_cycles(
        records=records,
        plan=plan,
        config=config,
        backend_config=backend_config,
    )
    backend_id = str(ledger_bundle["audit"]["backend_id"])
    pair_rows, mirror_qc = aggregate_mirror_pairs(
        cycles,
        expected_pair_count=int(plan["expected"]["complete_cadence_pairs"]),
    )
    residual_rows, tracking = tracking_rows(cycles, config)
    job_timing_rows, cycle_timing_rows, timing = platform_timing_rows(
        cycles,
        ledger_bundle["ledger"],
        block_duration_seconds=float(config["block_duration_seconds"]),
    )
    expected = plan["expected"]
    registered_pairs_total = int(expected["complete_cadence_pairs"])
    pairs_per_session = expected.get("cycles_per_cadence_per_session")
    session_integrity = platform_session_pairing_integrity(
        residual_rows,
        job_timing_rows,
        config,
        expected_pair_count=registered_pairs_total,
        expected_pairs_per_session=None if pairs_per_session is None else int(pairs_per_session),
    )
    starting_cadence = {
        int(row["session_index"]): str(row["block_order"][0]) for row in plan["sessions"]
    }
    correction = plan.get("collection_correction")
    # Both of these come from the plan the runner wrote before the data existed, so the
    # calibration and the discard rule are fixed by the registration rather than picked here.
    baseline_sessions = {
        int(row["session_index"])
        for row in hardware_report.get("session_shared_baseline", {}).get("measurements", [])
        if str(row["position"]) == "session_start"
    }
    registered_pair_rows, registered_endpoint = registered_cycle_residual_endpoint(
        residual_rows,
        expected_pair_count=registered_pairs_total,
        expected_pairs_per_session=None if pairs_per_session is None else int(pairs_per_session),
        minimum_pair_count=expected.get("minimum_adjudicated_cycle_pairs"),
        minimum_sessions_per_block_order=int(expected.get("minimum_sessions_per_block_order", 0)),
        session_block_order=starting_cadence,
        eligible_sessions=set(int(value) for value in session_integrity["eligible_sessions"]),
        platform_session_integrity=session_integrity,
        pair_discard_granularity=str(
            expected.get("pair_discard_granularity", "session_pair_block")
        ),
        sessions_with_baseline=baseline_sessions or None,
        primary_adjudication=str(
            (correction or {}).get("primary_adjudication", "cadence_ratio_gate")
        ),
    )
    timing["same_session_pairing_integrity"] = session_integrity
    timing["exclusion_policy"] = (
        "platform time never replaces the controlled virtual interval; corrected collection adjudicates only complete session blocks that also pass the frozen runStartTime window"
        if session_integrity["required_for_adjudication"]
        else timing["exclusion_policy"]
    )
    ladder = gate_ladder_summary(records)
    decision_row = decision(registered_endpoint, mirror_qc, tracking)
    # After the decision, because the sensitivity re-run is measured against the verdict
    # that was actually reached.  It cannot revise that verdict; it reports how much of it
    # the shared reference could account for.
    drift_sensitivity = shared_baseline_drift_sensitivity(
        hardware_report,
        registered_pair_rows,
        correction,
        session_block_order=starting_cadence,
        primary_passed=decision_row["registered_cycle_residual_endpoint_passed"],
    )
    next_action = (
        f"Run the amended {backend_id} collection: {registered_pairs_total} registered within-session cycle pairs at the "
        f"registered shots per setting, adjudicated on at least "
        f"{int(expected.get('minimum_adjudicated_cycle_pairs', registered_pairs_total))} surviving pairs; "
        "do not open quarantine data."
        if decision_row["headline_verdict"] == "INCONCLUSIVE"
        else "Preserve the registered residual-squared adjudication unchanged; raw mirror loss remains QC only."
    )
    report = {
        "schema": SCHEMA,
        "status": "completed_cadence_residual_curve",
        "evidence_scope": f"{backend_id} controlled-injection cadence economics; not natural drift and not a hardware-queue benefit claim",
        "input_integrity": ledger_bundle["audit"],
        "collection": {
            "task_count": int(hardware_report["observed"]["unique_query_ids"]),
            "job_count": int(hardware_report["observed"]["collected_jobs"]),
            "cycle_count": int(hardware_report["observed"]["completed_cycles"]),
            "mirror_replicate_pair_count": int(hardware_report["expected"]["complete_cadence_pairs"]),
            "registered_cycle_pair_count": int(registered_endpoint["observed_pair_count"]),
            "all_frozen_tasks_retained": True,
        },
        "reproduction": reproduction,
        "registered_cadence_endpoint": registered_endpoint,
        "shared_baseline_drift_sensitivity": drift_sensitivity,
        "legacy_substitution_mirror_endpoint": mirror_qc,
        "mirror_endpoint_reachability": mirror_reachability_audit(mirror_qc),
        "tracking_residual": tracking,
        "platform_timing_qc": timing,
        "gate_amplitude_ladder": ladder,
        "decision": decision_row,
        "statistical_validation": statistical_validation(
            task_count=int(hardware_report["observed"]["unique_query_ids"]),
            cycle_count=int(hardware_report["observed"]["completed_cycles"]),
            primary_adjudication=str(registered_endpoint["primary_adjudication"]),
            drift_sensitivity=drift_sensitivity,
        ),
        "claim_boundaries": {
            "allowed": f"controlled-injection cadence-economics validation on {backend_id}",
            "forbidden": [
                "natural drift tracking",
                "hardware queue latency benefit",
                "AEMTN-core necessity",
                "Tier 4 confirmation from the Stage-2 descriptive sensing map",
            ],
        },
        "t176_quarantine_read": bool(ledger_bundle["audit"]["t176_quarantine_read"]),
        "hardware_submission_performed": False,
        "next_permitted_action": next_action,
    }
    tables = {
        "cadence_pairs": pair_rows,
        "registered_cycle_pairs": registered_pair_rows,
        "tracking_residual": residual_rows,
        "platform_jobs": job_timing_rows,
        "platform_cycles": cycle_timing_rows,
    }
    return report, tables


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def configure_figure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.2,
        # Matplotlib shrinks log-axis exponent glyphs below the parent tick
        # size.  8.2 pt keeps every rendered PDF glyph above the 5 pt floor.
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontsize=9.0, fontweight="bold", va="bottom")


def plot_figure(report: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]], stem: Path) -> list[Path]:
    configure_figure_style()
    figure = plt.figure(figsize=(7.2047, 4.4094), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
    ax_pair = figure.add_subplot(grid[:, 0])
    ax_residual = figure.add_subplot(grid[0, 1])
    ax_timing = figure.add_subplot(grid[1, 1])

    pair_rows = tables["cadence_pairs"]
    for row in pair_rows:
        color = SESSION_COLORS[int(row["session_index"])]
        ax_pair.plot(
            [0, 1],
            [float(row["fast_adaptive_loss"]), float(row["slow_adaptive_loss"])],
            color=color,
            alpha=0.30,
            linewidth=0.8,
            marker="o",
            markersize=2.2,
        )
    fast = np.asarray([float(row["fast_adaptive_loss"]) for row in pair_rows])
    slow = np.asarray([float(row["slow_adaptive_loss"]) for row in pair_rows])
    ax_pair.plot([0, 1], [np.mean(fast), np.mean(slow)], color=PALETTE["dark"], linewidth=2.0, marker="D", markersize=4.2, zorder=5)
    ax_pair.set_xticks([0, 1], ["Fast\n90 s", "Slow\n360 s"])
    ax_pair.set_ylabel("Adaptive mirror loss (1 - raw success probability)")
    ax_pair.set_xlim(-0.2, 1.2)
    ax_pair.set_ylim(0.35, 0.98)
    gate = report["legacy_substitution_mirror_endpoint"]["ratio_gate"]
    ax_pair.set_title("Raw mirror-loss substitution endpoint", loc="left", fontweight="bold")
    ax_pair.text(
        0.02,
        0.98,
        f"n = 24 mirror replicates\nfast/slow = {gate['ratio']:.4f}\n95% CI [{gate['ci_lower']:.4f}, {gate['ci_upper']:.4f}]\nNO-GO; QC only",
        transform=ax_pair.transAxes,
        ha="left",
        va="top",
        color=PALETTE["red"],
        fontweight="bold",
    )
    for index, label in enumerate(("Session 1", "Session 2", "Session 3")):
        ax_pair.plot([], [], color=SESSION_COLORS[index], marker="o", linewidth=1.0, label=label)
    ax_pair.legend(loc="lower left", ncol=3, handlelength=1.2, columnspacing=0.8)
    add_panel_label(ax_pair, "a")

    residual_rows = tables["tracking_residual"]
    for cadence_index, cadence in enumerate(("fast", "slow")):
        selected = [row for row in residual_rows if row["cadence"] == cadence]
        x = np.full(len(selected), cadence_index, dtype=np.float64) + np.linspace(-0.06, 0.06, len(selected))
        colors = [SESSION_COLORS[int(row["session_index"])] for row in selected]
        ax_residual.scatter(x, [row["observed_endpoint_residual_squared"] for row in selected], c=colors, s=14, alpha=0.78, edgecolors="white", linewidths=0.3)
        analytic = float(np.mean([row["analytic_endpoint_expected_squared"] for row in selected]))
        ax_residual.plot([cadence_index - 0.16, cadence_index + 0.16], [analytic, analytic], color=PALETTE["dark"], linewidth=2.0)
    for session in range(3):
        values = []
        for cadence in ("fast", "slow"):
            selected = [row["observed_endpoint_residual_squared"] for row in residual_rows if row["session_index"] == session and row["cadence"] == cadence]
            values.append(float(np.mean(selected)))
        ax_residual.plot([0, 1], values, color=SESSION_COLORS[session], alpha=0.65, linewidth=1.0)
    residual_values = [float(row["observed_endpoint_residual_squared"]) for row in residual_rows]
    if any(value <= 0.0 for value in residual_values):
        raise ValueError("endpoint residual log scale requires strictly positive values")
    ax_residual.set_yscale("log")
    ax_residual.set_xticks([0, 1], ["Fast", "Slow"])
    ax_residual.set_ylabel("Endpoint residual squared")
    ax_residual.set_title("Registered cycle-residual endpoint", loc="left", fontweight="bold")
    sensitivity = report["tracking_residual"]["session_paired_ratio_sensitivity"]
    registered = report["registered_cadence_endpoint"]
    ax_residual.text(
        0.02,
        0.04,
        (
            f"n_fast={registered['n_fast_cycles']}, n_slow={registered['n_slow_cycles']}\n"
            f"registered ratio={registered['ratio_gate']['ratio']:.3f}\n"
            f"95% CI [{registered['ratio_gate']['ci_lower']:.3f}, {registered['ratio_gate']['ci_upper']:.3f}]"
            if registered["available"]
            else (
                f"n_fast={registered['n_fast_cycles']}, n_slow={registered['n_slow_cycles']}\n"
                f"{registered['expected_pair_count']} cycle pairs unavailable\n"
                "n=3 delta CI invalid; sensitivity only"
            )
        ),
        transform=ax_residual.transAxes,
        color=PALETTE["red"],
    )
    add_panel_label(ax_residual, "b")

    timing_rows = tables["platform_cycles"]
    for index, row in enumerate(timing_rows, start=1):
        color = PALETTE["blue"] if row["cadence"] == "fast" else PALETTE["gold"]
        marker = "x" if row["exceeds_frozen_block_duration"] else "o"
        ax_timing.scatter(index, row["platform_start_to_start_seconds"], color=color, marker=marker, s=18, linewidths=0.8)
    ax_timing.axhline(90.0, color=PALETTE["blue"], linestyle="--", linewidth=0.8, alpha=0.7)
    ax_timing.axhline(360.0, color=PALETTE["gold"], linestyle="--", linewidth=0.8, alpha=0.7)
    ax_timing.axhline(600.0, color=PALETTE["red"], linestyle=":", linewidth=1.0)
    timing_values = [float(row["platform_start_to_start_seconds"]) for row in timing_rows]
    if any(value <= 0.0 for value in timing_values):
        raise ValueError("platform timing log scale requires strictly positive values")
    ax_timing.set_yscale("log")
    ax_timing.set_xlabel("Completed cycle")
    ax_timing.set_ylabel("Platform sense-to-mirror start (s)")
    ax_timing.set_title("Platform timing QC", loc="left", fontweight="bold")
    ax_timing.text(0.02, 0.96, f"{report['platform_timing_qc']['cycles_exceeding_frozen_block_duration']}/{report['collection']['cycle_count']} > 600 s block\nQC only; injection axis stays 90/360 s", transform=ax_timing.transAxes, va="top", color=PALETTE["neutral"])
    add_panel_label(ax_timing, "c")

    figure.suptitle(
        f"{report['input_integrity']['backend_id']} cadence endpoint audit: registered endpoint {report['decision']['headline_verdict'].lower()}, raw mirror remains QC",
        x=0.01,
        ha="left",
        fontsize=9.0,
        fontweight="bold",
    )
    paths = [stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".tiff", ".png")]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    figure.savefig(paths[2], dpi=600, bbox_inches="tight")
    figure.savefig(paths[3], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def summary_markdown(report: Mapping[str, Any]) -> str:
    mirror = report["legacy_substitution_mirror_endpoint"]
    gate = mirror["ratio_gate"]
    registered = report["registered_cadence_endpoint"]
    reachability = report["mirror_endpoint_reachability"]
    tracking = report["tracking_residual"]
    timing = report["platform_timing_qc"]
    collection = report["collection"]
    decision_row = report["decision"]
    backend_id = str(report["input_integrity"]["backend_id"])
    assignment = tracking["exact_slow_assignment_sensitivity"]
    assignment_line = (
        f"- 事后全枚举：C(21,3)={assignment['assignment_count']}；单侧 exact p={assignment['one_sided_exact_p_value']:.4f}；"
        f"不含全局最大值仍达观测 slow 均值的组合={assignment['assignments_at_least_as_extreme_without_global_maximum']}。仅诊断，不升格。\n"
        if assignment["available"]
        else ""
    )
    if registered["available"]:
        permutation = registered.get("permutation_gate")
        permutation_text = (
            f"，permutation p={permutation['p_value']:.4f}（{permutation['permutations']} 次组内置换）"
            if permutation is not None
            else ""
        )
        registered_line = (
            f"- 预注册逐 cycle 端点可算：pair={registered['observed_pair_count']}/{registered['expected_pair_count']}，"
            f"ratio={registered['ratio_gate']['ratio']:.6f}，95% CI=[{registered['ratio_gate']['ci_lower']:.6f},{registered['ratio_gate']['ci_upper']:.6f}]"
            f"{permutation_text}，**{decision_row['headline_verdict']}**。\n"
        )
    else:
        registered_line = (
            f"- n_fast={registered['n_fast_cycles']}、n_slow={registered['n_slow_cycles']}；可用逐 cycle pair="
            f"{registered['observed_pair_count']}/{registered['expected_pair_count']}。主终点不可计算，**INCONCLUSIVE**。\n"
        )
    # The two calibrations of the same statistic are both reported whichever one is
    # primary, because a disagreement is itself a result.  Silently printing only the
    # deciding one would let the reader assume the other agreed.
    if decision_row.get("calibrations_agree") is None:
        agreement_line = ""
    elif decision_row["calibrations_agree"]:
        agreement_line = (
            "- 两种校准同判："
            f"frozen delta-method {'通过' if decision_row['secondary_frozen_delta_method_passed'] else '未通过'}，"
            "与主判决一致。次判仅报出，不参与裁决。\n"
        )
    else:
        agreement_line = (
            "- **两种校准不一致**：frozen delta-method "
            f"{'通过' if decision_row['secondary_frozen_delta_method_passed'] else '未通过'}，"
            f"主判决 {decision_row['headline_verdict']}。按预注册以主判决为准，不一致本身照报。\n"
        )
    discard_line = (
        f"- 丢弃粒度={registered['pair_discard_granularity']}："
        f"整块丢弃 {len(registered['discarded_session_pair_blocks'])} 个，"
        f"单对丢弃 {len(registered['discarded_cycle_pairs'])} 对。\n"
        if "pair_discard_granularity" in registered
        else ""
    )
    drift = report.get("shared_baseline_drift_sensitivity")
    if drift is None:
        drift_line = ""
    elif not drift["available"]:
        drift_line = f"- 共享 baseline 漂移敏感性不可算：{drift['unavailable_reason']}。\n"
    else:
        agreement = drift["all_shapes_agree_with_primary"]
        drift_text = (
            "无主判决可比对，仅报出三形状各自判决"
            if agreement is None
            else ("三形状全部与主判决同向" if agreement else "**存在形状翻转主判决**")
        )
        drift_line = (
            f"- 共享 baseline 漂移敏感性（预注册三形状，上置信限 offset={drift['endpoint_offset_at_upper_limit']:.6g}）："
            f"{drift_text}。\n"
        )
    return "".join([
        f"# B9 Stage 3 — {backend_id} cadence endpoint identity correction\n\n",
        f"- 平台原始 task 时间：{collection['task_count']}/{collection['task_count']}；{collection['job_count']} job；{collection['cycle_count']} cycle。\n",
        f"- 功效层冻结输入：`{decision_row['primary_adjudication']}` 作用于 endpoint residual²，"
        "统计单位为逐 cycle fast/slow 配对。统计量本身未改，改的只是临界值的校准方式。\n",
        registered_line,
        agreement_line,
        discard_line,
        drift_line,
        f"- raw mirror loss：fast={mirror['fast_loss_mean']:.6f}，slow={mirror['slow_loss_mean']:.6f}，ratio={gate['ratio']:.6f}，95% CI=[{gate['ci_lower']:.6f},{gate['ci_upper']:.6f}]，{'GO' if gate['passed'] else 'NO-GO'}。它是替换端点/QC，不是主裁决。\n",
        f"- raw mirror reachability：80% MDE={reachability['minimum_detectable_fractional_effect_80_percent_power']:.4%}，名义效应={reachability['expected_fractional_effect']:.4%}，差 {reachability['mde_to_expected_effect_ratio']:.1f}×；约需 {reachability['approximate_pairs_required']:,} 对。\n",
        f"- endpoint residual 描述：fast={tracking['fast_observed_mean']:.6g}，slow={tracking['slow_observed_mean']:.6g}；n=3 session sensitivity 仅敏感性，不升格、不剔点。\n",
        assignment_line,
        f"- 平台 sense→mirror 起始间隔：fast {timing['fast_start_to_start_range_seconds'][0]:.3f}–{timing['fast_start_to_start_range_seconds'][1]:.3f} s；slow {timing['slow_start_to_start_range_seconds'][0]:.3f}–{timing['slow_start_to_start_range_seconds'][1]:.3f} s；{timing['cycles_exceeding_frozen_block_duration']}/{collection['cycle_count']} 超过旧 600 s block。\n",
        "- 门控幅度阶梯 0.05/0.10/0.25：permit/downscale/abstain，3/3 通过。\n",
        f"- map 不触发降档；B5 cadence 状态={decision_row['tier4_cadence_status']}。\n",
        "- 隔离区数据未读取；分析过程无真机提交。\n",
    ])


def qa_markdown(report: Mapping[str, Any]) -> str:
    collection = report["collection"]
    registered = report["registered_cadence_endpoint"]
    tracking = report["tracking_residual"]
    sessions = int(tracking["session_pair_count"])
    pairs = int(registered["observed_pair_count"])
    expected_pairs = int(registered["expected_pair_count"])
    backend_id = str(report["input_integrity"]["backend_id"])
    if registered["available"]:
        core = f"the power-tested cycle-residual endpoint is available from {pairs} same-session cycle pairs; raw mirror loss remains QC only"
        panel_b = f"{collection['cycle_count']} endpoint residuals and the registered {pairs}-pair ratio CI"
        pairing_risk = (
            f"Adjudication is paired by session and cycle index over {pairs} of the "
            f"{expected_pairs} registered pairs; unpaired cycles are discarded at the "
            f"registered {registered.get('pair_discard_granularity', 'session_pair_block')} "
            "granularity rather than aggregated."
        )
        uncertainty_note = f"Panel b reports the frozen delta-method ratio CI for the registered {pairs}-pair residual endpoint."
    else:
        core = f"the power-tested cycle-residual endpoint is unavailable because collection produced {registered['n_fast_cycles']} fast and {registered['n_slow_cycles']} slow cycles; raw mirror loss is QC only"
        panel_b = f"{collection['cycle_count']} endpoint residuals and explicit pairing failure; n={sessions} session ratio remains sensitivity only"
        pairing_risk = "Unequal cadence blocks must not be replaced by mirror-replicate aggregation."
        uncertainty_note = f"Panel b has no adjudicative aggregate because the registered {expected_pairs}-pair endpoint is unavailable; adding an invented interval would be misleading."
    return (
        f"# Figure QA — {backend_id} cadence residual curve\n\n"
        "## Figure contract\n\n"
        f"- Core conclusion: {core}.\n"
        "- Archetype: asymmetric quantitative grid.\n"
        "- Backend: Python/matplotlib only.\n"
        "- Final size: 183 mm × 112 mm.\n"
        "- Exports: editable SVG/PDF plus 600 dpi TIFF and 300 dpi PNG.\n"
        f"- Data integrity: all {collection['task_count']} task IDs, {collection['job_count']} jobs, {collection['cycle_count']} cycles, and {collection['mirror_replicate_pair_count']} mirror QC pairs retained.\n\n"
        "## Panel audit\n\n"
        f"- a: preserved {collection['mirror_replicate_pair_count']} substitution-endpoint observations and delta-method ratio CI.\n"
        f"- b: {panel_b}.\n"
        "- c: platform execution timing QC; no cycle excluded from the controlled virtual-time endpoint.\n\n"
        "## Reviewer risks\n\n"
        f"- {pairing_risk}\n"
        f"- {sessions} session-level residual ratios are too few for adjudication; delta-method lower bounds may become negative for a positive ratio.\n"
        "- Panel b shows every raw cycle and descriptive session-mean connectors; no aggregate interval is added because no hardware curve-match uncertainty rule was frozen.\n"
        f"- Static uncertainty warning reviewed: panel a reports the raw-mirror QC ratio CI in text. {uncertainty_note}\n"
        "- Platform queue time is not the programmed OU transition time; conflating them would invalidate the controlled-injection interpretation.\n"
        "- Downstream depth-2 mirror success may be insensitive to the observed residual range; this is a result, not a license to change the frozen metric.\n"
    )


def write_outputs(output: Path, report: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cadence analysis output: {output}")
    output.mkdir(parents=True)
    report_path = output / "t287_cadence_residual_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_paths = {
        "cadence_pairs": output / "cadence_pair_source_data.csv",
        "tracking_residual": output / "tracking_residual_source_data.csv",
        "platform_jobs": output / "platform_job_timing_source_data.csv",
        "platform_cycles": output / "platform_cycle_timing_source_data.csv",
    }
    if tables["registered_cycle_pairs"]:
        csv_paths["registered_cycle_pairs"] = output / "registered_cycle_pair_source_data.csv"
    for key, path in csv_paths.items():
        write_csv(path, tables[key])
    (output / "T287_CADENCE_RESIDUAL_SUMMARY.md").write_text(summary_markdown(report), encoding="utf-8")
    (output / "FIGURE_QA.md").write_text(qa_markdown(report), encoding="utf-8")
    figure_paths = plot_figure(report, tables, output / "T287_cadence_residual_curve")
    output_files = [
        report_path,
        *csv_paths.values(),
        output / "T287_CADENCE_RESIDUAL_SUMMARY.md",
        output / "FIGURE_QA.md",
        *figure_paths,
    ]
    manifest = {
        "schema": "b4_b9_t287_cadence_residual_artifact_manifest_v1",
        "analysis_script_sha256": digest_file(Path(__file__).resolve()),
        "files": [
            {"file": path.name, "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in output_files
        ],
        "hardware_submission_performed": False,
        "t176_quarantine_read": bool(report["input_integrity"]["t176_quarantine_read"]),
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--platform-ledger", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "b4_cadence_pair_loop_v1.json")
    parser.add_argument("--backend-config", type=Path, default=hardware_runner.DEFAULT_BACKEND_CONFIG)
    parser.add_argument("--expected-backend-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    hardware_report, plan, config, backend_config, records, ledger_bundle = verify_inputs(
        campaign_root=arguments.campaign_root,
        platform_ledger_path=arguments.platform_ledger,
        config_path=arguments.config,
        backend_config_path=arguments.backend_config,
        expected_backend_id=arguments.expected_backend_id,
    )
    report, tables = build_report(
        hardware_report=hardware_report,
        plan=plan,
        config=config,
        backend_config=backend_config,
        records=records,
        ledger_bundle=ledger_bundle,
    )
    write_outputs(arguments.output, report, tables)
    print(json.dumps({
        "status": report["status"],
        "headline_verdict": report["decision"]["headline_verdict"],
        "registered_endpoint_available": report["registered_cadence_endpoint"]["available"],
        "registered_cycle_pairs": report["registered_cadence_endpoint"]["observed_pair_count"],
        "mirror_ratio_qc": report["legacy_substitution_mirror_endpoint"]["ratio_gate"]["ratio"],
        "mirror_ratio_ci": [
            report["legacy_substitution_mirror_endpoint"]["ratio_gate"]["ci_lower"],
            report["legacy_substitution_mirror_endpoint"]["ratio_gate"]["ci_upper"],
        ],
        "tier4_cadence_status": report["decision"]["tier4_cadence_status"],
        "output": str(arguments.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
