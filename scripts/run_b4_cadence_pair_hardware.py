#!/usr/bin/env python3
"""Run the frozen B-4 cadence-pair endpoint on TianYan hardware."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import drift_campaign as campaign
from scripts import drift_campaign_v4
from scripts import run_cadence_pair_hardware_smoke as hardware
from scripts import run_cadence_pair_loop as cadence
from scripts import simulate_b4_cadence_endpoint_power as endpoint_power
from src.adaptive import shared_baseline_sensing as shared_baseline
from src.adaptive.bandit import Action, shield
from src.adaptive.cadence_permutation import permutation_claim_rate
from src.adaptive.task_metric_mirror import success_probability_from_raw_counts


DEFAULT_LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json"
DEFAULT_CORRECTED_LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v2.json"
DEFAULT_BACKEND_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"
DEFAULT_PEER_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"
DEFAULT_STAGE1_MANIFEST = ROOT / "docs" / "B4_T176_MIGRATION_CURATED_MANIFEST_20260823.json"
DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\quarantine\tianyan176\B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823")
PLAN_NAME = "supplement_plan.json"
REPORT_NAME = "cadence_hardware_report.json"


class SubmissionLimitReached(RuntimeError):
    pass


class SessionPairBlockExpired(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(campaign.json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def collection_correction(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = config.get("collection_correction")
    return value if isinstance(value, Mapping) else None


def registered_backend_id(loop_config: Mapping[str, Any]) -> str:
    """Return exact platform spelling of backend frozen in loop config."""
    correction = collection_correction(loop_config)
    correction_value = None if correction is None else correction.get("registered_backend_id")
    top_level_value = loop_config.get("registered_backend_id")
    if correction_value is not None and top_level_value is not None and correction_value != top_level_value:
        raise ValueError("loop config carries conflicting registered_backend_id values")
    value = correction_value if correction_value is not None else top_level_value
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("loop config must declare a non-empty registered_backend_id")
    return value


def validate_backend_pin(
    loop_config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
) -> str:
    registered = registered_backend_id(loop_config)
    actual = str(backend_config["backend"]["backend_id"])
    if actual != registered:
        raise ValueError(
            f"backend config id {actual!r} does not match loop config "
            f"registered_backend_id {registered!r}"
        )
    return actual


def mirror_qc_cycle_indices(correction: Mapping[str, Any] | None) -> list[int] | None:
    """Cycle indices that carry the matched mirror QC job, or None for every cycle.

    The registered endpoint is ``|mirror_fields + compensation|**2``; both terms come
    from the injected OU truth and the sensing loop, so the mirror job is descriptive
    QC and may be pre-registered on a subset without touching the endpoint.
    """
    if correction is None:
        return None
    declared = correction.get("mirror_qc_cycle_indices")
    if declared is None:
        return None
    return sorted({int(value) for value in declared})


LEGACY_UNREACHABLE_CORRECTION_STATUS = "frozen_after_endpoint_identity_audit_20260815"


def shared_baseline_plan(correction: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The session-shared zero-field baseline, or None for the four-setting cycle.

    Corrections up to v3 measure the zero-field condition inside every cycle, so a cycle is
    four settings and there is no session-level baseline job.  From v4 the zero-field
    condition is one job per position per session, which halves the endpoint shot floor;
    everything downstream branches on whether this returns None so the earlier plans keep
    producing byte-identical schedules.
    """
    if correction is None or "baseline_measurements_per_session" not in correction:
        return None
    positions = [str(value) for value in correction["baseline_measurement_positions"]]
    if positions[:1] != ["session_start"]:
        raise ValueError("a session's first baseline measurement must be its session_start one")
    if len(set(positions)) != len(positions):
        raise ValueError("baseline measurement positions must be distinct")
    if int(correction["baseline_measurements_per_session"]) != len(positions):
        raise ValueError("baseline measurement count disagrees with the declared positions")
    if str(correction["baseline_used_by_estimate"]) != "session_start":
        # The compensation is computed online by the shield from each cycle's own sensed
        # estimate, so a cycle can only subtract a baseline that already exists when it runs.
        raise ValueError("the differential estimate can only use the session_start baseline")
    if str(correction["baseline_sharing_scope"]) != "whole_session_both_cadence_blocks":
        # One baseline serving both cadence blocks puts the identical offset into the fast
        # and slow arm of every registered pair, which is what keeps within-pair label swaps
        # measure preserving under the boundary null.
        raise ValueError("the shared baseline must serve both cadence blocks of its session")
    if int(correction["sensing_settings_per_cycle"]) != 2:
        raise ValueError("an amortised baseline leaves two settings per sensing cycle")
    shots = {
        "session_start": int(correction["baseline_shots_per_setting"]),
        "session_end": int(correction["baseline_end_shots_per_setting"]),
    }
    if any(shots[position] <= 0 for position in positions):
        raise ValueError("every baseline measurement needs a positive shot count")
    return {
        "positions": positions,
        "settings_per_measurement": int(correction["baseline_settings_per_measurement"]),
        "shots_by_position": {position: shots[position] for position in positions},
        "lead_seconds": float(correction.get("baseline_lead_seconds", 0.0)),
        "trail_seconds": float(correction.get("baseline_trail_seconds", 0.0)),
    }


def sensing_settings_per_cycle(correction: Mapping[str, Any] | None) -> int:
    return 4 if correction is None else int(correction.get("sensing_settings_per_cycle", 4))


def session_block_order(correction: Mapping[str, Any], config: Mapping[str, Any]) -> list[list[str]]:
    """Session schedule for the corrected collection.

    The three-day ``block_order_by_day`` contract belongs to the frozen T-B5
    simulation config and is validated by ``cadence.validate_config``; it is
    never rewritten here.  A correction may carry its own longer schedule.
    """
    declared = correction.get("session_block_order")
    if declared is None:
        return [[str(value) for value in row] for row in config["block_order_by_day"]]
    return [[str(value) for value in row] for row in declared]


def validate_collection_correction(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    correction = collection_correction(config)
    if correction is None:
        return None
    if correction.get("block_definition") != "fixed_cycle_count":
        raise ValueError("collection correction blocks must be defined by cycle count")
    sessions = int(correction["sessions"])
    cycles_per_cadence = int(correction["cycles_per_cadence_per_session"])
    orders = session_block_order(correction, config)
    if sessions != len(orders):
        raise ValueError("collection correction session count must match its own session block order")
    if any(sorted(order) != ["fast", "slow"] for order in orders):
        raise ValueError("every session must run one fast block and one slow block")
    if cycles_per_cadence <= 0:
        raise ValueError("collection correction needs a positive cycle count per cadence")
    if int(correction["registered_cycle_pairs_total"]) != sessions * cycles_per_cadence:
        raise ValueError("registered cycle-pair total does not match sessions times cycles per cadence")
    if int(correction["mirror_repetitions_per_cycle"]) != int(config["mirror"]["repetitions_per_block"]):
        raise ValueError("collection correction mirror repetitions must match the hardware config")
    mirror_cycles = mirror_qc_cycle_indices(correction)
    if mirror_cycles is not None and not set(mirror_cycles) <= set(range(cycles_per_cadence)):
        raise ValueError("mirror QC cycle indices must fall inside the registered cycle range")
    sensing_shots_per_cycle = int(correction["sensing_settings_per_cycle"]) * int(
        correction["sensing_shots_per_setting"]
    )
    declared_sensing = correction.get("sensing_shots_per_cycle")
    if declared_sensing is not None and int(declared_sensing) != sensing_shots_per_cycle:
        raise ValueError("declared sensing shots per cycle disagree with settings times shots per setting")
    legacy_shots_per_cycle = correction.get("shots_per_cycle")
    if legacy_shots_per_cycle is not None:
        # Historic corrections declared one combined figure covering the mirror pair on every cycle.
        combined = sensing_shots_per_cycle + (
            int(correction["mirror_repetitions_per_cycle"])
            * int(correction["mirror_settings_per_repetition"])
            * int(config["mirror"]["shots_per_task"])
        )
        if int(legacy_shots_per_cycle) != combined:
            raise ValueError("declared shots per cycle disagree with the sensing and mirror composition")
    minimum_pairs = int(correction.get("minimum_adjudicated_cycle_pairs", correction["registered_cycle_pairs_total"]))
    if not 0 < minimum_pairs <= int(correction["registered_cycle_pairs_total"]):
        raise ValueError("minimum adjudicated pair count must be positive and no larger than the registered total")
    granularity = str(correction.get("pair_discard_granularity", "session_pair_block"))
    if granularity not in ("session_pair_block", "cycle_pair"):
        raise ValueError("pair discard granularity must be session_pair_block or cycle_pair")
    if granularity == "session_pair_block" and minimum_pairs % cycles_per_cadence != 0:
        # Under session granularity a lost cycle costs its whole session, so any minimum
        # that is not a whole number of sessions is unreachable by construction.  A shared
        # baseline breaks that coupling -- it is a separate job that survives a lost cycle --
        # which is what lets v4 register a minimum in the middle of a session.
        raise ValueError("minimum adjudicated pair count must be a whole number of session pair blocks")
    baseline = shared_baseline_plan(correction)
    if baseline is not None:
        sessions_required = int(correction.get("minimum_sessions_per_block_order", 0))
        if sessions_required * 2 > sessions:
            raise ValueError("the registered per-block-order session minimum cannot be met by this schedule")
        if baseline["lead_seconds"] < 0.0 or baseline["trail_seconds"] < 0.0:
            raise ValueError("baseline scheduling allowances cannot be negative")
    if bool(correction.get("optional_stopping_permitted")):
        raise ValueError("the corrected collection is a fixed-size design; optional stopping is not permitted")
    span = float(correction.get("operational_session_programmed_span_seconds", 0.0))
    wallclock = float(correction["operational_session_wallclock_seconds"])
    if wallclock <= 0.0:
        raise ValueError("collection correction needs a positive operational session window")
    lead = 0.0 if baseline is None else baseline["lead_seconds"]
    trail = 0.0 if baseline is None else baseline["trail_seconds"]
    if lead + span > wallclock:
        raise ValueError("the sensing-loop deadline cannot precede the programmed block span")
    completion_window = float(correction["operational_session_completion_window_seconds"])
    if completion_window < wallclock + trail:
        raise ValueError("session completion window cannot be shorter than the sensing-loop window")
    spacing = correction.get("operational_inter_session_seconds")
    if spacing is not None and completion_window > float(spacing):
        raise ValueError("a session's completion window cannot run into the next session's start")
    ceiling = correction.get("machine_time_ceiling_seconds")
    modelled = correction.get("modelled_busy_seconds_total")
    if ceiling is not None and modelled is not None and float(modelled) > float(ceiling):
        raise ValueError("the modelled collection exceeds its own registered machine-time ceiling")
    if bool(correction.get("gate_module_change_permitted")) or bool(correction.get("outlier_exclusion_permitted")):
        raise ValueError("collection correction cannot alter the gate or permit outcome-based exclusions")
    if correction.get("status") != LEGACY_UNREACHABLE_CORRECTION_STATUS:
        # Every correction after the 2026-08-15 reachability audit must carry its own
        # endpoint-scale power and boundary-size criteria; see para 286.
        if "minimum_power" not in correction or "maximum_boundary_size" not in correction:
            raise ValueError(
                "collection correction must register minimum_power and maximum_boundary_size "
                "so that the shots per setting can be checked for endpoint reachability"
            )
    return correction


def measured_collection_budget(
    cycles: Sequence[Mapping[str, Any]],
    backend_config: Mapping[str, Any],
    correction: Mapping[str, Any] | None = None,
    *,
    daily_window_seconds: float = 1200.0,
) -> dict[str, Any]:
    """Quota and execution time implied by schedule under registered timing model.

    The backend's own measured shot rate is the *best* of the T-B6 roundtrips; a correction
    that declares ``shot_rate_per_second_used`` was sized at the worst one, so the budget is
    modelled at the declared rate and the measured rate is reported beside it.  Charging the
    optimistic rate here would let a plan pass a ceiling its own design would fail.  Newer
    corrections may instead register role-specific queue-free task-runtime envelopes: quota
    sums task runtimes, while execution wall time sums one parallel envelope per job.
    """
    timing = backend_config["backend"]["tb6_measured_timing"]
    measured_rate = float(timing["effective_shots_per_second"])
    overhead = float(timing["fixed_overhead_seconds_per_setting"])
    rate = float(correction["shot_rate_per_second_used"]) if correction is not None and "shot_rate_per_second_used" in correction else measured_rate
    timing_budget_model = (
        None if correction is None else correction.get("timing_budget_model")
    )
    role_envelope_model = timing_budget_model == "role_envelope_sum_task_runtime"
    if timing_budget_model not in (None, "role_envelope_sum_task_runtime"):
        raise ValueError(f"unsupported timing_budget_model: {timing_budget_model!r}")
    role_names = ("baseline", "sense", "mirror")
    if role_envelope_model:
        role_task_runtime_seconds = {
            role: float(correction["role_task_runtime_seconds"][role])
            for role in role_names
        }
        role_settings_per_job = {
            role: int(correction["role_settings_per_job"][role])
            for role in role_names
        }
        declared_role_jobs_per_session = {
            role: int(correction["role_jobs_per_session"][role])
            for role in role_names
        }
        if any(value <= 0.0 for value in role_task_runtime_seconds.values()):
            raise ValueError("role task-runtime envelopes must be positive")
        if any(value <= 0 for value in role_settings_per_job.values()):
            raise ValueError("role settings per job must be positive")
        if any(value < 0 for value in declared_role_jobs_per_session.values()):
            raise ValueError("role jobs per session cannot be negative")
        declared_quota_seconds = float(correction["quota_seconds_per_session"])
        declared_execution_wall_seconds = float(correction["execution_wall_seconds_per_session"])
    else:
        role_task_runtime_seconds = None
        role_settings_per_job = None
        declared_role_jobs_per_session = None
        declared_quota_seconds = None
        declared_execution_wall_seconds = None
    settings_per_cycle = sensing_settings_per_cycle(correction)
    baseline = shared_baseline_plan(correction)
    session_rows: list[dict[str, Any]] = []
    for session_index in sorted({int(row["session_index"]) for row in cycles}):
        selected = [row for row in cycles if int(row["session_index"]) == session_index]
        sensing_settings = settings_per_cycle * len(selected)
        mirror_settings = sum(2 * len(row["mirror_seeds"]) for row in selected)
        sensing_shots = sum(settings_per_cycle * int(row["sensing_shots_per_setting"]) for row in selected)
        mirror_shots = sum(
            2 * len(row["mirror_seeds"]) * int(row["mirror_shots_per_task"])
            for row in selected
        )
        baseline_jobs = 0 if baseline is None else len(baseline["positions"])
        baseline_settings = 0 if baseline is None else baseline_jobs * baseline["settings_per_measurement"]
        baseline_shots = (
            0
            if baseline is None
            else baseline["settings_per_measurement"]
            * sum(baseline["shots_by_position"][position] for position in baseline["positions"])
        )
        total_shots = sensing_shots + mirror_shots + baseline_shots
        total_settings = sensing_settings + mirror_settings + baseline_settings
        role_job_counts = {
            "baseline": baseline_jobs,
            "sense": len(selected),
            "mirror": sum(bool(row["mirror_seeds"]) for row in selected),
        }
        role_setting_counts = {
            "baseline": baseline_settings,
            "sense": sensing_settings,
            "mirror": mirror_settings,
        }
        if role_envelope_model:
            if role_job_counts != declared_role_jobs_per_session:
                raise ValueError(
                    "actual per-session role job counts disagree with timing envelope: "
                    f"actual={role_job_counts}, declared={declared_role_jobs_per_session}"
                )
            expected_role_settings = {
                role: role_job_counts[role] * role_settings_per_job[role]
                for role in role_names
            }
            if role_setting_counts != expected_role_settings:
                raise ValueError(
                    "actual per-session role setting counts disagree with timing envelope: "
                    f"actual={role_setting_counts}, expected={expected_role_settings}"
                )
            quota_seconds = sum(
                role_setting_counts[role] * role_task_runtime_seconds[role]
                for role in role_names
            )
            execution_wall_seconds = sum(
                role_job_counts[role] * role_task_runtime_seconds[role]
                for role in role_names
            )
            if not math.isclose(quota_seconds, declared_quota_seconds, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "recomputed per-session quota seconds disagree with timing envelope: "
                    f"actual={quota_seconds}, declared={declared_quota_seconds}"
                )
            if not math.isclose(
                execution_wall_seconds,
                declared_execution_wall_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "recomputed per-session execution wall seconds disagree with timing envelope: "
                    f"actual={execution_wall_seconds}, declared={declared_execution_wall_seconds}"
                )
            estimated_busy_seconds = quota_seconds
        else:
            quota_seconds = None
            execution_wall_seconds = None
            estimated_busy_seconds = total_shots / rate + total_settings * overhead
        session_rows.append({
            "session_index": session_index,
            "cycle_count": len(selected),
            "baseline_job_count": baseline_jobs,
            "role_job_counts": role_job_counts,
            "role_setting_counts": role_setting_counts,
            "setting_count": total_settings,
            "shot_count": total_shots,
            "baseline_shot_count": baseline_shots,
            "estimated_busy_seconds": estimated_busy_seconds,
            "quota_seconds": quota_seconds,
            "execution_wall_seconds": execution_wall_seconds,
            "programmed_hold_seconds": sum(float(row["cadence_seconds"]) for row in selected),
        })
    wallclock_model = (
        "quota = sum(role task runtime * setting count); execution wall = sum(role task runtime * job count)"
        if role_envelope_model
        else "busy_seconds = total_shots / R + settings * c"
    )
    return {
        "timing_budget_model": (
            "role_envelope_sum_task_runtime"
            if role_envelope_model
            else "shots_plus_per_setting_overhead"
        ),
        "wallclock_model": wallclock_model,
        "measured_shots_per_second": measured_rate,
        "modelled_shots_per_second": rate,
        "measured_overhead_seconds_per_setting": overhead,
        "role_task_runtime_seconds": role_task_runtime_seconds,
        "role_settings_per_job": role_settings_per_job,
        "role_jobs_per_session": declared_role_jobs_per_session,
        "sensing_settings_per_cycle": settings_per_cycle,
        "daily_window_seconds": float(daily_window_seconds),
        "session_rows": session_rows,
        "total_cycles": len(cycles),
        "total_settings": sum(int(row["setting_count"]) for row in session_rows),
        "total_shots": sum(int(row["shot_count"]) for row in session_rows),
        "total_baseline_jobs": sum(int(row["baseline_job_count"]) for row in session_rows),
        "estimated_busy_seconds_total": sum(float(row["estimated_busy_seconds"]) for row in session_rows),
        "quota_seconds_total": (
            sum(float(row["quota_seconds"]) for row in session_rows)
            if role_envelope_model
            else None
        ),
        "execution_wall_seconds_total": (
            sum(float(row["execution_wall_seconds"]) for row in session_rows)
            if role_envelope_model
            else None
        ),
        "daily_budget_metric": "quota_seconds" if role_envelope_model else "estimated_busy_seconds",
        "programmed_hold_seconds_total": sum(float(row["programmed_hold_seconds"]) for row in session_rows),
        "daily_budget_passed": all(
            float(row["estimated_busy_seconds"]) <= float(daily_window_seconds) for row in session_rows
        ),
        # Retained under its historic name: the daily window has always been 1200 s and the
        # existing preflight and reports key off this field.
        "twenty_minute_daily_budget_passed": all(
            float(row["estimated_busy_seconds"]) <= float(daily_window_seconds) for row in session_rows
        ),
    }


def verify_stage1_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    for row in manifest["files"]:
        candidate = Path(str(row["path"]))
        target = candidate if candidate.is_absolute() else ROOT / candidate
        actual_bytes = target.stat().st_size if target.is_file() else None
        actual_hash = digest_file(target) if target.is_file() else None
        expected_hash = str(row["sha256"]).upper()
        if actual_bytes != int(row["bytes"]) or actual_hash != expected_hash:
            mismatches.append({
                "path": str(target),
                "expected_bytes": int(row["bytes"]),
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            })
    if mismatches:
        raise RuntimeError(f"Stage-1 curated manifest mismatch: {mismatches}")
    return {
        "path": str(path.resolve()),
        "sha256": digest_file(path),
        "verified_files": len(manifest["files"]),
        "status": "passed",
    }


def advance_ou(
    state: np.ndarray,
    elapsed_seconds: float,
    generator: np.random.Generator,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    controlled = config["controlled_ou"]
    tau = float(controlled["tau_seconds"])
    variance = float(controlled["stationary_process_variance"])
    clip = float(controlled["hard_clip_absolute"])
    elapsed = float(elapsed_seconds)
    if elapsed < 0.0:
        raise ValueError("OU virtual time cannot move backward")
    rho = math.exp(-elapsed / tau)
    innovation_sigma = math.sqrt(variance * (1.0 - rho * rho))
    innovation = generator.normal(0.0, innovation_sigma, len(state)) if elapsed > 0.0 else np.zeros_like(state)
    unconstrained = rho * state + innovation
    after = np.clip(unconstrained, -clip, clip)
    return after, {
        "elapsed_seconds": elapsed,
        "rho": rho,
        "innovation_sigma": innovation_sigma,
        "state_before": state.tolist(),
        "innovation": innovation.tolist(),
        "state_unclipped": unconstrained.tolist(),
        "state_after": after.tolist(),
        "clipped": [bool(abs(value) > clip) for value in unconstrained],
    }


REACHABILITY_REPLICATES = 6000
REACHABILITY_PERMUTATIONS = 600
# Three Monte Carlo standard errors at the replicate count above, near the 0.8 power bar.
# The check is a guard against a grossly unreachable design -- the superseded v2 plan
# claimed 0.910 and realised 0.462 -- not a precision estimate of the operating point, so
# it is stated with its sampling error rather than as a bare inequality.
REACHABILITY_ALLOWANCE = 0.016


def registered_endpoint_reachability(
    loop_config: Mapping[str, Any],
    correction: Mapping[str, Any],
) -> dict[str, Any]:
    """Can this configuration reach the rule it registers, at its own shot levels?

    Two adjudication rules are in play and they have different operating characteristics,
    so the check has to follow whichever the correction registers as primary.  The frozen
    delta-method gate is checked by ``simulate_b4_cadence_endpoint_power``; the within-pair
    permutation calibration is checked here against the shot floor the amortised baseline
    actually produces, because the four-setting floor the older module assumes is not the
    floor a two-setting cycle with a shared baseline has.
    """
    if str(correction.get("primary_adjudication", "")) != "cadence_ratio_permutation_gate":
        legacy = endpoint_power.evaluate_config(loop_config, replicates=2000, seed=20260815)
        binding = next(row for row in legacy["rows"] if row["pair_count"] == legacy["binding_pair_count"])
        return {
            **legacy,
            "rule": "cadence_ratio_gate",
            "binding_power": float(binding["power"]),
            "binding_boundary_size": float(binding["boundary_size"]),
        }
    floor = shared_baseline.endpoint_shot_floor(
        injected_shots_per_setting=int(correction["sensing_shots_per_setting"]),
        baseline_shots_per_setting=int(correction["baseline_shots_per_setting"]),
    )
    ou = loop_config["controlled_ou"]
    variance = float(ou["stationary_process_variance"])
    tau = float(ou["tau_seconds"])
    means = {
        label: floor["total_floor"]
        + endpoint_power.drift_endpoint_term(float(loop_config["cadence"][f"{label}_seconds"]), variance, tau)
        for label in ("fast", "slow")
    }
    for label, declared_key in (("fast", "expected_fast_endpoint_mean"), ("slow", "expected_slow_endpoint_mean")):
        declared = correction.get(declared_key)
        if declared is not None and not math.isclose(means[label], float(declared), rel_tol=1e-9, abs_tol=0.0):
            # A declared mean that does not follow from the config's own shots and OU
            # process is the failure mode this whole check exists to catch.
            raise ValueError(
                f"declared {declared_key} does not follow from the configured shot floor and cadence"
            )
    minimum_pairs = int(correction.get("minimum_adjudicated_cycle_pairs", correction["registered_cycle_pairs_total"]))
    minimum_power = float(correction.get("minimum_power", 0.8))
    maximum_size = float(correction.get("maximum_boundary_size", 0.05))
    power = permutation_claim_rate(
        pair_count=minimum_pairs,
        fast_mean=means["fast"],
        slow_mean=means["slow"],
        replicates=REACHABILITY_REPLICATES,
        permutations=REACHABILITY_PERMUTATIONS,
        seed=20260815,
    )
    size = permutation_claim_rate(
        pair_count=minimum_pairs,
        fast_mean=means["fast"],
        slow_mean=means["fast"],
        replicates=REACHABILITY_REPLICATES,
        permutations=REACHABILITY_PERMUTATIONS,
        seed=20260816,
    )
    return {
        "schema": "b4_cadence_permutation_reachability_v1",
        "rule": "cadence_ratio_permutation_gate",
        "registered_cycle_pairs_total": int(correction["registered_cycle_pairs_total"]),
        "minimum_adjudicated_cycle_pairs": minimum_pairs,
        "binding_pair_count": minimum_pairs,
        "replicates": REACHABILITY_REPLICATES,
        "permutations": REACHABILITY_PERMUTATIONS,
        "monte_carlo_allowance": REACHABILITY_ALLOWANCE,
        "endpoint_shot_floor": floor,
        "fast_endpoint_mean": means["fast"],
        "slow_endpoint_mean": means["slow"],
        "expected_ratio": means["fast"] / means["slow"],
        "minimum_power": minimum_power,
        "maximum_boundary_size": maximum_size,
        "binding_power": power,
        "binding_boundary_size": size,
        "power_pass": bool(power >= minimum_power - REACHABILITY_ALLOWANCE),
        "size_pass": bool(size <= maximum_size + REACHABILITY_ALLOWANCE),
        "reachable": bool(
            power >= minimum_power - REACHABILITY_ALLOWANCE
            and size <= maximum_size + REACHABILITY_ALLOWANCE
        ),
    }


def build_plan_payload(
    loop_config: Mapping[str, Any],
    backend_config: Mapping[str, Any],
    *,
    operational_start_utc: datetime,
    validate_registered_reachability: bool = True,
) -> dict[str, Any]:
    cadence.validate_config(loop_config)
    correction = validate_collection_correction(loop_config)
    backend_id = validate_backend_pin(loop_config, backend_config)
    generator = np.random.default_rng(int(loop_config["controlled_ou"]["rng_seed"]))
    state = np.zeros(2, dtype=np.float64)
    last_virtual_seconds = 0.0
    sessions: list[dict[str, Any]] = []
    all_cycles: list[dict[str, Any]] = []
    legacy_block_duration = float(loop_config["block_duration_seconds"])
    operational_spacing = (
        float(correction["operational_inter_session_seconds"])
        if correction is not None
        else float(loop_config["daily_window_seconds"])
    )
    virtual_spacing = float(loop_config["inter_day_seconds"])
    mirror_repetitions = (
        int(correction["mirror_repetitions_per_cycle"])
        if correction is not None
        else int(loop_config["mirror"]["repetitions_per_block"])
    )
    corrected_cycles_per_cadence = None if correction is None else int(correction["cycles_per_cadence_per_session"])
    corrected_sensing_shots = None if correction is None else int(correction["sensing_shots_per_setting"])
    mirror_cycles = mirror_qc_cycle_indices(correction)
    baseline_plan = shared_baseline_plan(correction)
    baseline_lead = 0.0 if baseline_plan is None else baseline_plan["lead_seconds"]
    schedule = (
        session_block_order(correction, loop_config)
        if correction is not None
        else [[str(value) for value in row] for row in loop_config["block_order_by_day"]]
    )
    ladder_amplitudes = [float(value) for value in loop_config["gate_amplitude_ladder"]["amplitudes"]]
    for session_index, order in enumerate(schedule):
        operational_session_start = operational_start_utc + timedelta(seconds=session_index * operational_spacing)
        virtual_session_start = session_index * virtual_spacing
        state, session_advance = advance_ou(
            state,
            virtual_session_start - last_virtual_seconds,
            generator,
            loop_config,
        )
        last_virtual_seconds = virtual_session_start
        session_cycles: list[dict[str, Any]] = []
        virtual_block_offset = 0.0
        # The session-start baseline job is submitted at the operational start and the first
        # cycle is held until the lead has elapsed, so a slow baseline queue cannot push a
        # cycle off its cadence tick.  The lead is wall clock only; it buys no shots.
        operational_block_offset = baseline_lead
        for block_index, cadence_label in enumerate(order):
            cadence_seconds = float(loop_config["cadence"][f"{cadence_label}_seconds"])
            cycle_count = (
                corrected_cycles_per_cadence
                if corrected_cycles_per_cadence is not None
                else int(math.floor(legacy_block_duration / cadence_seconds + 1e-12))
            )
            block_duration = cycle_count * cadence_seconds if correction is not None else legacy_block_duration
            virtual_block_start = virtual_session_start + virtual_block_offset
            operational_block_start = operational_session_start + timedelta(seconds=operational_block_offset)
            state, block_advance = advance_ou(
                state,
                virtual_block_start - last_virtual_seconds,
                generator,
                loop_config,
            )
            last_virtual_seconds = virtual_block_start
            for cycle_index in range(cycle_count):
                virtual_sense = virtual_block_start + cycle_index * cadence_seconds
                state, sense_advance = advance_ou(
                    state,
                    virtual_sense - last_virtual_seconds,
                    generator,
                    loop_config,
                )
                last_virtual_seconds = virtual_sense
                sense_fields = state.copy()
                virtual_mirror = virtual_sense + cadence_seconds
                state, mirror_advance = advance_ou(
                    state,
                    virtual_mirror - last_virtual_seconds,
                    generator,
                    loop_config,
                )
                last_virtual_seconds = virtual_mirror
                registered_pair_id = (
                    f"session{session_index:02d}-cyclepair{cycle_index:02d}"
                    if correction is not None
                    else None
                )
                if correction is not None:
                    carries_mirror = mirror_cycles is None or cycle_index in mirror_cycles
                    mirror_seeds = [
                        int(loop_config["mirror"]["rng_seed"]) + session_index * 1000 + cycle_index * 10 + replicate
                        for replicate in range(mirror_repetitions)
                    ] if carries_mirror else []
                else:
                    mirror_seeds = [
                        int(loop_config["mirror"]["rng_seed"]) + session_index * 1000 + replicate
                        for replicate in range(mirror_repetitions)
                    ]
                cycle = {
                    "cycle_id": f"session{session_index:02d}-block{block_index:02d}-cycle{cycle_index:02d}",
                    "session_index": session_index,
                    "block_index": block_index,
                    "cadence": str(cadence_label),
                    "cadence_seconds": cadence_seconds,
                    "cycle_index": cycle_index,
                    "registered_pair_id": registered_pair_id,
                    "virtual_sense_seconds": virtual_sense,
                    "virtual_mirror_seconds": virtual_mirror,
                    "sense_target_utc": iso(operational_block_start + timedelta(seconds=cycle_index * cadence_seconds)),
                    "mirror_target_utc": iso(operational_block_start + timedelta(seconds=(cycle_index + 1) * cadence_seconds)),
                    "sense_fields": sense_fields.tolist(),
                    "mirror_fields": state.tolist(),
                    "sense_ou_advance": sense_advance,
                    "mirror_ou_advance": mirror_advance,
                    "sensing_shots_per_setting": (
                        corrected_sensing_shots
                        if corrected_sensing_shots is not None
                        else cadence.shots_per_setting(loop_config, cadence_seconds)
                    ),
                    "mirror_shots_per_task": int(loop_config["mirror"]["shots_per_task"]),
                    "mirror_seeds": mirror_seeds,
                }
                session_cycles.append(cycle)
                all_cycles.append(cycle)
            virtual_block_end = virtual_block_start + block_duration
            state, block_end_advance = advance_ou(
                state,
                virtual_block_end - last_virtual_seconds,
                generator,
                loop_config,
            )
            last_virtual_seconds = virtual_block_end
            session_cycles[-1]["block_start_ou_advance"] = block_advance
            session_cycles[-1]["block_end_ou_advance"] = block_end_advance
            virtual_block_offset += block_duration
            operational_block_offset += block_duration
        sessions.append({
            "session_index": session_index,
            "block_order": [str(value) for value in order],
            "operational_start_utc": iso(operational_session_start),
            "operational_deadline_utc": (
                iso(
                    operational_session_start
                    + timedelta(seconds=float(correction["operational_session_wallclock_seconds"]))
                )
                if correction is not None
                else None
            ),
            "operational_completion_deadline_utc": (
                iso(
                    operational_session_start
                    + timedelta(seconds=float(correction["operational_session_completion_window_seconds"]))
                )
                if correction is not None
                else None
            ),
            "virtual_start_seconds": virtual_session_start,
            "session_start_ou_advance": session_advance,
            # The three-rung ladder is a shield-behaviour contract, not a per-session
            # treatment, and it is never submitted to hardware; it cycles when the
            # corrected collection runs more sessions than the ladder has rungs.
            "gate_amplitude": ladder_amplitudes[session_index % len(ladder_amplitudes)],
            "programmed_session_wallclock_seconds": operational_block_offset,
            "session_deadline_slack_seconds": (
                None
                if correction is None
                else float(correction["operational_session_wallclock_seconds"]) - operational_block_offset
            ),
            "baseline_measurements": (
                []
                if baseline_plan is None
                else [
                    {
                        "position": position,
                        "session_index": session_index,
                        "measurement_id": f"session{session_index:02d}-baseline-{position}",
                        "target_utc": iso(
                            operational_session_start
                            + timedelta(seconds=0.0 if position == "session_start" else operational_block_offset)
                        ),
                        "settings": baseline_plan["settings_per_measurement"],
                        "shots_per_setting": baseline_plan["shots_by_position"][position],
                        "used_by_estimate": position == "session_start",
                        "role": (
                            "shared zero-field reference subtracted by every cycle of both cadence blocks"
                            if position == "session_start"
                            else "drift readout only; it does not exist when the cycles run and never enters an estimate"
                        ),
                    }
                    for position in baseline_plan["positions"]
                ]
            ),
            "cycles": session_cycles,
        })
        if correction is not None:
            programmed_span = float(correction.get("operational_session_programmed_span_seconds", operational_block_offset - baseline_lead))
            if not math.isclose(operational_block_offset - baseline_lead, programmed_span, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("fixed cycle-count blocks do not match the registered programmed session span")
            slack = float(correction["operational_session_wallclock_seconds"]) - operational_block_offset
            if slack < 0.0:
                raise ValueError("the sensing-loop deadline precedes the last programmed cycle")
            declared_slack = correction.get("operational_session_slack_seconds")
            if declared_slack is not None and not math.isclose(slack, float(declared_slack), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("realized session deadline slack disagrees with the registered slack")
    expected_loop_tasks = len(all_cycles) * sensing_settings_per_cycle(correction)
    mirror_cycle_rows = [row for row in all_cycles if row["mirror_seeds"]]
    expected_mirror_tasks = sum(2 * len(row["mirror_seeds"]) for row in all_cycles)
    expected_baseline_jobs = 0 if baseline_plan is None else len(sessions) * len(baseline_plan["positions"])
    expected_baseline_tasks = (
        0 if baseline_plan is None else expected_baseline_jobs * baseline_plan["settings_per_measurement"]
    )
    expected_complete_pairs = (
        int(correction["registered_cycle_pairs_total"])
        if correction is not None
        else int(loop_config["simulation_days"]) * mirror_repetitions
    )
    daily_window = float(loop_config["daily_window_seconds"])
    budget = measured_collection_budget(
        all_cycles, backend_config, correction, daily_window_seconds=daily_window
    )
    reachability = None
    if correction is not None:
        if not bool(budget["daily_budget_passed"]):
            raise ValueError(
                f"corrected collection exceeds the {daily_window:.0f}-second daily busy-time budget"
            )
        if int(budget["total_shots"]) > int(loop_config["shield"]["shot_budget_cap"]):
            raise ValueError("corrected collection exceeds the frozen shot budget cap")
        if (
            correction.get("status") != LEGACY_UNREACHABLE_CORRECTION_STATUS
            and validate_registered_reachability
        ):
            # Para 286: a gate is only registered once its own configuration has been shown
            # to be able to reach it.  The shots per setting fix the estimator noise floor,
            # the floor is common to both cadences, and the frozen ratio gate is not
            # additive-offset invariant, so this has to be measured, not assumed.
            reachability = registered_endpoint_reachability(loop_config, correction)
            if not bool(reachability["reachable"]):
                raise ValueError(
                    "collection is not registered-endpoint reachable at the configured shots per setting: "
                    f"expected ratio {reachability['expected_ratio']:.4f} gives "
                    f"power {reachability['binding_power']:.3f} and boundary size "
                    f"{reachability['binding_boundary_size']:.3f} at "
                    f"{reachability['binding_pair_count']} pairs"
                )
        elif correction.get("status") != LEGACY_UNREACHABLE_CORRECTION_STATUS:
            reachability = {
                "validation_performed": False,
                "reason": "caller requested timing-only plan construction without statistical recomputation",
            }
    return {
        "schema": "b4_cadence_pair_hardware_supplement_plan_v1",
        "created_at_utc": iso(utc_now()),
        "backend_id": backend_id,
        "operational_start_utc": iso(operational_start_utc),
        "calendar_execution_amendment": {
            "reason": (
                "endpoint-identity collection correction with cycle-count blocks"
                if correction is not None
                else "user-authorized session compression after measured runtime corrected the original three-day capacity estimate"
            ),
            "frozen_virtual_inter_session_seconds": virtual_spacing,
            "operational_inter_session_seconds": operational_spacing,
            "statistical_session_index_and_alternating_order_preserved": True,
            "controlled_ou_virtual_spacing_preserved": True,
        },
        "collection_correction": None if correction is None else dict(correction),
        "measured_collection_budget": budget,
        "registered_endpoint_reachability": reachability,
        "source_hashes": {
            "cadence_module_sha256": digest_file(ROOT / "scripts" / "run_cadence_pair_loop.py"),
            "hardware_smoke_module_sha256": digest_file(ROOT / "scripts" / "run_cadence_pair_hardware_smoke.py"),
        },
        "expected": {
            "sessions": len(sessions),
            "blocks": 2 * len(sessions),
            "cycles": len(all_cycles),
            "loop_jobs": len(all_cycles),
            "mirror_jobs": len(mirror_cycle_rows),
            "baseline_jobs": expected_baseline_jobs,
            "loop_tasks": expected_loop_tasks,
            "mirror_tasks": expected_mirror_tasks,
            "baseline_tasks": expected_baseline_tasks,
            "total_tasks": expected_loop_tasks + expected_mirror_tasks + expected_baseline_tasks,
            "complete_cadence_pairs": expected_complete_pairs,
            **(
                {}
                if correction is None
                else {
                    "cycles_per_cadence_per_session": corrected_cycles_per_cadence,
                    "sensing_settings_per_cycle": sensing_settings_per_cycle(correction),
                    "mirror_repetitions_per_cycle": mirror_repetitions,
                    "mirror_qc_cycle_indices": mirror_cycles,
                    "minimum_adjudicated_cycle_pairs": int(
                        correction.get("minimum_adjudicated_cycle_pairs", expected_complete_pairs)
                    ),
                    "minimum_sessions_per_block_order": int(correction.get("minimum_sessions_per_block_order", 0)),
                    "pair_discard_granularity": str(
                        correction.get("pair_discard_granularity", "session_pair_block")
                    ),
                    "pair_block_discard_condition": str(
                        correction.get(
                            "pair_block_discard_condition",
                            "either cadence block incomplete",
                        )
                    ),
                }
            ),
        },
        "sessions": sessions,
    }


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign.json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_plan(
    loop_config_path: Path,
    backend_config_path: Path,
    peer_config_path: Path,
    stage1_manifest_path: Path,
    output: Path,
    *,
    start_utc: datetime,
) -> dict[str, Any]:
    verify_stage1_manifest(stage1_manifest_path)
    drift_campaign_v4.prepare_v4(backend_config_path, peer_config_path, output)
    loop_config = cadence.load_config(loop_config_path)
    backend_config = drift_campaign_v4.load_config(backend_config_path)
    backend_id = validate_backend_pin(loop_config, backend_config)
    plan_path = output / PLAN_NAME
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        source_hashes = plan.get("source_hashes", {})
        if source_hashes.get("loop_config_sha256") != digest_file(loop_config_path):
            raise RuntimeError("existing supplement plan loop-config hash changed")
        if source_hashes.get("backend_config_sha256") != digest_file(backend_config_path):
            raise RuntimeError("existing supplement plan backend-config hash changed")
        if plan.get("backend_id") != backend_id:
            raise RuntimeError("existing supplement plan backend id changed")
        return plan
    plan = build_plan_payload(loop_config, backend_config, operational_start_utc=start_utc)
    plan["source_hashes"] = {
        **plan["source_hashes"],
        "loop_config_sha256": digest_file(loop_config_path),
        "backend_config_sha256": digest_file(backend_config_path),
        "peer_config_sha256": digest_file(peer_config_path),
        "stage1_manifest_sha256": digest_file(stage1_manifest_path),
        "runner_sha256": digest_file(Path(__file__).resolve()),
    }
    write_new_json(plan_path, plan)
    return plan


def wait_until(target_utc: str, *, sleeper: Callable[[float], None] = time.sleep) -> float:
    target = parse_utc(target_utc)
    remaining = (target - utc_now()).total_seconds()
    while remaining > 0.0:
        sleeper(min(remaining, 30.0))
        remaining = (target - utc_now()).total_seconds()
    return max(0.0, (utc_now() - target).total_seconds())


def require_session_open(
    deadline_utc: str | None,
    *,
    stage: str,
    now_utc: datetime | None = None,
) -> None:
    if deadline_utc is None:
        return
    now = utc_now() if now_utc is None else now_utc.astimezone(timezone.utc)
    deadline = parse_utc(deadline_utc)
    if now > deadline:
        raise SessionPairBlockExpired(
            f"same-session deadline expired before {stage}: now={iso(now)}, deadline={iso(deadline)}"
        )


def role_snapshot_id(config: Mapping[str, Any], role: str, target_utc: str) -> str:
    return campaign.digest_payload({
        "campaign_id": config["campaign_id"],
        "backend_id": config["backend"]["backend_id"],
        "job_role": role,
        "planned_target_utc": campaign.iso(campaign.parse_utc(target_utc)),
        "burst_flag": False,
    })[:20].lower()


def safe_task_metadata(programs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **{key: value for key, value in row.items() if key != "qcis"},
            "qcis_sha256": sha256(str(row["qcis"]).encode("utf-8")).hexdigest().upper(),
        }
        for row in programs
    ]


def submit_job(
    *,
    platform: Any,
    config: Mapping[str, Any],
    store: campaign.CampaignStore,
    programs: Sequence[Mapping[str, Any]],
    role: str,
    target_utc: str,
    shots: int,
    cycle_id: str,
    submission_budget: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    identifier = role_snapshot_id(config, role, target_utc)
    submitted = store.latest(identifier, "submitted")
    if submitted is not None:
        return submitted
    if (
        submission_budget is not None
        and submission_budget["limit"] is not None
        and int(submission_budget["used"] or 0) >= int(submission_budget["limit"])
    ):
        raise SubmissionLimitReached("new job submission limit reached")
    prior_started = store.latest(identifier, "submission_started")
    prior_rejected = store.latest(identifier, "submission_rejected")
    retry_allowed = bool(
        prior_rejected
        and (
            prior_rejected.get("known_no_submission")
            or any(
                token in str(prior_rejected.get("error", ""))
                for token in ("Insufficient remaining computing time", "additional machine time")
            )
        )
    )
    if prior_started is not None and not retry_allowed:
        raise RuntimeError(f"uncertain prior submission; refusing resubmit: {identifier}")
    try:
        download_config = campaign.json_ready(platform.download_config(machine=str(config["backend"]["backend_id"])))
        download_error = None
    except Exception as error:
        download_config = None
        download_error = f"{type(error).__name__}: {error}"
    telemetry = {
        "backend_id": config["backend"]["backend_id"],
        "captured_at_utc": campaign.iso(),
        "raw": {"download_config": download_config},
        "errors": {"download_config": download_error},
        "available": download_config is not None,
        "execution_time_available": False,
    }
    previous_regime = next((row for row in reversed(store.records) if row.get("event") == "calibration_regime"), None)
    transition = drift_campaign_v4.regime_transition(previous_regime, drift_campaign_v4.calibration_time_raw(telemetry))
    if previous_regime is None:
        transition["regime_id"] = "regime-0000"
    store.append("calibration_regime", {
        "snapshot_id": identifier,
        "backend_id": config["backend"]["backend_id"],
        **transition,
        "telemetry": telemetry,
        "job_role": role,
        "burst_flag": bool(transition["flipped"]),
        "cycle_id": cycle_id,
    })
    tasks = safe_task_metadata(programs)
    attempt_index = sum(
        row.get("event") == "submission_started"
        for row in store.by_snapshot(identifier)
    )
    store.append("submission_started", {
        "snapshot_id": identifier,
        "backend_id": config["backend"]["backend_id"],
        "job_role": role,
        "planned_target_utc": target_utc,
        "cycle_id": cycle_id,
        "settings": len(programs),
        "shots_per_setting": int(shots),
        "total_shots": len(programs) * int(shots),
        "tasks": tasks,
        "attempt_index": attempt_index,
    })
    try:
        query_ids = platform.submit_experiment(
            circuit=[str(row["qcis"]) for row in programs],
            name=f"XA202609_B4_{role.upper()}_{identifier[:12]}",
            num_shots=int(shots),
            machine_name=str(config["backend"]["backend_id"]),
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        known_no_submission = any(
            token in message
            for token in ("正在校准中", "The quantum computer is Calibrating", "temporarily unavailable")
        )
        known_no_submission = known_no_submission or any(
            token in message for token in ("Insufficient remaining computing time", "additional machine time")
        )
        store.append("submission_rejected", {
            "snapshot_id": identifier,
            "backend_id": config["backend"]["backend_id"],
            "job_role": role,
            "cycle_id": cycle_id,
            "planned_target_utc": target_utc,
            "attempt_index": attempt_index,
            "known_no_submission": known_no_submission,
            "error": message,
        })
        raise
    if not isinstance(query_ids, list) or len(query_ids) != len(programs):
        raise RuntimeError("platform returned incomplete task-ID list")
    row = store.append("submitted", {
        "snapshot_id": identifier,
        "backend_id": config["backend"]["backend_id"],
        "job_role": role,
        "planned_target_utc": target_utc,
        "wallclock_submit_utc": campaign.iso(),
        "cycle_id": cycle_id,
        "regime_id": transition["regime_id"],
        "burst_flag": bool(transition["flipped"]),
        "settings": len(programs),
        "shots_per_setting": int(shots),
        "total_shots": len(programs) * int(shots),
        "tasks": [
            {"query_id": str(query_id), **task}
            for query_id, task in zip(query_ids, tasks, strict=True)
        ],
    })
    if submission_budget is not None:
        submission_budget["used"] = int(submission_budget["used"] or 0) + 1
    return row


def collect_job(
    *,
    platform: Any,
    config: Mapping[str, Any],
    store: campaign.CampaignStore,
    output: Path,
    submitted: Mapping[str, Any],
    max_wait_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    identifier = str(submitted["snapshot_id"])
    existing = store.latest(identifier, "collected")
    if existing is not None:
        return existing
    query_ids = [str(row["query_id"]) for row in submitted["tasks"]]
    results = platform.query_experiment(query_ids, max_wait_time=max_wait_seconds, sleep_time=poll_seconds)
    if not isinstance(results, list):
        raise RuntimeError("platform query did not return a result list")
    by_id = {
        str(row["experimentTaskId"]): row
        for row in results
        if isinstance(row, Mapping) and row.get("experimentTaskId") is not None
    }
    missing = [query_id for query_id in query_ids if query_id not in by_id]
    if missing:
        store.append("partial", {
            "snapshot_id": identifier,
            "backend_id": config["backend"]["backend_id"],
            "job_role": submitted["job_role"],
            "cycle_id": submitted["cycle_id"],
            "missing_query_ids": missing,
        })
        raise RuntimeError(f"platform result missing {len(missing)} task IDs")
    raw_path = output / "raw" / f"{identifier}_query.json"
    if not raw_path.exists():
        campaign.write_json_new(raw_path, {
            "schema": "b4_cadence_hardware_raw_query_v1",
            "snapshot_id": identifier,
            "job_role": submitted["job_role"],
            "cycle_id": submitted["cycle_id"],
            "requested_query_ids": query_ids,
            "results": campaign.json_ready(results),
        })
    physical = list(config["backend"]["physical_qubits"])
    shots = int(submitted["shots_per_setting"])
    counts = [campaign.result_counts(by_id[str(task["query_id"])], physical, shots) for task in submitted["tasks"]]
    counts_path = output / "raw" / f"{identifier}_counts.npz"
    if not counts_path.exists():
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            counts_path,
            labels=np.asarray([task["label"] for task in submitted["tasks"]]),
            strategies=np.asarray([str(task.get("strategy", "")) for task in submitted["tasks"]]),
            pair_ids=np.asarray([str(task.get("pair_id", "")) for task in submitted["tasks"]]),
            ideal_bitstrings=np.asarray([str(task.get("ideal_bitstring", "")) for task in submitted["tasks"]]),
            counts=np.stack(counts),
            shots=np.asarray(shots),
        )
    return store.append("collected", {
        "snapshot_id": identifier,
        "backend_id": config["backend"]["backend_id"],
        "job_role": submitted["job_role"],
        "cycle_id": submitted["cycle_id"],
        "query_ids": query_ids,
        "raw_results_path": str(raw_path),
        "raw_results_sha256": digest_file(raw_path),
        "counts_path": str(counts_path),
        "counts_sha256": digest_file(counts_path),
    })


def estimate_fields_from_counts(counts: np.ndarray, shots: int, phase_time_seconds: float) -> dict[str, Any]:
    if counts.shape != (4, 64):
        raise ValueError("cadence sensing counts must be 4x64")
    estimates: list[float] = []
    sigmas: list[float] = []
    fields: list[dict[str, Any]] = []
    for field_index in range(2):
        phases: list[float] = []
        phase_variances: list[float] = []
        condition_rows: list[dict[str, float]] = []
        for offset in (0, 2):
            observed_y = hardware.single_qubit_expectation(counts[offset], field_index, shots)
            observed_z = hardware.single_qubit_expectation(counts[offset + 1], field_index, shots)
            radius_squared = max(observed_y * observed_y + observed_z * observed_z, 1e-12)
            phase = math.atan2(-observed_y, observed_z)
            variance_y = max((1.0 - observed_y * observed_y) / shots, 1.0 / (shots * shots))
            variance_z = max((1.0 - observed_z * observed_z) / shots, 1.0 / (shots * shots))
            derivative_y = -observed_z / radius_squared
            derivative_z = observed_y / radius_squared
            phase_variance = max(derivative_y * derivative_y * variance_y + derivative_z * derivative_z * variance_z, 1e-15)
            phases.append(phase)
            phase_variances.append(phase_variance)
            condition_rows.append({
                "observed_y": observed_y,
                "observed_z": observed_z,
                "phase": phase,
                "phase_variance": phase_variance,
            })
        phase_difference = math.atan2(math.sin(phases[1] - phases[0]), math.cos(phases[1] - phases[0]))
        estimate = phase_difference / (2.0 * phase_time_seconds)
        sigma = math.sqrt(sum(phase_variances)) / (2.0 * phase_time_seconds)
        estimates.append(estimate)
        sigmas.append(sigma)
        fields.append({
            "field": f"h{field_index + 1}",
            "baseline": condition_rows[0],
            "injected": condition_rows[1],
            "wrapped_phase_difference": phase_difference,
            "estimate": estimate,
            "shot_sigma": sigma,
        })
    return {
        "estimator": "baseline_differential_two_axis_atan2_effective_field",
        "estimates": estimates,
        "shot_sigmas": sigmas,
        "fields": fields,
        "shots_used_total": 4 * shots,
    }


def loop_programs(
    physical_qubits: Sequence[int],
    fields: Sequence[float],
    phase_time: float,
    cycle: Mapping[str, Any],
    *,
    conditions: Sequence[str] = ("baseline", "injected"),
) -> list[dict[str, Any]]:
    rows = hardware.build_sensing_programs(physical_qubits, fields, phase_time, conditions=conditions)
    return [
        {
            **row,
            "label": f"{cycle['cycle_id']}_{row['condition']}_{row['axis']}",
            "kind": "cadence_sense",
            "analysis_role": "cadence_sense_estimator_input",
        }
        for row in rows
    ]


def baseline_programs(
    physical_qubits: Sequence[int],
    phase_time: float,
    measurement: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """The session's shared zero-field measurement: the baseline half of a sensing cycle."""
    rows = hardware.build_sensing_programs(
        physical_qubits, (0.0, 0.0), phase_time, conditions=("baseline",)
    )
    return [
        {
            **row,
            "label": f"{measurement['measurement_id']}_{row['axis']}",
            "kind": "session_shared_baseline",
            "analysis_role": (
                "session_shared_baseline_estimator_input"
                if bool(measurement["used_by_estimate"])
                else "session_baseline_drift_readout"
            ),
        }
        for row in rows
    ]


def axis_expectations(counts: np.ndarray, shots: int) -> list[dict[str, float]]:
    """A two-setting job's raw counts, Y then Z, as one expectation pair per field."""
    if counts.shape != (2, 64):
        raise ValueError("a two-setting sensing job must return two six-qubit count rows")
    return [
        {
            "observed_y": hardware.single_qubit_expectation(counts[0], field_index, shots),
            "observed_z": hardware.single_qubit_expectation(counts[1], field_index, shots),
        }
        for field_index in range(2)
    ]


def mirror_programs(
    physical_qubits: Sequence[int],
    fields: Sequence[float],
    compensation: Sequence[float],
    phase_time: float,
    depth: int,
    cycle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, seed in enumerate(cycle["mirror_seeds"]):
        registered_pair_id = cycle.get("registered_pair_id")
        pair_id = (
            f"{registered_pair_id}-mirror{replicate:02d}"
            if registered_pair_id
            else f"session{int(cycle['session_index']):02d}-mirror{replicate:02d}"
        )
        for row in hardware.build_mirror_pair(
            physical_qubits,
            depth=depth,
            seed=int(seed),
            injected_fields=fields,
            compensation=compensation,
            phase_time_seconds=phase_time,
        ):
            rows.append({
                **row,
                "label": f"{cycle['cycle_id']}_{pair_id}_{row['strategy']}",
                "kind": "matched_mirror",
                "analysis_role": "cadence_primary_raw_success_probability",
                "pair_id": pair_id,
                "replicate": replicate,
                "time_window_id": cycle["cycle_id"],
                "task_family": "random_native_clifford_mirror_v1",
            })
    return rows


def load_counts(collected: Mapping[str, Any]) -> np.ndarray:
    with np.load(str(collected["counts_path"]), allow_pickle=False) as blob:
        return np.asarray(blob["counts"], dtype=np.int64)


def score_mirror(
    counts: np.ndarray,
    programs: Sequence[Mapping[str, Any]],
    shots: int,
    cycle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(counts) != len(programs):
        raise ValueError("mirror counts and program metadata differ")
    rows: list[dict[str, Any]] = []
    width = int(round(math.log2(counts.shape[1])))
    for vector, program in zip(counts, programs, strict=True):
        raw = {
            format(index, f"0{width}b"): int(value)
            for index, value in enumerate(vector)
            if int(value) > 0
        }
        score = success_probability_from_raw_counts(raw, str(program["ideal_bitstring"]), shots=shots)
        rows.append({
            "cycle_id": cycle["cycle_id"],
            "session_index": cycle["session_index"],
            "cadence": cycle["cadence"],
            "pair_id": program["pair_id"],
            "replicate": program["replicate"],
            "strategy": program["strategy"],
            "seed": program["seed"],
            "depth": program["depth"],
            "ideal_bitstring": program["ideal_bitstring"],
            **score,
        })
    return rows


def run_baseline_measurement(
    *,
    platform: Any,
    config: Mapping[str, Any],
    loop_config: Mapping[str, Any],
    store: campaign.CampaignStore,
    output: Path,
    measurement: Mapping[str, Any],
    max_wait_seconds: int,
    poll_seconds: int,
    deadline_utc: str | None = None,
    submission_budget: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """One session-shared zero-field measurement, submitted outside the cadence grid."""
    completed = next(
        (
            row
            for row in reversed(store.records)
            if row.get("event") == "session_baseline_measured"
            and row.get("measurement_id") == measurement["measurement_id"]
        ),
        None,
    )
    if completed is not None:
        return completed
    physical = list(config["backend"]["physical_qubits"])
    phase_time = float(loop_config["sensing"]["phase_time_seconds"])
    lateness = wait_until(str(measurement["target_utc"]))
    require_session_open(deadline_utc, stage=f"baseline submission for {measurement['measurement_id']}")
    programs = baseline_programs(physical, phase_time, measurement)
    shots = int(measurement["shots_per_setting"])
    submitted = submit_job(
        platform=platform,
        config=config,
        store=store,
        programs=programs,
        role="baseline",
        target_utc=str(measurement["target_utc"]),
        shots=shots,
        cycle_id=str(measurement["measurement_id"]),
        submission_budget=submission_budget,
    )
    collected = collect_job(
        platform=platform,
        config=config,
        store=store,
        output=output,
        submitted=submitted,
        max_wait_seconds=max_wait_seconds,
        poll_seconds=poll_seconds,
    )
    record = shared_baseline.baseline_record(
        axis_expectations(load_counts(collected), shots),
        shots,
        session_index=int(measurement["session_index"]),
        position=str(measurement["position"]),
    )
    row = store.append("session_baseline_measured", {
        "measurement_id": measurement["measurement_id"],
        "session_index": int(measurement["session_index"]),
        "position": str(measurement["position"]),
        "target_utc": measurement["target_utc"],
        "lateness_seconds": lateness,
        "shots_per_setting": shots,
        "used_by_estimate": bool(measurement["used_by_estimate"]),
        "role": measurement["role"],
        "snapshot_id": submitted["snapshot_id"],
        "baseline": record,
    })
    print(canonical_json({
        "event": "session_baseline_measured",
        "measurement_id": measurement["measurement_id"],
        "position": measurement["position"],
        "used_by_estimate": bool(measurement["used_by_estimate"]),
    }), flush=True)
    return row


def run_cycle(
    *,
    platform: Any,
    config: Mapping[str, Any],
    loop_config: Mapping[str, Any],
    store: campaign.CampaignStore,
    output: Path,
    cycle: Mapping[str, Any],
    max_wait_seconds: int,
    poll_seconds: int,
    session_deadline_utc: str | None = None,
    session_completion_deadline_utc: str | None = None,
    submission_budget: dict[str, int | None] | None = None,
    session_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completed = next(
        (
            row
            for row in reversed(store.records)
            if row.get("event") == "cadence_cycle_completed" and row.get("cycle_id") == cycle["cycle_id"]
        ),
        None,
    )
    if completed is not None:
        return completed
    physical = list(config["backend"]["physical_qubits"])
    phase_time = float(loop_config["sensing"]["phase_time_seconds"])
    sense_lateness = wait_until(str(cycle["sense_target_utc"]))
    require_session_open(session_deadline_utc, stage=f"loop submission for {cycle['cycle_id']}")
    conditions = ("injected",) if session_baseline is not None else ("baseline", "injected")
    sensing_programs = loop_programs(
        physical, cycle["sense_fields"], phase_time, cycle, conditions=conditions
    )
    loop_submitted = submit_job(
        platform=platform,
        config=config,
        store=store,
        programs=sensing_programs,
        role="loop",
        target_utc=str(cycle["sense_target_utc"]),
        shots=int(cycle["sensing_shots_per_setting"]),
        cycle_id=str(cycle["cycle_id"]),
        submission_budget=submission_budget,
    )
    loop_collected = collect_job(
        platform=platform,
        config=config,
        store=store,
        output=output,
        submitted=loop_submitted,
        max_wait_seconds=max_wait_seconds,
        poll_seconds=poll_seconds,
    )
    shots_per_setting = int(cycle["sensing_shots_per_setting"])
    if session_baseline is None:
        sensed = estimate_fields_from_counts(load_counts(loop_collected), shots_per_setting, phase_time)
    else:
        # Same arithmetic as the four-setting path -- wrapped phase difference over 2T with
        # the two conditions' variances added -- with the zero-field condition coming from
        # the session's opening measurement instead of from this cycle's own shots.  The
        # closing measurement does not exist yet and never enters here.
        sensed = shared_baseline.differential_estimate(
            axis_expectations(load_counts(loop_collected), shots_per_setting),
            shots_per_setting,
            session_baseline,
            phase_time,
        )
    observable = cadence.observable_state(sensed, loop_config)
    decision = shield(observable, Action(1.0, 2, "act"), cadence.shield_config(loop_config))
    compensation = list(decision.compensation[:2]) if decision.permitted else [0.0, 0.0]
    # The cadence interval is always held, whether or not this cycle carries the
    # descriptive mirror QC job, so the registered pairing timing is unchanged.
    mirror_lateness = wait_until(str(cycle["mirror_target_utc"]))
    if cycle["mirror_seeds"]:
        require_session_open(
            session_completion_deadline_utc,
            stage=f"mirror QC submission for {cycle['cycle_id']}",
        )
        matched_programs = mirror_programs(
            physical,
            cycle["mirror_fields"],
            compensation,
            phase_time,
            int(loop_config["mirror"]["depth"]),
            cycle,
        )
        mirror_submitted = submit_job(
            platform=platform,
            config=config,
            store=store,
            programs=matched_programs,
            role="mirror",
            target_utc=str(cycle["mirror_target_utc"]),
            shots=int(cycle["mirror_shots_per_task"]),
            cycle_id=str(cycle["cycle_id"]),
            submission_budget=submission_budget,
        )
        mirror_collected = collect_job(
            platform=platform,
            config=config,
            store=store,
            output=output,
            submitted=mirror_submitted,
            max_wait_seconds=max_wait_seconds,
            poll_seconds=poll_seconds,
        )
        scores = score_mirror(
            load_counts(mirror_collected),
            matched_programs,
            int(cycle["mirror_shots_per_task"]),
            cycle,
        )
        mirror_snapshot_id = mirror_submitted["snapshot_id"]
    else:
        scores = []
        mirror_snapshot_id = None
    row = store.append("cadence_cycle_completed", {
        "cycle_id": cycle["cycle_id"],
        "session_index": cycle["session_index"],
        "block_index": cycle["block_index"],
        "cadence": cycle["cadence"],
        "cadence_seconds": cycle["cadence_seconds"],
        "cycle_index": cycle.get("cycle_index"),
        "registered_pair_id": cycle.get("registered_pair_id"),
        "sense_target_utc": cycle["sense_target_utc"],
        "mirror_target_utc": cycle["mirror_target_utc"],
        "sense_lateness_seconds": sense_lateness,
        "mirror_lateness_seconds": mirror_lateness,
        "sense_fields": cycle["sense_fields"],
        "mirror_fields": cycle["mirror_fields"],
        "sensing": sensed,
        "sensing_settings": len(sensing_programs),
        "session_baseline_position": None if session_baseline is None else str(session_baseline["position"]),
        "observable_shield_state": observable,
        "shield": {
            "permitted": decision.permitted,
            "gate": decision.gate,
            "reason": decision.reason,
            "action": asdict(decision.action),
            "compensation": list(decision.compensation),
        },
        "loop_snapshot_id": loop_submitted["snapshot_id"],
        "mirror_snapshot_id": mirror_snapshot_id,
        "mirror_carried": bool(cycle["mirror_seeds"]),
        "mirror_scores": scores,
        "event_order": ["sense", "five_gate_shield", "digital_inverse_compensation", "matched_mirror_pair"],
    })
    print(canonical_json({
        "event": "cycle_completed",
        "cycle_id": cycle["cycle_id"],
        "cadence": cycle["cadence"],
        "completed_cycles": sum(item.get("event") == "cadence_cycle_completed" for item in store.records),
    }), flush=True)
    return row


def registered_pair_block_status(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_per_cadence = plan["expected"].get("cycles_per_cadence_per_session")
    if expected_per_cadence is None:
        return {
            "policy": "legacy mirror-replicate aggregation; no registered per-cycle block adjudication",
            "discard_granularity": "not_applicable",
            "expected_cycles_per_cadence_per_session": None,
            "complete_session_pair_blocks": [],
            "discarded_session_pair_blocks": [],
            "registered_cycle_pairs": 0,
        }
    expected_count = int(expected_per_cadence)
    granularity = str(plan["expected"].get("pair_discard_granularity", "session_pair_block"))
    completed_ids = {
        str(row["cycle_id"])
        for row in records
        if row.get("event") == "cadence_cycle_completed"
    }
    forced_discard = {
        int(row["session_index"]): dict(row)
        for row in records
        if row.get("event") == "session_pair_block_discarded"
    }
    measured_baselines = {
        (int(row["session_index"]), str(row["position"]))
        for row in records
        if row.get("event") == "session_baseline_measured"
    }
    complete_sessions: list[int] = []
    discarded_sessions: list[dict[str, Any]] = []
    complete_pairs: list[str] = []
    discarded_pairs: list[dict[str, Any]] = []
    for session in plan["sessions"]:
        session_index = int(session["session_index"])
        counts: dict[str, int] = {}
        for cadence_label in ("fast", "slow"):
            planned = [
                str(row["cycle_id"])
                for row in session["cycles"]
                if str(row["cadence"]) == cadence_label
            ]
            counts[cadence_label] = sum(cycle_id in completed_ids for cycle_id in planned)

        # The block-level discard condition.  Under session_pair_block granularity a single
        # lost cycle poisons the whole block, because in that design each cycle carried its
        # own zero-field condition and the block was the only unit whose composition was
        # known.  Under cycle_pair granularity the zero-field condition is one separate
        # job that both members of every pair subtract, so it -- not any individual cycle --
        # is what the block depends on.
        block_reason: str | None = None
        if session_index in forced_discard:
            block_reason = str(
                forced_discard[session_index].get("reason", "session pair block explicitly discarded")
            )
        elif granularity == "cycle_pair":
            needs_baseline = any(
                str(row["position"]) == "session_start" and bool(row["used_by_estimate"])
                for row in session.get("baseline_measurements", [])
            )
            if needs_baseline and (session_index, "session_start") not in measured_baselines:
                block_reason = (
                    "missing session_start baseline measurement, which every cycle in both "
                    "cadence blocks subtracts, so no pair in this session is adjudicable"
                )
        elif counts != {"fast": expected_count, "slow": expected_count}:
            block_reason = (
                "entire session pair block is non-adjudicative until both cadence blocks complete"
            )

        if block_reason is not None:
            discarded_sessions.append({
                "session_index": session_index,
                "fast_completed": counts["fast"],
                "slow_completed": counts["slow"],
                "reason": block_reason,
            })
            if granularity == "cycle_pair":
                for pair_id in sorted({
                    str(row["registered_pair_id"])
                    for row in session["cycles"]
                    if row.get("registered_pair_id")
                }):
                    discarded_pairs.append({
                        "registered_pair_id": pair_id,
                        "session_index": session_index,
                        "reason": block_reason,
                    })
            continue

        complete_sessions.append(session_index)
        if granularity != "cycle_pair":
            continue
        members: dict[str, dict[str, str]] = {}
        for row in session["cycles"]:
            pair_id = row.get("registered_pair_id")
            if pair_id:
                members.setdefault(str(pair_id), {})[str(row["cadence"])] = str(row["cycle_id"])
        for pair_id in sorted(members):
            arms = members[pair_id]
            missing = sorted(
                cadence_label
                for cadence_label in ("fast", "slow")
                if arms.get(cadence_label) not in completed_ids
            )
            if missing:
                discarded_pairs.append({
                    "registered_pair_id": pair_id,
                    "session_index": session_index,
                    "reason": f"registered pair incomplete; missing {', '.join(missing)} cycle",
                })
            else:
                complete_pairs.append(pair_id)

    if granularity == "cycle_pair":
        return {
            "policy": (
                "retain raw records; adjudicate every registered cycle pair whose fast and slow "
                "members both completed, and discard a whole session pair block only when its "
                "session_start baseline measurement is missing"
            ),
            "discard_granularity": granularity,
            "expected_cycles_per_cadence_per_session": expected_count,
            "complete_session_pair_blocks": complete_sessions,
            "discarded_session_pair_blocks": discarded_sessions,
            "complete_cycle_pairs": complete_pairs,
            "discarded_cycle_pairs": discarded_pairs,
            "registered_cycle_pairs": len(complete_pairs),
        }
    return {
        "policy": "retain raw records; adjudicate only complete same-session fast/slow cycle blocks",
        "discard_granularity": granularity,
        "expected_cycles_per_cadence_per_session": expected_count,
        "complete_session_pair_blocks": complete_sessions,
        "discarded_session_pair_blocks": discarded_sessions,
        "registered_cycle_pairs": len(complete_sessions) * expected_count,
    }


def session_baseline_drift(
    records: Sequence[Mapping[str, Any]],
    *,
    phase_time_seconds: float | None,
) -> list[dict[str, Any]]:
    """Per-session opening-versus-closing drift on the shared zero-field bias.

    Reported, never gating.  A session whose closing measurement was lost simply has no
    drift readout; its registered pairs are unaffected, because no cycle subtracted that
    measurement.
    """
    if phase_time_seconds is None:
        return []
    by_session: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        if row.get("event") != "session_baseline_measured":
            continue
        by_session.setdefault(int(row["session_index"]), {})[str(row["position"])] = row["baseline"]
    rows: list[dict[str, Any]] = []
    for session_index in sorted(by_session):
        positions = by_session[session_index]
        first = positions.get("session_start")
        last = positions.get("session_end")
        if first is None or last is None:
            rows.append({
                "session_index": session_index,
                "measured": False,
                "reason": "the session has no opening/closing baseline pair to difference",
            })
            continue
        rows.append({
            "session_index": session_index,
            "measured": True,
            **shared_baseline.baseline_drift_qc(
                first, last, phase_time_seconds=float(phase_time_seconds)
            ),
        })
    return rows


def finalize_report(
    output: Path,
    plan: Mapping[str, Any],
    stage1_audit: Mapping[str, Any],
    *,
    backend_id: str,
    phase_time_seconds: float | None = None,
) -> dict[str, Any]:
    if str(plan.get("backend_id")) != backend_id:
        raise ValueError("plan backend id does not match executing backend config")
    store = campaign.CampaignStore(output)
    submitted = [row for row in store.records if row.get("event") == "submitted"]
    collected = [row for row in store.records if row.get("event") == "collected"]
    cycles = [row for row in store.records if row.get("event") == "cadence_cycle_completed"]
    gates = [row for row in store.records if row.get("event") == "session_gate_test"]
    expected = plan["expected"]
    pair_block_status = registered_pair_block_status(plan, store.records)
    # "baseline" only exists in plans that amortise the zero-field condition into one shared
    # measurement per session.  Plans without it declare zero expected baseline jobs, so the
    # role is counted unconditionally and compares equal on the older schedules.
    roles = ("loop", "mirror", "baseline")
    role_jobs = {
        role: sum(row.get("job_role") == role for row in submitted)
        for role in roles
    }
    role_tasks = {
        role: sum(len(row["tasks"]) for row in submitted if row.get("job_role") == role)
        for role in roles
    }
    baselines = [row for row in store.records if row.get("event") == "session_baseline_measured"]
    lost_baselines = [
        row for row in store.records if row.get("event") == "session_baseline_measurement_lost"
    ]
    collected_ids = {str(row["snapshot_id"]) for row in collected}
    query_ids = [str(task["query_id"]) for row in submitted for task in row["tasks"]]
    file_hashes_passed = all(
        Path(str(row["raw_results_path"])).is_file()
        and Path(str(row["counts_path"])).is_file()
        and digest_file(Path(str(row["raw_results_path"]))) == str(row["raw_results_sha256"])
        and digest_file(Path(str(row["counts_path"]))) == str(row["counts_sha256"])
        for row in collected
    )
    drift_rows = session_baseline_drift(store.records, phase_time_seconds=phase_time_seconds)
    completed = bool(
        role_jobs == {
            "loop": int(expected["loop_jobs"]),
            "mirror": int(expected["mirror_jobs"]),
            "baseline": int(expected.get("baseline_jobs", 0)),
        }
        and role_tasks == {
            "loop": int(expected["loop_tasks"]),
            "mirror": int(expected["mirror_tasks"]),
            "baseline": int(expected.get("baseline_tasks", 0)),
        }
        and len(collected_ids) == len(submitted)
        and len(cycles) == int(expected["cycles"])
        and (
            expected.get("cycles_per_cadence_per_session") is None
            or int(pair_block_status["registered_cycle_pairs"]) == int(expected["complete_cadence_pairs"])
        )
        and len(gates) == int(expected["sessions"])
        and len(query_ids) == len(set(query_ids))
        and file_hashes_passed
    )
    report = {
        "schema": "b4_cadence_pair_hardware_supplement_report_v1",
        "created_at_utc": iso(utc_now()),
        "completed": completed,
        "backend_id": backend_id,
        "plan_path": str((output / PLAN_NAME).resolve()),
        "plan_sha256": digest_file(output / PLAN_NAME),
        "stage1_audit": stage1_audit,
        "expected": expected,
        "observed": {
            "role_jobs": role_jobs,
            "role_tasks": role_tasks,
            "collected_jobs": len(collected_ids),
            "completed_cycles": len(cycles),
            "cycles_by_cadence": {
                cadence_label: sum(str(row.get("cadence")) == cadence_label for row in cycles)
                for cadence_label in ("fast", "slow")
            },
            "registered_cycle_pairs": int(pair_block_status["registered_cycle_pairs"]),
            "complete_session_pair_blocks": pair_block_status["complete_session_pair_blocks"],
            "discarded_session_pair_blocks": pair_block_status["discarded_session_pair_blocks"],
            "baseline_measurements": len(baselines),
            "baseline_measurements_lost": len(lost_baselines),
            "session_gate_tests": len(gates),
            "unique_query_ids": len(set(query_ids)),
            "duplicate_query_ids": len(query_ids) - len(set(query_ids)),
            "file_hashes_passed": file_hashes_passed,
            "journal_records": len(store.records),
            "journal_tail_sha256": store.last_hash,
        },
        "registered_pair_block_status": pair_block_status,
        "session_shared_baseline": {
            "measurements": [
                {
                    "measurement_id": row["measurement_id"],
                    "session_index": int(row["session_index"]),
                    "position": str(row["position"]),
                    "shots_per_setting": int(row["shots_per_setting"]),
                    "used_by_estimate": bool(row["used_by_estimate"]),
                    "lateness_seconds": row["lateness_seconds"],
                    "snapshot_id": row["snapshot_id"],
                }
                for row in baselines
            ],
            "lost_measurements": [
                {
                    "measurement_id": row["measurement_id"],
                    "session_index": int(row["session_index"]),
                    "position": str(row["position"]),
                    "failure_class": row["failure_class"],
                }
                for row in lost_baselines
            ],
            "drift_qc": drift_rows,
            "drift_qc_role": (
                "reported readout and input to the pre-registered sensitivity re-run; it "
                "never gates the endpoint"
            ),
        },
    }
    report_path = output / REPORT_NAME
    if report_path.exists():
        report_path.unlink()
    write_new_json(report_path, report)
    return report


def execute(
    loop_config_path: Path,
    backend_config_path: Path,
    peer_config_path: Path,
    stage1_manifest_path: Path,
    output: Path,
    *,
    confirm_hardware: bool,
    max_wait_seconds: int,
    poll_seconds: int,
    max_new_jobs: int | None = None,
    platform_factory: Callable[[Mapping[str, Any]], Any] = drift_campaign_v4.platform_from_config,
) -> dict[str, Any]:
    if not confirm_hardware:
        raise RuntimeError("hardware cadence supplement requires --confirm-hardware")
    preflight_config = json.loads(Path(loop_config_path).read_text(encoding="utf-8"))
    preflight_correction = collection_correction(preflight_config)
    if (
        preflight_correction is not None
        and preflight_correction.get("status") == LEGACY_UNREACHABLE_CORRECTION_STATUS
    ):
        # The 2026-08-15 audit showed this plan cannot reach its own registered endpoint:
        # 3072 shots per setting install a common additive noise floor inside a
        # scale-invariant ratio gate, giving power 0.46 at 24 pairs rather than 0.91.
        # It stays loadable so its defect can be tested, but it is never submitted.
        raise RuntimeError(
            f"{loop_config_path} is the superseded collection plan and is not registered-endpoint "
            "reachable; submit the amended plan instead"
        )
    stage1_audit = verify_stage1_manifest(stage1_manifest_path)
    plan = prepare_plan(
        loop_config_path,
        backend_config_path,
        peer_config_path,
        stage1_manifest_path,
        output,
        start_utc=utc_now() + timedelta(seconds=5),
    )
    loop_config = cadence.load_config(loop_config_path)
    config = drift_campaign_v4.load_config(backend_config_path)
    platform = platform_factory(config)
    store = campaign.CampaignStore(output)
    cadence.shield_self_audit(loop_config)
    gate_rows = cadence.gate_ladder(loop_config)
    submission_budget: dict[str, int | None] = {"limit": max_new_jobs, "used": 0}
    try:
        for session in plan["sessions"]:
            session_index = int(session["session_index"])
            already_discarded = any(
                row.get("event") == "session_pair_block_discarded"
                and int(row.get("session_index", -1)) == session_index
                for row in store.records
            )
            if already_discarded:
                continue
            try:
                require_session_open(
                    session.get("operational_deadline_utc"),
                    stage=f"session {session_index} resume",
                )
                measurements = list(session.get("baseline_measurements", []))
                opening = next(
                    (row for row in measurements if str(row["position"]) == "session_start"), None
                )
                session_baseline = None
                if opening is not None:
                    # Every cycle in both cadence blocks subtracts this one measurement, so
                    # it has to exist before the first cycle runs.  A session whose opening
                    # baseline fails carries no adjudicable pair at all, which is why this
                    # sits inside the per-session try and discards the block.
                    session_baseline = run_baseline_measurement(
                        platform=platform,
                        config=config,
                        loop_config=loop_config,
                        store=store,
                        output=output,
                        measurement=opening,
                        max_wait_seconds=max_wait_seconds,
                        poll_seconds=poll_seconds,
                        deadline_utc=session.get("operational_deadline_utc"),
                        submission_budget=submission_budget,
                    )["baseline"]
                for cycle in session["cycles"]:
                    run_cycle(
                        platform=platform,
                        config=config,
                        loop_config=loop_config,
                        store=store,
                        output=output,
                        cycle=cycle,
                        max_wait_seconds=max_wait_seconds,
                        poll_seconds=poll_seconds,
                        session_deadline_utc=session.get("operational_deadline_utc"),
                        session_completion_deadline_utc=session.get("operational_completion_deadline_utc"),
                        submission_budget=submission_budget,
                        session_baseline=session_baseline,
                    )
                for closing in measurements:
                    if str(closing["position"]) == "session_start":
                        continue
                    # A drift readout, not an endpoint input: it runs after the last mirror,
                    # no cycle ever subtracted it, and every registered pair is already
                    # adjudicable without it.  Losing it therefore costs the drift
                    # sensitivity analysis and must not discard the session's pairs, so its
                    # failure is recorded and swallowed rather than raised.
                    try:
                        run_baseline_measurement(
                            platform=platform,
                            config=config,
                            loop_config=loop_config,
                            store=store,
                            output=output,
                            measurement=closing,
                            max_wait_seconds=max_wait_seconds,
                            poll_seconds=poll_seconds,
                            deadline_utc=session.get("operational_completion_deadline_utc"),
                            submission_budget=submission_budget,
                        )
                    except SubmissionLimitReached:
                        raise
                    except Exception as error:
                        store.append("session_baseline_measurement_lost", {
                            "measurement_id": closing["measurement_id"],
                            "session_index": session_index,
                            "position": str(closing["position"]),
                            "failure_class": type(error).__name__,
                            "reason": str(error),
                            "consequence": (
                                "the baseline drift upper limit is unavailable for this session; "
                                "its registered cycle pairs are unaffected because no cycle "
                                "subtracted this measurement"
                            ),
                        })
                        print(canonical_json({
                            "event": "session_baseline_measurement_lost",
                            "measurement_id": closing["measurement_id"],
                            "failure_class": type(error).__name__,
                        }), flush=True)
            except SubmissionLimitReached:
                raise
            except Exception as error:
                # A session is the unit of loss.  Anything that kills one session --
                # its deadline, a submission failure, a collection timeout -- discards
                # that pair block and leaves the remaining sessions to run, because an
                # unattended multi-day collection must not lose every later session to
                # one transient platform fault.
                expired = isinstance(error, SessionPairBlockExpired)
                completed_session_cycles = [
                    row
                    for row in store.records
                    if row.get("event") == "cadence_cycle_completed"
                    and int(row.get("session_index", -1)) == session_index
                ]
                store.append("session_pair_block_discarded", {
                    "session_index": session_index,
                    "operational_start_utc": session["operational_start_utc"],
                    "operational_deadline_utc": session.get("operational_deadline_utc"),
                    "operational_completion_deadline_utc": session.get("operational_completion_deadline_utc"),
                    "completed_cycles": len(completed_session_cycles),
                    "failure_class": "session_pair_block_expired" if expired else type(error).__name__,
                    "reason": str(error),
                    "policy": "retain raw records; never resume an incomplete pair block after its same-session deadline",
                })
                print(canonical_json({
                    "event": "session_pair_block_discarded",
                    "session_index": session_index,
                    "failure_class": "session_pair_block_expired" if expired else type(error).__name__,
                    "completed_cycles": len(completed_session_cycles),
                }), flush=True)
                continue
            gate_exists = any(
                row.get("event") == "session_gate_test" and row.get("session_index") == session["session_index"]
                for row in store.records
            )
            if not gate_exists:
                # The ladder is a three-rung shield contract, not a per-session
                # treatment; every session records the whole contract.
                store.append("session_gate_test", {
                    "session_index": session["session_index"],
                    "execution_role": "full_shield_ladder_contract_at_session_end",
                    "hardware_job_submitted": False,
                    "rungs": list(gate_rows),
                })
    except SubmissionLimitReached:
        print(canonical_json({
            "event": "submission_limit_reached",
            "max_new_jobs": max_new_jobs,
            "new_jobs_submitted": submission_budget["used"],
        }), flush=True)
    report = finalize_report(
        output,
        plan,
        stage1_audit,
        backend_id=str(config["backend"]["backend_id"]),
        phase_time_seconds=float(loop_config["sensing"]["phase_time_seconds"]),
    )
    if not report["completed"] and max_new_jobs is None:
        raise RuntimeError(f"cadence supplement incomplete: {report['observed']}")
    return report


def wait_for_running(
    platform: Any,
    *,
    backend_id: str,
    poll_seconds: int,
    stable_polls: int,
) -> None:
    consecutive = 0
    while consecutive < stable_polls:
        machines = platform.query_quantum_computer_list()
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
        print(canonical_json({
            "event": "machine_status",
            "recorded_at_utc": iso(utc_now()),
            "backend_id": backend_id,
            "status": status,
            "consecutive_running_polls": consecutive,
            "required_running_polls": stable_polls,
        }), flush=True)
        if consecutive < stable_polls:
            time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-config", type=Path, default=DEFAULT_LOOP_CONFIG)
    parser.add_argument("--backend-config", type=Path, default=DEFAULT_BACKEND_CONFIG)
    parser.add_argument("--peer-config", type=Path, default=DEFAULT_PEER_CONFIG)
    parser.add_argument("--stage1-manifest", type=Path, default=DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-wait-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--max-new-jobs", type=int)
    parser.add_argument("--wait-for-running", action="store_true")
    parser.add_argument("--status-poll-seconds", type=int, default=30)
    parser.add_argument("--stable-running-polls", type=int, default=3)
    parser.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()
    platform = None
    platform_factory = drift_campaign_v4.platform_from_config
    if arguments.wait_for_running:
        backend_config = drift_campaign_v4.load_config(arguments.backend_config.resolve())
        platform = drift_campaign_v4.platform_from_config(backend_config)
        wait_for_running(
            platform,
            backend_id=str(backend_config["backend"]["backend_id"]),
            poll_seconds=arguments.status_poll_seconds,
            stable_polls=arguments.stable_running_polls,
        )
        platform_factory = lambda _config: platform
    report = execute(
        arguments.loop_config.resolve(),
        arguments.backend_config.resolve(),
        arguments.peer_config.resolve(),
        arguments.stage1_manifest.resolve(),
        arguments.output.resolve(),
        confirm_hardware=arguments.confirm_hardware,
        max_wait_seconds=arguments.max_wait_seconds,
        poll_seconds=arguments.poll_seconds,
        max_new_jobs=arguments.max_new_jobs,
        platform_factory=platform_factory,
    )
    print(json.dumps({"completed": report["completed"], **report["observed"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
