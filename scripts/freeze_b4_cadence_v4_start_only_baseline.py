"""Freeze v4 cadence timing without silently re-freezing registered statistics.

The registered endpoint is ``|mirror_fields + compensation|**2`` and the compensation is
computed *online* by the shield from each cycle's own sensed estimate.  A cycle can
therefore only subtract a baseline that already exists when it runs, so the session-end
baseline measurement cannot enter the estimate and the shared variance term cannot be
halved by averaging the two.  This script rewrites ``collection_correction`` in the v4
config at the design point that follows from that ordering.  The original script wrote a
new pre-registration artifact and replaced the whole ``collection_correction`` block.
That is unsafe for a backend migration: changing timing would also appear to re-bless the
registered statistics.  The default operation is therefore timing-only.  It emits a
backend-migration timing artifact, preserves the original statistical evidence and its
hash, and refuses to continue if any registered statistic no longer matches the frozen
40-pair design.  ``--rewrite-statistics`` is an explicit escape hatch, never the default.

It writes nothing to hardware and submits no jobs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOOP_CONFIG_PATH = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json"
DEFAULT_BACKEND_CONFIG_PATH = ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"
DEFAULT_ARTIFACT_DIR = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_V4_T176_TIMING_MIGRATION_20260823"
)
DEFAULT_ARTIFACT_NAME = "backend_migration_timing_evidence.json"

# Design point.  Frozen; see docs/B4_B5_COLLECTION_AMENDMENT_v3_TO_v4_20260815.md.
SESSIONS = 2
PAIRS_PER_SESSION = 20
INJECTED_SHOTS = 6186
BASELINE_SHOTS = 27664
BASELINE_END_SHOTS = 27664
SENSING_SETTINGS_PER_CYCLE = 2
BASELINE_SETTINGS = 2
BASELINE_MEASUREMENTS = 2
MIRROR_QC_JOBS = 2
MIRROR_SETTINGS = 2
# Shots for one whole QC cycle, i.e. the fixed and adaptive settings together.  The runner
# submits ``mirror.shots_per_task`` per setting, so this is MIRROR_SETTINGS times that, and
# multiplying it by MIRROR_SETTINGS again would charge the mirror twice.  ``design`` asserts
# the identity against the loop config rather than trusting this comment.
MIRROR_QC_SHOTS = 8192
CEILING_SECONDS = 2400.0
MINIMUM_PAIRS = 30
FLOOR_CONSTANT = 2.373
# Wall-clock room reserved around the cadence grid for the two baseline jobs.  Each costs
# about 65 s of busy time, so these are queue-wait allowances, not work: the whole point of
# holding them outside the grid is that a slow baseline queue must never push a cycle off
# its cadence tick.  They consume no busy time and no shots.
BASELINE_LEAD_SECONDS = 1200.0
BASELINE_TRAIL_SECONDS = 1200.0

# Measured operating characteristic, 40000 replicates / 2000 permutations.
POWER = 0.90155
BOUNDARY_SIZE = 0.048075
MAXIMUM_BOUNDARY_SIZE = 0.0533
LOSS_TOLERANCE = {
    40: 0.90000, 38: 0.88285, 36: 0.86995, 34: 0.85085, 32: 0.83000,
    30: 0.80460, 28: 0.78300, 24: 0.71910, 20: 0.64015,
}
# Three pre-registered drift shapes at the no-drift one-sided 95% upper limit z = 2.3264
# and above.  z is the start-versus-end drift in units of the start baseline's own phase
# sigma; the endpoint offset is D = z**2 * shared_floor.
DRIFT_SENSITIVITY = {
    "linear_ramp": {"block_offsets_of_D": [1 / 16, 9 / 16],
                    "rows": [(2.3264, 0.0488, 0.8460), (2.5, 0.0484, 0.8369),
                             (3.0, 0.0465, 0.8048), (4.0, 0.0424, 0.7180)]},
    "step_at_block_boundary": {"block_offsets_of_D": [0.0, 1.0],
                    "rows": [(2.3264, 0.0452, 0.8040), (2.5, 0.0438, 0.7875),
                             (3.0, 0.0410, 0.7267), (4.0, 0.0325, 0.5694)]},
    "early_saturating_transient": {"block_offsets_of_D": [1.0, 1.0],
                    "rows": [(2.3264, 0.0503, 0.7289), (2.5, 0.0503, 0.7022),
                             (3.0, 0.0503, 0.6210), (4.0, 0.0503, 0.4609)]},
}
PREDICTION = {
    "predicted_ratio_median": 0.5070213621785998,
    "predicted_ratio_mean": 0.5199006129148598,
    "prediction_interval_lower": 0.3260557361133296,
    "prediction_interval_upper": 0.7884314361673062,
    "prediction_mass": 0.95,
    "prediction_replicates": 400000,
    "prediction_seed": 20260816,
}

TIMING_BUDGET_MODEL = "role_envelope_sum_task_runtime"
ROLE_NAMES = ("baseline", "sense", "mirror")

# These values are independently re-derived below.  A timing-only migration is allowed
# to proceed only when the existing registered values match them exactly.  The output
# write then preserves *every* non-timing value from the input object, not merely these.
REGISTERED_STATISTICAL_FIELDS = (
    "sessions",
    "cycles_per_cadence_per_session",
    "session_block_order",
    "registered_cycle_pairs_total",
    "minimum_adjudicated_cycle_pairs",
    "sensing_settings_per_cycle",
    "sensing_shots_per_setting",
    "sensing_shots_per_cycle",
    "baseline_measurements_per_session",
    "baseline_settings_per_measurement",
    "baseline_shots_per_setting",
    "baseline_end_shots_per_setting",
    "baseline_used_by_estimate",
    "baseline_sharing_scope",
    "shots_per_session",
    "settings_per_session",
    "jobs_per_session",
    "shots_total",
    "machine_time_ceiling_seconds",
    "registered_endpoint",
    "primary_adjudication",
    "secondary_adjudication",
    "gate_module_change_permitted",
    "outlier_exclusion_permitted",
    "optional_stopping_permitted",
    "expected_fast_endpoint_mean",
    "expected_slow_endpoint_mean",
    "expected_ratio",
    "pure_drift_ratio_limit",
    "endpoint_shot_floor_per_cycle",
    "endpoint_shot_floor_shared",
    "endpoint_shot_floor_total",
    "preregistered_ratio_prediction",
    "preregistered_ratio_interval",
    "preregistered_ratio_interval_mass",
    "simulation_pooling_permitted",
    "minimum_power",
    "expected_power",
    "measured_boundary_size",
    "maximum_boundary_size",
)

# Only these existing keys may change in default timing-only mode.  New migration keys
# are added separately.  In particular, the original reachability evidence and hash are
# absent: they remain the evidence for the registered statistical design.
TIMING_UPDATE_FIELDS = (
    "modelled_busy_seconds_per_session",
    "modelled_busy_seconds_total",
    "machine_time_ceiling_slack_seconds",
    "machine_time_ceiling_slack_note",
    "machine_time_ceiling_slack_not_reclaimed_reason",
    "shot_rate_per_second_used",
    "seconds_per_job_used",
    "seconds_per_setting_used",
    "seconds_per_job_note",
    "shot_rate_note",
    "operational_window_note",
)


def _require_number(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping or isinstance(mapping[key], bool):
        raise ValueError(f"cadence_supplement_timing missing numeric field {key!r}")
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"cadence_supplement_timing field {key!r} must be finite")
    return value


def _require_sha256(mapping: Mapping[str, Any], key: str) -> str:
    value = str(mapping.get(key, ""))
    if len(value) != 64:
        raise ValueError(f"{key} must be a 64-character sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{key} must be hexadecimal") from exc
    return value.lower()


def read_cadence_supplement_timing(backend_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate migration timing provenance without using legacy T-B6 timing."""
    backend = backend_config.get("backend")
    if not isinstance(backend, Mapping):
        raise ValueError("backend config has no backend object")
    raw = backend.get("cadence_supplement_timing")
    if not isinstance(raw, Mapping):
        raise ValueError("backend.cadence_supplement_timing is required")

    source_ledger = str(raw.get("source_ledger", "")).strip()
    if not source_ledger:
        raise ValueError("cadence_supplement_timing.source_ledger is required")
    source_hash = _require_sha256(raw, "source_ledger_sha256")
    rate = _require_number(raw, "queue_free_slope_shots_per_second")
    intercept = _require_number(raw, "unconstrained_intercept_seconds")
    overhead = _require_number(raw, "nonnegative_overhead_seconds_used")
    if rate <= 0.0:
        raise ValueError("queue_free_slope_shots_per_second must be positive")
    if overhead < 0.0:
        raise ValueError("nonnegative_overhead_seconds_used cannot be negative")
    if overhead != max(0.0, intercept):
        raise ValueError(
            "nonnegative_overhead_seconds_used must equal max(0, unconstrained_intercept_seconds)"
        )
    if raw.get("budget_model") != TIMING_BUDGET_MODEL:
        raise ValueError(f"budget_model must be {TIMING_BUDGET_MODEL!r}")

    role_runtime_raw = raw.get("role_task_runtime_seconds")
    role_settings_raw = raw.get("role_settings_per_job")
    if not isinstance(role_runtime_raw, Mapping) or not isinstance(role_settings_raw, Mapping):
        raise ValueError("role runtime and settings maps are required")
    if set(role_runtime_raw) != set(ROLE_NAMES) or set(role_settings_raw) != set(ROLE_NAMES):
        raise ValueError(f"role maps must have exactly {ROLE_NAMES!r}")
    role_runtime = {name: float(role_runtime_raw[name]) for name in ROLE_NAMES}
    role_settings = {name: int(role_settings_raw[name]) for name in ROLE_NAMES}
    if any(not math.isfinite(value) or value <= 0.0 for value in role_runtime.values()):
        raise ValueError("every role task runtime must be finite and positive")
    if any(
        value <= 0 or value != role_settings_raw[name]
        for name, value in role_settings.items()
    ):
        raise ValueError("every role settings-per-job value must be a positive integer")

    floor = raw.get("floor_constant_verification")
    if not isinstance(floor, Mapping):
        raise ValueError("cadence_supplement_timing.floor_constant_verification is required")
    registered_constant = _require_number(floor, "registered_constant")
    measured_constant = _require_number(floor, "measured_constant")
    ratio = _require_number(floor, "ratio_to_registered")
    if registered_constant != FLOOR_CONSTANT:
        raise ValueError("floor verification must name the unchanged registered constant")
    if ratio != measured_constant / registered_constant:
        raise ValueError("floor verification ratio does not match measured/registered")
    floor_source = str(floor.get("source_artifact", "")).strip()
    if not floor_source:
        raise ValueError("floor verification source_artifact is required")
    floor_hash = _require_sha256(floor, "source_artifact_sha256")
    if floor.get("pooling_permitted") is not False:
        raise ValueError("floor verification must remain non-poolable")
    if floor.get("registered_endpoint_contribution") != "none":
        raise ValueError("floor verification must contribute no registered endpoint records")

    return {
        "backend_id": str(backend.get("backend_id", "")),
        "source_ledger": source_ledger,
        "source_ledger_sha256": source_hash,
        "queue_free_slope_shots_per_second": rate,
        "unconstrained_intercept_seconds": intercept,
        "nonnegative_overhead_seconds_used": overhead,
        "budget_model": TIMING_BUDGET_MODEL,
        "role_task_runtime_seconds": role_runtime,
        "role_settings_per_job": role_settings,
        "floor_constant_verification": {
            "registered_constant": registered_constant,
            "measured_constant": measured_constant,
            "ratio_to_registered": ratio,
            "source_artifact": floor_source,
            "source_artifact_sha256": floor_hash,
            "pooling_permitted": False,
            "registered_endpoint_contribution": "none",
        },
    }


def drift_endpoint_term(cadence_seconds: float, variance: float, tau: float) -> float:
    return 4.0 * variance * (1.0 - math.exp(-cadence_seconds / tau))


def design(loop_config: dict[str, Any], timing: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute design counts and conservative role-envelope timing."""
    ou = loop_config["controlled_ou"]
    cadence = loop_config["cadence"]
    per_cycle = FLOOR_CONSTANT / INJECTED_SHOTS
    shared = FLOOR_CONSTANT / BASELINE_SHOTS
    floor = per_cycle + shared
    drift_fast = drift_endpoint_term(
        cadence["fast_seconds"], ou["stationary_process_variance"], ou["tau_seconds"]
    )
    drift_slow = drift_endpoint_term(
        cadence["slow_seconds"], ou["stationary_process_variance"], ou["tau_seconds"]
    )
    sensing_shots = 2 * PAIRS_PER_SESSION * SENSING_SETTINGS_PER_CYCLE * INJECTED_SHOTS
    baseline_shots = BASELINE_SETTINGS * (BASELINE_SHOTS + BASELINE_END_SHOTS)
    if MIRROR_QC_SHOTS != MIRROR_SETTINGS * int(loop_config["mirror"]["shots_per_task"]):
        raise ValueError(
            "the declared mirror QC cycle shots must equal what the runner submits, which is "
            "mirror.shots_per_task on each of the fixed and adaptive settings"
        )
    mirror_shots = MIRROR_QC_JOBS * MIRROR_QC_SHOTS
    shots = sensing_shots + baseline_shots + mirror_shots
    settings = (
        2 * PAIRS_PER_SESSION * SENSING_SETTINGS_PER_CYCLE
        + BASELINE_MEASUREMENTS * BASELINE_SETTINGS
        + MIRROR_QC_JOBS * MIRROR_SETTINGS
    )
    jobs = 2 * PAIRS_PER_SESSION + BASELINE_MEASUREMENTS + MIRROR_QC_JOBS
    correction = loop_config["collection_correction"]
    role_jobs = {
        "baseline": int(correction["baseline_measurements_per_session"]),
        "sense": 2 * int(correction["cycles_per_cadence_per_session"]),
        "mirror": int(correction["mirror_qc_jobs_per_session"]),
    }
    role_settings = {
        "baseline": int(correction["baseline_settings_per_measurement"]),
        "sense": int(correction["sensing_settings_per_cycle"]),
        "mirror": int(correction["mirror_settings_per_repetition"]),
    }
    if role_jobs != {"baseline": 2, "sense": 40, "mirror": 2}:
        raise ValueError(
            "the registered 40-pair design must recompute role jobs as "
            "baseline=2, sense=40, mirror=2"
        )
    declared_role_settings = dict(timing["role_settings_per_job"])
    if role_settings != declared_role_settings:
        raise ValueError(
            f"backend role settings {declared_role_settings!r} do not match loop design "
            f"{role_settings!r}"
        )
    role_runtime = timing["role_task_runtime_seconds"]
    wall_busy = sum(role_jobs[name] * role_runtime[name] for name in ROLE_NAMES)
    # Platform quota accounting is not yet independently documented.  Sum every task's
    # queue-free runtime, even though same-job tasks share exact start/finish timestamps.
    # This is the conservative envelope Murphy's observed decrement can falsify later.
    quota_busy = sum(
        role_jobs[name] * role_settings[name] * role_runtime[name] for name in ROLE_NAMES
    )
    span = PAIRS_PER_SESSION * (cadence["fast_seconds"] + cadence["slow_seconds"])
    return {
        "per_cycle_floor": per_cycle,
        "shared_floor": shared,
        "total_floor": floor,
        "four_setting_floor_at_same_shots": 2.0 * per_cycle,
        "fast_mean": floor + drift_fast,
        "slow_mean": floor + drift_slow,
        "ratio": (floor + drift_fast) / (floor + drift_slow),
        "pure_drift_ratio": drift_fast / drift_slow,
        "shots_per_session": shots,
        "settings_per_session": settings,
        "jobs_per_session": jobs,
        "busy_per_session": quota_busy,
        "busy_total": SESSIONS * quota_busy,
        "execution_wall_seconds_per_session": wall_busy,
        "execution_wall_seconds_total": SESSIONS * wall_busy,
        "quota_seconds_per_session": quota_busy,
        "quota_seconds_total": SESSIONS * quota_busy,
        "role_jobs_per_session": role_jobs,
        "role_settings_per_job": role_settings,
        "role_task_runtime_seconds": dict(role_runtime),
        "sensing_job_busy_seconds": role_runtime["sense"],
        "baseline_job_busy_seconds": role_runtime["baseline"],
        "legacy_rate_only_seconds_per_session": shots
        / timing["queue_free_slope_shots_per_second"],
        "programmed_span_seconds": span,
    }


def correction_block(
    loop_config: dict[str, Any],
    d: dict[str, Any],
    previous: dict[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    new: "collections.OrderedDict[str, Any]" = collections.OrderedDict()

    def keep(*names: str) -> None:
        for name in names:
            new[name] = previous[name]

    keep("status", "supersedes")
    # This script rewrites the same config it reads, so the clause has to be appended
    # idempotently or a second run states the ordering defect twice.  It did, four times,
    # before this guard.
    ordering_clause = (
        "the amortised baseline then had to be sized against the loop's own ordering, "
        "because the compensation each cycle applies is computed online by the shield from "
        "that cycle's estimate, so a cycle can only subtract the baseline measured at "
        "session start"
    )
    previous_reason = str(previous["reason"])
    new["reason"] = (
        previous_reason
        if ordering_clause in previous_reason
        else previous_reason + "; " + ordering_clause
    )
    keep("amendment_scope", "machine_time_ceiling_seconds", "machine_time_ceiling_source",
         "block_definition")
    new["sessions"] = SESSIONS
    new["cycles_per_cadence_per_session"] = PAIRS_PER_SESSION
    new["session_block_order"] = [["fast", "slow"], ["slow", "fast"]]
    keep("session_block_order_note")
    new["session_block_order_drift_note"] = (
        "the balanced starting cadence is also the baseline-drift protection, and this is "
        "measured rather than argued: a linear drift loads the first block and the second "
        "block unequally, but because session 0 runs fast first and session 1 runs slow "
        "first, the two loadings land on opposite arms and cancel to first order in the "
        "pooled statistic; injecting the same drift with both sessions in the same order "
        "raises the realized boundary size from 0.05 to 0.20"
    )
    new["registered_cycle_pairs_total"] = SESSIONS * PAIRS_PER_SESSION
    new["minimum_adjudicated_cycle_pairs"] = MINIMUM_PAIRS
    new["minimum_sessions_per_block_order"] = 1
    new["pair_discard_granularity"] = "cycle_pair"
    new["pair_discard_granularity_note"] = (
        "a lost cycle discards its own registered pair, not the whole session pair block, "
        "because the shared baseline both members subtract is a separate job that survives "
        "a lost cycle; the whole session pair block is discarded only if its session-start "
        "baseline measurement is missing"
    )
    new["pair_block_discard_condition"] = "missing session_start baseline measurement"
    new["pair_count_note"] = (
        f"the primary rule is the within-pair permutation calibration, which is exact at "
        f"every pair count, so the pair count is set by power rather than by size; measured "
        f"permutation power is {POWER:.4f} at {SESSIONS * PAIRS_PER_SESSION} pairs and "
        f"{LOSS_TOLERANCE[MINIMUM_PAIRS]:.4f} at {MINIMUM_PAIRS}, which is why "
        f"{MINIMUM_PAIRS} is the registered minimum; the pair count itself was chosen by "
        "the registered rule 'largest pair count within two Monte Carlo standard errors of "
        "the maximum', which at 30000 replicates admits 36 and 40 and excludes 44"
    )
    new["adjudication_rule"] = (
        "run both registered sessions; adjudicate every registered cycle pair whose fast "
        "and slow members both completed inside a session whose session-start baseline "
        "exists; the endpoint is adjudicated only if the realized pair count is at least "
        "minimum_adjudicated_cycle_pairs and at least minimum_sessions_per_block_order "
        "complete sessions carry each starting cadence; otherwise INCONCLUSIVE"
    )
    new["whole_session_loss_policy"] = (
        f"losing an entire session leaves {PAIRS_PER_SESSION} pairs from a single starting "
        f"cadence, which is both underpowered ({LOSS_TOLERANCE[PAIRS_PER_SESSION]:.3f}) and "
        "block-order confounded, so it is reported INCONCLUSIVE and is not patched by "
        "re-running one session; splitting the same forty minutes into three or four "
        "sessions was evaluated and rejected because it lowers nominal power without "
        "lifting the lost-session power above 0.8"
    )
    keep("optional_stopping_permitted", "optional_stopping_note", "pairing_rule")
    new["incomplete_session_pair_block_policy"] = (
        "retain raw records but discard the individual cycle pair from adjudication unless "
        "both members completed; discard the whole session pair block only if its "
        "session-start baseline measurement is missing"
    )
    new["sensing_settings_per_cycle"] = SENSING_SETTINGS_PER_CYCLE
    keep("sensing_settings_note")
    new["sensing_shots_per_setting"] = INJECTED_SHOTS
    new["sensing_shots_per_cycle"] = SENSING_SETTINGS_PER_CYCLE * INJECTED_SHOTS
    new["baseline_measurements_per_session"] = BASELINE_MEASUREMENTS
    new["baseline_measurement_positions"] = ["session_start", "session_end"]
    new["baseline_settings_per_measurement"] = BASELINE_SETTINGS
    new["baseline_shots_per_setting"] = BASELINE_SHOTS
    new["baseline_end_shots_per_setting"] = BASELINE_END_SHOTS
    new["baseline_used_by_estimate"] = "session_start"
    new["baseline_online_ordering_note"] = (
        "the registered endpoint is |mirror_fields + compensation|**2 and the compensation "
        "is computed online by the shield from that cycle's sensed estimate, so a cycle can "
        "only subtract a baseline that already exists when it runs; the session-end "
        "measurement does not exist at cycle time, the start/end average is therefore "
        "unavailable to the estimate, and the shared variance term is 2.373/S_start with no "
        "factor of two; the closing measurement is a drift readout, not an endpoint input"
    )
    new["baseline_end_sizing_rule"] = (
        "the closing measurement carries the same shots as the opening one, which is the "
        "tightest drift upper limit obtainable without taking shots from the endpoint; "
        "reaching the drift level at which power visibly degrades would need "
        "S_end >= 3 * S_start, about 8.5 percent of the session budget, which lowers power "
        "by more than the tighter bound recovers, so it was evaluated and rejected"
    )
    new["baseline_sharing_scope"] = "whole_session_both_cadence_blocks"
    keep("baseline_sharing_note", "baseline_offset_direction_note")
    new["baseline_drift_sensitivity_rule"] = (
        "form the one-sided 95 percent upper confidence limit on the measured "
        "start-versus-end baseline drift, expressed as z = drift / start-baseline phase "
        "sigma, and re-run the verdict with endpoint offset D = z**2 * shared_floor "
        "distributed over the two cadence blocks under three pre-registered drift shapes: "
        "a linear ramp (block offsets D/16 and 9D/16), a step at the block boundary (0 and "
        "D) and an early-saturating transient (D and D); report the primary verdict and all "
        "three sensitivity verdicts whatever the drift turns out to be"
    )
    new["baseline_drift_measured_effect"] = (
        "measured at this design point across all three shapes and z up to 4: the realized "
        "boundary size never exceeds 0.0512 and falls to 0.0325 under the asymmetric "
        "shapes, so baseline drift cannot manufacture a pass; the cost is power, and the "
        "binding shape is the symmetric early-saturating one, which at the no-drift upper "
        f"limit z = 2.326 leaves power 0.729 against the {POWER:.3f} nominal"
    )
    new["baseline_drift_undetectable_mode"] = (
        "a transient that departs and returns between the two baseline measurements is "
        "invisible to the drift readout at any shot count; it is bounded by no measurement "
        "in this design and is stated as a limitation rather than as a controlled risk"
    )
    new["shot_allocation_rule"] = (
        "minimise the total endpoint shot floor 2.373/S_injected + 2.373/S_baseline subject "
        "to the session shot budget; with pairs_per_session cycles per cadence drawing on "
        "one shared baseline this gives S_baseline = S_injected * sqrt(pairs_per_session)"
    )
    new["mirror_repetitions_per_cycle"] = int(loop_config["mirror"]["repetitions_per_block"])
    new["mirror_settings_per_repetition"] = MIRROR_SETTINGS
    keep("mirror_qc_jobs_per_session", "mirror_qc_cycle_indices",
         "mirror_qc_cycle_indices_note", "mirror_shots_per_qc_cycle", "mirror_subset_note",
         "raw_mirror_role", "operational_inter_session_seconds")
    span = d["programmed_span_seconds"]
    new["operational_session_programmed_span_seconds"] = span
    new["baseline_lead_seconds"] = BASELINE_LEAD_SECONDS
    new["baseline_trail_seconds"] = BASELINE_TRAIL_SECONDS
    new["baseline_scheduling_note"] = (
        "the session-start baseline job is submitted at the session's operational start and "
        f"the first cycle's sensing job is held until {BASELINE_LEAD_SECONDS:.0f} s later, so "
        "a slow baseline queue can never push a cycle off its cadence tick; the session-end "
        "baseline follows the last mirror job and has "
        f"{BASELINE_TRAIL_SECONDS:.0f} s of the completion window to itself; both allowances "
        "are wall clock only and add no busy time, no shots and no jobs beyond the two "
        "already counted in jobs_per_session"
    )
    new["operational_session_slack_seconds"] = span * 1.5 - BASELINE_LEAD_SECONDS - span
    new["operational_session_wallclock_seconds"] = span * 1.5
    new["operational_session_completion_window_seconds"] = span * 1.75
    new["operational_window_note"] = (
        f"each cycle's sensing job costs {d['sensing_job_busy_seconds']:.1f} s of busy time "
        f"against a {loop_config['cadence']['fast_seconds']:.0f} s fast cadence period and a "
        f"measured {loop_config['cadence']['measured_interface_p90_seconds']:.1f} s interface "
        f"P90, leaving {loop_config['cadence']['fast_seconds'] - d['sensing_job_busy_seconds']:.1f} s "
        f"of the period unused against a registered "
        f"{loop_config['cadence']['hardware_lower_bound_seconds']:.1f} s lower bound, so the "
        f"nominal grid already absorbs per-cycle queue jitter; the two baseline jobs cost "
        f"{d['baseline_job_busy_seconds']:.1f} s each and run outside the cadence grid; the "
        "session slack is wall clock only, consumes no busy time, and cannot contaminate the "
        "endpoint because the injected OU truth and the compensation are both evaluated on "
        "the nominal virtual grid rather than on the realized submission clock"
    )
    new["wall_clock_note"] = (
        f"the forty minutes of busy time are spread over {SESSIONS} unattended sessions of "
        f"about {span * 1.5 / 3600.0:.2f} h of wall clock each, because {PAIRS_PER_SESSION} "
        f"pairs at {loop_config['cadence']['fast_seconds']:.0f} s plus "
        f"{loop_config['cadence']['slow_seconds']:.0f} s per pair is an irreducible "
        f"{span:.0f} s of nominal grid; the cadence pair is frozen and defines the endpoint, "
        "so the wall clock cannot be compressed without changing the contrast"
    )
    new["shots_per_session"] = d["shots_per_session"]
    new["settings_per_session"] = d["settings_per_session"]
    new["jobs_per_session"] = d["jobs_per_session"]
    new["shots_total"] = SESSIONS * d["shots_per_session"]
    new["modelled_busy_seconds_per_session"] = d["busy_per_session"]
    new["modelled_busy_seconds_total"] = d["busy_total"]
    new["machine_time_ceiling_slack_seconds"] = CEILING_SECONDS - d["busy_total"]
    new["machine_time_ceiling_slack_note"] = (
        "T176 migration uses the conservative sum of queue-free per-task runtimes by role; "
        "same-job settings share exact start and finish timestamps, but summing them is the "
        "upper envelope for quota accounting until the platform billing rule is documented"
    )
    new["machine_time_ceiling_slack_not_reclaimed_reason"] = (
        "the backend migration keeps all 40 registered pairs and their frozen shot allocation; "
        "reallocating newly available timing slack would change endpoint shot floors and require "
        "new statistical registration, so no slack is reclaimed"
    )
    new["shot_rate_per_second_used"] = timing["queue_free_slope_shots_per_second"]
    new["seconds_per_job_used"] = timing["nonnegative_overhead_seconds_used"]
    new["seconds_per_setting_used"] = timing["nonnegative_overhead_seconds_used"]
    new["seconds_per_job_note"] = (
        "deprecated compatibility fields only; the unconstrained queue-free two-point fit has "
        f"intercept {timing['unconstrained_intercept_seconds']:.12g} s, so its nonnegative "
        "overhead is zero.  The daily and total gates must use timing_budget_model and the "
        "role envelope, not settings times this value"
    )
    new["shot_rate_note"] = (
        f"queue-free BASELINE/SENSE slope from the T176 platform ledger; "
        f"source {timing['source_ledger']} sha256 {timing['source_ledger_sha256']}; mirror is "
        "excluded from this slope because circuit role/depth is confounded with shot count"
    )
    keep("registered_endpoint", "endpoint_distribution", "primary_adjudication",
         "primary_adjudication_note", "secondary_adjudication",
         "secondary_adjudication_note", "gate_module_change_permitted",
         "outlier_exclusion_permitted")
    new["expected_fast_endpoint_mean"] = d["fast_mean"]
    new["expected_slow_endpoint_mean"] = d["slow_mean"]
    new["expected_ratio"] = d["ratio"]
    new["pure_drift_ratio_limit"] = d["pure_drift_ratio"]
    new["endpoint_shot_floor_per_cycle"] = d["per_cycle_floor"]
    new["endpoint_shot_floor_shared"] = d["shared_floor"]
    new["endpoint_shot_floor_total"] = d["total_floor"]
    new["endpoint_shot_floor_note"] = (
        f"{FLOOR_CONSTANT}/{INJECTED_SHOTS} per cycle plus {FLOOR_CONSTANT}/{BASELINE_SHOTS} "
        "shared, the latter carrying no factor of two because the estimate uses the "
        "session-start measurement alone; the four-setting path would carry "
        f"{d['four_setting_floor_at_same_shots']:.4e} at the same shots per setting, and a "
        "four-setting design refitted into the same session budget reaches only about 3975 "
        "shots per setting for a floor of 1.19e-3, which puts the expected ratio at 0.63 and "
        "the design far below the power bar, so amortising the baseline is what makes forty "
        "minutes reachable at all"
    )
    new["endpoint_shot_floor_shared_fraction_note"] = (
        "the Lagrange optimum sets shared/per_cycle = 1/sqrt(pairs_per_session) = "
        f"{1.0 / math.sqrt(PAIRS_PER_SESSION):.3f}, so the shared term is deliberately not "
        "negligible; forcing it smaller would raise the total floor, and it is safe to leave "
        "large because a common additive offset only dilutes the ratio toward 1.0 and the "
        "measured boundary size confirms it cannot manufacture a pass"
    )
    new["preregistered_ratio_prediction"] = PREDICTION["predicted_ratio_median"]
    new["preregistered_ratio_interval"] = [
        PREDICTION["prediction_interval_lower"], PREDICTION["prediction_interval_upper"]
    ]
    new["preregistered_ratio_interval_mass"] = PREDICTION["prediction_mass"]
    new["prediction_role"] = (
        "this interval is committed before the first hardware job and is adjudicated by hit "
        f"or miss; it stays informative at {SESSIONS * PAIRS_PER_SESSION} pairs in a way a "
        "p-value alone does not, and it is reported whether it hits or misses"
    )
    keep("simulation_pooling_permitted", "simulation_pooling_note", "minimum_power")
    new["expected_power"] = POWER
    new["expected_power_note"] = (
        "nominal, at 40000 replicates and 2000 permutations, under no baseline drift; a "
        "measured drift lowers the realized power and is reported through "
        "baseline_drift_sensitivity_rule rather than absorbed"
    )
    new["measured_boundary_size"] = BOUNDARY_SIZE
    new["maximum_boundary_size"] = MAXIMUM_BOUNDARY_SIZE
    new["maximum_boundary_size_note"] = (
        "the within-pair permutation test is exact a priori, so the simulation checks the "
        "implementation rather than estimating an unknown size; at 40000 replicates the "
        "Monte Carlo standard error on a nominal 0.05 is 0.0011, and the criterion is set at "
        f"three of those, so a point estimate of {BOUNDARY_SIZE:.4f} is a pass and a point "
        f"estimate above {MAXIMUM_BOUNDARY_SIZE} would indicate a coding error rather than "
        "an anti-conservative rule"
    )
    keep("reachability_evidence", "reachability_evidence_sha256", "reachability_evidence_note")
    return new


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistical_mismatches(
    previous: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for field in REGISTERED_STATISTICAL_FIELDS:
        if field not in previous or field not in candidate:
            mismatches.append({
                "field": field,
                "previous": previous.get(field),
                "candidate": candidate.get(field),
            })
        elif previous[field] != candidate[field]:
            mismatches.append(
                {"field": field, "previous": previous[field], "candidate": candidate[field]}
            )
    return mismatches


def migration_artifact(
    *,
    loop_config: Mapping[str, Any],
    loop_config_path: Path,
    backend_config_path: Path,
    timing: Mapping[str, Any],
    d: Mapping[str, Any],
    statistical_mismatches: Sequence[Mapping[str, Any]],
    rewrite_statistics: bool,
) -> dict[str, Any]:
    previous = loop_config["collection_correction"]
    statistical_snapshot = {
        field: previous[field] for field in REGISTERED_STATISTICAL_FIELDS if field in previous
    }
    return {
        "schema": "b4_cadence_v4_backend_migration_timing_v1",
        "artifact_role": "post_registration_backend_migration_timing_evidence",
        "not_a_new_preregistration": True,
        "backend_id": timing["backend_id"],
        "loop_config": str(loop_config_path),
        "backend_config": str(backend_config_path),
        "source_hashes": {
            "loop_config_sha256_before_timing_update": _sha256_file(loop_config_path),
            "backend_config_sha256": _sha256_file(backend_config_path),
            "platform_task_time_ledger_sha256": timing["source_ledger_sha256"],
        },
        "original_registered_statistical_evidence": {
            "path": previous["reachability_evidence"],
            "sha256": previous["reachability_evidence_sha256"],
            "preserved_by_timing_migration": not rewrite_statistics,
            "note": (
                "this remains the authority for expected_ratio, prediction interval, power, "
                "boundary size and all other registered statistical fields; this migration "
                "artifact does not replace it"
            ),
        },
        "statistics_guard": {
            "rewrite_statistics_requested": rewrite_statistics,
            "registered_fields_compared": list(REGISTERED_STATISTICAL_FIELDS),
            "mismatches": list(statistical_mismatches),
            "registered_statistics_snapshot": statistical_snapshot,
        },
        "migration_timing": {
            "source_ledger": timing["source_ledger"],
            "queue_free_slope_shots_per_second": timing[
                "queue_free_slope_shots_per_second"
            ],
            "unconstrained_intercept_seconds": timing["unconstrained_intercept_seconds"],
            "nonnegative_overhead_seconds_used": timing[
                "nonnegative_overhead_seconds_used"
            ],
            "budget_model": TIMING_BUDGET_MODEL,
            "role_task_runtime_seconds": d["role_task_runtime_seconds"],
            "role_settings_per_job": d["role_settings_per_job"],
            "role_jobs_per_session": d["role_jobs_per_session"],
            "execution_wall_seconds_per_session": d["execution_wall_seconds_per_session"],
            "execution_wall_seconds_total": d["execution_wall_seconds_total"],
            "quota_seconds_per_session": d["quota_seconds_per_session"],
            "quota_seconds_total": d["quota_seconds_total"],
            "daily_window_seconds": loop_config["daily_window_seconds"],
            "machine_time_ceiling_seconds": previous["machine_time_ceiling_seconds"],
            "registered_pairs_retained": previous["registered_cycle_pairs_total"],
            "mirror_runtime_note": (
                "the 72.057 s runtime comes from a 16384-shot depth-2 task and is retained as "
                "a conservative upper envelope for each registered 4096-shot depth-2 mirror task"
            ),
            "parallelism_note": (
                "all twenty mirror tasks share exactly the same runStartTime and finishTime; "
                "execution-wall accounting therefore charges one runtime per mirror job, while "
                "the quota envelope sums both registered tasks per job"
            ),
        },
        "floor_constant_backend_verification": {
            **timing["floor_constant_verification"],
            "use_in_this_artifact": "provenance_only",
            "statistical_constant_changed": False,
            "note": (
                "the measured T176 value verifies the frozen 2.373 constant on the safe side; "
                "it is not substituted into any endpoint, ratio, prediction or power calculation"
            ),
        },
        "integrity": {
            "pooling_permitted": False,
            "registered_endpoint_contribution": "none",
            "hardware_submission_performed": False,
            "statistical_fields_written": rewrite_statistics,
            "monte_carlo_recomputed_by_this_artifact": False,
        },
    }


def migrated_correction(
    *,
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
    timing: Mapping[str, Any],
    d: Mapping[str, Any],
    timing_artifact_path: Path,
    timing_artifact_sha256: str,
    rewrite_statistics: bool,
) -> collections.OrderedDict[str, Any]:
    mismatches = _statistical_mismatches(previous, candidate)
    if mismatches and not rewrite_statistics:
        fields = ", ".join(item["field"] for item in mismatches)
        raise ValueError(
            "timing-only freeze would change registered statistical fields: " + fields
        )

    updated: "collections.OrderedDict[str, Any]" = collections.OrderedDict(
        candidate if rewrite_statistics else previous
    )
    for field in TIMING_UPDATE_FIELDS:
        updated[field] = candidate[field]
    updated["timing_migration_status"] = "t176_timing_only_post_registration_amendment"
    updated["timing_backend_id"] = timing["backend_id"]
    updated["timing_budget_model"] = TIMING_BUDGET_MODEL
    updated["role_task_runtime_seconds"] = d["role_task_runtime_seconds"]
    updated["role_settings_per_job"] = d["role_settings_per_job"]
    updated["role_jobs_per_session"] = d["role_jobs_per_session"]
    updated["execution_wall_seconds_per_session"] = d["execution_wall_seconds_per_session"]
    updated["execution_wall_seconds_total"] = d["execution_wall_seconds_total"]
    updated["quota_seconds_per_session"] = d["quota_seconds_per_session"]
    updated["quota_seconds_total"] = d["quota_seconds_total"]
    updated["queue_free_slope_shots_per_second"] = timing[
        "queue_free_slope_shots_per_second"
    ]
    updated["unconstrained_intercept_seconds"] = timing["unconstrained_intercept_seconds"]
    updated["nonnegative_overhead_seconds_used"] = timing[
        "nonnegative_overhead_seconds_used"
    ]
    updated["timing_source_ledger"] = timing["source_ledger"]
    updated["timing_source_ledger_sha256"] = timing["source_ledger_sha256"]
    updated["floor_constant_backend_verification"] = timing[
        "floor_constant_verification"
    ]
    updated["timing_reachability_evidence"] = str(timing_artifact_path)
    updated["timing_reachability_evidence_sha256"] = timing_artifact_sha256
    updated["timing_reachability_evidence_note"] = (
        "post-registration backend-migration timing evidence only; it neither supersedes nor "
        "re-hashes the original registered statistical reachability evidence"
    )

    if not rewrite_statistics:
        for field in REGISTERED_STATISTICAL_FIELDS:
            if updated.get(field) != previous.get(field):
                raise AssertionError(f"timing-only write changed statistical field {field}")
        for field in ("reachability_evidence", "reachability_evidence_sha256"):
            if updated.get(field) != previous.get(field):
                raise AssertionError(f"timing-only write changed original {field}")
    return updated


def freeze(
    *,
    loop_config_path: Path,
    backend_config_path: Path,
    artifact_dir: Path,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    allow_overwrite: bool = False,
    rewrite_statistics: bool = False,
) -> dict[str, Any]:
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise ValueError("artifact_name must be one filename")
    loop_config = json.loads(
        loop_config_path.read_text(encoding="utf-8"),
        object_pairs_hook=collections.OrderedDict,
    )
    backend_config = json.loads(
        backend_config_path.read_text(encoding="utf-8"),
        object_pairs_hook=collections.OrderedDict,
    )
    timing = read_cadence_supplement_timing(backend_config)
    d = design(loop_config, timing)
    previous = loop_config["collection_correction"]
    daily_window = float(loop_config["daily_window_seconds"])
    ceiling = float(previous["machine_time_ceiling_seconds"])
    if d["quota_seconds_per_session"] > daily_window:
        raise ValueError(
            f"design exceeds daily quota envelope: {d['quota_seconds_per_session']:.3f} s "
            f"> {daily_window:.3f} s"
        )
    if d["quota_seconds_total"] > ceiling:
        raise ValueError(
            f"design exceeds total quota envelope: {d['quota_seconds_total']:.3f} s "
            f"> {ceiling:.3f} s"
        )

    candidate = correction_block(loop_config, d, previous, timing)
    mismatches = _statistical_mismatches(previous, candidate)
    if mismatches and not rewrite_statistics:
        fields = ", ".join(item["field"] for item in mismatches)
        raise ValueError(
            "timing-only freeze would change registered statistical fields: " + fields
        )

    target = artifact_dir / artifact_name
    if target.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite existing evidence: {target}")
    payload = migration_artifact(
        loop_config=loop_config,
        loop_config_path=loop_config_path,
        backend_config_path=backend_config_path,
        timing=timing,
        d=d,
        statistical_mismatches=mismatches,
        rewrite_statistics=rewrite_statistics,
    )
    artifact_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    digest = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()
    correction = migrated_correction(
        previous=previous,
        candidate=candidate,
        timing=timing,
        d=d,
        timing_artifact_path=target,
        timing_artifact_sha256=digest,
        rewrite_statistics=rewrite_statistics,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Write the exact bytes that were hashed.  ``Path.write_text`` performs CRLF
    # translation on Windows, which would make the on-disk evidence hash differ from
    # ``digest`` even though the JSON content is unchanged.
    target.write_bytes(artifact_text.encode("utf-8"))
    loop_config["collection_correction"] = correction
    loop_config_path.write_text(
        json.dumps(loop_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "artifact": target,
        "artifact_sha256": digest,
        "loop_config": loop_config_path,
        "backend_id": timing["backend_id"],
        "quota_seconds_per_session": d["quota_seconds_per_session"],
        "quota_seconds_total": d["quota_seconds_total"],
        "execution_wall_seconds_per_session": d["execution_wall_seconds_per_session"],
        "registered_pairs": previous["registered_cycle_pairs_total"],
        "statistics_rewritten": rewrite_statistics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-config", type=Path, default=DEFAULT_LOOP_CONFIG_PATH)
    parser.add_argument("--backend-config", type=Path, default=DEFAULT_BACKEND_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="overwrite the migration timing artifact if it already exists",
    )
    parser.add_argument(
        "--rewrite-statistics",
        action="store_true",
        help="explicitly permit registered statistical fields to be rewritten",
    )
    args = parser.parse_args(argv)

    result = freeze(
        loop_config_path=args.loop_config,
        backend_config_path=args.backend_config,
        artifact_dir=args.artifact_dir,
        artifact_name=args.artifact_name,
        allow_overwrite=args.allow_overwrite,
        rewrite_statistics=args.rewrite_statistics,
    )
    print(f"artifact  {result['artifact']}")
    print(f"sha256    {result['artifact_sha256']}")
    print(
        f"quota     {result['quota_seconds_per_session']:.3f} s/session, "
        f"{result['quota_seconds_total']:.3f} s total"
    )
    print(f"wall      {result['execution_wall_seconds_per_session']:.3f} s/session")
    print(f"pairs     {result['registered_pairs']} retained")
    print(f"config    {result['loop_config']} timing updated for {result['backend_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
