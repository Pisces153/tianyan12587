#!/usr/bin/env python3
"""B-4 three-day design power and size simulation.

The analysis path is imported from ``src.adaptive.sensing_economics``.  Truth
parameters are used only after analysis to score recovery; they are never
passed to an estimator or gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive import sensing_economics


DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_design_power_20260804_timing_grid_v6")
SHOT_RATES_PER_SECOND = (490, 600, 750, 850, 1000)
FIXED_OVERHEADS_SECONDS_PER_SETTING = (0.0, 0.5, 1.1)
TIMING_PROFILES = tuple(
    (rate, overhead)
    for rate in SHOT_RATES_PER_SECOND
    for overhead in FIXED_OVERHEADS_SECONDS_PER_SETTING
)
# Kept as an alias for downstream imports.  Timing decisions must use the
# two-dimensional TIMING_PROFILES grid, never this scalar list alone.
THROUGHPUTS = SHOT_RATES_PER_SECOND
ANCHOR_SHOTS_PER_SETTING = 1024
PROBE_SHOTS_PER_SETTING = 16384
PROBE_SETTINGS_PER_JOB = 2
CALENDAR_DAYS = 3
SESSION_GAP_DAYS = 1
BURST_MINUTES = (0, 1, 2, 4, 8, 16)
REFERENCE_POSITIONS_ZERO_INDEXED = (0, 11, 22, 32)
OU_TAU_MINUTES = (10, 15, 30, 60)
PROCESS_VARIANCES = (2.0e-4, 6.1e-4, 1.2e-3)
REGRET_LIMIT = 1.25
DGP_NAMES = (
    "null_flat",
    "ou",
    "pink",
    "step_calendar",
    "step_triggered",
    "step_as_ramp_artifact",
)
PRIMARY_PROBABILITY = 0.11137820512820513
E0_PROBABILITY = 0.08169320913461539
EVENT_JUMP = 0.02537


@dataclass(frozen=True)
class LatentSeries:
    probability: np.ndarray
    regime_ids: np.ndarray
    event_indices: tuple[int, ...]
    artifact_projection: np.ndarray | None
    calibration: dict[str, float | str | list[float]]


def setting_duration_seconds(
    shots_per_setting: int,
    shot_rate_per_second: float,
    fixed_overhead_seconds_per_setting: float,
) -> float:
    """Wallclock model frozen for T-B6: shots/rate plus fixed setting cost."""
    if shots_per_setting <= 0 or shot_rate_per_second <= 0.0:
        raise ValueError("shots and shot rate must be positive")
    if fixed_overhead_seconds_per_setting < 0.0:
        raise ValueError("fixed setting overhead must be non-negative")
    return float(shots_per_setting) / float(shot_rate_per_second) + float(fixed_overhead_seconds_per_setting)


def effective_probe_shots_per_second(
    shot_rate_per_second: float,
    fixed_overhead_seconds_per_setting: float,
) -> float:
    """Per-channel rate for one channel observed in a two-setting probe job."""
    setting_seconds = setting_duration_seconds(
        PROBE_SHOTS_PER_SETTING,
        shot_rate_per_second,
        fixed_overhead_seconds_per_setting,
    )
    return PROBE_SHOTS_PER_SETTING / (PROBE_SETTINGS_PER_JOB * setting_seconds)


def reference_offsets_seconds(
    shot_rate_per_second: float,
    fixed_overhead_seconds_per_setting: float = 0.0,
) -> np.ndarray:
    """Anchor-reference offsets under the two-parameter wallclock model."""
    setting_seconds = setting_duration_seconds(
        ANCHOR_SHOTS_PER_SETTING,
        shot_rate_per_second,
        fixed_overhead_seconds_per_setting,
    )
    return np.asarray(REFERENCE_POSITIONS_ZERO_INDEXED, dtype=np.float64) * setting_seconds


def probe_observation_offsets_seconds(
    shot_rate_per_second: float,
    fixed_overhead_seconds_per_setting: float = 0.0,
) -> np.ndarray:
    """Realized burst timestamps after serial-job duration clips target times."""
    setting_seconds = setting_duration_seconds(
        PROBE_SHOTS_PER_SETTING,
        shot_rate_per_second,
        fixed_overhead_seconds_per_setting,
    )
    job_seconds = PROBE_SETTINGS_PER_JOB * setting_seconds
    next_available = 0.0
    observations: list[float] = []
    for target_minutes in BURST_MINUTES:
        start = max(float(target_minutes) * 60.0, next_available)
        # Channel-neutral midpoint.  Both map channels use identical timing;
        # the two-setting duration still shifts later jobs when targets overlap.
        observations.append(start + job_seconds / 2.0)
        next_available = start + job_seconds
    return np.asarray(observations, dtype=np.float64)


def design_schedule(
    shot_rate_per_second: float = 490,
    fixed_overhead_seconds_per_setting: float = 0.0,
) -> dict[str, np.ndarray]:
    readout_times = np.asarray(
        [
            session_index * SESSION_GAP_DAYS * 86400.0 + offset
            for session_index in range(CALENDAR_DAYS)
            for offset in probe_observation_offsets_seconds(
                shot_rate_per_second,
                fixed_overhead_seconds_per_setting,
            )
        ],
        dtype=np.float64,
    )
    reference_offsets = reference_offsets_seconds(
        shot_rate_per_second,
        fixed_overhead_seconds_per_setting,
    )
    reference_times = np.asarray(
        [
            session_index * SESSION_GAP_DAYS * 86400.0 + offset
            for session_index in range(CALENDAR_DAYS)
            for offset in reference_offsets
        ],
        dtype=np.float64,
    )
    return {
        "readout_probe_times_seconds": readout_times,
        "reference_probe_times_seconds": reference_times,
    }


def analysis_schedule(
    shot_rate_per_second: float,
    fixed_overhead_seconds_per_setting: float = 0.0,
) -> dict[str, np.ndarray]:
    schedule = design_schedule(shot_rate_per_second, fixed_overhead_seconds_per_setting)
    readout = schedule["readout_probe_times_seconds"]
    reference = schedule["reference_probe_times_seconds"]
    times = np.concatenate([readout, reference])
    shots = np.concatenate([
        np.full(len(readout), PROBE_SHOTS_PER_SETTING, dtype=np.int64),
        np.full(len(reference), ANCHOR_SHOTS_PER_SETTING, dtype=np.int64),
    ])
    instruments = np.asarray(["probe_burst"] * len(readout) + ["anchor_33"] * len(reference))
    order = np.argsort(times, kind="stable")
    return {"times_seconds": times[order], "shots": shots[order], "instrument_ids": instruments[order]}


def design_resolvable_floor_seconds(
    shot_rate_per_second: float,
    fixed_overhead_seconds_per_setting: float = 0.0,
) -> float:
    schedule = design_schedule(shot_rate_per_second, fixed_overhead_seconds_per_setting)
    gaps: list[float] = []
    for key in ("readout_probe_times_seconds", "reference_probe_times_seconds"):
        times = schedule[key]
        within_day = times[times < 86400.0]
        gaps.extend(float(value) for value in np.diff(within_day) if value > 0.0)
    return min(gaps)


def lag_edges(times_seconds: Sequence[float], minimum_lag_seconds: float) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64)
    return np.geomspace(max(minimum_lag_seconds * 0.8, 1e-3), max(80.0, float((times[-1] - times[0]) * 0.75)), 16)


def _ou_component(times: np.ndarray, variance: float, tau_seconds: float, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(len(times), dtype=np.float64)
    values[0] = rng.normal(0.0, np.sqrt(variance))
    for index in range(1, len(times)):
        coefficient = float(np.exp(-(times[index] - times[index - 1]) / tau_seconds))
        values[index] = coefficient * values[index - 1] + np.sqrt(variance * (1.0 - coefficient**2)) * rng.normal()
    return values


def generate_latent(
    dgp_name: str,
    times_seconds: Sequence[float],
    rng: np.random.Generator,
    *,
    mean_probability: float = PRIMARY_PROBABILITY,
    tau_minutes: float | None = None,
    process_variance: float | None = None,
) -> LatentSeries:
    times = np.asarray(times_seconds, dtype=np.float64)
    regimes = np.zeros(len(times), dtype=np.int32)
    events: list[int] = []
    artifact: np.ndarray | None = None
    calibration: dict[str, float | str | list[float]] = {}
    if dgp_name == "null_flat":
        probability = np.full(len(times), mean_probability)
    elif dgp_name == "ou":
        if tau_minutes is None or process_variance is None:
            raise ValueError("OU requires tau_minutes and process_variance")
        probability = mean_probability + _ou_component(times, process_variance, tau_minutes * 60.0, rng)
        calibration = {"tau_minutes": float(tau_minutes), "process_variance": float(process_variance)}
    elif dgp_name == "pink":
        if process_variance is None:
            raise ValueError("pink requires process_variance")
        kappa = np.power(10.0, np.arange(9, dtype=np.float64) / 4.0)
        taus = 10.0 * 60.0 * kappa
        components = [_ou_component(times, process_variance / len(taus), float(tau), rng) for tau in taus]
        probability = mean_probability + np.sum(components, axis=0)
        calibration = {
            "process_variance": float(process_variance),
            "kappa": [float(value) for value in kappa],
            "tau_minutes": [float(value / 60.0) for value in taus],
        }
    elif dgp_name in {"step_calendar", "step_as_ramp_artifact"}:
        event_time = 1.5 * 86400.0
        event_index = int(np.searchsorted(times, event_time, side="left"))
        direction = float(rng.choice((-1.0, 1.0)))
        probability = np.full(len(times), mean_probability)
        probability[event_index:] += direction * EVENT_JUMP
        regimes[event_index:] = 1
        events = [event_index]
        calibration = {"event_time_days": 1.5, "jump": direction * EVENT_JUMP, "latent_form": "piecewise_constant_step"}
        if dgp_name == "step_as_ramp_artifact":
            artifact = mean_probability + direction * EVENT_JUMP * (times - times[0]) / max(times[-1] - times[0], 1.0)
    elif dgp_name == "step_triggered":
        threshold = 0.018
        mean_crossing_seconds = 1.5 * 86400.0
        drift_per_second = threshold / mean_crossing_seconds
        diffusion_per_sqrt_second = threshold / np.sqrt(mean_crossing_seconds) * 0.18
        direction = float(rng.choice((-1.0, 1.0)))
        hidden_trigger = 0.0
        level = mean_probability
        probability = np.full(len(times), level, dtype=np.float64)
        current_regime = 0
        for index in range(1, len(times)):
            elapsed = times[index] - times[index - 1]
            hidden_trigger += direction * drift_per_second * elapsed + diffusion_per_sqrt_second * np.sqrt(elapsed) * rng.normal()
            if abs(hidden_trigger) >= threshold:
                events.append(index)
                current_regime += 1
                regimes[index:] = current_regime
                level = float(np.clip(level + rng.choice((-1.0, 1.0)) * EVENT_JUMP, 1e-5, 1.0 - 1e-5))
                hidden_trigger = rng.normal(0.0, threshold * 0.05)
                direction = float(rng.choice((-1.0, 1.0)))
            probability[index] = level
        calibration = {
            "trigger_threshold": threshold,
            "drift_per_day": drift_per_second * 86400.0,
            "diffusion_per_sqrt_day": diffusion_per_sqrt_second * np.sqrt(86400.0),
            "trigger_rule": "hidden controller state crossing; measured channel is piecewise constant and jumps only at events",
        }
    else:
        raise ValueError(f"unknown DGP: {dgp_name}")
    return LatentSeries(
        probability=np.clip(probability, 1.0e-5, 1.0 - 1.0e-5),
        regime_ids=regimes,
        event_indices=tuple(events),
        artifact_projection=None if artifact is None else np.clip(artifact, 1.0e-5, 1.0 - 1.0e-5),
        calibration=calibration,
    )


def simulate_observations(probability: np.ndarray, shots: Sequence[int] | int, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(shots, probability).astype(np.float64) / shots


def cell_grid(
    timing_profiles: Sequence[tuple[int, float]] = TIMING_PROFILES,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for shot_rate, overhead in timing_profiles:
        timing = {
            "shot_rate_per_second": int(shot_rate),
            "fixed_overhead_seconds_per_setting": float(overhead),
        }
        cells.append({"dgp": "null_flat", **timing, "tau_minutes": None, "process_variance": None})
        for tau_minutes in OU_TAU_MINUTES:
            for process_variance in PROCESS_VARIANCES:
                cells.append({
                    "dgp": "ou",
                    **timing,
                    "tau_minutes": tau_minutes,
                    "process_variance": process_variance,
                })
        for process_variance in PROCESS_VARIANCES:
            cells.append({
                "dgp": "pink",
                **timing,
                "tau_minutes": None,
                "process_variance": process_variance,
            })
        for dgp_name in ("step_calendar", "step_triggered", "step_as_ramp_artifact"):
            cells.append({"dgp": dgp_name, **timing, "tau_minutes": None, "process_variance": None})
    return cells


def _seed_for(payload: Mapping[str, Any], seed: int) -> int:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") + seed) % (2**32 - 1)


def _analyze_observed(
    observed: np.ndarray,
    latent: LatentSeries,
    times: np.ndarray,
    shots: np.ndarray,
    instrument_ids: np.ndarray,
    shot_rate: float,
    fixed_overhead_seconds_per_setting: float,
) -> dict[str, Any]:
    floor = design_resolvable_floor_seconds(shot_rate, fixed_overhead_seconds_per_setting)
    return sensing_economics.analyze_ou_sensing(
        values=observed,
        times_seconds=times,
        shots=shots,
        regime_ids=latent.regime_ids,
        burst_flags=np.zeros(len(times), dtype=bool),
        instrument_ids=instrument_ids,
        lag_edges_seconds=lag_edges(times, floor),
        effective_shots_per_second=effective_probe_shots_per_second(
            shot_rate,
            fixed_overhead_seconds_per_setting,
        ),
        maximum_interval_seconds=float(times[-1] - times[0]),
        interface_floor_seconds=floor,
    )


def run_cell(cell: Mapping[str, Any], replicates: int, seed: int) -> dict[str, Any]:
    shot_rate = float(cell["shot_rate_per_second"])
    overhead = float(cell["fixed_overhead_seconds_per_setting"])
    schedule = analysis_schedule(shot_rate, overhead)
    times = schedule["times_seconds"]
    shots = schedule["shots"]
    instrument_ids = schedule["instrument_ids"]
    design_floor = design_resolvable_floor_seconds(shot_rate, overhead)
    rng = np.random.default_rng(_seed_for(cell, seed))
    interior_claims: list[float] = []
    worth_claims: list[float] = []
    recovery: list[float] = []
    estimate_available: list[float] = []
    point_factor2: list[float] = []
    ci_covers_truth: list[float] = []
    ci_intersects_factor2: list[float] = []
    estimate_ratios: list[float] = []
    ci_width_octaves: list[float] = []
    regret_gate_condition: list[float] = []
    regret_success_unconditional: list[float] = []
    regret_success_conditional: list[float] = []
    regret_ratios: list[float] = []
    event_counts: list[int] = []
    artifact_claims: list[float] = []
    calibration: dict[str, Any] | None = None
    for _ in range(replicates):
        latent = generate_latent(
            str(cell["dgp"]),
            times,
            rng,
            tau_minutes=cell.get("tau_minutes"),
            process_variance=cell.get("process_variance"),
        )
        calibration = latent.calibration
        observed = simulate_observations(latent.probability, shots, rng)
        analysis = _analyze_observed(observed, latent, times, shots, instrument_ids, shot_rate, overhead)
        interior_claims.append(float(bool(analysis["interior_optimum_claim"])))
        worth_claims.append(float(bool(analysis["worth_sensing"])))
        event_counts.append(len(latent.event_indices))
        if cell["dgp"] == "ou":
            truth = sensing_economics.optimum_ou_interval(
                mean_probability=PRIMARY_PROBABILITY,
                effective_shots_per_second=effective_probe_shots_per_second(shot_rate, overhead),
                process_variance=float(cell["process_variance"]),
                tau_seconds=float(cell["tau_minutes"]) * 60.0,
                maximum_interval_seconds=float(times[-1] - times[0]),
            )
            true_interval = float(truth["interval_seconds"])
            available = analysis["t_star_ci_lower_seconds"] is not None
            estimate_available.append(float(available))
            if available:
                estimate = float(analysis["t_star_seconds"])
                lower = float(analysis["t_star_ci_lower_seconds"])
                upper = float(analysis["t_star_ci_upper_seconds"])
                point_factor2.append(float(true_interval / 2.0 <= estimate <= true_interval * 2.0))
                ci_covers_truth.append(float(lower <= true_interval <= upper))
                ci_intersects_factor2.append(float(upper >= true_interval / 2.0 and lower <= true_interval * 2.0))
                recovery.append(float(lower >= true_interval / 2.0 and upper <= true_interval * 2.0))
                estimate_ratios.append(estimate / true_interval)
                ci_width_octaves.append(float(np.log2(max(upper / max(lower, np.finfo(float).tiny), 1.0))))
            else:
                point_factor2.append(0.0)
                ci_covers_truth.append(0.0)
                ci_intersects_factor2.append(0.0)
                recovery.append(0.0)
            conditioned = bool(available and analysis["variance_gate"]["passed"] and analysis["worth_sensing"])
            regret_gate_condition.append(float(conditioned))
            if conditioned:
                estimated_interval = max(float(analysis["t_star_seconds"]), design_floor)
                attainable_truth = sensing_economics.optimum_ou_interval(
                    mean_probability=PRIMARY_PROBABILITY,
                    effective_shots_per_second=effective_probe_shots_per_second(shot_rate, overhead),
                    process_variance=float(cell["process_variance"]),
                    tau_seconds=float(cell["tau_minutes"]) * 60.0,
                    maximum_interval_seconds=float(times[-1] - times[0]),
                    minimum_interval_seconds=design_floor,
                )
                estimated_residual = float(sensing_economics.ou_residual_variance(
                    estimated_interval,
                    PRIMARY_PROBABILITY,
                    effective_probe_shots_per_second(shot_rate, overhead),
                    float(cell["process_variance"]),
                    float(cell["tau_minutes"]) * 60.0,
                ))
                regret = estimated_residual / float(attainable_truth["minimum_residual_variance"])
                passed_regret = float(regret <= REGRET_LIMIT)
                regret_ratios.append(regret)
                regret_success_conditional.append(passed_regret)
                regret_success_unconditional.append(passed_regret)
            else:
                regret_success_unconditional.append(0.0)
        if latent.artifact_projection is not None:
            artifact_observed = simulate_observations(latent.artifact_projection, shots, rng)
            artifact_latent = LatentSeries(
                probability=latent.artifact_projection,
                regime_ids=np.zeros(len(times), dtype=np.int32),
                event_indices=(),
                artifact_projection=None,
                calibration={"role": "deliberate_old_step_projection_only"},
            )
            artifact_analysis = _analyze_observed(
                artifact_observed,
                artifact_latent,
                times,
                shots,
                instrument_ids,
                shot_rate,
                overhead,
            )
            artifact_claims.append(float(bool(artifact_analysis["interior_optimum_claim"])))
    claim_rate = float(np.mean(interior_claims))
    expects_continuous = cell["dgp"] in {"ou", "pink"}
    return {
        "dgp": str(cell["dgp"]),
        "shot_rate_per_second": int(shot_rate),
        "fixed_overhead_seconds_per_setting": overhead,
        "anchor_setting_duration_seconds": setting_duration_seconds(ANCHOR_SHOTS_PER_SETTING, shot_rate, overhead),
        "probe_setting_duration_seconds": setting_duration_seconds(PROBE_SHOTS_PER_SETTING, shot_rate, overhead),
        "effective_probe_shots_per_second": effective_probe_shots_per_second(shot_rate, overhead),
        "calendar_days": CALENDAR_DAYS,
        "session_gap_days": SESSION_GAP_DAYS,
        "anchor_shots_per_setting": ANCHOR_SHOTS_PER_SETTING,
        "probe_shots_per_setting": PROBE_SHOTS_PER_SETTING,
        "design_resolvable_floor_seconds": design_floor,
        "n_sequences": int(replicates),
        "tau_minutes": cell.get("tau_minutes"),
        "process_variance": cell.get("process_variance"),
        "size": None,
        "power": claim_rate,
        "interior_optimum_claim_rate": claim_rate,
        "worth_sensing_rate": float(np.mean(worth_claims)),
        "tstar_ci_factor2_rate": float(np.mean(recovery)) if recovery else None,
        "tstar_estimate_available_rate": float(np.mean(estimate_available)) if estimate_available else None,
        "tstar_point_factor2_rate": float(np.mean(point_factor2)) if point_factor2 else None,
        "tstar_ci_covers_truth_rate": float(np.mean(ci_covers_truth)) if ci_covers_truth else None,
        "tstar_ci_intersects_factor2_rate": float(np.mean(ci_intersects_factor2)) if ci_intersects_factor2 else None,
        "tstar_estimate_to_truth_median": float(np.median(estimate_ratios)) if estimate_ratios else None,
        "tstar_ci_width_octaves_median": float(np.median(ci_width_octaves)) if ci_width_octaves else None,
        "regret_limit": REGRET_LIMIT if recovery else None,
        "regret_gate_condition_rate": float(np.mean(regret_gate_condition)) if regret_gate_condition else None,
        "tstar_regret_conditional_power": float(np.mean(regret_success_conditional)) if regret_success_conditional else None,
        "tstar_regret_unconditional_power": float(np.mean(regret_success_unconditional)) if regret_success_unconditional else None,
        "tstar_regret_median": float(np.median(regret_ratios)) if regret_ratios else None,
        "event_continuous_false_positive_rate": claim_rate if str(cell["dgp"]).startswith("step_") else None,
        "correct_behavior_rate": claim_rate if expects_continuous else 1.0 - claim_rate,
        "n_events_realized_mean": float(np.mean(event_counts)),
        "artifact_projection_claim_rate": float(np.mean(artifact_claims)) if artifact_claims else None,
        "expected_continuous_process": bool(expects_continuous),
        "size_pass": None,
        "power_pass": bool(claim_rate >= 0.8) if expects_continuous else bool(claim_rate <= 0.05),
        "joint_pass": None,
        "calibration": json.dumps(calibration or {}, ensure_ascii=False, sort_keys=True),
    }


def timing_fields(shot_rate: float, overhead: float) -> dict[str, float | int]:
    return {
        "shot_rate_per_second": int(shot_rate),
        "fixed_overhead_seconds_per_setting": float(overhead),
        "anchor_setting_duration_seconds": setting_duration_seconds(ANCHOR_SHOTS_PER_SETTING, shot_rate, overhead),
        "probe_setting_duration_seconds": setting_duration_seconds(PROBE_SHOTS_PER_SETTING, shot_rate, overhead),
        "effective_probe_shots_per_second": effective_probe_shots_per_second(shot_rate, overhead),
    }


def run_map_endpoint(
    shot_rate: int,
    replicates: int,
    seed: int,
    fixed_overhead_seconds_per_setting: float = 0.0,
) -> dict[str, Any]:
    overhead = float(fixed_overhead_seconds_per_setting)
    schedule = analysis_schedule(shot_rate, overhead)
    times = schedule["times_seconds"]
    shots = schedule["shots"]
    instruments = schedule["instrument_ids"]
    floor = design_resolvable_floor_seconds(shot_rate, overhead)
    edges = lag_edges(times, floor)
    rng = np.random.default_rng(_seed_for({"endpoint": "map", "shot_rate": shot_rate, "overhead": overhead}, seed))
    effective_rate = effective_probe_shots_per_second(shot_rate, overhead)
    e0_claims: list[float] = []
    e1_claims: list[float] = []
    for replicate in range(replicates):
        e0_latent = LatentSeries(np.full(len(times), E0_PROBABILITY), np.zeros(len(times), dtype=int), (), None, {})
        e0_analysis = sensing_economics.analyze_ou_sensing(
            values=simulate_observations(e0_latent.probability, shots, rng),
            times_seconds=times,
            shots=shots,
            regime_ids=e0_latent.regime_ids,
            burst_flags=np.zeros(len(times), dtype=bool),
            instrument_ids=instruments,
            lag_edges_seconds=edges,
            effective_shots_per_second=effective_rate,
            maximum_interval_seconds=float(times[-1] - times[0]),
            interface_floor_seconds=floor,
        )
        tau_minutes = 15.0 if replicate % 2 == 0 else 30.0
        e1_probability = PRIMARY_PROBABILITY + _ou_component(times, 6.1e-4, tau_minutes * 60.0, rng)
        e1_latent = LatentSeries(np.clip(e1_probability, 1e-5, 1 - 1e-5), np.zeros(len(times), dtype=int), (), None, {})
        e1_analysis = sensing_economics.analyze_ou_sensing(
            values=simulate_observations(e1_latent.probability, shots, rng),
            times_seconds=times,
            shots=shots,
            regime_ids=e1_latent.regime_ids,
            burst_flags=np.zeros(len(times), dtype=bool),
            instrument_ids=instruments,
            lag_edges_seconds=edges,
            effective_shots_per_second=effective_rate,
            maximum_interval_seconds=float(times[-1] - times[0]),
            interface_floor_seconds=floor,
        )
        e0_claims.append(float(bool(e0_analysis["worth_sensing"])))
        e1_claims.append(float(bool(e1_analysis["worth_sensing"])))
    return {
        "endpoint": "worth_sensing_map",
        **timing_fields(shot_rate, overhead),
        "design_resolvable_floor_seconds": floor,
        "size": float(np.mean(e0_claims)),
        "power": float(np.mean(e1_claims)),
        "size_pass": bool(np.mean(e0_claims) <= 0.05),
        "power_pass": bool(np.mean(e1_claims) >= 0.8),
    }


def run_cadence_endpoint(
    shot_rate: int,
    replicates: int,
    seed: int,
    fixed_overhead_seconds_per_setting: float = 0.0,
) -> dict[str, Any]:
    overhead = float(fixed_overhead_seconds_per_setting)
    rng = np.random.default_rng(_seed_for({"endpoint": "cadence", "shot_rate": shot_rate, "overhead": overhead}, seed))
    null_claims: list[float] = []
    injected_claims: list[float] = []
    pair_count = 24
    effective_rate = effective_probe_shots_per_second(shot_rate, overhead)
    p = PRIMARY_PROBABILITY
    null_fast_variance = p * (1.0 - p) / (effective_rate * 90.0)
    null_slow_variance = p * (1.0 - p) / (effective_rate * 360.0)
    injected_fast_variance = float(sensing_economics.ou_residual_variance(90.0, p, effective_rate, 6.1e-4, 300.0))
    injected_slow_variance = float(sensing_economics.ou_residual_variance(360.0, p, effective_rate, 6.1e-4, 300.0))
    for _ in range(replicates):
        null_fast = rng.normal(0.0, np.sqrt(null_fast_variance), pair_count) ** 2
        null_slow = rng.normal(0.0, np.sqrt(null_slow_variance), pair_count) ** 2
        injected_fast = rng.normal(0.0, np.sqrt(injected_fast_variance), pair_count) ** 2
        injected_slow = rng.normal(0.0, np.sqrt(injected_slow_variance), pair_count) ** 2
        null_claims.append(float(bool(sensing_economics.cadence_ratio_gate(null_fast, null_slow)["passed"])))
        injected_claims.append(float(bool(sensing_economics.cadence_ratio_gate(injected_fast, injected_slow)["passed"])))
    return {
        "endpoint": "cadence_pair_ratio",
        **timing_fields(shot_rate, overhead),
        "size": float(np.mean(null_claims)),
        "power": float(np.mean(injected_claims)),
        "size_pass": bool(np.mean(null_claims) <= 0.05),
        "power_pass": bool(np.mean(injected_claims) >= 0.8),
        "fast_seconds": 90.0,
        "slow_seconds": 360.0,
        "injection_tau_seconds": 300.0,
        "injection_process_variance": 6.1e-4,
    }


def _write_outputs(output: Path, report: Mapping[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite simulation output: {output}")
    output.mkdir(parents=True)
    results = list(report["results"])
    fieldnames = list(results[0].keys())
    if any(set(row) != set(fieldnames) for row in results):
        raise ValueError("simulation result rows do not share one field set")
    with (output / "simulation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    csv_fields = set(fieldnames)
    json_fields = set(results[0])
    if csv_fields != json_fields:
        raise ValueError("JSON/CSV result field sets differ")
    payload = dict(report)
    payload["field_set_audit"] = {
        "json_result_fields": sorted(json_fields),
        "csv_fields": sorted(csv_fields),
        "difference": sorted(json_fields.symmetric_difference(csv_fields)),
        "passed": True,
    }
    (output / "simulation_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    conclusion = report["conclusion"]
    endpoint_lines = "\n".join(
        f"- {row['endpoint']} @ (R={row['shot_rate_per_second']} shots/s, "
        f"c={row['fixed_overhead_seconds_per_setting']:.3f} s/setting): "
        f"size={row['size']:.4f}, power={row['power']:.4f}."
        for row in report["endpoint_summary"]
    )
    (output / "B4_B1_CONCLUSION_20260804.md").write_text(
        "# B-4 T-B1 功效与 size 结论\n\n"
        f"**Headline:** {conclusion['headline']}\n\n"
        "本结果仅为仿真，不是真机证据。所有门只读取观测值、时间戳、shots、regime 与 burst 元数据。\n\n"
        "## 成对 size / power\n\n"
        f"{endpoint_lines}\n\n"
        "## 事件误读\n\n"
        f"最大事件型连续漂移误报率：{conclusion['maximum_event_misread_rate']:.4f}。\n\n"
        "## 判定\n\n"
        f"{conclusion['decision']}\n",
        encoding="utf-8",
    )


def run(
    *,
    output: Path,
    replicates: int,
    seed: int,
    workers: int,
    timing_profiles: Sequence[tuple[int, float]] = TIMING_PROFILES,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    profiles = tuple(sorted({(int(rate), float(overhead)) for rate, overhead in timing_profiles}))
    if not profiles:
        raise ValueError("at least one timing profile is required")
    cells = cell_grid(profiles)
    results: list[dict[str, Any]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_cell = {executor.submit(run_cell, cell, replicates, seed): cell for cell in cells}
            for completed, future in enumerate(as_completed(future_to_cell), start=1):
                results.append(future.result())
                print(f"completed B1 cells {completed}/{len(cells)}", flush=True)
    else:
        for index, cell in enumerate(cells, start=1):
            results.append(run_cell(cell, replicates, seed))
            print(f"completed B1 cells {index}/{len(cells)}", flush=True)
    results.sort(key=lambda row: (
        int(row["shot_rate_per_second"]),
        float(row["fixed_overhead_seconds_per_setting"]),
        str(row["dgp"]),
        -1 if row["tau_minutes"] is None else float(row["tau_minutes"]),
        -1 if row["process_variance"] is None else float(row["process_variance"]),
    ))
    null_size = {
        (int(row["shot_rate_per_second"]), float(row["fixed_overhead_seconds_per_setting"])): float(row["interior_optimum_claim_rate"])
        for row in results if row["dgp"] == "null_flat"
    }
    for row in results:
        profile = (int(row["shot_rate_per_second"]), float(row["fixed_overhead_seconds_per_setting"]))
        size = null_size[profile]
        row["size"] = size
        row["size_pass"] = bool(size <= 0.05)
        row["joint_pass"] = bool(row["size_pass"] and row["power_pass"])

    endpoint_summary: list[dict[str, Any]] = []
    for shot_rate, overhead in profiles:
        profile_fields = timing_fields(shot_rate, overhead)
        profile_size = null_size[(shot_rate, overhead)]
        endpoint_summary.append({
            "endpoint": "interior_optimum_null",
            **profile_fields,
            "size": profile_size,
            "power": 1.0 - profile_size,
            "size_pass": bool(profile_size <= 0.05),
            "power_pass": bool(1.0 - profile_size >= 0.8),
        })
        reference = [
            row for row in results
            if row["dgp"] == "ou"
            and int(row["shot_rate_per_second"]) == shot_rate
            and float(row["fixed_overhead_seconds_per_setting"]) == overhead
            and float(row["process_variance"]) == 6.1e-4
            and float(row["tau_minutes"]) in {15.0, 30.0}
        ]
        tstar_power = float(np.mean([float(row["tstar_regret_conditional_power"] or 0.0) for row in reference]))
        tstar_unconditional_power = float(np.mean([float(row["tstar_regret_unconditional_power"] or 0.0) for row in reference]))
        gate_condition_rate = float(np.mean([float(row["regret_gate_condition_rate"] or 0.0) for row in reference]))
        factor2_descriptive = float(np.mean([float(row["tstar_ci_factor2_rate"] or 0.0) for row in reference]))
        endpoint_summary.append({
            "endpoint": "tstar_regret_c1p25_conditional",
            **profile_fields,
            "size": profile_size,
            "power": tstar_power,
            "size_pass": bool(profile_size <= 0.05),
            "power_pass": bool(tstar_power >= 0.8),
            "unconditional_power": tstar_unconditional_power,
            "gate_condition_rate": gate_condition_rate,
            "factor2_ci_recovery_descriptive": factor2_descriptive,
        })
        endpoint_summary.append(run_map_endpoint(shot_rate, replicates, seed, overhead))
        endpoint_summary.append(run_cadence_endpoint(shot_rate, replicates, seed, overhead))

    event_rows = [row for row in results if str(row["dgp"]).startswith("step_")]
    maximum_event_misread = max(float(row["event_continuous_false_positive_rate"]) for row in event_rows)
    for shot_rate, overhead in profiles:
        profile_event_rows = [
            row for row in event_rows
            if int(row["shot_rate_per_second"]) == shot_rate
            and float(row["fixed_overhead_seconds_per_setting"]) == overhead
        ]
        size = max(float(row["event_continuous_false_positive_rate"]) for row in profile_event_rows)
        endpoint_summary.append({
            "endpoint": "event_not_misread_as_continuous",
            **timing_fields(shot_rate, overhead),
            "size": size,
            "power": 1.0 - size,
            "size_pass": bool(size <= 0.05),
            "power_pass": bool(1.0 - size >= 0.8),
        })

    timing_profile_pass: dict[str, bool] = {}
    for shot_rate, overhead in profiles:
        rows = [
            row for row in endpoint_summary
            if int(row["shot_rate_per_second"]) == shot_rate
            and float(row["fixed_overhead_seconds_per_setting"]) == overhead
        ]
        key = f"R={shot_rate},c={overhead:.3f}"
        timing_profile_pass[key] = bool(all(bool(row["size_pass"]) and bool(row["power_pass"]) for row in rows))
    any_feasible = any(timing_profile_pass.values())
    headline = (
        "预注册前重定义的 regret(c=1.25) 终点存在至少一个吞吐档同时达到 size≤0.05 与全部目标 power≥0.8；factor-2 CI 坐标恢复仅保留为描述性副终点。"
        if any_feasible
        else "没有可行 cell 能在 regret(c=1.25) 终点同时达到 power≥0.8 与 size≤0.05；T*/U 曲线终点应从预注册删除。"
    )
    decision = "T-B1 regret 科学门通过；可在 Stage-1 冻结 regret 主终点，factor-2 坐标恢复降为描述性。" if any_feasible else "按工单失败分支降级为检测 + 判据地图（无最优节奏）；采集设计不变。"
    reference_schedule = design_schedule(*profiles[0])
    report = {
        "schema": "b4_design_power_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "replicates_per_cell": replicates,
        "seed": seed,
        "analysis_module": "src.adaptive.sensing_economics",
        "design": {
            "calendar_days": CALENDAR_DAYS,
            "session_gap_days": SESSION_GAP_DAYS,
            "anchor_shots_per_setting": ANCHOR_SHOTS_PER_SETTING,
            "probe_shots_per_setting": PROBE_SHOTS_PER_SETTING,
            "shot_rates_per_second": sorted({rate for rate, _ in profiles}),
            "fixed_overheads_seconds_per_setting": sorted({overhead for _, overhead in profiles}),
            "timing_profiles": [
                {"shot_rate_per_second": rate, "fixed_overhead_seconds_per_setting": overhead}
                for rate, overhead in profiles
            ],
            "burst_minutes_each_day": list(BURST_MINUTES),
            "reference_positions_zero_indexed": list(REFERENCE_POSITIONS_ZERO_INDEXED),
            "planning_reference_offsets_seconds_by_timing_profile": {
                f"R={rate},c={overhead:.3f}": [float(value) for value in reference_offsets_seconds(rate, overhead)]
                for rate, overhead in profiles
            },
            "planning_probe_offsets_seconds_by_timing_profile": {
                f"R={rate},c={overhead:.3f}": [float(value) for value in probe_observation_offsets_seconds(rate, overhead)]
                for rate, overhead in profiles
            },
            "reference_observations": int(len(reference_schedule["reference_probe_times_seconds"])),
            "readout_observations": int(len(reference_schedule["readout_probe_times_seconds"])),
            "instrument_ids": ["anchor_33", "probe_burst"],
            "pairing_rule": "same instrument only; no cross-role pairs",
            "wallclock_model": "setting_seconds = shots / shot_rate + fixed_overhead_seconds_per_setting",
            "anchor_pacing_status": "planning offsets use the two-parameter wallclock grid; T-B6 point estimates select the frozen cell",
            "cross_day_lags_present": True,
            "binomial_sampling": True,
        },
        "dgp_grid": {
            "names": list(DGP_NAMES),
            "ou_tau_minutes": list(OU_TAU_MINUTES),
            "process_variances": list(PROCESS_VARIANCES),
            "pink_kappa_rule": "10^(i/4), i=0..8; equal marginal variance per OU component",
        },
        "endpoint_definition": {
            "primary_tstar_endpoint": "P[sigma_res2(max(T_hat,T_floor(R)))/min_{T>=T_floor(R)} sigma_res2(T) <= 1.25 | detection gate and economic gate pass]",
            "regret_limit_frozen_for_this_run": REGRET_LIMIT,
            "design_resolvable_floor_seconds_by_timing_profile": {
                f"R={rate},c={overhead:.3f}": design_resolvable_floor_seconds(rate, overhead)
                for rate, overhead in profiles
            },
            "floor_role": "role-specific planning lag proxy; T-B6 interface P50/P90 remains unfrozen",
            "coordinate_factor2_recovery_role": "descriptive_secondary",
            "redefinition_timing": "before Stage-1 freeze and before data collection",
        },
        "results": results,
        "endpoint_summary": endpoint_summary,
        "conclusion": {
            "headline": headline,
            "decision": decision,
            "timing_profile_pass": timing_profile_pass,
            "maximum_event_misread_rate": maximum_event_misread,
        },
        "oracle_audit": {
            "gate_inputs": ["observed probabilities", "timestamps", "shots", "regime_id", "burst_flag", "instrument_id", "effective shot rate", "interface floor"],
            "truth_used_only_for_scoring": True,
            "scenario_branch_in_analysis_module": False,
        },
    }
    _write_outputs(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shot-rates", type=int, nargs="+", default=list(SHOT_RATES_PER_SECOND))
    parser.add_argument(
        "--setting-overheads",
        type=float,
        nargs="+",
        default=list(FIXED_OVERHEADS_SECONDS_PER_SETTING),
    )
    arguments = parser.parse_args()
    timing_profiles = tuple(
        (rate, overhead)
        for rate in arguments.shot_rates
        for overhead in arguments.setting_overheads
    )
    run(
        output=arguments.output,
        replicates=arguments.replicates,
        seed=arguments.seed,
        workers=max(1, arguments.workers),
        timing_profiles=timing_profiles,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
