#!/usr/bin/env python3
"""B-4 T-B5 controlled-injection cadence-pair simulation preflight.

The simulation executes the same scheduler order required for later hardware
use: sense -> five-gate shield -> digital inverse compensation -> mirror probe.
Randomized interface delay advances a controlled OU process through the same
clock used by the scheduler.  This file contains no hardware submission path;
simulation delay is not evidence about hardware queue latency.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive.bandit import Action, ShieldConfig, ShieldDecision, shield
from src.adaptive.sensing_economics import cadence_ratio_gate
from src.adaptive.task_metric_mirror import success_probability_from_raw_counts


SCHEMA = "b4_cadence_pair_loop_v1"
DEFAULT_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_v1.json"
DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_TB5_simulation_preflight_20260804")
CADENCE_LABELS = ("fast", "slow")


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().lower()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shots_per_setting(config: Mapping[str, Any], cadence_seconds: float) -> int:
    sensing = config["sensing"]
    calculated = int(round(float(sensing["effective_shots_per_second"]) * cadence_seconds / int(sensing["settings_per_update"])))
    return max(int(sensing["minimum_shots_per_setting"]), calculated)


def fast_increment_to_sensing_sigma_ratio(config: Mapping[str, Any]) -> float:
    controlled = config["controlled_ou"]
    sensing = config["sensing"]
    fast = float(config["cadence"]["fast_seconds"])
    tau = float(controlled["tau_seconds"])
    variance = float(controlled["stationary_process_variance"])
    phase_time = float(sensing["phase_time_seconds"])
    shots = shots_per_setting(config, fast)
    increment_sigma = math.sqrt(2.0 * variance * (1.0 - math.exp(-fast / tau)))
    sensing_sigma = 1.0 / (2.0 * phase_time * math.sqrt(shots))
    return increment_sigma / sensing_sigma


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA or config.get("status") not in {
        "simulation_preflight_frozen_before_tb6",
        "hardware_parameters_frozen_after_tb6",
    }:
        raise ValueError("unexpected T-B5 config schema or status")
    if int(config.get("simulation_days", 0)) != 3:
        raise ValueError("T-B5 simulation preflight requires exactly three days")
    orders = config.get("block_order_by_day")
    if orders != [["fast", "slow"], ["slow", "fast"], ["fast", "slow"]]:
        raise ValueError("fast/slow block order must alternate by day")
    if float(config["daily_window_seconds"]) != 2.0 * float(config["block_duration_seconds"]):
        raise ValueError("two cadence blocks must fill the registered continuous daily window")
    cadence = config["cadence"]
    if (float(cadence["fast_seconds"]), float(cadence["slow_seconds"])) != (90.0, 360.0):
        raise ValueError("pre-T-B6 simulation cadences must remain 90/360 seconds")
    if cadence.get("measured_interface_p90_seconds") is not None:
        p90 = float(cadence["measured_interface_p90_seconds"])
        if float(cadence["fast_seconds"]) < 2.0 * p90:
            raise ValueError("T_fast violates the frozen two-times-P90 lower bound")
    controlled = config["controlled_ou"]
    if float(controlled["tau_seconds"]) != 300.0 or float(controlled["stationary_process_variance"]) != 6.1e-4:
        raise ValueError("OU injection must retain T-B1 tau and process variance")
    if float(controlled["hard_clip_absolute"]) != 0.08:
        raise ValueError("controlled OU hard clip must remain 0.08")
    target = [float(value) for value in controlled["target_fast_increment_to_sensing_sigma_ratio"]]
    ratio = fast_increment_to_sensing_sigma_ratio(config)
    if not target[0] <= ratio <= target[1]:
        raise ValueError(f"T-B1 injection calibration ratio {ratio:.6f} is outside [{target[0]}, {target[1]}]")
    delay = config["simulated_interface_latency"]
    if delay.get("distribution") != "uniform" or not 0.0 < float(delay["minimum_seconds"]) < float(delay["maximum_seconds"]):
        raise ValueError("simulation latency requires a positive non-degenerate uniform distribution")
    shield_config(config)
    ladder = config["gate_amplitude_ladder"]
    if ladder.get("amplitudes") != [0.05, 0.1, 0.25] or ladder.get("expected_behaviors") != ["permit", "downscale", "abstain"]:
        raise ValueError("gate ladder must retain 0.05/0.10/0.25 three-behavior contract")
    mirror = config["mirror"]
    if int(mirror["repetitions_per_block"]) <= 0 or int(mirror["shots_per_task"]) <= 0:
        raise ValueError("mirror repetitions and shots must be positive")
    if mirror.get("depth_status") not in {
        "simulation_only_placeholder_pending_tb6",
        "frozen_after_tb6_hardware_ladder",
    }:
        raise ValueError("mirror depth status is outside the registered T-B6 transition")
    if config["claim_boundary"]["allowed"] != "controlled-injection cadence-economics validation":
        raise ValueError("T-B5 claim boundary changed")


def shield_config(config: Mapping[str, Any]) -> ShieldConfig:
    row = config["shield"]
    result = ShieldConfig(
        max_uncertainty=float(row["max_uncertainty"]),
        h1_range=(float(row["h1_range"][0]), float(row["h1_range"][1])),
        h2_range=(float(row["h2_range"][0]), float(row["h2_range"][1])),
        max_action_amplitude=float(row["max_action_amplitude"]),
        shot_budget_cap=int(row["shot_budget_cap"]),
    )
    if result.max_action_amplitude != 0.1:
        raise ValueError("shield amplitude threshold must remain 0.10")
    return result


@dataclass
class VirtualClock:
    seconds: float = 0.0

    def now(self) -> float:
        return float(self.seconds)

    def advance(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError("virtual clock cannot move backward")
        self.seconds += float(seconds)


@dataclass(frozen=True)
class AdvanceReceipt:
    planned_target_seconds: float
    started_seconds: float
    finished_seconds: float
    waited_seconds: float
    lateness_seconds: float
    tracked_duration_seconds: float
    integrated_residual_squared: float


class FakeCadenceBackend:
    """Controlled OU backend with independent OU, sensing, delay and mirror RNGs."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        validate_config(config)
        self.config = config
        self.clock = VirtualClock()
        controlled = config["controlled_ou"]
        self.tau = float(controlled["tau_seconds"])
        self.variance = float(controlled["stationary_process_variance"])
        self.clip = float(controlled["hard_clip_absolute"])
        self.integration_step = float(controlled["integration_step_seconds"])
        self.ou_random = np.random.default_rng(int(controlled["rng_seed"]))
        self.sensing_random = np.random.default_rng(int(config["sensing"]["rng_seed"]))
        self.latency_random = np.random.default_rng(int(config["simulated_interface_latency"]["rng_seed"]))
        self.mirror_random = np.random.default_rng(int(config["mirror"]["rng_seed"]))
        self.controlled_state = np.zeros(2, dtype=np.float64)
        self.compensation = np.zeros(2, dtype=np.float64)
        self.injection_log: list[dict[str, Any]] = []
        self.latencies: list[dict[str, Any]] = []
        self.shots_used = 0

    def _advance_one(self, seconds: float, reason: str, *, track_residual: bool) -> tuple[float, float]:
        if seconds <= 0.0:
            return 0.0, 0.0
        before = self.controlled_state.copy()
        rho = math.exp(-seconds / self.tau)
        innovation_sigma = math.sqrt(self.variance * (1.0 - rho * rho))
        innovation = self.ou_random.normal(0.0, innovation_sigma, 2)
        unconstrained = rho * before + innovation
        after = np.clip(unconstrained, -self.clip, self.clip)
        self.controlled_state = after
        self.clock.advance(seconds)
        residual_squared = float(np.dot(after + self.compensation, after + self.compensation))
        integrated = residual_squared * seconds if track_residual else 0.0
        self.injection_log.append({
            "event_index": len(self.injection_log),
            "reason": reason,
            "start_seconds": self.clock.now() - seconds,
            "finish_seconds": self.clock.now(),
            "elapsed_seconds": seconds,
            "rho": rho,
            "innovation_sigma": innovation_sigma,
            "controlled_state_before": before.tolist(),
            "innovation": innovation.tolist(),
            "controlled_state_unclipped": unconstrained.tolist(),
            "controlled_state_after": after.tolist(),
            "hard_clip_absolute": self.clip,
            "clipped": [bool(abs(unconstrained[index]) > self.clip) for index in range(2)],
            "tracked_residual": bool(track_residual),
        })
        return seconds if track_residual else 0.0, integrated

    def advance_to(self, target_seconds: float, reason: str, *, track_residual: bool) -> AdvanceReceipt:
        started = self.clock.now()
        target = float(target_seconds)
        wait = max(0.0, target - started)
        tracked = 0.0
        integrated = 0.0
        remaining = wait
        while remaining > 1e-12:
            step = min(remaining, self.integration_step) if track_residual else remaining
            duration, area = self._advance_one(step, reason, track_residual=track_residual)
            tracked += duration
            integrated += area
            remaining -= step
        finished = self.clock.now()
        return AdvanceReceipt(
            planned_target_seconds=target,
            started_seconds=started,
            finished_seconds=finished,
            waited_seconds=wait,
            lateness_seconds=max(0.0, started - target),
            tracked_duration_seconds=tracked,
            integrated_residual_squared=integrated,
        )

    def random_interface_delay(self, role: str, *, track_residual: bool) -> AdvanceReceipt:
        latency = self.config["simulated_interface_latency"]
        delay = float(self.latency_random.uniform(float(latency["minimum_seconds"]), float(latency["maximum_seconds"])))
        started = self.clock.now()
        receipt = self.advance_to(started + delay, f"simulated_interface_delay:{role}", track_residual=track_residual)
        self.latencies.append({"role": role, "delay_seconds": delay, "started_seconds": started, "finished_seconds": self.clock.now()})
        return receipt

    def sense(self, cadence_seconds: float) -> dict[str, Any]:
        latency = self.random_interface_delay("sense", track_residual=False)
        sensing = self.config["sensing"]
        phase_time = float(sensing["phase_time_seconds"])
        shots = shots_per_setting(self.config, cadence_seconds)
        field_rows: list[dict[str, Any]] = []
        estimates: list[float] = []
        sigmas: list[float] = []
        for field_index, value in enumerate(self.controlled_state):
            phase = 2.0 * float(value) * phase_time
            mean_y = -math.sin(phase)
            mean_z = math.cos(phase)
            plus_y = int(self.sensing_random.binomial(shots, (1.0 + mean_y) / 2.0))
            plus_z = int(self.sensing_random.binomial(shots, (1.0 + mean_z) / 2.0))
            observed_y = 2.0 * plus_y / shots - 1.0
            observed_z = 2.0 * plus_z / shots - 1.0
            radius_squared = max(observed_y * observed_y + observed_z * observed_z, 1e-12)
            estimate = math.atan2(-observed_y, observed_z) / (2.0 * phase_time)
            variance_y = max((1.0 - observed_y * observed_y) / shots, 1.0 / (shots * shots))
            variance_z = max((1.0 - observed_z * observed_z) / shots, 1.0 / (shots * shots))
            derivative_y = -observed_z / (2.0 * phase_time * radius_squared)
            derivative_z = observed_y / (2.0 * phase_time * radius_squared)
            sigma = math.sqrt(max(derivative_y * derivative_y * variance_y + derivative_z * derivative_z * variance_z, 1e-15))
            estimates.append(estimate)
            sigmas.append(sigma)
            field_rows.append({
                "field": f"h{field_index + 1}",
                "shots_per_axis": shots,
                "y_counts": {"plus": plus_y, "minus": shots - plus_y},
                "z_counts": {"plus": plus_z, "minus": shots - plus_z},
                "observed_y": observed_y,
                "observed_z": observed_z,
                "estimate": estimate,
                "shot_sigma": sigma,
            })
        self.shots_used += 2 * shots
        return {
            "estimator": sensing["estimator"],
            "estimator_lineage": sensing["estimator_lineage"],
            "execution_seconds": self.clock.now(),
            "simulated_interface_delay": asdict(latency),
            "estimates": estimates,
            "shot_sigmas": sigmas,
            "fields": field_rows,
            "shots_used_total": self.shots_used,
        }

    def apply_compensation(self, decision: ShieldDecision) -> None:
        self.compensation = np.asarray(decision.compensation[:2], dtype=np.float64)

    def release_compensation(self) -> None:
        self.compensation = np.zeros(2, dtype=np.float64)

    def mirror_batch(self) -> dict[str, Any]:
        latency = self.random_interface_delay("mirror", track_residual=True)
        mirror = self.config["mirror"]
        residual = self.controlled_state + self.compensation
        residual_squared = float(np.dot(residual, residual))
        probability = float(mirror["baseline_success_probability"]) * math.exp(-float(mirror["residual_sensitivity"]) * residual_squared)
        probability = float(np.clip(probability, float(mirror["minimum_success_probability"]), 1.0))
        shots = int(mirror["shots_per_task"])
        ideal = str(mirror["ideal_bitstring"])
        failure = str(mirror["failure_bitstring"])
        rows: list[dict[str, Any]] = []
        for replicate in range(int(mirror["repetitions_per_block"])):
            successes = int(self.mirror_random.binomial(shots, probability))
            raw_counts = {ideal: successes, failure: shots - successes}
            score = success_probability_from_raw_counts(raw_counts, ideal, shots=shots)
            rows.append({
                "replicate": replicate,
                "depth": int(mirror["depth"]),
                "depth_status": mirror["depth_status"],
                "ideal_bitstring": ideal,
                "raw_counts": raw_counts,
                "primary_metric": "success_probability",
                "primary_source": "raw_counts",
                "success_probability": float(score["success_probability"]),
                "loss": 1.0 - float(score["success_probability"]),
            })
        self.shots_used += shots * len(rows)
        return {
            "execution_seconds": self.clock.now(),
            "simulated_interface_delay": asdict(latency),
            "residual": residual.tolist(),
            "residual_squared": residual_squared,
            "sampling_probability": probability,
            "tasks": rows,
            "shots_used_total": self.shots_used,
        }


class CadenceScheduler:
    def __init__(self, backend: FakeCadenceBackend) -> None:
        self.backend = backend

    def wait_until(self, target_seconds: float, role: str, *, track_residual: bool = False) -> AdvanceReceipt:
        return self.backend.advance_to(target_seconds, f"scheduler:{role}", track_residual=track_residual)


def exceedance_probability(estimate: float, sigma: float, threshold: float) -> float:
    if sigma <= 0.0:
        return float(abs(estimate) > threshold)
    z = (threshold - abs(estimate)) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def observable_state(sensing: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    estimates = [float(value) for value in sensing["estimates"]]
    sigmas = [float(value) for value in sensing["shot_sigmas"]]
    threshold = float(config["shield"]["max_action_amplitude"])
    return {
        "mu_h1": estimates[0],
        "mu_h2": estimates[1],
        "sigma_h1": sigmas[0],
        "sigma_h2": sigmas[1],
        "z_features": [abs(estimates[0]), abs(estimates[1]), sigmas[0], sigmas[1]],
        "p_exceed": max(exceedance_probability(estimates[0], sigmas[0], threshold), exceedance_probability(estimates[1], sigmas[1], threshold)),
        "shots_used": int(sensing["shots_used_total"]),
        "ood_score": 0.0,
        "in_support": True,
    }


def analytic_average_tracking_mse(cadence_seconds: float, actual_interval_seconds: float, shot_sigmas: Sequence[float], config: Mapping[str, Any]) -> float:
    if actual_interval_seconds <= 0.0:
        raise ValueError("tracking interval must be positive")
    controlled = config["controlled_ou"]
    tau = float(controlled["tau_seconds"])
    variance = float(controlled["stationary_process_variance"])
    interval = float(actual_interval_seconds)
    drift_per_field = 2.0 * variance * (1.0 - (tau / interval) * (1.0 - math.exp(-interval / tau)))
    shot_term = sum(float(value) ** 2 for value in shot_sigmas)
    return shot_term + 2.0 * drift_per_field


def _base_shield_state(amplitude: float = 0.02, *, sigma: float = 0.005, shots_used: int = 0) -> dict[str, Any]:
    return {
        "mu_h1": float(amplitude),
        "mu_h2": 0.0,
        "sigma_h1": float(sigma),
        "sigma_h2": float(sigma),
        "z_features": [abs(float(amplitude)), 0.0, float(sigma), float(sigma)],
        "p_exceed": 0.0,
        "shots_used": int(shots_used),
        "ood_score": 0.0,
        "in_support": True,
    }


def shield_self_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    registered = shield_config(config)
    cases = {
        "permit": shield(_base_shield_state(), Action(1.0, 2, "act"), registered),
        "confidence": shield(_base_shield_state(sigma=registered.max_uncertainty * 2.0), Action(1.0, 2, "act"), registered),
        "physical_range": shield(_base_shield_state(amplitude=registered.h1_range[1] + 0.1), Action(0.0, 2, "act"), registered),
        "jz_reject": shield(_base_shield_state(), Action(1.0, 2, "act", gain_jz=0.01), registered),
        "action_amplitude": shield(_base_shield_state(amplitude=0.25), Action(1.0, 2, "act"), registered),
        "budget": shield(_base_shield_state(shots_used=registered.shot_budget_cap), Action(1.0, 2, "act"), registered),
    }
    expected = {
        "permit": None,
        "confidence": "confidence",
        "physical_range": "physical_range",
        "jz_reject": "jz_reject",
        "action_amplitude": "action_amplitude",
        "budget": "budget",
    }
    observed = {name: decision.gate for name, decision in cases.items()}
    if observed != expected or not cases["permit"].permitted:
        raise AssertionError(f"five-gate shield self-audit failed: {observed}")
    return {"passed": True, "expected": expected, "observed": observed}


def gate_ladder(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    registered = shield_config(config)
    ladder = config["gate_amplitude_ladder"]
    rows: list[dict[str, Any]] = []
    for amplitude, expected in zip(ladder["amplitudes"], ladder["expected_behaviors"], strict=True):
        value = float(amplitude)
        gain = float(ladder["middle_gain"]) if math.isclose(value, registered.max_action_amplitude) else 1.0
        decision = shield(_base_shield_state(amplitude=value), Action(gain, 2, "act"), registered)
        behavior = "abstain" if not decision.permitted else ("downscale" if gain < 1.0 else "permit")
        row = {
            "amplitude": value,
            "requested_gain": gain,
            "expected_behavior": expected,
            "observed_behavior": behavior,
            "shield_permitted": decision.permitted,
            "shield_gate": decision.gate,
            "compensation": list(decision.compensation),
            "passed": behavior == expected,
        }
        rows.append(row)
    if not all(row["passed"] for row in rows):
        raise AssertionError("0.05/0.10/0.25 gate ladder did not produce three registered behaviors")
    return rows


def load_tb1_cadence_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    registered = config["tb1_cadence_evidence"]
    path = Path(str(registered["path"]))
    actual_hash = digest_file(path)
    if actual_hash != str(registered["sha256"]).lower():
        raise ValueError("T-B1 cadence evidence SHA256 changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in report["rows"] if row.get("endpoint") == registered["required_endpoint"]]
    if len(rows) != 15:
        raise ValueError("T-B1 cadence evidence must cover all 15 timing cells")
    maximum_size = max(float(row["size"]) for row in rows)
    minimum_power = min(float(row["power"]) for row in rows)
    passed = bool(
        maximum_size <= float(registered["maximum_size"])
        and minimum_power >= float(registered["minimum_power"])
        and all(bool(row["joint_pass"]) for row in rows)
    )
    if not passed:
        raise RuntimeError("T-B1 cadence endpoint no longer passes size/power across timing grid")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "cell_count": len(rows),
        "maximum_size": maximum_size,
        "minimum_power": minimum_power,
        "passed": passed,
    }


def run_block(
    backend: FakeCadenceBackend,
    scheduler: CadenceScheduler,
    config: Mapping[str, Any],
    *,
    day_index: int,
    block_index: int,
    cadence_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if cadence_label not in CADENCE_LABELS:
        raise ValueError("unknown cadence label")
    cadence_seconds = float(config["cadence"][f"{cadence_label}_seconds"])
    block_duration = float(config["block_duration_seconds"])
    block_start = backend.clock.now()
    block_end = block_start + block_duration
    cycle_rows: list[dict[str, Any]] = []
    task_losses: dict[int, list[float]] = {index: [] for index in range(int(config["mirror"]["repetitions_per_block"]))}
    cycle_index = 0
    target = block_start
    while target + cadence_seconds <= block_end + 1e-9:
        sense_target = scheduler.wait_until(target, f"day{day_index}:{cadence_label}:sense{cycle_index}")
        sensed = backend.sense(cadence_seconds)
        state = observable_state(sensed, config)
        decision = shield(state, Action(1.0, 2, "act"), shield_config(config))
        backend.apply_compensation(decision)
        compensation_applied_seconds = backend.clock.now()
        hold = scheduler.wait_until(target + cadence_seconds, f"day{day_index}:{cadence_label}:hold{cycle_index}", track_residual=True)
        mirrored = backend.mirror_batch()
        mirror_delay = mirrored["simulated_interface_delay"]
        tracked_duration = float(hold.tracked_duration_seconds) + float(mirror_delay["tracked_duration_seconds"])
        integrated = float(hold.integrated_residual_squared) + float(mirror_delay["integrated_residual_squared"])
        tracking_mse = integrated / tracked_duration if tracked_duration > 0.0 else float("nan")
        predicted = analytic_average_tracking_mse(cadence_seconds, tracked_duration, sensed["shot_sigmas"], config)
        for task in mirrored["tasks"]:
            task_losses[int(task["replicate"])].append(float(task["loss"]))
        cycle_rows.append({
            "day_index": day_index,
            "block_index": block_index,
            "cadence": cadence_label,
            "cycle_index": cycle_index,
            "event_order": ["sense", "shield", "digital_inverse_compensation", "mirror_probe"],
            "planned_sense_target_seconds": target,
            "sense_target_receipt": asdict(sense_target),
            "sense": sensed,
            "observable_shield_state": state,
            "shield": {
                "permitted": decision.permitted,
                "gate": decision.gate,
                "reason": decision.reason,
                "action": asdict(decision.action),
                "compensation": list(decision.compensation),
            },
            "compensation_applied_seconds": compensation_applied_seconds,
            "planned_mirror_target_seconds": target + cadence_seconds,
            "hold_receipt": asdict(hold),
            "mirror": mirrored,
            "actual_tracking_interval_seconds": tracked_duration,
            "tracking_residual_mse": tracking_mse,
            "analytic_tracking_residual_mse": predicted,
            "prediction_ratio_observed_to_analytic": tracking_mse / predicted if predicted > 0.0 else None,
            "hardware_job_submitted": False,
        })
        cycle_index += 1
        target = block_start + cycle_index * cadence_seconds
    backend.release_compensation()
    scheduler.wait_until(max(block_end, backend.clock.now()), f"day{day_index}:{cadence_label}:block_end")
    if not cycle_rows:
        raise RuntimeError(f"cadence block {cadence_label} produced no complete cycle")
    task_summary = [
        {
            "pair_id": f"day{day_index:02d}-mirror{replicate:02d}",
            "day_index": day_index,
            "mirror_replicate": replicate,
            "cadence": cadence_label,
            "mean_loss": float(np.mean(losses)),
            "mean_success_probability": 1.0 - float(np.mean(losses)),
            "cycle_count": len(losses),
        }
        for replicate, losses in sorted(task_losses.items())
    ]
    block = {
        "day_index": day_index,
        "block_index": block_index,
        "cadence": cadence_label,
        "cadence_seconds": cadence_seconds,
        "planned_block_start_seconds": block_start,
        "planned_block_end_seconds": block_end,
        "actual_block_end_seconds": backend.clock.now(),
        "cycle_count": len(cycle_rows),
        "shield_permitted_cycles": sum(bool(row["shield"]["permitted"]) for row in cycle_rows),
        "tracking_residual_mse_mean": float(np.mean([row["tracking_residual_mse"] for row in cycle_rows])),
        "analytic_tracking_residual_mse_mean": float(np.mean([row["analytic_tracking_residual_mse"] for row in cycle_rows])),
        "task_summary": task_summary,
    }
    return block, cycle_rows


def paired_endpoint(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[int, int], dict[str, float]] = {}
    for block in blocks:
        for task in block["task_summary"]:
            key = (int(task["day_index"]), int(task["mirror_replicate"]))
            by_key.setdefault(key, {})[str(task["cadence"])] = float(task["mean_loss"])
    incomplete = [key for key, value in by_key.items() if set(value) != set(CADENCE_LABELS)]
    if incomplete:
        raise RuntimeError(f"incomplete fast/slow mirror pairs: {incomplete}")
    ordered = sorted(by_key)
    fast = np.asarray([by_key[key]["fast"] for key in ordered], dtype=np.float64)
    slow = np.asarray([by_key[key]["slow"] for key in ordered], dtype=np.float64)
    gate = cadence_ratio_gate(fast, slow)
    return {
        "metric": "raw-count mirror success_probability loss",
        "contrast": "fast_loss / slow_loss",
        "pair_count": len(ordered),
        "pair_ids": [f"day{day:02d}-mirror{replicate:02d}" for day, replicate in ordered],
        "fast_loss_mean": float(np.mean(fast)),
        "slow_loss_mean": float(np.mean(slow)),
        "fast_success_probability_mean": float(1.0 - np.mean(fast)),
        "slow_success_probability_mean": float(1.0 - np.mean(slow)),
        "direction_fast_loss_lower": bool(np.mean(fast) < np.mean(slow)),
        "ratio_gate": gate,
    }


def run_simulation(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T-B5 output: {output}")
    config = load_config(config_path)
    validate_config(config)
    tb1 = load_tb1_cadence_evidence(config)
    self_audit = shield_self_audit(config)
    ladder_templates = gate_ladder(config)
    daily_ladder: list[dict[str, Any]] = []
    backend = FakeCadenceBackend(config)
    scheduler = CadenceScheduler(backend)
    blocks: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    for day_index, order in enumerate(config["block_order_by_day"]):
        planned_day_start = day_index * float(config["inter_day_seconds"])
        backend.release_compensation()
        scheduler.wait_until(planned_day_start, f"day{day_index}:start")
        actual_day_start = backend.clock.now()
        day_blocks: list[dict[str, Any]] = []
        for block_index, cadence_label in enumerate(order):
            block, block_cycles = run_block(
                backend,
                scheduler,
                config,
                day_index=day_index,
                block_index=block_index,
                cadence_label=str(cadence_label),
            )
            blocks.append(block)
            day_blocks.append(block)
            cycles.extend(block_cycles)
        ladder_row = {
            **ladder_templates[day_index],
            "day_index": day_index,
            "execution_role": "single_gate_test_at_daily_block_end",
            "executed_seconds": backend.clock.now(),
            "hardware_job_submitted": False,
        }
        daily_ladder.append(ladder_row)
        day_rows.append({
            "day_index": day_index,
            "planned_start_seconds": planned_day_start,
            "actual_start_seconds": actual_day_start,
            "block_order": list(order),
            "actual_end_seconds": backend.clock.now(),
            "planned_window_seconds": float(config["daily_window_seconds"]),
            "block_count": len(day_blocks),
            "gate_test": ladder_row,
        })
    endpoint = paired_endpoint(blocks)
    clip_passed = all(
        max(abs(float(value)) for value in row["controlled_state_after"]) <= float(config["controlled_ou"]["hard_clip_absolute"]) + 1e-12
        for row in backend.injection_log
    )
    randomized_latency_passed = len({round(float(row["delay_seconds"]), 9) for row in backend.latencies}) > 1
    event_order_passed = all(row["event_order"] == ["sense", "shield", "digital_inverse_compensation", "mirror_probe"] for row in cycles)
    preflight_passed = bool(
        tb1["passed"]
        and self_audit["passed"]
        and all(row["passed"] for row in daily_ladder)
        and clip_passed
        and randomized_latency_passed
        and event_order_passed
        and endpoint["pair_count"] == int(config["simulation_days"]) * int(config["mirror"]["repetitions_per_block"])
        and endpoint["direction_fast_loss_lower"]
    )
    report = {
        "schema": "b4_tb5_simulation_preflight_v1",
        "simulation_only": True,
        "hardware_job_submitted": False,
        "claim": config["claim_boundary"]["allowed"],
        "forbidden_claims": config["claim_boundary"]["forbidden"],
        "config_path": str(config_path.resolve()),
        "config_sha256": digest_file(config_path),
        "calibration": {
            "fast_increment_to_sensing_sigma_ratio": fast_increment_to_sensing_sigma_ratio(config),
            "fast_shots_per_setting": shots_per_setting(config, float(config["cadence"]["fast_seconds"])),
            "slow_shots_per_setting": shots_per_setting(config, float(config["cadence"]["slow_seconds"])),
        },
        "tb1_cadence_endpoint_evidence": tb1,
        "shield_self_audit": self_audit,
        "gate_amplitude_ladder": daily_ladder,
        "days": day_rows,
        "blocks": blocks,
        "cadence_endpoint": endpoint,
        "tracking_curve": {
            "definition": "observed time-average squared controlled-field residual versus independent OU-plus-atan2 analytic prediction",
            "rows": [
                {
                    "day_index": row["day_index"],
                    "cadence": row["cadence"],
                    "cycle_index": row["cycle_index"],
                    "actual_interval_seconds": row["actual_tracking_interval_seconds"],
                    "observed_mse": row["tracking_residual_mse"],
                    "analytic_mse": row["analytic_tracking_residual_mse"],
                    "observed_to_analytic_ratio": row["prediction_ratio_observed_to_analytic"],
                }
                for row in cycles
            ],
        },
        "acceptance": {
            "tb1_endpoint4_size_power_passed": tb1["passed"],
            "fake_backend_true_scheduler_path_passed": event_order_passed,
            "randomized_simulated_interface_delay_passed": randomized_latency_passed,
            "controlled_ou_hard_clip_passed": clip_passed,
            "seed_and_injection_log_pending_seal": True,
            "preflight_passed_before_log_seal": preflight_passed,
        },
    }
    if not preflight_passed:
        raise RuntimeError(f"T-B5 simulation preflight failed: {report['acceptance']}, endpoint={endpoint}")

    output.mkdir(parents=True)
    injection_path = output / "controlled_ou_injection.jsonl"
    cycle_path = output / "cadence_observations.jsonl"
    ladder_path = output / "gate_amplitude_ladder.jsonl"
    injection_path.write_text("".join(canonical_json(row) + "\n" for row in backend.injection_log), encoding="utf-8")
    cycle_path.write_text("".join(canonical_json(row) + "\n" for row in cycles), encoding="utf-8")
    ladder_path.write_text("".join(canonical_json(row) + "\n" for row in daily_ladder), encoding="utf-8")
    pair_rows = [task for block in blocks for task in block["task_summary"]]
    with (output / "cadence_pair_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    report["acceptance"]["seed_and_injection_log_pending_seal"] = False
    report["acceptance"]["seed_and_injection_log_sealed"] = True
    report["acceptance"]["preflight_passed"] = True
    (output / "simulation_preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_files = [injection_path, cycle_path, ladder_path, output / "cadence_pair_rows.csv", output / "simulation_preflight_report.json"]
    manifest = {
        "schema": "b4_tb5_simulation_manifest_v1",
        "simulation_only": True,
        "hardware_job_submitted": False,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": digest_file(Path(__file__).resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": digest_file(config_path),
        "rng_seeds": {
            "controlled_ou": int(config["controlled_ou"]["rng_seed"]),
            "sensing": int(config["sensing"]["rng_seed"]),
            "simulated_interface_latency": int(config["simulated_interface_latency"]["rng_seed"]),
            "mirror": int(config["mirror"]["rng_seed"]),
        },
        "sealed_injection_log": {
            "path": str(injection_path.resolve()),
            "rows": len(backend.injection_log),
            "sha256": digest_file(injection_path),
        },
        "outputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in output_files
        ],
        "tb1_cadence_endpoint_evidence": tb1,
        "claim": config["claim_boundary"]["allowed"],
        "forbidden_claims": config["claim_boundary"]["forbidden"],
        "preflight_passed": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "B4_TB5_SIMULATION_PREFLIGHT.md").write_text(
        "# B-4 T-B5 仿真预飞\n\n"
        "状态：**通过（仅仿真）**。证据范围：受控注入下的节奏经济学验证；不是自然漂移或真机排队证据。\n\n"
        f"- T-B1 终点4：15 格全过；max size={tb1['maximum_size']:.4f}，min power={tb1['minimum_power']:.4f}。\n"
        f"- OU：tau=300 s，Var=6.1e-4，硬裁剪 ±0.08；fast 漂移/感知 sigma={report['calibration']['fast_increment_to_sensing_sigma_ratio']:.3f}。\n"
        f"- 调度：3 天；块顺序 fast/slow、slow/fast、fast/slow；完整镜像配对={endpoint['pair_count']}。\n"
        f"- 镜像 raw-count loss：fast={endpoint['fast_loss_mean']:.6f}，slow={endpoint['slow_loss_mean']:.6f}，ratio={endpoint['ratio_gate']['ratio']:.6f}。\n"
        f"- 五门 shield 自检：{self_audit['passed']}；0.05/0.10/0.25 行为：permit/downscale/abstain。\n"
        f"- 注入日志 SHA256：`{manifest['sealed_injection_log']['sha256']}`。\n\n"
        "待 T-B6：回填接口 P90，执行 `T_fast >= 2*P90`；冻结镜像深度；再启真机路径。\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simulate", action="store_true")
    arguments = parser.parse_args()
    if not arguments.simulate:
        raise RuntimeError("T-B5 currently exposes simulation preflight only; pass --simulate")
    report = run_simulation(arguments.config.resolve(), arguments.output.resolve())
    print(json.dumps({"preflight_passed": report["acceptance"]["preflight_passed"], "cadence_endpoint": report["cadence_endpoint"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
