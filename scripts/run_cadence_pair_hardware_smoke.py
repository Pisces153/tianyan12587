#!/usr/bin/env python3
"""T-B5 real-hardware smoke for the frozen cadence-pair control path.

This is deliberately smaller than the registered three-day endpoint.  It
executes one controlled OU injection through the real queue in this order:
sense -> five-gate shield -> digital inverse compensation -> matched mirror
pair.  It validates hardware plumbing and raw-count recovery only; it cannot
be reported as the cadence-pair efficacy endpoint or natural-drift tracking.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import b4_dry_run_common as common
from scripts import drift_campaign as campaign
from scripts import run_cadence_pair_loop as cadence
from src.adaptive.bandit import Action, shield
from src.adaptive.task_metric_mirror import build_random_clifford_mirror, success_probability_from_raw_counts
from src.backends.tianyan_native import assert_native_qcis
from src.backends.tianyan_v8_entangling import measurement_rotation, rx


DEFAULT_LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_v1.json"
DEFAULT_BACKEND_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"
DEFAULT_DEPTH_ARTIFACT = Path(r"E:\TianYan\XA-202609\artifacts\hardware\B4_TB6\mirror_depth.json")
DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\hardware\B4_TB5_hardware_smoke_20260805_v2")


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().lower()


def controlled_injection(config: Mapping[str, Any]) -> dict[str, Any]:
    row = config["controlled_ou"]
    elapsed = float(config["cadence"]["fast_seconds"])
    tau = float(row["tau_seconds"])
    variance = float(row["stationary_process_variance"])
    rho = math.exp(-elapsed / tau)
    innovation_sigma = math.sqrt(variance * (1.0 - rho * rho))
    generator = np.random.default_rng(int(row["rng_seed"]))
    unconstrained = generator.normal(0.0, innovation_sigma, 2)
    clipped = np.clip(unconstrained, -float(row["hard_clip_absolute"]), float(row["hard_clip_absolute"]))
    return {
        "rng_seed": int(row["rng_seed"]),
        "elapsed_seconds": elapsed,
        "rho": rho,
        "innovation_sigma": innovation_sigma,
        "unconstrained": unconstrained.tolist(),
        "injected_effective_fields": clipped.tolist(),
        "hard_clip_absolute": float(row["hard_clip_absolute"]),
        "clipped": [bool(left != right) for left, right in zip(unconstrained, clipped, strict=True)],
    }


def build_sensing_programs(
    physical_qubits: Sequence[int],
    fields: Sequence[float],
    phase_time_seconds: float,
    *,
    conditions: Sequence[str] = ("baseline", "injected"),
) -> list[dict[str, Any]]:
    """Sensing programs for the requested conditions, in the order requested.

    Both conditions together are the four-setting cycle.  ``conditions=("injected",)`` is
    the two-setting cycle used once the zero-field condition is amortised into one shared
    measurement per session, and ``conditions=("baseline",)`` is that shared measurement.
    The programs emitted are byte-identical either way, so amortising changes what is
    submitted and when, never what is computed.
    """
    qubits = [int(value) for value in physical_qubits]
    if len(qubits) != 6 or len(fields) != 2:
        raise ValueError("hardware sensing requires two fields on six registered qubits")
    available = {"baseline": (0.0, 0.0), "injected": tuple(float(value) for value in fields)}
    requested = [str(name) for name in conditions]
    if not requested or any(name not in available for name in requested):
        raise ValueError("sensing conditions must be a non-empty selection of baseline and injected")
    if len(set(requested)) != len(requested):
        raise ValueError("sensing conditions must be distinct")
    rows: list[dict[str, Any]] = []
    for condition, effective_fields in ((name, available[name]) for name in requested):
        for axis in ("Y", "Z"):
            lines = [
                *rx(qubits[0], 2.0 * float(effective_fields[0]) * phase_time_seconds),
                *rx(qubits[1], 2.0 * float(effective_fields[1]) * phase_time_seconds),
                *measurement_rotation(qubits[0], axis),
                *measurement_rotation(qubits[1], axis),
                *(f"M Q{qubit}" for qubit in qubits),
            ]
            qcis = "\n".join(lines)
            assert_native_qcis(qcis)
            rows.append({"condition": condition, "axis": axis, "qcis": qcis})
    return rows


def single_qubit_expectation(counts: np.ndarray, bit_position: int, shots: int) -> float:
    if counts.shape != (64,) or not 0 <= bit_position < 6 or int(counts.sum()) != shots:
        raise ValueError("invalid six-qubit raw-count vector")
    values = np.arange(64, dtype=np.int64)
    bits = (values >> (5 - bit_position)) & 1
    return float(np.dot(counts.astype(np.float64), 1.0 - 2.0 * bits) / shots)


def estimate_fields(
    results: Sequence[Mapping[str, Any]],
    physical_qubits: Sequence[int],
    shots: int,
    phase_time_seconds: float,
) -> dict[str, Any]:
    if len(results) != 4:
        raise ValueError("differential two-axis estimator requires baseline Y/Z and injected Y/Z results")
    counts = [campaign.result_counts(row, physical_qubits, shots) for row in results]
    estimates: list[float] = []
    sigmas: list[float] = []
    fields: list[dict[str, Any]] = []
    for field_index in range(2):
        condition_rows: list[dict[str, float]] = []
        phases: list[float] = []
        phase_variances: list[float] = []
        for offset in (0, 2):
            observed_y = single_qubit_expectation(counts[offset], field_index, shots)
            observed_z = single_qubit_expectation(counts[offset + 1], field_index, shots)
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


def build_mirror_pair(
    physical_qubits: Sequence[int],
    *,
    depth: int,
    seed: int,
    injected_fields: Sequence[float],
    compensation: Sequence[float],
    phase_time_seconds: float,
) -> list[dict[str, Any]]:
    qubits = [int(value) for value in physical_qubits]
    mirror = build_random_clifford_mirror(qubits, depth=depth, seed=seed)
    injection_lines = [
        *rx(qubits[0], 2.0 * float(injected_fields[0]) * phase_time_seconds),
        *rx(qubits[1], 2.0 * float(injected_fields[1]) * phase_time_seconds),
    ]
    compensation_lines = [
        *rx(qubits[0], 2.0 * float(compensation[0]) * phase_time_seconds),
        *rx(qubits[1], 2.0 * float(compensation[1]) * phase_time_seconds),
    ]
    rows: list[dict[str, Any]] = []
    for strategy, prefix in (
        ("fixed", injection_lines),
        ("adaptive", [*injection_lines, *compensation_lines]),
    ):
        qcis = "\n".join([*prefix, mirror.qcis])
        assert_native_qcis(qcis)
        rows.append({
            "strategy": strategy,
            "depth": depth,
            "seed": seed,
            "ideal_bitstring": mirror.ideal_bitstring,
            "qcis": qcis,
        })
    return rows


def execute(
    loop_config_path: Path,
    backend_config_path: Path,
    depth_artifact_path: Path,
    output: Path,
    *,
    confirm_hardware: bool,
    sensing_shots: int | None = None,
    mirror_shots: int = 4096,
    platform_factory: Callable[[Mapping[str, Any]], Any] = common.platform_from_config,
) -> dict[str, Any]:
    if not confirm_hardware:
        raise RuntimeError("T-B5 hardware smoke requires --confirm-hardware")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T-B5 hardware smoke: {output}")
    loop_config = json.loads(loop_config_path.read_text(encoding="utf-8"))
    cadence.validate_config(loop_config)
    if sensing_shots is None:
        sensing_shots = cadence.shots_per_setting(
            loop_config,
            float(loop_config["cadence"]["fast_seconds"]),
        )
    backend_config = common.load_config(backend_config_path)
    if backend_config["backend"]["backend_id"] != "tianyan-287":
        raise ValueError("T-B5 hardware smoke is frozen to the T287 primary backend")
    depth_artifact = json.loads(depth_artifact_path.read_text(encoding="utf-8"))
    depth = depth_artifact.get("selected_depth")
    if not depth_artifact.get("selection_passed") or depth is None:
        raise ValueError("T-B6 mirror-depth selection did not pass")
    phase_time = float(loop_config["sensing"]["phase_time_seconds"])
    physical = list(backend_config["backend"]["physical_qubits"])
    injection = controlled_injection(loop_config)
    platform = platform_factory(backend_config)

    sensing_programs = build_sensing_programs(physical, injection["injected_effective_fields"], phase_time)
    sensing_job, sensing_results_raw = common.run_job(
        platform=platform,
        config=backend_config,
        circuits=[row["qcis"] for row in sensing_programs],
        shots_per_setting=sensing_shots,
        name="XA202609_B4_TB5_HARDWARE_SMOKE_SENSE",
        max_wait_seconds=600,
        poll_seconds=5,
    )
    sensing_by_id = {str(row["experimentTaskId"]): row for row in sensing_results_raw}
    sensing_results = [sensing_by_id[value] for value in sensing_job["query_ids"]]
    sensing = estimate_fields(sensing_results, physical, sensing_shots, phase_time)
    observable = cadence.observable_state(sensing, loop_config)
    decision = shield(observable, Action(1.0, 2, "act"), cadence.shield_config(loop_config))
    compensation = list(decision.compensation[:2]) if decision.permitted else [0.0, 0.0]

    mirror_programs = build_mirror_pair(
        physical,
        depth=int(depth),
        seed=int(loop_config["mirror"]["rng_seed"]),
        injected_fields=injection["injected_effective_fields"],
        compensation=compensation,
        phase_time_seconds=phase_time,
    )
    mirror_job, mirror_results_raw = common.run_job(
        platform=platform,
        config=backend_config,
        circuits=[row["qcis"] for row in mirror_programs],
        shots_per_setting=mirror_shots,
        name="XA202609_B4_TB5_HARDWARE_SMOKE_MIRROR_PAIR",
        max_wait_seconds=600,
        poll_seconds=5,
    )
    mirror_by_id = {str(row["experimentTaskId"]): row for row in mirror_results_raw}
    mirror_scores: list[dict[str, Any]] = []
    for query_id, task in zip(mirror_job["query_ids"], mirror_programs, strict=True):
        counts = common.raw_counts(mirror_by_id[query_id], physical, mirror_shots)
        score = success_probability_from_raw_counts(counts, task["ideal_bitstring"], shots=mirror_shots)
        mirror_scores.append({
            "query_id": query_id,
            "strategy": task["strategy"],
            "depth": task["depth"],
            "seed": task["seed"],
            "ideal_bitstring": task["ideal_bitstring"],
            **score,
        })

    smoke_passed = bool(
        decision.permitted
        and len(sensing_results) == 4
        and len(mirror_scores) == 2
        and all(row["shots"] == mirror_shots for row in mirror_scores)
    )
    report = {
        "schema": "b4_tb5_hardware_smoke_v2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend_id": backend_config["backend"]["backend_id"],
        "evidence_scope": "real-queue controlled-injection closed-loop plumbing smoke only",
        "not_main_endpoint": True,
        "forbidden_claims": ["natural drift tracking", "three-day cadence-pair efficacy", "AEMTN-core necessity"],
        "event_order": ["controlled_injection", "sense", "five_gate_shield", "digital_inverse_compensation", "matched_mirror_pair"],
        "controlled_injection": injection,
        "sensing_job": sensing_job,
        "sensing": sensing,
        "observable_shield_state": observable,
        "shield": {
            "permitted": decision.permitted,
            "gate": decision.gate,
            "reason": decision.reason,
            "action": asdict(decision.action),
            "compensation": list(decision.compensation),
        },
        "mirror_job": mirror_job,
        "mirror_scores": mirror_scores,
        "hardware_jobs_submitted": 2,
        "hardware_job_submitted": True,
        "smoke_passed": smoke_passed,
        "source_artifacts": {
            "loop_config": str(loop_config_path.resolve()),
            "loop_config_sha256": digest_file(loop_config_path),
            "backend_config": str(backend_config_path.resolve()),
            "backend_config_sha256": digest_file(backend_config_path),
            "depth_artifact": str(depth_artifact_path.resolve()),
            "depth_artifact_sha256": digest_file(depth_artifact_path),
        },
    }
    output.mkdir(parents=True)
    raw_path = output / "raw_query_results.json"
    raw_path.write_text(json.dumps(campaign.json_ready({
        "sensing": sensing_results_raw,
        "mirror": mirror_results_raw,
    }), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output / "hardware_smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "b4_tb5_hardware_smoke_manifest_v2",
        "smoke_passed": smoke_passed,
        "hardware_jobs_submitted": 2,
        "outputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in (raw_path, report_path)
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop-config", type=Path, default=DEFAULT_LOOP_CONFIG)
    parser.add_argument("--backend-config", type=Path, default=DEFAULT_BACKEND_CONFIG)
    parser.add_argument("--depth-artifact", type=Path, default=DEFAULT_DEPTH_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sensing-shots", type=int)
    parser.add_argument("--mirror-shots", type=int, default=4096)
    parser.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()
    report = execute(
        arguments.loop_config.resolve(),
        arguments.backend_config.resolve(),
        arguments.depth_artifact.resolve(),
        arguments.output.resolve(),
        confirm_hardware=arguments.confirm_hardware,
        sensing_shots=arguments.sensing_shots,
        mirror_shots=arguments.mirror_shots,
    )
    print(json.dumps({
        "smoke_passed": report["smoke_passed"],
        "shield": report["shield"],
        "mirror_scores": report["mirror_scores"],
    }, ensure_ascii=False))
    return 0 if report["smoke_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
