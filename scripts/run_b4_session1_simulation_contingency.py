#!/usr/bin/env python3
"""Freeze and run the isolated B4/T176 Session 1 simulation contingency.

This module deliberately has no TianYan client, login, query, or submission path.  It
replays only Session 1 from the already-frozen hardware supplement plan.  The OU truth,
virtual timestamps, cadence order, shots, cycle identifiers, and registered pair mapping
come from that plan; measurement noise is generated from the frozen sensing model.

Synthetic records are written to a separate analysis artifact and carry an explicit
``record_origin=simulation`` tag.  They are never appended to the hardware journal and
are never admitted to the registered hardware adjudication endpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import platform as runtime_platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_cadence_pair_loop as cadence
from src.adaptive.bandit import Action, shield
from src.adaptive.cadence_permutation import cadence_ratio_permutation_gate
from src.adaptive import shared_baseline_sensing as shared_baseline


PLAN_SCHEMA = "b4_t176_session1_simulation_contingency_plan_v1"
REPORT_SCHEMA = "b4_t176_session1_simulation_contingency_report_v1"
MANIFEST_SCHEMA = "b4_t176_session1_simulation_contingency_manifest_v1"
JOURNAL_SCHEMA = "b4_t176_session1_simulation_journal_v1"
SOURCE_PLAN_SCHEMA = "b4_cadence_pair_hardware_supplement_plan_v1"
SESSION_INDEX = 1
EXPECTED_BLOCK_ORDER = ["slow", "fast"]
EXPECTED_CYCLES_PER_CADENCE = 20
EXPECTED_CYCLES = 2 * EXPECTED_CYCLES_PER_CADENCE
EXPECTED_PAIRS = EXPECTED_CYCLES_PER_CADENCE
EXPECTED_BASELINES = 2
RECORD_TAGS = {
    "record_origin": "simulation",
    "simulation_only": True,
    "hardware_job_submitted": False,
    "registered_hardware_endpoint_contribution": "none",
    "pooling_permitted": False,
}
FORBIDDEN_RECORD_KEYS = {
    "query_id",
    "query_ids",
    "counts_path",
    "raw_counts",
    "hardware_snapshot_id",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest().lower()


def digest_payload(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_new_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _source_hash(plan: Mapping[str, Any], key: str) -> str:
    source_hashes = plan.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or key not in source_hashes:
        raise ValueError(f"source plan is missing source_hashes.{key}")
    return str(source_hashes[key]).lower()


def _session(source_plan: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    matches = [
        row
        for row in source_plan.get("sessions", [])
        if int(row.get("session_index", -1)) == int(index)
    ]
    if len(matches) != 1:
        raise ValueError(f"source plan needs exactly one Session {index}")
    return matches[0]


def _validate_session1_contract(
    source_plan: Mapping[str, Any],
    loop_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if source_plan.get("schema") != SOURCE_PLAN_SCHEMA:
        raise ValueError("unexpected B4 hardware supplement-plan schema")
    if str(source_plan.get("backend_id")) != "tianyan176":
        raise ValueError("simulation contingency is pinned to the T176 supplement plan")
    if str(loop_config.get("schema")) != "b4_cadence_pair_loop_v1":
        raise ValueError("unexpected B4 cadence loop-config schema")

    correction = loop_config.get("collection_correction")
    source_correction = source_plan.get("collection_correction")
    if not isinstance(correction, Mapping) or not isinstance(source_correction, Mapping):
        raise ValueError("v4 collection correction is required")
    if canonical_json(correction) != canonical_json(source_correction):
        raise ValueError("source plan and loop config carry different collection corrections")
    if correction.get("simulation_pooling_permitted") is not False:
        raise ValueError("simulation pooling must be explicitly forbidden")
    if str(correction.get("primary_adjudication")) != "cadence_ratio_permutation_gate":
        raise ValueError("unexpected primary hardware adjudication")
    if int(correction.get("sessions", 0)) != 2:
        raise ValueError("the registered v4 design requires two sessions")
    if int(correction.get("cycles_per_cadence_per_session", 0)) != EXPECTED_CYCLES_PER_CADENCE:
        raise ValueError("unexpected registered cycles per cadence")

    expected = source_plan.get("expected")
    if not isinstance(expected, Mapping):
        raise ValueError("source plan is missing expected collection totals")
    expected_contract = {
        "sessions": 2,
        "complete_cadence_pairs": 40,
        "cycles_per_cadence_per_session": EXPECTED_CYCLES_PER_CADENCE,
        "minimum_adjudicated_cycle_pairs": 30,
        "minimum_sessions_per_block_order": 1,
    }
    for key, value in expected_contract.items():
        if int(expected.get(key, -1)) != value:
            raise ValueError(f"source expected.{key} changed")

    session = _session(source_plan, SESSION_INDEX)
    if list(session.get("block_order", [])) != EXPECTED_BLOCK_ORDER:
        raise ValueError("Session 1 must retain the slow-first, fast-second block order")
    if not math.isclose(float(session.get("virtual_start_seconds", -1.0)), 86400.0):
        raise ValueError("Session 1 virtual start changed")

    baselines = list(session.get("baseline_measurements", []))
    if len(baselines) != EXPECTED_BASELINES:
        raise ValueError("Session 1 needs opening and closing baseline measurements")
    if [str(row.get("position")) for row in baselines] != ["session_start", "session_end"]:
        raise ValueError("Session 1 baseline positions changed")

    cycles = list(session.get("cycles", []))
    if len(cycles) != EXPECTED_CYCLES:
        raise ValueError("Session 1 needs exactly forty cycles")
    counts = {
        label: sum(str(row.get("cadence")) == label for row in cycles)
        for label in ("fast", "slow")
    }
    if counts != {"fast": EXPECTED_CYCLES_PER_CADENCE, "slow": EXPECTED_CYCLES_PER_CADENCE}:
        raise ValueError(f"Session 1 cadence counts changed: {counts}")
    observed_order = []
    for row in cycles:
        label = str(row.get("cadence"))
        if not observed_order or observed_order[-1] != label:
            observed_order.append(label)
        if len(row.get("sense_fields", [])) != 2 or len(row.get("mirror_fields", [])) != 2:
            raise ValueError("every cycle needs two frozen controlled-field values")
        if int(row.get("sensing_shots_per_setting", 0)) != int(
            correction["sensing_shots_per_setting"]
        ):
            raise ValueError("cycle sensing shots changed")
        if int(row.get("mirror_shots_per_task", 0)) != int(loop_config["mirror"]["shots_per_task"]):
            raise ValueError("cycle mirror shots changed")
    if observed_order != EXPECTED_BLOCK_ORDER:
        raise ValueError(f"Session 1 cycle order changed: {observed_order}")

    by_pair: dict[str, set[str]] = {}
    for row in cycles:
        pair_id = str(row.get("registered_pair_id", ""))
        if not pair_id:
            raise ValueError("every Session 1 cycle needs a registered pair ID")
        by_pair.setdefault(pair_id, set()).add(str(row["cadence"]))
    if len(by_pair) != EXPECTED_PAIRS or any(value != {"fast", "slow"} for value in by_pair.values()):
        raise ValueError("Session 1 pair mapping is incomplete")

    mirror_jobs = sum(bool(row.get("mirror_seeds")) for row in cycles)
    if mirror_jobs != int(correction["mirror_qc_jobs_per_session"]):
        raise ValueError("Session 1 mirror-QC subset changed")
    return session


def _derived_seed(base_plan_sha256: str, loop_config_sha256: str, label: str) -> int:
    material = f"{base_plan_sha256}|{loop_config_sha256}|{label}|v1".encode("utf-8")
    # NumPy accepts an unsigned 64-bit seed.  Keeping the high bit clear also makes the
    # value portable to runtimes that only admit signed 64-bit integers.
    return int.from_bytes(sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def freeze_plan(
    *,
    base_plan_path: Path,
    loop_config_path: Path,
    plan_path: Path,
    output_dir: Path,
    monte_carlo_replicates: int = 40_000,
    permutation_count: int = 20_000,
    created_at_utc: datetime | None = None,
) -> dict[str, Any]:
    base_plan_path = base_plan_path.resolve()
    loop_config_path = loop_config_path.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if plan_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen simulation plan: {plan_path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to target an existing output directory: {output_dir}")
    if int(monte_carlo_replicates) < 100:
        raise ValueError("Monte Carlo envelope needs at least 100 replicates")
    if int(permutation_count) < 100:
        raise ValueError("counterfactual trace gate needs at least 100 permutations")

    source_plan = load_json(base_plan_path)
    loop_config = load_json(loop_config_path)
    session = _validate_session1_contract(source_plan, loop_config)
    base_sha = digest_file(base_plan_path)
    config_sha = digest_file(loop_config_path)
    if config_sha != _source_hash(source_plan, "loop_config_sha256"):
        raise ValueError("current v4 loop config does not match the frozen supplement plan")

    seeds = {
        label: _derived_seed(base_sha, config_sha, label)
        for label in (
            "session_baseline",
            "session_sensing",
            "session_mirror_qc",
            "session1_endpoint_ensemble",
            "session1_trace_permutation",
        )
    }
    correction = loop_config["collection_correction"]
    runner_path = Path(__file__).resolve()
    payload = {
        "schema": PLAN_SCHEMA,
        "status": "frozen_before_hardware_outcome_unsealing",
        "created_at_utc": iso(created_at_utc or utc_now()),
        "purpose": (
            "complete the B4/T176 project workflow with an isolated simulated Session 1 "
            "while the hardware backend is unavailable"
        ),
        "project_status_on_success": "COMPLETED_WITH_SIMULATION_CONTINGENCY",
        "registered_hardware_endpoint_status_on_success": "PENDING_HARDWARE",
        "source": {
            "base_hardware_plan": str(base_plan_path),
            "base_hardware_plan_sha256": base_sha,
            "loop_config": str(loop_config_path),
            "loop_config_sha256": config_sha,
            "runner": str(runner_path),
            "runner_sha256": digest_file(runner_path),
            "source_plan_runner_sha256": _source_hash(source_plan, "runner_sha256"),
            "session1_payload_sha256": digest_payload(session),
        },
        "session1_contract": {
            "session_index": SESSION_INDEX,
            "block_order": EXPECTED_BLOCK_ORDER,
            "virtual_start_seconds": float(session["virtual_start_seconds"]),
            "cycles": EXPECTED_CYCLES,
            "cycles_per_cadence": EXPECTED_CYCLES_PER_CADENCE,
            "registered_pair_mappings": EXPECTED_PAIRS,
            "baseline_measurements": EXPECTED_BASELINES,
            "sensing_shots_per_setting": int(correction["sensing_shots_per_setting"]),
            "baseline_start_shots_per_setting": int(correction["baseline_shots_per_setting"]),
            "baseline_end_shots_per_setting": int(correction["baseline_end_shots_per_setting"]),
            "mirror_qc_jobs": int(correction["mirror_qc_jobs_per_session"]),
            "mirror_shots_per_task": int(loop_config["mirror"]["shots_per_task"]),
        },
        "measurement_model": {
            "controlled_truth": "sense_fields and mirror_fields from frozen Session 1 plan",
            "axis_expectations": "Y=-sin(2*h*T), Z=cos(2*h*T)",
            "sampling": "independent binomial shot sampling for each controlled field and axis",
            "baseline": "one synthetic zero-field opening measurement shared by both cadence blocks",
            "estimator": str(loop_config["sensing"]["estimator"]),
            "shield": "the same five-gate shield and Action(1.0, 2, act) computation used by the hardware runner",
            "registered_endpoint_formula": "sum((mirror_fields + compensation)**2)",
            "mirror_qc": (
                "simulation-only fixed-zero/adaptive matched-pair model; descriptive and not an endpoint input"
            ),
        },
        "rng_seeds": seeds,
        "monte_carlo": {
            "replicates": int(monte_carlo_replicates),
            "endpoint_law": "independent Exponential draws at the frozen expected fast/slow means",
            "scope": "simulation-only Session 1 ratio envelope",
            "permutation_count_for_single_trace": int(permutation_count),
        },
        "record_tags": dict(RECORD_TAGS),
        "separation_contract": {
            "output_dir": str(output_dir),
            "hardware_journal_read": False,
            "hardware_raw_counts_read": False,
            "hardware_npz_read": False,
            "hardware_scientific_result_read": False,
            "hardware_directory_modified": False,
            "query_ids_generated": False,
            "simulation_pooling_permitted": False,
            "registered_hardware_endpoint_contribution": "none",
        },
        "claim_boundary": {
            "allowed": [
                "simulation-assisted project workflow completion",
                "frozen-model controlled-injection mechanism validation",
                "counterfactual Session 1 sensitivity analysis",
            ],
            "forbidden": [
                "claiming that Session 1 was collected on hardware",
                "claiming a registered hardware endpoint PASS from synthetic pairs",
                "pooling synthetic pairs into the hardware adjudication endpoint",
                *list(loop_config["claim_boundary"]["forbidden"]),
            ],
        },
    }
    write_new_json(plan_path, payload)
    return payload


def validate_frozen_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    plan_path = plan_path.resolve()
    plan = load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unexpected simulation-contingency plan schema")
    if plan.get("status") != "frozen_before_hardware_outcome_unsealing":
        raise ValueError("simulation-contingency plan is not frozen")
    if plan.get("record_tags") != RECORD_TAGS:
        raise ValueError("simulation record tags changed")
    separation = plan.get("separation_contract")
    if not isinstance(separation, Mapping):
        raise ValueError("missing separation contract")
    required_false = (
        "hardware_journal_read",
        "hardware_raw_counts_read",
        "hardware_npz_read",
        "hardware_scientific_result_read",
        "hardware_directory_modified",
        "query_ids_generated",
        "simulation_pooling_permitted",
    )
    if any(separation.get(key) is not False for key in required_false):
        raise ValueError("simulation/hardware separation contract changed")

    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("frozen plan is missing source hashes")
    runner = Path(str(source["runner"]))
    base_plan_path = Path(str(source["base_hardware_plan"]))
    loop_config_path = Path(str(source["loop_config"]))
    expected_hashes = {
        runner: str(source["runner_sha256"]),
        base_plan_path: str(source["base_hardware_plan_sha256"]),
        loop_config_path: str(source["loop_config_sha256"]),
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen source is missing: {path}")
        if digest_file(path) != expected.lower():
            raise ValueError(f"frozen source hash changed: {path}")
    if runner.resolve() != Path(__file__).resolve():
        raise ValueError("frozen plan names a different simulation runner")

    source_plan = load_json(base_plan_path)
    loop_config = load_json(loop_config_path)
    session = _validate_session1_contract(source_plan, loop_config)
    if digest_payload(session) != str(source["session1_payload_sha256"]).lower():
        raise ValueError("frozen Session 1 payload changed")
    output = Path(str(separation["output_dir"]))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite simulation output: {output}")
    return plan, source_plan, loop_config, session


def _sample_axes(
    true_fields: Sequence[float],
    *,
    shots: int,
    phase_time_seconds: float,
    generator: np.random.Generator,
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    if len(true_fields) != 2:
        raise ValueError("the controlled simulation requires exactly two fields")
    if int(shots) <= 0:
        raise ValueError("shots must be positive")
    expectations: list[dict[str, float]] = []
    synthetic_counts: list[dict[str, Any]] = []
    for field_index, value in enumerate(true_fields):
        phase = 2.0 * float(value) * float(phase_time_seconds)
        mean_y = -math.sin(phase)
        mean_z = math.cos(phase)
        plus_y = int(generator.binomial(int(shots), (1.0 + mean_y) / 2.0))
        plus_z = int(generator.binomial(int(shots), (1.0 + mean_z) / 2.0))
        observed_y = 2.0 * plus_y / int(shots) - 1.0
        observed_z = 2.0 * plus_z / int(shots) - 1.0
        expectations.append({"observed_y": observed_y, "observed_z": observed_z})
        synthetic_counts.append({
            "field": f"h{field_index + 1}",
            "true_field": float(value),
            "shots_per_axis": int(shots),
            "y": {"plus": plus_y, "minus": int(shots) - plus_y},
            "z": {"plus": plus_z, "minus": int(shots) - plus_z},
        })
    return expectations, synthetic_counts


def _simulate_baseline(
    measurement: Mapping[str, Any],
    *,
    phase_time_seconds: float,
    generator: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shots = int(measurement["shots_per_setting"])
    expectations, counts = _sample_axes(
        (0.0, 0.0),
        shots=shots,
        phase_time_seconds=phase_time_seconds,
        generator=generator,
    )
    record = shared_baseline.baseline_record(
        expectations,
        shots,
        session_index=int(measurement["session_index"]),
        position=str(measurement["position"]),
    )
    return record, counts


def _mirror_qc(
    cycle: Mapping[str, Any],
    compensation: Sequence[float],
    loop_config: Mapping[str, Any],
    generator: np.random.Generator,
) -> list[dict[str, Any]]:
    if not cycle.get("mirror_seeds"):
        return []
    mirror = loop_config["mirror"]
    fields = np.asarray(cycle["mirror_fields"], dtype=np.float64)
    adaptive = fields + np.asarray(compensation, dtype=np.float64)
    residuals = {
        "fixed_zero": fields,
        "adaptive": adaptive,
    }
    rows: list[dict[str, Any]] = []
    for replicate, seed in enumerate(cycle["mirror_seeds"]):
        for strategy, residual in residuals.items():
            residual_squared = float(np.dot(residual, residual))
            probability = float(mirror["baseline_success_probability"]) * math.exp(
                -float(mirror["residual_sensitivity"]) * residual_squared
            )
            probability = float(
                np.clip(probability, float(mirror["minimum_success_probability"]), 1.0)
            )
            shots = int(cycle["mirror_shots_per_task"])
            successes = int(generator.binomial(shots, probability))
            rows.append({
                **RECORD_TAGS,
                "cycle_id": str(cycle["cycle_id"]),
                "source_registered_pair_id": str(cycle["registered_pair_id"]),
                "replicate": int(replicate),
                "seed": int(seed),
                "strategy": strategy,
                "depth": int(mirror["depth"]),
                "shots": shots,
                "sampling_probability": probability,
                "synthetic_counts": {
                    "ideal": successes,
                    "non_ideal": shots - successes,
                },
                "success_probability": successes / float(shots),
                "residual_squared": residual_squared,
                "analysis_role": "simulation_only_descriptive_mirror_qc",
            })
    return rows


def _assert_no_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_RECORD_KEYS:
                raise ValueError(f"forbidden hardware-origin key in simulation record: {path}.{key}")
            _assert_no_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_no_forbidden_keys(item, path=f"{path}[{index}]")


def _append_chained(records: list[dict[str, Any]], event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "schema": JOURNAL_SCHEMA,
        **RECORD_TAGS,
        "event": str(event),
        "sequence": len(records),
        "previous_record_sha256": records[-1]["record_sha256"] if records else None,
        **dict(payload),
    }
    _assert_no_forbidden_keys(row)
    row["record_sha256"] = digest_payload(row)
    records.append(row)
    return row


def _pair_rows(cycles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in cycles:
        pair_id = str(row["source_registered_pair_id"])
        by_pair.setdefault(pair_id, {})[str(row["cadence"])] = row
    incomplete = [key for key, value in by_pair.items() if set(value) != {"fast", "slow"}]
    if incomplete:
        raise RuntimeError(f"simulated Session 1 has incomplete cadence pairs: {incomplete}")
    rows = []
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        rows.append({
            **RECORD_TAGS,
            "source_registered_pair_id": pair_id,
            "session_index": SESSION_INDEX,
            "fast_cycle_id": str(pair["fast"]["cycle_id"]),
            "slow_cycle_id": str(pair["slow"]["cycle_id"]),
            "fast_endpoint_squared_residual": float(pair["fast"]["endpoint_squared_residual"]),
            "slow_endpoint_squared_residual": float(pair["slow"]["endpoint_squared_residual"]),
        })
    if len(rows) != EXPECTED_PAIRS:
        raise RuntimeError("simulated Session 1 did not produce twenty pair rows")
    return rows


def _monte_carlo_envelope(
    loop_config: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    correction = loop_config["collection_correction"]
    fast_mean = float(correction["expected_fast_endpoint_mean"])
    slow_mean = float(correction["expected_slow_endpoint_mean"])
    generator = np.random.default_rng(int(seed))
    fast = generator.exponential(fast_mean, (int(replicates), EXPECTED_PAIRS))
    slow = generator.exponential(slow_mean, (int(replicates), EXPECTED_PAIRS))
    ratios = fast.mean(axis=1) / slow.mean(axis=1)
    interval = np.quantile(ratios, [0.025, 0.5, 0.975], method="linear")
    return {
        "evidence_scope": "simulation_only_session1_endpoint_law",
        "replicates": int(replicates),
        "pairs_per_replicate": EXPECTED_PAIRS,
        "seed": int(seed),
        "fast_endpoint_mean": fast_mean,
        "slow_endpoint_mean": slow_mean,
        "expected_ratio": fast_mean / slow_mean,
        "ratio_quantiles": {
            "q025": float(interval[0]),
            "q500": float(interval[1]),
            "q975": float(interval[2]),
        },
        "probability_ratio_below_one": float(np.mean(ratios < 1.0)),
        "interpretation": (
            "model envelope for the missing Session 1 only; it is not a hardware confidence interval"
        ),
    }


def simulate_session1(
    frozen_plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    loop_config: Mapping[str, Any],
    session: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = frozen_plan["rng_seeds"]
    baseline_rng = np.random.default_rng(int(seeds["session_baseline"]))
    sensing_rng = np.random.default_rng(int(seeds["session_sensing"]))
    mirror_rng = np.random.default_rng(int(seeds["session_mirror_qc"]))
    phase_time = float(loop_config["sensing"]["phase_time_seconds"])
    records: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    mirror_rows: list[dict[str, Any]] = []

    baselines = list(session["baseline_measurements"])
    opening, opening_counts = _simulate_baseline(
        baselines[0], phase_time_seconds=phase_time, generator=baseline_rng
    )
    _append_chained(records, "simulated_session_baseline", {
        "measurement_id": str(baselines[0]["measurement_id"]),
        "session_index": SESSION_INDEX,
        "position": "session_start",
        "used_by_estimate": True,
        "shots_per_setting": int(baselines[0]["shots_per_setting"]),
        "baseline": opening,
        "synthetic_marginal_counts": opening_counts,
    })

    for completed, cycle in enumerate(session["cycles"], start=1):
        shots = int(cycle["sensing_shots_per_setting"])
        injected_expectations, synthetic_counts = _sample_axes(
            cycle["sense_fields"],
            shots=shots,
            phase_time_seconds=phase_time,
            generator=sensing_rng,
        )
        sensed = shared_baseline.differential_estimate(
            injected_expectations,
            shots,
            opening,
            phase_time,
        )
        observable = cadence.observable_state(sensed, loop_config)
        decision = shield(
            observable,
            Action(1.0, 2, "act"),
            cadence.shield_config(loop_config),
        )
        compensation = list(decision.compensation[:2]) if decision.permitted else [0.0, 0.0]
        residual = np.asarray(cycle["mirror_fields"], dtype=np.float64) + np.asarray(
            compensation, dtype=np.float64
        )
        endpoint = float(np.dot(residual, residual))
        cycle_mirror_rows = _mirror_qc(cycle, compensation, loop_config, mirror_rng)
        mirror_rows.extend(cycle_mirror_rows)
        row = {
            **RECORD_TAGS,
            "cycle_id": str(cycle["cycle_id"]),
            "session_index": SESSION_INDEX,
            "block_index": int(cycle["block_index"]),
            "cadence": str(cycle["cadence"]),
            "cadence_seconds": float(cycle["cadence_seconds"]),
            "cycle_index": int(cycle["cycle_index"]),
            "source_registered_pair_id": str(cycle["registered_pair_id"]),
            "virtual_sense_seconds": float(cycle["virtual_sense_seconds"]),
            "virtual_mirror_seconds": float(cycle["virtual_mirror_seconds"]),
            "sense_fields": list(cycle["sense_fields"]),
            "mirror_fields": list(cycle["mirror_fields"]),
            "sensing_shots_per_setting": shots,
            "synthetic_marginal_counts": synthetic_counts,
            "sensing": sensed,
            "observable_shield_state": observable,
            "shield": {
                "permitted": bool(decision.permitted),
                "gate": decision.gate,
                "reason": decision.reason,
                "action": asdict(decision.action),
                "compensation": list(decision.compensation),
            },
            "endpoint_squared_residual": endpoint,
            "mirror_qc_carried": bool(cycle["mirror_seeds"]),
            "mirror_qc_rows": cycle_mirror_rows,
            "event_order": [
                "simulated_sense",
                "five_gate_shield",
                "digital_inverse_compensation",
                "simulated_mirror_qc",
            ],
        }
        _assert_no_forbidden_keys(row)
        cycle_rows.append(row)
        _append_chained(records, "simulated_cadence_cycle_completed", row)
        if completed % 5 == 0 or completed == EXPECTED_CYCLES:
            print(canonical_json({
                "event": "simulation_progress",
                "completed_cycles": completed,
                "expected_cycles": EXPECTED_CYCLES,
            }), flush=True)

    closing, closing_counts = _simulate_baseline(
        baselines[1], phase_time_seconds=phase_time, generator=baseline_rng
    )
    _append_chained(records, "simulated_session_baseline", {
        "measurement_id": str(baselines[1]["measurement_id"]),
        "session_index": SESSION_INDEX,
        "position": "session_end",
        "used_by_estimate": False,
        "shots_per_setting": int(baselines[1]["shots_per_setting"]),
        "baseline": closing,
        "synthetic_marginal_counts": closing_counts,
    })

    pairs = _pair_rows(cycle_rows)
    fast = [float(row["fast_endpoint_squared_residual"]) for row in pairs]
    slow = [float(row["slow_endpoint_squared_residual"]) for row in pairs]
    trace_gate = cadence_ratio_permutation_gate(
        fast,
        slow,
        permutations=int(frozen_plan["monte_carlo"]["permutation_count_for_single_trace"]),
        seed=int(seeds["session1_trace_permutation"]),
    )
    trace_gate.update({
        "evidence_scope": "simulation_only_session1_counterfactual_trace",
        "registered_hardware_verdict": False,
        "interpretation": (
            "diagnostic gate on one synthetic Session 1 trace; never pooled with hardware"
        ),
    })
    _append_chained(records, "simulated_session1_counterfactual_gate", {
        "session_index": SESSION_INDEX,
        "pair_count": len(pairs),
        "counterfactual_gate": trace_gate,
    })

    drift_qc = shared_baseline.baseline_drift_qc(
        opening,
        closing,
        phase_time_seconds=phase_time,
    )
    floor = shared_baseline.endpoint_shot_floor(
        injected_shots_per_setting=int(
            loop_config["collection_correction"]["sensing_shots_per_setting"]
        ),
        baseline_shots_per_setting=int(
            loop_config["collection_correction"]["baseline_shots_per_setting"]
        ),
    )
    drift_offsets = shared_baseline.drift_sensitivity_offsets(
        drift_qc,
        shared_baseline_floor=float(floor["shared_baseline_floor"]),
    )
    ensemble = _monte_carlo_envelope(
        loop_config,
        replicates=int(frozen_plan["monte_carlo"]["replicates"]),
        seed=int(seeds["session1_endpoint_ensemble"]),
    )

    adaptive_qc = [row for row in mirror_rows if row["strategy"] == "adaptive"]
    fixed_qc = [row for row in mirror_rows if row["strategy"] == "fixed_zero"]
    report = {
        "schema": REPORT_SCHEMA,
        "completed_at_utc": iso(utc_now()),
        "project_status": "COMPLETED_WITH_SIMULATION_CONTINGENCY",
        "simulation_contingency_status": "COMPLETE",
        "registered_hardware_endpoint_status": "PENDING_HARDWARE",
        "hardware_session1_status": "NOT_COLLECTED_PLATFORM_CALIBRATION",
        "simulation_only": True,
        "hardware_jobs_submitted": 0,
        "hardware_results_read": False,
        "registered_hardware_endpoint_contribution": "none",
        "pooling_permitted": False,
        "source_hardware_plan": frozen_plan["source"]["base_hardware_plan"],
        "source_hardware_plan_sha256": frozen_plan["source"]["base_hardware_plan_sha256"],
        "loop_config": frozen_plan["source"]["loop_config"],
        "loop_config_sha256": frozen_plan["source"]["loop_config_sha256"],
        "session1": {
            "session_index": SESSION_INDEX,
            "record_origin": "simulation",
            "block_order": list(session["block_order"]),
            "virtual_start_seconds": float(session["virtual_start_seconds"]),
            "cycles": len(cycle_rows),
            "fast_cycles": sum(row["cadence"] == "fast" for row in cycle_rows),
            "slow_cycles": sum(row["cadence"] == "slow" for row in cycle_rows),
            "counterfactual_pair_count": len(pairs),
            "baseline_measurements": 2,
            "mirror_qc_jobs": sum(bool(row["mirror_qc_carried"]) for row in cycle_rows),
            "mirror_qc_tasks": len(mirror_rows),
            "shield_permitted_cycles": sum(bool(row["shield"]["permitted"]) for row in cycle_rows),
            "shield_abstained_cycles": sum(not bool(row["shield"]["permitted"]) for row in cycle_rows),
        },
        "simulation_only_counterfactual_endpoint": trace_gate,
        "session1_model_envelope": ensemble,
        "simulated_baseline_drift_qc": drift_qc,
        "simulated_baseline_drift_sensitivity_offsets": drift_offsets,
        "simulated_mirror_qc": {
            "analysis_role": "simulation_only_descriptive_mirror_qc",
            "fixed_zero_mean_success_probability": (
                float(np.mean([row["success_probability"] for row in fixed_qc]))
                if fixed_qc else None
            ),
            "adaptive_mean_success_probability": (
                float(np.mean([row["success_probability"] for row in adaptive_qc]))
                if adaptive_qc else None
            ),
            "task_count": len(mirror_rows),
            "registered_endpoint_input": False,
        },
        "claim_boundary": frozen_plan["claim_boundary"],
        "reporting_statement": (
            "Session 1 was completed in simulation-contingency mode because the T176 "
            "hardware backend was unavailable. Synthetic records are reported separately "
            "and do not adjudicate the registered hardware endpoint."
        ),
    }
    _assert_no_forbidden_keys(report)
    return report, records, pairs, mirror_rows


def _markdown_report(report: Mapping[str, Any], plan_path: Path) -> str:
    session = report["session1"]
    endpoint = report["simulation_only_counterfactual_endpoint"]
    envelope = report["session1_model_envelope"]
    return (
        "# B4/T176 Session 1 Simulation Contingency\n\n"
        "## Material Passport\n\n"
        "- Origin Skill: experiment-agent\n"
        "- Origin Mode: run\n"
        f"- Origin Date: {report['completed_at_utc']}\n"
        "- Verification Status: SIMULATED\n"
        "- Version Label: b4_t176_session1_simulation_contingency_v1\n\n"
        "## Status\n\n"
        "- Project: **COMPLETED_WITH_SIMULATION_CONTINGENCY**\n"
        "- Simulation Session 1: **COMPLETE**\n"
        "- Registered hardware endpoint: **PENDING_HARDWARE**\n"
        "- Hardware jobs submitted by this run: **0**\n"
        "- Hardware scientific results read by this run: **no**\n"
        "- Synthetic/hardware pooling: **forbidden**\n\n"
        "## Session 1 execution\n\n"
        f"- Frozen plan: `{plan_path}`\n"
        f"- Block order: `{' -> '.join(session['block_order'])}`\n"
        f"- Cycles: {session['cycles']} ({session['slow_cycles']} slow + {session['fast_cycles']} fast)\n"
        f"- Counterfactual pair mappings: {session['counterfactual_pair_count']}\n"
        f"- Baselines: {session['baseline_measurements']}\n"
        f"- Mirror-QC jobs/tasks: {session['mirror_qc_jobs']}/{session['mirror_qc_tasks']}\n"
        f"- Shield permitted/abstained: {session['shield_permitted_cycles']}/{session['shield_abstained_cycles']}\n\n"
        "## Simulation-only readout\n\n"
        f"- Single-trace ratio: {float(endpoint['ratio']):.9f}\n"
        f"- Single-trace permutation p-value: {float(endpoint['p_value']):.9f}\n"
        f"- Diagnostic trace gate passed: {bool(endpoint['passed'])}\n"
        f"- Model-envelope replicates: {envelope['replicates']}\n"
        f"- Model ratio median [2.5%, 97.5%]: {float(envelope['ratio_quantiles']['q500']):.6f} "
        f"[{float(envelope['ratio_quantiles']['q025']):.6f}, {float(envelope['ratio_quantiles']['q975']):.6f}]\n\n"
        "The readout above is a counterfactual simulation diagnostic. It is not a hardware "
        "confidence interval and does not produce a registered hardware PASS/FAIL verdict.\n\n"
        "## Required disclosure\n\n"
        f"> {report['reporting_statement']}\n"
    )


def run_frozen_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan, source_plan, loop_config, session = validate_frozen_plan(plan_path)
    output = Path(str(plan["separation_contract"]["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    print(canonical_json({
        "event": "simulation_started",
        "session_index": SESSION_INDEX,
        "expected_cycles": EXPECTED_CYCLES,
        "output": str(output),
    }), flush=True)

    report, records, pairs, mirror_rows = simulate_session1(
        plan,
        source_plan,
        loop_config,
        session,
    )
    journal_path = output / "session1_simulated_snapshots.jsonl"
    pair_path = output / "session1_simulated_pair_rows.csv"
    mirror_path = output / "session1_simulated_mirror_qc.jsonl"
    report_path = output / "session1_simulation_report.json"
    markdown_path = output / "B4_T176_SESSION1_SIMULATION_CONTINGENCY.md"

    write_new_text(
        journal_path,
        "".join(canonical_json(row) + "\n" for row in records),
    )
    write_new_text(
        mirror_path,
        "".join(canonical_json(row) + "\n" for row in mirror_rows),
    )
    if not pairs:
        raise RuntimeError("cannot write an empty Session 1 pair table")
    with pair_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    write_new_json(report_path, report)
    write_new_text(markdown_path, _markdown_report(report, plan_path))

    evidence_files = [journal_path, pair_path, mirror_path, report_path, markdown_path]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": iso(utc_now()),
        "simulation_only": True,
        "hardware_job_submitted": False,
        "hardware_scientific_results_read": False,
        "registered_hardware_endpoint_contribution": "none",
        "pooling_permitted": False,
        "plan": str(plan_path),
        "plan_sha256": digest_file(plan_path),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": digest_file(Path(__file__).resolve()),
        "python": sys.version,
        "platform": runtime_platform.platform(),
        "source": dict(plan["source"]),
        "record_chain": {
            "rows": len(records),
            "first_record_sha256": records[0]["record_sha256"],
            "last_record_sha256": records[-1]["record_sha256"],
        },
        "outputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
            }
            for path in evidence_files
        ],
        "success": True,
    }
    manifest_path = output / "simulation_manifest.json"
    write_new_json(manifest_path, manifest)
    print(canonical_json({
        "event": "simulation_completed",
        "session_index": SESSION_INDEX,
        "cycles": report["session1"]["cycles"],
        "pairs": report["session1"]["counterfactual_pair_count"],
        "hardware_jobs_submitted": 0,
        "manifest": str(manifest_path),
    }), flush=True)
    return {"report": report, "manifest": manifest, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze the isolated simulation plan")
    freeze.add_argument("--base-plan", type=Path, required=True)
    freeze.add_argument("--loop-config", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--monte-carlo-replicates", type=int, default=40_000)
    freeze.add_argument("--permutation-count", type=int, default=20_000)

    run = subparsers.add_parser("run", help="run an already-frozen simulation plan")
    run.add_argument("--plan", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze_plan(
            base_plan_path=args.base_plan,
            loop_config_path=args.loop_config,
            plan_path=args.plan,
            output_dir=args.output,
            monte_carlo_replicates=args.monte_carlo_replicates,
            permutation_count=args.permutation_count,
        )
        print(canonical_json({
            "event": "simulation_plan_frozen",
            "plan": str(args.plan.resolve()),
            "plan_sha256": digest_file(args.plan.resolve()),
            "session1": plan["session1_contract"],
        }))
        return 0
    result = run_frozen_plan(args.plan)
    return 0 if result["manifest"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
