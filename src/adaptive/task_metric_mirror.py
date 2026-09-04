"""Raw-count mirror-task metric and paired strategy comparison for B-4.

The primary endpoint is always the probability of the ideal mirror output
computed directly from raw counts.  Any readout-mitigated value is retained as
secondary metadata and never replaces the primary score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


PRIMARY_METRIC = "success_probability"
PRIMARY_SOURCE = "raw_counts"
PAIR_MATCH_FIELDS = (
    "backend_id",
    "time_window_id",
    "shots",
    "task_family",
    "depth",
    "task_duration_seconds",
    "total_strategy_shots_in_window",
)

_RANDOM_CLIFFORD_GATES = ("I", "X2P", "X2M", "Y2P", "Y2M")
_INVERSE_GATE = {"I": "I", "X2P": "X2M", "X2M": "X2P", "Y2P": "Y2M", "Y2M": "Y2P", "CZ": "CZ"}


@dataclass(frozen=True)
class MirrorScore:
    pair_id: str
    strategy: str
    backend_id: str
    time_window_id: str
    shots: int
    task_family: str
    depth: int
    task_duration_seconds: float
    total_strategy_shots_in_window: int
    ideal_bitstring: str
    success_count: int
    success_probability: float
    readout_mitigated_success_probability_secondary: float | None


@dataclass(frozen=True)
class MirrorCircuit:
    seed: int
    depth: int
    physical_qubits: tuple[int, ...]
    prepared_bits: str
    ideal_bitstring: str
    forward_layers: tuple[tuple[str, ...], ...]
    inverse_layers: tuple[tuple[str, ...], ...]
    qcis: str


def _invert_operation(operation: str) -> str:
    pieces = operation.split()
    gate = pieces[0]
    if gate not in _INVERSE_GATE:
        raise ValueError(f"unsupported Clifford operation: {operation}")
    return " ".join([_INVERSE_GATE[gate], *pieces[1:]])


def build_random_clifford_mirror(
    physical_qubits: Sequence[int],
    *,
    depth: int,
    seed: int,
    prepared_bits: str = "000000",
) -> MirrorCircuit:
    """Build randomized native-Clifford layers followed by exact inverse layers.

    Each forward layer contains random local pi/2 Clifford rotations and an
    alternating nearest-neighbour CZ matching.  Reversing layer and operation
    order while replacing every gate with its inverse guarantees ideal output
    equals the prepared state.
    """
    qubits = tuple(int(value) for value in physical_qubits)
    prepared = _canonical_bitstring(prepared_bits)
    if len(qubits) < 2 or len(set(qubits)) != len(qubits):
        raise ValueError("mirror task requires at least two distinct physical qubits")
    if len(prepared) != len(qubits) or depth <= 0:
        raise ValueError("prepared state width must match qubits and depth must be positive")
    generator = np.random.default_rng(seed)
    forward: list[tuple[str, ...]] = []
    for layer_index in range(depth):
        operations: list[str] = []
        for qubit in qubits:
            gate = str(generator.choice(_RANDOM_CLIFFORD_GATES))
            if gate != "I":
                operations.append(f"{gate} Q{qubit}")
        offset = layer_index % 2
        for index in range(offset, len(qubits) - 1, 2):
            operations.append(f"CZ Q{qubits[index]} Q{qubits[index + 1]}")
        forward.append(tuple(operations))
    inverse = [tuple(_invert_operation(operation) for operation in reversed(layer)) for layer in reversed(forward)]
    qcis_lines: list[str] = []
    for bit, qubit in zip(prepared, qubits, strict=True):
        if bit == "1":
            qcis_lines.extend((f"X2P Q{qubit}", f"X2P Q{qubit}"))
    for layer in (*forward, *inverse):
        qcis_lines.extend(layer)
    qcis_lines.extend(f"M Q{qubit}" for qubit in qubits)
    return MirrorCircuit(
        seed=int(seed),
        depth=int(depth),
        physical_qubits=qubits,
        prepared_bits=prepared,
        ideal_bitstring=prepared,
        forward_layers=tuple(forward),
        inverse_layers=tuple(inverse),
        qcis="\n".join(qcis_lines),
    )


def build_depth_ladder_manifest(
    physical_qubits: Sequence[int],
    depths: Sequence[int],
    *,
    seeds_per_depth: int,
    seed: int,
    prepared_bits: str = "000000",
) -> dict[str, Any]:
    if not depths or seeds_per_depth <= 0:
        raise ValueError("depth ladder requires depths and positive seeds_per_depth")
    tasks: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(depths):
        if int(depth) <= 0:
            raise ValueError("mirror depths must be positive")
        for replicate in range(seeds_per_depth):
            task_seed = int(seed + depth_index * 100_000 + replicate)
            circuit = build_random_clifford_mirror(
                physical_qubits,
                depth=int(depth),
                seed=task_seed,
                prepared_bits=prepared_bits,
            )
            tasks.append({
                "task_id": f"mirror-d{int(depth)}-r{replicate:03d}",
                "task_family": "random_native_clifford_mirror_v1",
                "depth": int(depth),
                "seed": task_seed,
                "prepared_bits": circuit.prepared_bits,
                "ideal_bitstring": circuit.ideal_bitstring,
                "physical_qubits": list(circuit.physical_qubits),
                "qcis": circuit.qcis,
                "qcis_sha256": sha256(circuit.qcis.encode("utf-8")).hexdigest().upper(),
            })
    return {
        "schema": "b4_mirror_depth_ladder_v1",
        "selection_rule": "choose depth whose raw-count success_probability lies in [0.3, 0.7] during T-B6 dry-run; freeze before collection",
        "primary_metric": PRIMARY_METRIC,
        "primary_source": PRIMARY_SOURCE,
        "prepared_bits": _canonical_bitstring(prepared_bits),
        "depths": [int(value) for value in depths],
        "seeds_per_depth": int(seeds_per_depth),
        "seed_origin": int(seed),
        "tasks": tasks,
    }


def _canonical_bitstring(value: str) -> str:
    cleaned = value.replace(" ", "").replace("_", "")
    if not cleaned or set(cleaned).difference({"0", "1"}):
        raise ValueError("bitstrings must contain only zero and one")
    return cleaned


def success_probability_from_raw_counts(
    counts: Mapping[str, int],
    ideal_bitstring: str,
    *,
    shots: int | None = None,
) -> dict[str, int | float]:
    """Return ideal-output probability without mitigation or smoothing."""
    if not counts:
        raise ValueError("raw counts must not be empty")
    normalized: dict[str, int] = {}
    for bitstring, count in counts.items():
        key = _canonical_bitstring(str(bitstring))
        value = int(count)
        if value < 0:
            raise ValueError("raw counts must be non-negative")
        normalized[key] = normalized.get(key, 0) + value
    widths = {len(key) for key in normalized}
    ideal = _canonical_bitstring(ideal_bitstring)
    if widths != {len(ideal)}:
        raise ValueError("raw-count and ideal bitstring widths differ")
    total = int(sum(normalized.values()))
    if total <= 0:
        raise ValueError("raw-count total must be positive")
    if shots is not None and total != int(shots):
        raise ValueError(f"raw-count total {total} does not equal registered shots {int(shots)}")
    success_count = int(normalized.get(ideal, 0))
    return {
        "success_count": success_count,
        "shots": total,
        PRIMARY_METRIC: success_count / total,
    }


def score_observation(row: Mapping[str, Any]) -> MirrorScore:
    required = {
        "pair_id",
        "strategy",
        "backend_id",
        "time_window_id",
        "shots",
        "task_family",
        "depth",
        "task_duration_seconds",
        "total_strategy_shots_in_window",
        "ideal_bitstring",
        "raw_counts",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"mirror observation is missing fields: {missing}")
    strategy = str(row["strategy"])
    if strategy not in {"adaptive", "fixed"}:
        raise ValueError("strategy must be adaptive or fixed")
    raw = success_probability_from_raw_counts(
        row["raw_counts"],
        str(row["ideal_bitstring"]),
        shots=int(row["shots"]),
    )
    mitigated = row.get("readout_mitigated_success_probability")
    if mitigated is not None and not 0.0 <= float(mitigated) <= 1.0:
        raise ValueError("readout-mitigated success probability must lie in [0, 1]")
    return MirrorScore(
        pair_id=str(row["pair_id"]),
        strategy=strategy,
        backend_id=str(row["backend_id"]),
        time_window_id=str(row["time_window_id"]),
        shots=int(row["shots"]),
        task_family=str(row["task_family"]),
        depth=int(row["depth"]),
        task_duration_seconds=float(row["task_duration_seconds"]),
        total_strategy_shots_in_window=int(row["total_strategy_shots_in_window"]),
        ideal_bitstring=_canonical_bitstring(str(row["ideal_bitstring"])),
        success_count=int(raw["success_count"]),
        success_probability=float(raw[PRIMARY_METRIC]),
        readout_mitigated_success_probability_secondary=None if mitigated is None else float(mitigated),
    )


def validate_pair(adaptive: MirrorScore, fixed: MirrorScore) -> None:
    if adaptive.strategy != "adaptive" or fixed.strategy != "fixed":
        raise ValueError("paired comparison requires adaptive and fixed rows")
    if adaptive.pair_id != fixed.pair_id:
        raise ValueError("paired rows have different pair_id values")
    left = asdict(adaptive)
    right = asdict(fixed)
    mismatch = [field for field in PAIR_MATCH_FIELDS if left[field] != right[field]]
    if mismatch:
        raise ValueError(f"paired rows differ on frozen matching fields: {mismatch}")


def paired_bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> list[float]:
    observed = np.asarray(values, dtype=np.float64)
    if observed.ndim != 1 or len(observed) < 2:
        raise ValueError("paired bootstrap requires at least two paired differences")
    if resamples < 100 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid bootstrap configuration")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(observed), size=(resamples, len(observed)))
    means = observed[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return [float(value) for value in np.quantile(means, (tail, 1.0 - tail))]


def compare_strategies(
    observations: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
    minimum_actual_improvement: float = 0.0,
) -> dict[str, Any]:
    """Pair adaptive/fixed observations and bootstrap adaptive-minus-fixed.

    Incomplete pairs are reported and excluded without imputation.  Complete
    pairs must match on backend, window, shots, task family, depth, and task
    duration before any endpoint is computed.
    """
    if not 0.0 <= float(minimum_actual_improvement) < 1.0:
        raise ValueError("minimum_actual_improvement must lie in [0, 1)")
    grouped: dict[str, dict[str, MirrorScore]] = {}
    for raw in observations:
        score = score_observation(raw)
        strategies = grouped.setdefault(score.pair_id, {})
        if score.strategy in strategies:
            raise ValueError(f"duplicate {score.strategy} row for pair {score.pair_id}")
        strategies[score.strategy] = score

    paired_rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        strategies = grouped[pair_id]
        if set(strategies) != {"adaptive", "fixed"}:
            incomplete.append({"pair_id": pair_id, "present_strategies": sorted(strategies)})
            continue
        adaptive = strategies["adaptive"]
        fixed = strategies["fixed"]
        validate_pair(adaptive, fixed)
        difference = adaptive.success_probability - fixed.success_probability
        paired_rows.append({
            "pair_id": pair_id,
            "backend_id": adaptive.backend_id,
            "time_window_id": adaptive.time_window_id,
            "shots": adaptive.shots,
            "task_family": adaptive.task_family,
            "depth": adaptive.depth,
            "task_duration_seconds": adaptive.task_duration_seconds,
            "adaptive_success_probability": adaptive.success_probability,
            "fixed_success_probability": fixed.success_probability,
            "adaptive_minus_fixed": difference,
            "adaptive_mitigated_secondary": adaptive.readout_mitigated_success_probability_secondary,
            "fixed_mitigated_secondary": fixed.readout_mitigated_success_probability_secondary,
        })
    differences = np.asarray([row["adaptive_minus_fixed"] for row in paired_rows], dtype=np.float64)
    if len(differences) < 2:
        return {
            "endpoint": {
                "metric": PRIMARY_METRIC,
                "source": PRIMARY_SOURCE,
                "contrast": "adaptive_minus_fixed",
                "available": False,
                "reason": "fewer than two complete matched pairs",
                "mean_difference": None,
                "paired_bootstrap_interval": None,
                "minimum_actual_improvement": float(minimum_actual_improvement),
                "passed": None,
            },
            "pairing": {
                "matching_fields": list(PAIR_MATCH_FIELDS),
                "complete_pairs": len(paired_rows),
                "incomplete_pairs": len(incomplete),
                "missing_policy": "exclude incomplete pair; report; no imputation",
                "incomplete_pair_details": incomplete,
            },
            "bootstrap": {
                "method": "paired case bootstrap over complete task pairs",
                "resamples": int(resamples),
                "seed": int(seed),
                "confidence_level": float(confidence_level),
                "available": False,
            },
            "secondary_only": {
                "readout_mitigated_metric": "retained per pair when supplied; never used by primary gate",
            },
            "paired_rows": paired_rows,
        }
    interval = paired_bootstrap_interval(
        differences,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    mean_difference = float(np.mean(differences))
    endpoint = {
        "metric": PRIMARY_METRIC,
        "source": PRIMARY_SOURCE,
        "contrast": "adaptive_minus_fixed",
        "available": True,
        "mean_difference": mean_difference,
        "paired_bootstrap_interval": interval,
        "minimum_actual_improvement": float(minimum_actual_improvement),
        "passed": bool(interval[0] > float(minimum_actual_improvement)),
    }
    return {
        "endpoint": endpoint,
        "pairing": {
            "matching_fields": list(PAIR_MATCH_FIELDS),
            "complete_pairs": len(paired_rows),
            "incomplete_pairs": len(incomplete),
            "missing_policy": "exclude incomplete pair; report; no imputation",
            "incomplete_pair_details": incomplete,
        },
        "bootstrap": {
            "method": "paired case bootstrap over complete task pairs",
            "resamples": int(resamples),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
        },
        "secondary_only": {
            "readout_mitigated_metric": "retained per pair when supplied; never used by primary gate",
        },
        "paired_rows": paired_rows,
    }


def budget_summary(
    *,
    pairs: int,
    shots_per_task: int,
    throughput_shots_per_second: float,
    fixed_overhead_seconds_per_job: float,
    jobs_per_pair: int = 2,
) -> dict[str, float | int]:
    if pairs <= 0 or shots_per_task <= 0 or throughput_shots_per_second <= 0.0:
        raise ValueError("budget inputs must be positive")
    if fixed_overhead_seconds_per_job < 0.0 or jobs_per_pair <= 0:
        raise ValueError("job overhead must be non-negative and jobs_per_pair positive")
    total_jobs = pairs * jobs_per_pair
    total_shots = total_jobs * shots_per_task
    measurement_seconds = total_shots / throughput_shots_per_second
    overhead_seconds = total_jobs * fixed_overhead_seconds_per_job
    return {
        "pairs": pairs,
        "jobs": total_jobs,
        "total_shots": total_shots,
        "measurement_seconds": measurement_seconds,
        "fixed_overhead_seconds": overhead_seconds,
        "wallclock_seconds": measurement_seconds + overhead_seconds,
    }
