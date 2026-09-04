"""Compare CQlib local full-amplitude Trotter circuits with exact QuTiP evolution.

This script never creates a TianYan experiment and never uses credentials.  It
is the G0 evidence used to choose a candidate Trotter order and step count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from qutip import Qobj, expect

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.circuits.trotter import build_trotter_circuit, circuit_statistics, cqlib_statevector_q0_leftmost
from src.physics.hamiltonian import NUM_QUBITS, X, Y, Z, build_hamiltonian, fixed_neel_density_matrix, local_op, sample_parameters, two_body_op


OBSERVABLES = (
    ("X0", local_op(0, X)), ("Y0", local_op(0, Y)), ("Z0", local_op(0, Z)),
    ("X1", local_op(1, X)), ("Y1", local_op(1, Y)), ("Z1", local_op(1, Z)),
    ("X0X1", two_body_op(0, 1, X, X)), ("X0Y1", two_body_op(0, 1, X, Y)),
    ("X0Z1", two_body_op(0, 1, X, Z)), ("Y0X1", two_body_op(0, 1, Y, X)),
    ("Y0Y1", two_body_op(0, 1, Y, Y)), ("Y0Z1", two_body_op(0, 1, Y, Z)),
    ("Z0X1", two_body_op(0, 1, Z, X)), ("Z0Y1", two_body_op(0, 1, Z, Y)),
    ("Z0Z1", two_body_op(0, 1, Z, Z)),
)


def statevector_density(circuit) -> Qobj:
    state = cqlib_statevector_q0_leftmost(circuit)
    return Qobj(state, dims=[[2] * NUM_QUBITS, [1] * NUM_QUBITS]).proj()


def ideal_density(parameters: dict[str, object], evolution_time: float) -> Qobj:
    propagator = (-1.0j * build_hamiltonian(parameters) * evolution_time).expm()
    initial = fixed_neel_density_matrix()
    return propagator * initial * propagator.dag()


def observable_values(density: Qobj) -> np.ndarray:
    return np.asarray([float(np.real(expect(operator, density))) for _, operator in OBSERVABLES])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "logs" / "trotter_reference.json")
    arguments = parser.parse_args()
    if arguments.samples <= 0:
        parser.error("--samples must be positive")

    rng = np.random.default_rng(arguments.seed)
    candidates = [(order, steps) for order in (1, 2) for steps in (1, 2, 4, 6, 8)]
    errors: dict[tuple[int, int], list[float]] = {candidate: [] for candidate in candidates}
    statistics: dict[tuple[int, int], dict[str, object]] = {}
    for sample_index in range(arguments.samples):
        parameters = sample_parameters(rng)
        evolution_time = float(rng.uniform(0.05, 1.5))
        exact = observable_values(ideal_density(parameters, evolution_time))
        for order, steps in candidates:
            circuit = build_trotter_circuit(parameters, evolution_time, order=order, steps=steps)
            approximate = observable_values(statevector_density(circuit))
            errors[(order, steps)].append(float(np.sqrt(np.mean((approximate - exact) ** 2))))
            statistics.setdefault((order, steps), circuit_statistics(circuit))
        print(f"completed ideal-reference sample {sample_index + 1}/{arguments.samples}")

    table = []
    for order, steps in candidates:
        values = np.asarray(errors[(order, steps)], dtype=float)
        table.append({
            "order": order,
            "steps": steps,
            "mean_pauli15_rmse": float(values.mean()),
            "max_pauli15_rmse": float(values.max()),
            "circuit_statistics_last_sample": statistics[(order, steps)],
        })
    table.sort(
        key=lambda row: (
            row["mean_pauli15_rmse"],
            row["max_pauli15_rmse"],
            row["circuit_statistics_last_sample"]["estimated_logical_depth"],
        )
    )
    report = {
        "purpose": "local CQlib full-amplitude versus ideal QuTiP reference; no platform request was made",
        "samples": arguments.samples,
        "seed": arguments.seed,
        "observable_order": [name for name, _ in OBSERVABLES],
        "candidates": table,
        "recommended_candidate": {"order": table[0]["order"], "steps": table[0]["steps"]},
        "protocol_config_was_not_modified": True,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report["recommended_candidate"], ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
