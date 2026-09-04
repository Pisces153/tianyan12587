"""Measurement sampling and counts-to-Pauli reconstruction for logical qubits 0 and 1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import numpy as np
from qutip import Qobj, identity, tensor


NUM_QUBITS = 6
BASIS_ORDER = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")
PAULI15_ORDER = (
    "X0", "Y0", "Z0", "X1", "Y1", "Z1",
    "X0X1", "X0Y1", "X0Z1", "Y0X1", "Y0Y1", "Y0Z1", "Z0X1", "Z0Y1", "Z0Z1",
)
XOBS_INDICES = (0, 1, 2, 6, 10, 14)
XOBS_ORDER = tuple(PAULI15_ORDER[index] for index in XOBS_INDICES)

I2 = identity(2)
H = Qobj(np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0))
SDG = Qobj(np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128))


def measurement_rotation(axis: str) -> Qobj:
    if axis == "X":
        return H
    if axis == "Y":
        return H * SDG
    if axis == "Z":
        return I2
    raise ValueError(f"Unknown measurement axis: {axis}")


def measurement_unitary(basis: str) -> Qobj:
    if basis not in BASIS_ORDER:
        raise ValueError(f"Unsupported measurement basis: {basis}")
    return tensor(measurement_rotation(basis[0]), measurement_rotation(basis[1]), *([I2] * 4))


def outcome_probabilities(density_matrix: Qobj, basis: str) -> np.ndarray:
    unitary = measurement_unitary(basis)
    measured_state = unitary * density_matrix * unitary.dag()
    probabilities = np.real(np.diag(measured_state.full())).astype(np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Measurement probabilities are invalid.")
    return probabilities / total


def sample_counts(density_matrix: Qobj, basis: str, shots: int, rng: np.random.Generator) -> np.ndarray:
    if shots <= 0:
        raise ValueError("shots must be positive")
    return rng.multinomial(shots, outcome_probabilities(density_matrix, basis)).astype(np.int32)


def sample_all_counts(density_matrix: Qobj, shots: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([sample_counts(density_matrix, basis, shots, rng) for basis in BASIS_ORDER], axis=0)


def _coerce_counts(values: Sequence[int] | Mapping[str, int], shots: int | None) -> np.ndarray:
    if isinstance(values, Mapping):
        result = np.zeros(2**NUM_QUBITS, dtype=np.int64)
        for bitstring, count in values.items():
            if len(bitstring) != NUM_QUBITS or set(bitstring).difference({"0", "1"}):
                raise ValueError(f"Invalid q0-leftmost bitstring: {bitstring!r}")
            result[int(bitstring, 2)] += int(count)
    else:
        result = np.asarray(values, dtype=np.int64)
        if result.shape != (2**NUM_QUBITS,):
            raise ValueError(f"Counts must have 64 outcomes; got {result.shape}")
    if np.any(result < 0):
        raise ValueError("Counts cannot be negative")
    count_sum = int(result.sum())
    if shots is not None and count_sum != shots:
        raise ValueError(f"Counts sum {count_sum} does not equal expected shots {shots}")
    if count_sum == 0:
        raise ValueError("Counts sum to zero")
    return result


def _expectations_for_basis(counts: np.ndarray) -> tuple[float, float, float]:
    total = float(counts.sum())
    indices = np.arange(2**NUM_QUBITS, dtype=np.uint8)
    q0 = ((indices >> 5) & 1).astype(np.int8)
    q1 = ((indices >> 4) & 1).astype(np.int8)
    value0 = 1.0 - 2.0 * q0
    value1 = 1.0 - 2.0 * q1
    return (
        float(np.dot(counts, value0) / total),
        float(np.dot(counts, value1) / total),
        float(np.dot(counts, value0 * value1) / total),
    )


def counts_to_pauli15(
    counts_by_basis: Mapping[str, Sequence[int] | Mapping[str, int]], *, shots: int | None = 1024
) -> np.ndarray:
    if set(counts_by_basis) != set(BASIS_ORDER):
        missing = sorted(set(BASIS_ORDER).difference(counts_by_basis))
        unexpected = sorted(set(counts_by_basis).difference(BASIS_ORDER))
        raise ValueError(f"Counts bases mismatch; missing={missing}, unexpected={unexpected}")
    singles0: dict[str, list[float]] = {axis: [] for axis in "XYZ"}
    singles1: dict[str, list[float]] = {axis: [] for axis in "XYZ"}
    pairs: dict[str, float] = {}
    for basis in BASIS_ORDER:
        value0, value1, correlation = _expectations_for_basis(
            _coerce_counts(counts_by_basis[basis], shots)
        )
        singles0[basis[0]].append(value0)
        singles1[basis[1]].append(value1)
        pairs[basis] = correlation
    pauli15 = np.array(
        [
            *(np.mean(singles0[axis]) for axis in "XYZ"),
            *(np.mean(singles1[axis]) for axis in "XYZ"),
            *(pairs[basis] for basis in BASIS_ORDER),
        ],
        dtype=np.float32,
    )
    return pauli15


def counts_array_to_pauli15(counts: np.ndarray, *, shots: int = 1024) -> np.ndarray:
    array = np.asarray(counts)
    if array.shape != (len(BASIS_ORDER), 2**NUM_QUBITS):
        raise ValueError(f"Counts array must have shape (9, 64); got {array.shape}")
    return counts_to_pauli15({basis: array[index] for index, basis in enumerate(BASIS_ORDER)}, shots=shots)


def select_xobs6(pauli15: Sequence[float]) -> np.ndarray:
    return select_pauli_features(pauli15, XOBS_ORDER)


def select_pauli_features(pauli15: Sequence[float], order: Sequence[str]) -> np.ndarray:
    values = np.asarray(pauli15, dtype=np.float32)
    if values.shape != (len(PAULI15_ORDER),):
        raise ValueError(f"pauli15 must have shape (15,); got {values.shape}")
    names = tuple(str(name) for name in order)
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("AEMTN feature order must contain six unique Pauli15 names")
    unexpected = sorted(set(names).difference(PAULI15_ORDER))
    if unexpected:
        raise ValueError(f"Unknown Pauli15 feature names: {unexpected}")
    return values[[PAULI15_ORDER.index(name) for name in names]]
