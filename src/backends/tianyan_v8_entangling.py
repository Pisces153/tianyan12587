"""Shared fixed V8 entangling probe for TianYan-287.

This module is deliberately independent of the estimator.  It defines the
native program and the nominal forward model used only by the classical
baseline, so a checkpoint cannot silently be deployed on a different task.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from src.backends.tianyan_native import assert_native_qcis, gate_counts
from src.features.pauli import PAULI15_ORDER


LOCAL6 = ("X0", "Y0", "Z0", "X1", "Y1", "Z1")
CZ_COUNT = 2
FIXED_RZ = math.pi / 4.0


def rx(q: int, angle: float) -> list[str]:
    """Native, time-ordered implementation of ``Rx(angle)``."""
    return [f"Y2M Q{q}", f"RZ Q{q} {angle:.17g}", f"Y2P Q{q}"]


def h(q: int) -> list[str]:
    return [f"RZ Q{q} {math.pi:.17g}", f"Y2P Q{q}"]


def measurement_rotation(q: int, axis: str) -> list[str]:
    if axis == "Z":
        return []
    if axis == "X":
        return h(q)
    if axis == "Y":
        return [f"RZ Q{q} {-math.pi / 2.0:.17g}", *h(q)]
    raise ValueError(f"Unsupported measurement axis {axis!r}")


def program(parameters: Mapping[str, float], time: float, basis: str, qubits: list[int]) -> str:
    """Return the fixed two-CZ noncommuting V8 QCIS program."""
    if len(qubits) != 6 or len(basis) != 2:
        raise ValueError("V8 expects six physical qubits and a two-axis basis")
    q0, q1 = qubits[:2]
    lines = [
        *rx(q0, 2.0 * float(parameters["h1"]) * time),
        *rx(q1, 2.0 * float(parameters["h2"]) * time),
        f"RZ Q{q0} {FIXED_RZ:.17g}",
        *h(q0),
        f"CZ Q{q0} Q{q1}",
        f"Y2P Q{q1}",
        f"CZ Q{q0} Q{q1}",
        *h(q0),
        *measurement_rotation(q0, basis[0]),
        *measurement_rotation(q1, basis[1]),
        *(f"M Q{qubit}" for qubit in qubits),
    ]
    result = "\n".join(lines)
    assert_native_qcis(result)
    if gate_counts(result).get("CZ", 0) != CZ_COUNT:
        raise AssertionError("The V8 task must contain exactly two CZ gates")
    return result


_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _single(matrix: np.ndarray, qubit: int) -> np.ndarray:
    return np.kron(matrix, _I) if qubit == 0 else np.kron(_I, matrix)


def _rx(angle: float) -> np.ndarray:
    return math.cos(angle / 2.0) * _I - 1j * math.sin(angle / 2.0) * _X


def _rz(angle: float) -> np.ndarray:
    return np.diag((np.exp(-0.5j * angle), np.exp(0.5j * angle)))


_H = (_X + _Z) / math.sqrt(2.0)
_CZ = np.diag((1.0, 1.0, 1.0, -1.0)).astype(np.complex128)


def _cross_drive_pair(cross_drive: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(cross_drive, (int, float)):
        return float(cross_drive), float(cross_drive)
    values = tuple(float(value) for value in cross_drive)
    if len(values) != 2:
        raise ValueError("cross_drive must be a scalar or a two-element sequence")
    return values


def unitary(h1: float, h2: float, time: float, *, rotation_scale: tuple[float, float] = (1.0, 1.0), cz_phase_error: float = 0.0, cross_drive: float | Sequence[float] = 0.0) -> np.ndarray:
    """Nominal or randomized two-qubit V8 unitary in circuit time order."""
    first_cz = np.diag((1.0, 1.0, 1.0, -np.exp(1j * cz_phase_error)))
    cross_drive_1, cross_drive_2 = _cross_drive_pair(cross_drive)
    effective_h1 = rotation_scale[0] * h1 + cross_drive_1 * h2
    effective_h2 = rotation_scale[1] * h2 + cross_drive_2 * h1
    operations = (
        _single(_rx(2.0 * effective_h1 * time), 0),
        _single(_rx(2.0 * effective_h2 * time), 1),
        _single(_rz(FIXED_RZ), 0),
        _single(_H, 0),
        first_cz,
        _single(math.cos(math.pi / 4.0) * _I - 1j * math.sin(math.pi / 4.0) * _Y, 1),
        first_cz,
        _single(_H, 0),
    )
    result = np.eye(4, dtype=np.complex128)
    for operation in operations:
        result = operation @ result
    return result


def pauli15_expectation(h1: float, h2: float, time: float, **noise: object) -> np.ndarray:
    """Return ideal or randomized Pauli15 expectations for the V8 task."""
    state = unitary(h1, h2, time, **noise) @ np.array((1.0, 0.0, 0.0, 0.0), dtype=np.complex128)
    rho = np.outer(state, state.conj())
    singles = (_X, _Y, _Z)
    observables = [np.kron(item, _I) for item in singles] + [np.kron(_I, item) for item in singles]
    observables.extend(np.kron(left, right) for left in singles for right in singles)
    values = np.asarray([np.real(np.trace(rho @ observable)) for observable in observables], dtype=np.float64)
    if len(values) != len(PAULI15_ORDER):
        raise AssertionError("Pauli15 ordering mismatch")
    return values
