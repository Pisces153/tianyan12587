"""Native-gate lowering and offline verification for TianYan-287 QCIS.

The 287 native alphabet is deliberately restricted to
``X2P, X2M, Y2P, Y2M, RZ, CZ, I, M``.  In particular, this module never
relies on the platform expanding composite ``H``, ``S``, ``X`` or ``CX``
instructions.  The phase convention follows the usual QCIS convention
``X2P = Rx(+pi/2)`` and ``Y2P = Ry(+pi/2)``.  A hardware semantic probe must
confirm this convention before any scientific batch is submitted.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

import numpy as np


NATIVE_GATES = frozenset({"X2P", "X2M", "Y2P", "Y2M", "RZ", "CZ", "I", "M"})
_COMPOSITE_GATES = frozenset({"H", "S", "SD", "X", "CX"})


def _parts(line: str) -> list[str]:
    return line.strip().split()


def _h(qubit: str) -> list[str]:
    """Return native QCIS for H, up to a global phase.

    In time order this is Rz(pi), then Ry(pi/2), i.e. ``Y2P @ RZ(pi)``.
    """
    return [f"RZ {qubit} {math.pi}", f"Y2P {qubit}"]


def lower_qcis_native(qcis: str) -> str:
    """Lower a restricted CQlib/QCIS program to TianYan-287 native gates.

    This is intentionally fail-closed: adding a new composite operation to a
    source circuit requires an explicit, independently verified decomposition.
    """
    lowered: list[str] = []
    for raw in qcis.splitlines():
        parts = _parts(raw)
        if not parts:
            continue
        op = parts[0]
        if op in NATIVE_GATES:
            lowered.append(" ".join(parts))
        elif op == "H" and len(parts) == 2:
            lowered.extend(_h(parts[1]))
        elif op == "S" and len(parts) == 2:
            lowered.append(f"RZ {parts[1]} {math.pi / 2.0}")
        elif op == "SD" and len(parts) == 2:
            lowered.append(f"RZ {parts[1]} {-math.pi / 2.0}")
        elif op == "X" and len(parts) == 2:
            lowered.extend((f"X2P {parts[1]}", f"X2P {parts[1]}"))
        elif op == "CX" and len(parts) == 3:
            control, target = parts[1:]
            lowered.extend(_h(target))
            lowered.append(f"CZ {control} {target}")
            lowered.extend(_h(target))
        else:
            raise ValueError(f"Unsupported QCIS operation for native lowering: {raw!r}")
    if any(_parts(line)[0] not in NATIVE_GATES for line in lowered):
        raise AssertionError("Native lowering emitted a non-native instruction.")
    return "\n".join(lowered)


def gate_counts(qcis: str) -> dict[str, int]:
    return dict(sorted(Counter(_parts(line)[0] for line in qcis.splitlines() if _parts(line)).items()))


def assert_native_qcis(qcis: str) -> None:
    invalid = sorted({parts[0] for line in qcis.splitlines() if (parts := _parts(line)) and parts[0] not in NATIVE_GATES})
    if invalid:
        raise ValueError(f"QCIS contains non-native TianYan instructions: {invalid}")


def _single_gate(name: str, angle: float | None = None) -> np.ndarray:
    if name == "I":
        return np.eye(2, dtype=np.complex128)
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if name == "H":
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
    if name == "S":
        return np.diag([1.0, 1.0j]).astype(np.complex128)
    if name == "SD":
        return np.diag([1.0, -1.0j]).astype(np.complex128)
    if name == "RZ":
        if angle is None:
            raise ValueError("RZ requires an angle")
        return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    if name in {"X2P", "X2M", "Y2P", "Y2M"}:
        sign = 1.0 if name.endswith("P") else -1.0
        pauli = np.array([[0, 1], [1, 0]], dtype=np.complex128) if name.startswith("X") else np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        return math.cos(math.pi / 4.0) * np.eye(2) - 1j * sign * math.sin(math.pi / 4.0) * pauli
    raise ValueError(f"No single-qubit matrix for {name}")


def _qubit_index(token: str, num_qubits: int) -> int:
    if not token.startswith("Q"):
        raise ValueError(f"Invalid QCIS qubit token: {token!r}")
    index = int(token[1:])
    if not 0 <= index < num_qubits:
        raise ValueError(f"Qubit {token!r} is outside Q0..Q{num_qubits - 1}")
    return index


def _expanded_single(matrix: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    factors = [matrix if index == qubit else np.eye(2, dtype=np.complex128) for index in range(num_qubits)]
    output = factors[0]
    for factor in factors[1:]:
        output = np.kron(output, factor)
    return output


def qcis_unitary(qcis: str, *, num_qubits: int = 6) -> np.ndarray:
    """Return QCIS unitary for offline equivalence checks; ignores measurement."""
    dimension = 2**num_qubits
    unitary = np.eye(dimension, dtype=np.complex128)
    indices = np.arange(dimension)
    for raw in qcis.splitlines():
        parts = _parts(raw)
        if not parts or parts[0] == "M":
            continue
        op = parts[0]
        if op == "CZ" and len(parts) == 3:
            first, second = (_qubit_index(token, num_qubits) for token in parts[1:])
            mask_first, mask_second = 1 << (num_qubits - 1 - first), 1 << (num_qubits - 1 - second)
            unitary[((indices & mask_first != 0) & (indices & mask_second != 0)), :] *= -1.0
            continue
        if op in {"I", "X", "H", "S", "SD", "X2P", "X2M", "Y2P", "Y2M"} and len(parts) == 2:
            unitary = _expanded_single(_single_gate(op), _qubit_index(parts[1], num_qubits), num_qubits) @ unitary
            continue
        if op == "RZ" and len(parts) == 3:
            unitary = _expanded_single(_single_gate(op, float(parts[2])), _qubit_index(parts[1], num_qubits), num_qubits) @ unitary
            continue
        if op == "CX" and len(parts) == 3:
            control, target = (_qubit_index(token, num_qubits) for token in parts[1:])
            permutation = indices.copy()
            active = indices & (1 << (num_qubits - 1 - control)) != 0
            permutation[active] ^= 1 << (num_qubits - 1 - target)
            unitary = unitary[permutation, :]
            continue
        raise ValueError(f"Unsupported operation in offline unitary evaluator: {raw!r}")
    return unitary


def equivalence_report(source_qcis: str, lowered_qcis: str, *, num_qubits: int = 6) -> dict[str, float | bool | dict[str, int]]:
    """Compare source and lowered programs up to their irrelevant global phase."""
    assert_native_qcis(lowered_qcis)
    source, lowered = qcis_unitary(source_qcis, num_qubits=num_qubits), qcis_unitary(lowered_qcis, num_qubits=num_qubits)
    overlap = np.vdot(source, lowered)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    max_error = float(np.max(np.abs(source - lowered / phase)))
    return {
        "passed": bool(max_error < 1e-10),
        "max_elementwise_error_up_to_global_phase": max_error,
        "source_gate_counts": gate_counts(source_qcis),
        "native_gate_counts": gate_counts(lowered_qcis),
    }
