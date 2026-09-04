"""Digital simulation of the frozen six-qubit Hamiltonian.

The logical qubit order is q0..q5, with q0 as the leftmost bit in every
statevector and count bitstring saved by this project.  CQlib's local
statevector simulator is converted to this convention by its public API.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from cqlib import Circuit
from cqlib.simulator import SimpleSimulator
from scipy.linalg import expm

from src.features.pauli import BASIS_ORDER
from src.physics.hamiltonian import NUM_QUBITS


Axis = Literal["X", "Y", "Z"]
TrotterOrder = Literal[1, 2]
_IDENTITY = np.eye(2, dtype=np.complex128)
_PAULI: dict[Axis, np.ndarray] = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


@dataclass(frozen=True)
class PauliTerm:
    """One coefficient times one- or two-site Pauli product."""

    coefficient: float
    operators: tuple[tuple[int, Axis], ...]
    name: str

    def __post_init__(self) -> None:
        if len(self.operators) not in (1, 2):
            raise ValueError("A Trotter term must act on one or two qubits")
        sites = [site for site, _ in self.operators]
        if len(set(sites)) != len(sites) or any(site < 0 or site >= NUM_QUBITS for site in sites):
            raise ValueError(f"Invalid Pauli term sites: {sites}")


def hamiltonian_terms(parameters: Mapping[str, object]) -> tuple[PauliTerm, ...]:
    """Return the exact ordered term list for ``build_hamiltonian``.

    In particular, positive D contributes ``+D*XY`` and ``-D*YX`` on every
    open-boundary edge.  The explicit sign prevents a silent convention change.
    """
    required = ("Jx", "Jy", "Jz", "Jxz", "Jzx", "D", "hx", "hy", "hz")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"Hamiltonian parameters are missing: {missing}")
    hx = _field_values(parameters, "hx")
    hy = _field_values(parameters, "hy")
    hz = _field_values(parameters, "hz")
    terms: list[PauliTerm] = []
    for site in range(NUM_QUBITS - 1):
        edge = f"{site}{site + 1}"
        terms.extend(
            (
                PauliTerm(float(parameters["Jx"]), ((site, "X"), (site + 1, "X")), f"Jx_XX_{edge}"),
                PauliTerm(float(parameters["Jy"]), ((site, "Y"), (site + 1, "Y")), f"Jy_YY_{edge}"),
                PauliTerm(float(parameters["Jz"]), ((site, "Z"), (site + 1, "Z")), f"Jz_ZZ_{edge}"),
                PauliTerm(float(parameters["Jxz"]), ((site, "X"), (site + 1, "Z")), f"Jxz_XZ_{edge}"),
                PauliTerm(float(parameters["Jzx"]), ((site, "Z"), (site + 1, "X")), f"Jzx_ZX_{edge}"),
                PauliTerm(float(parameters["D"]), ((site, "X"), (site + 1, "Y")), f"D_XY_{edge}"),
                PauliTerm(-float(parameters["D"]), ((site, "Y"), (site + 1, "X")), f"minus_D_YX_{edge}"),
            )
        )
    for site in range(NUM_QUBITS):
        terms.extend(
            (
                PauliTerm(hx[site], ((site, "X"),), f"hx_X_{site}"),
                PauliTerm(hy[site], ((site, "Y"),), f"hy_Y_{site}"),
                PauliTerm(hz[site], ((site, "Z"),), f"hz_Z_{site}"),
            )
        )
    return tuple(terms)


def _field_values(parameters: Mapping[str, object], name: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in parameters[name])  # type: ignore[arg-type]
    if len(values) != NUM_QUBITS:
        raise ValueError(f"{name} must have {NUM_QUBITS} entries; got {len(values)}")
    return values


def append_neel_initial_state(circuit: Circuit) -> None:
    """Prepare ``|010101>`` from CQlib's all-zero computational state."""
    for qubit in (1, 3, 5):
        circuit.x(qubit)


def _append_basis_to_z(circuit: Circuit, qubit: int, axis: Axis) -> None:
    if axis == "X":
        circuit.h(qubit)
    elif axis == "Y":
        circuit.sd(qubit)
        circuit.h(qubit)
    elif axis != "Z":
        raise ValueError(f"Unsupported Pauli axis: {axis}")


def _append_basis_from_z(circuit: Circuit, qubit: int, axis: Axis) -> None:
    if axis == "X":
        circuit.h(qubit)
    elif axis == "Y":
        circuit.h(qubit)
        circuit.s(qubit)
    elif axis != "Z":
        raise ValueError(f"Unsupported Pauli axis: {axis}")


def append_pauli_rotation(circuit: Circuit, term: PauliTerm, theta: float) -> None:
    """Append ``exp(-i * theta * P)`` for a one- or two-site Pauli product."""
    if theta == 0.0:
        return
    for qubit, axis in term.operators:
        _append_basis_to_z(circuit, qubit, axis)
    if len(term.operators) == 1:
        circuit.rz(term.operators[0][0], 2.0 * theta)
    else:
        control, target = (term.operators[0][0], term.operators[1][0])
        circuit.cx(control, target)
        circuit.rz(target, 2.0 * theta)
        circuit.cx(control, target)
    for qubit, axis in reversed(term.operators):
        _append_basis_from_z(circuit, qubit, axis)


def _rotation_schedule(terms: Sequence[PauliTerm], dt: float, order: TrotterOrder) -> tuple[tuple[PauliTerm, float], ...]:
    if order == 1:
        return tuple((term, term.coefficient * dt) for term in terms)
    if order == 2:
        # Symmetric second order: H1/2 ... H(n-1)/2, Hn, H(n-1)/2 ... H1/2.
        forward = [(term, term.coefficient * dt / 2.0) for term in terms[:-1]]
        middle = [(terms[-1], terms[-1].coefficient * dt)]
        return tuple(forward + middle + list(reversed(forward)))
    raise ValueError("Trotter order must be 1 or 2")


def build_trotter_circuit(
    parameters: Mapping[str, object],
    evolution_time: float,
    *,
    order: TrotterOrder = 1,
    steps: int = 1,
    prepare_initial_state: bool = True,
) -> Circuit:
    """Build an unmeasured CQlib/QCIS circuit for first- or second-order Trotter evolution."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not np.isfinite(evolution_time) or evolution_time < 0.0:
        raise ValueError("evolution_time must be finite and non-negative")
    terms = hamiltonian_terms(parameters)
    circuit = Circuit(NUM_QUBITS)
    if prepare_initial_state:
        append_neel_initial_state(circuit)
    schedule = _rotation_schedule(terms, float(evolution_time) / steps, order)
    for _ in range(steps):
        for term, theta in schedule:
            append_pauli_rotation(circuit, term, theta)
    return circuit


def build_measurement_circuits(
    parameters: Mapping[str, object],
    evolution_time: float,
    *,
    order: TrotterOrder,
    steps: int,
) -> dict[str, Circuit]:
    """Create the nine measured circuit variants required by the data contract."""
    circuits: dict[str, Circuit] = {}
    for basis in BASIS_ORDER:
        circuit = build_trotter_circuit(parameters, evolution_time, order=order, steps=steps)
        _append_basis_to_z(circuit, 0, basis[0])
        _append_basis_to_z(circuit, 1, basis[1])
        circuit.measure_all()
        circuits[basis] = circuit
    return circuits


def circuit_statistics(circuit: Circuit) -> dict[str, object]:
    """Return deterministic logical gate counts and a greedy logical-depth estimate."""
    counts = Counter(item.instruction.name for item in circuit.instruction_sequence)
    qubit_layers = [0] * NUM_QUBITS
    for item in circuit.instruction_sequence:
        qubits = [qubit.index for qubit in item.qubits]
        if not qubits or item.instruction.name == "M":
            continue
        layer = 1 + max(qubit_layers[qubit] for qubit in qubits)
        for qubit in qubits:
            qubit_layers[qubit] = layer
    return {
        "gate_counts": dict(sorted(counts.items())),
        "total_instructions": int(sum(counts.values())),
        "estimated_logical_depth": int(max(qubit_layers, default=0)),
    }


def pauli_term_matrix(term: PauliTerm) -> np.ndarray:
    """Dense logical-q0-leftmost matrix for test and reference use only."""
    factors = [_IDENTITY] * NUM_QUBITS
    for qubit, axis in term.operators:
        factors[qubit] = _PAULI[axis]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def pauli_rotation_matrix(term: PauliTerm, theta: float) -> np.ndarray:
    return expm(-1.0j * theta * pauli_term_matrix(term))


def trotter_unitary(
    parameters: Mapping[str, object], evolution_time: float, *, order: TrotterOrder, steps: int
) -> np.ndarray:
    """Matrix semantics of ``build_trotter_circuit`` for independent verification."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    total = np.eye(2**NUM_QUBITS, dtype=np.complex128)
    schedule = _rotation_schedule(hamiltonian_terms(parameters), evolution_time / steps, order)
    for _ in range(steps):
        for term, theta in schedule:
            total = pauli_rotation_matrix(term, theta) @ total
    return total


def qcis_text(circuit: Circuit) -> str:
    """Return the exact QCIS text that CQlib will submit after a later approval gate."""
    return str(circuit.qcis)


def cqlib_statevector_q0_leftmost(circuit: Circuit) -> np.ndarray:
    """Run CQlib's local full-amplitude simulator in this project's bit order.

    ``SimpleSimulator.statevector`` returns amplitudes in q(n-1)-leftmost
    display order.  The data contract uses q0-leftmost order, so reverse the
    six-bit index once at this boundary before comparing with QuTiP or counts.
    This is a local-simulator adapter; hardware count ordering is separately
    calibrated before any hardware data are accepted.
    """
    native = np.asarray(
        SimpleSimulator(circuit, device="cpu").statevector(dict_format=False), dtype=np.complex128
    )
    if native.shape != (2**NUM_QUBITS,):
        raise RuntimeError(f"CQlib returned unexpected statevector shape: {native.shape}")
    canonical = np.empty_like(native)
    for index, amplitude in enumerate(native):
        canonical[int(f"{index:0{NUM_QUBITS}b}"[::-1], 2)] = amplitude
    return canonical
