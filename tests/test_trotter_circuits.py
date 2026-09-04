from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm

from src.circuits.trotter import (
    PauliTerm,
    build_measurement_circuits,
    build_trotter_circuit,
    circuit_statistics,
    cqlib_statevector_q0_leftmost,
    hamiltonian_terms,
    pauli_rotation_matrix,
    pauli_term_matrix,
    trotter_unitary,
)
from src.physics.hamiltonian import NUM_QUBITS, build_hamiltonian


def parameters() -> dict[str, object]:
    return {
        "Jx": 0.41, "Jy": -0.23, "Jz": 0.67, "Jxz": -0.31, "Jzx": 0.19, "D": 0.53,
        "hx": [0.12, -0.16, 0.21, -0.08, 0.05, -0.11],
        "hy": [-0.13, 0.22, -0.17, 0.09, 0.04, -0.06],
        "hz": [0.10, -0.04, 0.15, -0.20, 0.18, -0.07],
    }


class TrotterCircuitTests(unittest.TestCase):
    def test_all_pauli_rotation_matrices_match_exact_exponentials(self) -> None:
        terms = (
            PauliTerm(1.0, ((0, "X"),), "X"),
            PauliTerm(1.0, ((0, "Y"),), "Y"),
            PauliTerm(1.0, ((0, "Z"),), "Z"),
            PauliTerm(1.0, ((0, "X"), (1, "X")), "XX"),
            PauliTerm(1.0, ((0, "Y"), (1, "Y")), "YY"),
            PauliTerm(1.0, ((0, "Z"), (1, "Z")), "ZZ"),
            PauliTerm(1.0, ((0, "X"), (1, "Z")), "XZ"),
            PauliTerm(1.0, ((0, "Z"), (1, "X")), "ZX"),
            PauliTerm(1.0, ((0, "X"), (1, "Y")), "XY"),
            PauliTerm(1.0, ((0, "Y"), (1, "X")), "YX"),
        )
        for term in terms:
            expected = expm(-0.37j * pauli_term_matrix(term))
            np.testing.assert_allclose(pauli_rotation_matrix(term, 0.37), expected, atol=1e-12)

    def test_terms_reproduce_qutip_hamiltonian_and_dm_sign(self) -> None:
        values = parameters()
        dense = sum((term.coefficient * pauli_term_matrix(term) for term in hamiltonian_terms(values)), np.zeros((64, 64), complex))
        np.testing.assert_allclose(dense, build_hamiltonian(values).full(), atol=1e-12)
        dm_terms = [term for term in hamiltonian_terms(values) if "D_" in term.name]
        self.assertEqual(dm_terms[0].coefficient, values["D"])
        self.assertEqual(dm_terms[1].coefficient, -values["D"])

    def test_cqlib_statevector_matches_independent_trotter_matrix(self) -> None:
        values = parameters()
        circuit = build_trotter_circuit(values, 0.29, order=2, steps=2)
        actual = cqlib_statevector_q0_leftmost(circuit)
        neel_index = int("010101", 2)
        initial = np.zeros(2**NUM_QUBITS, dtype=np.complex128)
        initial[neel_index] = 1.0
        expected = trotter_unitary(values, 0.29, order=2, steps=2) @ initial
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_measurement_variants_and_statistics_are_deterministic(self) -> None:
        circuits = build_measurement_circuits(parameters(), 0.11, order=1, steps=1)
        self.assertEqual(set(circuits), {"XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"})
        stats = circuit_statistics(circuits["YY"])
        self.assertGreater(stats["gate_counts"]["M"], 0)
        self.assertGreater(stats["estimated_logical_depth"], 0)
        self.assertIn("M Q0", circuits["YY"].qcis)


if __name__ == "__main__":
    unittest.main()
