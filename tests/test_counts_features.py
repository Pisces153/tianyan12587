from __future__ import annotations

import unittest

import numpy as np
from qutip import basis, tensor

from src.features.pauli import BASIS_ORDER, PAULI15_ORDER, XOBS_INDICES, counts_array_to_pauli15, outcome_probabilities, select_xobs6
from src.physics.hamiltonian import NUM_QUBITS


def exact_counts(density_matrix, shots: int = 1024) -> np.ndarray:
    rows = []
    for basis_name in BASIS_ORDER:
        probabilities = outcome_probabilities(density_matrix, basis_name)
        row = np.rint(probabilities * shots).astype(np.int64)
        row[0] += shots - int(row.sum())
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


class CountsFeatureTests(unittest.TestCase):
    def test_zero_state_pauli_values(self) -> None:
        rho = tensor(*(basis(2, 0) for _ in range(NUM_QUBITS))).proj()
        pauli15 = counts_array_to_pauli15(exact_counts(rho))
        values = dict(zip(PAULI15_ORDER, pauli15, strict=True))
        self.assertAlmostEqual(values["Z0"], 1.0, places=7)
        self.assertAlmostEqual(values["Z1"], 1.0, places=7)
        self.assertAlmostEqual(values["Z0Z1"], 1.0, places=7)
        self.assertAlmostEqual(values["X0"], 0.0, places=7)
        self.assertAlmostEqual(values["Y1"], 0.0, places=7)
        self.assertTrue(np.array_equal(select_xobs6(pauli15), pauli15[list(XOBS_INDICES)]))

    def test_bell_state_correlators_and_y_sign(self) -> None:
        bell_pair = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
        rho = tensor(bell_pair, *(basis(2, 0) for _ in range(4))).proj()
        pauli15 = counts_array_to_pauli15(exact_counts(rho))
        values = dict(zip(PAULI15_ORDER, pauli15, strict=True))
        self.assertAlmostEqual(values["X0X1"], 1.0, places=7)
        self.assertAlmostEqual(values["Y0Y1"], -1.0, places=7)
        self.assertAlmostEqual(values["Z0Z1"], 1.0, places=7)
        self.assertAlmostEqual(values["X0"], 0.0, places=7)
        self.assertAlmostEqual(values["Z1"], 0.0, places=7)

    def test_counts_shape_and_shots_are_enforced(self) -> None:
        malformed = np.zeros((9, 64), dtype=np.int64)
        malformed[:, 0] = 1023
        with self.assertRaisesRegex(ValueError, "does not equal expected shots"):
            counts_array_to_pauli15(malformed)


if __name__ == "__main__":
    unittest.main()
