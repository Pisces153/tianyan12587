from __future__ import annotations

import unittest

import numpy as np

from src.physics.hamiltonian import (
    NUM_QUBITS,
    depolarizing_collapse_operators,
    fixed_neel_density_matrix,
    ground_state_density_matrix,
    sample_parameters,
)


class PhysicsContractTests(unittest.TestCase):
    def test_preparation_depolarization_is_a_valid_density_matrix(self) -> None:
        density = fixed_neel_density_matrix(0.2)
        self.assertAlmostEqual(float(density.tr()), 1.0, places=12)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(density.full()).min()), -1e-12)
        with self.assertRaises(ValueError):
            fixed_neel_density_matrix(0.21)

    def test_local_depolarization_has_three_channels_per_qubit(self) -> None:
        operators = depolarizing_collapse_operators(0.06)
        self.assertEqual(len(operators), 3 * NUM_QUBITS)

    def test_local_field_range_matches_the_manuscript(self) -> None:
        values = sample_parameters(np.random.default_rng(20260727))
        for name in ("hx", "hy", "hz"):
            self.assertTrue(np.all(np.abs(np.asarray(values[name], dtype=float)) <= 2.0))

    def test_ground_state_preparation_is_a_valid_density_matrix(self) -> None:
        parameters = sample_parameters(np.random.default_rng(20260727))
        density = ground_state_density_matrix(parameters, 0.2)
        self.assertAlmostEqual(float(density.tr()), 1.0, places=10)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(density.full()).min()), -1e-10)


if __name__ == "__main__":
    unittest.main()
