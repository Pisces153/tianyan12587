from __future__ import annotations

import numpy as np

from src.backends.tianyan_native import assert_native_qcis, gate_counts
from src.backends.tianyan_v8_entangling import CZ_COUNT, LOCAL6, pauli15_expectation, program
from src.baselines.tianyan_v8_nominal import estimate_offset, forward_local6


def test_v8_program_is_native_and_has_fixed_two_cz() -> None:
    qcis = program({"h1": 0.25, "h2": -0.35}, 0.31, "YZ", [0, 1, 2, 3, 4, 5])
    assert_native_qcis(qcis)
    assert gate_counts(qcis)["CZ"] == CZ_COUNT


def test_v8_nominal_baseline_recovers_noiseless_controlled_offset() -> None:
    times = (0.16, 0.31, 0.47)
    reference = {"h1": 0.25, "h2": -0.35}
    offset = {"h1": 0.40, "h2": -0.40}
    estimate, _ = estimate_offset(
        forward_local6(reference, times),
        forward_local6({name: reference[name] + offset[name] for name in reference}, times),
        times,
        lower=-1.5,
        upper=1.5,
        max_nfev=120,
    )
    assert np.allclose([estimate["h1"], estimate["h2"]], [offset["h1"], offset["h2"]], atol=0.01)


def test_v8_pauli_expectation_and_local_contract() -> None:
    values = pauli15_expectation(0.2, -0.1, 0.31)
    assert values.shape == (15,)
    assert len(LOCAL6) == 6
    assert np.max(np.abs(values)) <= 1.0


def test_fixed_cross_drive_is_explicit_simulator_physics() -> None:
    nominal = pauli15_expectation(0.4, -0.3, 0.31)
    crosstalk = pauli15_expectation(0.4, -0.3, 0.31, cross_drive=0.18)
    assert not np.allclose(nominal, crosstalk)
