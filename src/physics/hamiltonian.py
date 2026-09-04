"""Six-qubit OBC Hamiltonian and Lindblad reference evolution."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from qutip import Qobj, basis, fidelity, identity, mesolve, ptrace, sigmax, sigmay, sigmaz, tensor


NUM_QUBITS = 6
I2 = identity(2)
X = sigmax()
Y = sigmay()
Z = sigmaz()


def local_op(site: int, operator: Qobj) -> Qobj:
    operators = [I2] * NUM_QUBITS
    operators[site] = operator
    return tensor(*operators)


def two_body_op(site_a: int, site_b: int, operator_a: Qobj, operator_b: Qobj) -> Qobj:
    operators = [I2] * NUM_QUBITS
    operators[site_a] = operator_a
    operators[site_b] = operator_b
    return tensor(*operators)


def fixed_neel_density_matrix(preparation_depolarization: float = 0.0) -> Qobj:
    """Return the paper's depolarized |010101><010101| initial state."""
    if not 0.0 <= preparation_depolarization <= 0.2:
        raise ValueError("preparation_depolarization must be in [0, 0.2]")
    ket = tensor(*(basis(2, value) for value in (0, 1, 0, 1, 0, 1)))
    pure = ket.proj()
    if preparation_depolarization == 0.0:
        return pure
    mixed = tensor(*([I2] * NUM_QUBITS)) / (2**NUM_QUBITS)
    return (1.0 - preparation_depolarization) * pure + preparation_depolarization * mixed


def depolarize_initial_density_matrix(
    pure_density_matrix: Qobj, preparation_depolarization: float
) -> Qobj:
    """Apply the manuscript's preparation depolarization to a pure initial state."""
    if not 0.0 <= preparation_depolarization <= 0.2:
        raise ValueError("preparation_depolarization must be in [0, 0.2]")
    if pure_density_matrix.shape != (2**NUM_QUBITS, 2**NUM_QUBITS):
        raise ValueError("Initial density matrix has an unexpected Hilbert-space dimension.")
    trace = complex(pure_density_matrix.tr())
    if not np.isclose(trace, 1.0, atol=1e-10):
        raise ValueError("Initial density matrix must have unit trace.")
    maximally_mixed = tensor(*([I2] * NUM_QUBITS)) / (2**NUM_QUBITS)
    return (1.0 - preparation_depolarization) * pure_density_matrix + preparation_depolarization * maximally_mixed


def ground_state_density_matrix(
    parameters: Mapping[str, object], preparation_depolarization: float = 0.0
) -> Qobj:
    """Return the manuscript profile's depolarized instantaneous Hamiltonian ground state."""
    _, ground_state = build_hamiltonian(parameters).groundstate()
    return depolarize_initial_density_matrix(ground_state.proj(), preparation_depolarization)


def sample_parameters(rng: np.random.Generator) -> dict[str, object]:
    """Keep the old generator's parameter ranges while removing them from model inputs."""
    j_base = float(rng.uniform(-1.0, 2.0))
    epsilon = float(rng.uniform(-0.12, 0.12))
    jx = j_base * (1.0 + epsilon)
    jy = j_base * (1.0 - epsilon)
    # The manuscript fixes local-field sampling to [-2, 2].
    hz = rng.uniform(-2.0, 2.0, NUM_QUBITS).astype(float)
    hy = rng.uniform(-2.0, 2.0, NUM_QUBITS).astype(float)
    hx = rng.uniform(-2.0, 2.0, NUM_QUBITS).astype(float)
    return {
        "Jx": float(jx),
        "Jy": float(jy),
        "Jz": float(rng.uniform(-0.5, 1.8)),
        "Jxz": float(rng.uniform(-0.6, 0.6)),
        "Jzx": float(rng.uniform(-0.6, 0.6)),
        "D": float(rng.uniform(-1.0, 1.0)),
        "hx": hx.tolist(),
        "hy": hy.tolist(),
        "hz": hz.tolist(),
        "h1": float(hx[0]),
        "h2": float(hx[1]),
        "hy1": float(hy[0]),
        "hy2": float(hy[1]),
        "hz1": float(hz[0]),
        "hz2": float(hz[1]),
    }


def evolution_time_and_gamma(
    parameters: Mapping[str, object], rng: np.random.Generator
) -> tuple[float, float]:
    j_eff = math.sqrt(
        max(
            1e-8,
            float(parameters["Jz"]) ** 2
            + float(parameters["Jx"]) ** 2
            + float(parameters["Jy"]) ** 2
            + float(parameters["D"]) ** 2,
        )
    )
    evolution_time = float(
        np.clip(rng.normal(math.pi / (4.0 * max(j_eff, 1e-4)), 0.10), 0.05, 5.0)
    )
    gamma = float(rng.uniform(0.02, 0.08))
    return evolution_time, gamma


def sample_preparation_depolarization(rng: np.random.Generator) -> float:
    """Sample the manuscript's independent state-preparation depolarization p."""
    return float(rng.uniform(0.0, 0.2))


def build_hamiltonian(parameters: Mapping[str, object]) -> Qobj:
    hamiltonian: Qobj | int = 0
    jx = float(parameters["Jx"])
    jy = float(parameters["Jy"])
    jz = float(parameters["Jz"])
    jxz = float(parameters["Jxz"])
    jzx = float(parameters["Jzx"])
    dm = float(parameters["D"])
    hx = parameters["hx"]
    hy = parameters["hy"]
    hz = parameters["hz"]
    for site in range(NUM_QUBITS - 1):
        hamiltonian += jx * two_body_op(site, site + 1, X, X)
        hamiltonian += jy * two_body_op(site, site + 1, Y, Y)
        hamiltonian += jz * two_body_op(site, site + 1, Z, Z)
        hamiltonian += jxz * two_body_op(site, site + 1, X, Z)
        hamiltonian += jzx * two_body_op(site, site + 1, Z, X)
        hamiltonian += dm * (
            two_body_op(site, site + 1, X, Y) - two_body_op(site, site + 1, Y, X)
        )
    for site in range(NUM_QUBITS):
        hamiltonian += float(hx[site]) * local_op(site, X)
        hamiltonian += float(hy[site]) * local_op(site, Y)
        hamiltonian += float(hz[site]) * local_op(site, Z)
    return hamiltonian


def depolarizing_collapse_operators(gamma: float) -> list[Qobj]:
    """Local isotropic depolarization L_k^a=sqrt(gamma/3)*sigma_k^a."""
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    return [
        math.sqrt(gamma / 3.0) * local_op(site, operator)
        for site in range(NUM_QUBITS)
        for operator in (X, Y, Z)
    ]


def evolve_density_matrix(
    parameters: Mapping[str, object],
    evolution_time: float,
    gamma: float,
    preparation_depolarization: float = 0.0,
) -> Qobj:
    result = mesolve(
        build_hamiltonian(parameters),
        fixed_neel_density_matrix(preparation_depolarization),
        [0.0, evolution_time],
        c_ops=depolarizing_collapse_operators(gamma),
        options={"nsteps": 3000, "atol": 1e-8, "rtol": 1e-6, "store_final_state": True},
    )
    return result.states[-1]


def entropy_base2(density_matrix: Qobj) -> float:
    eigenvalues = np.clip(np.linalg.eigvalsh(density_matrix.full()), 1e-12, 1.0)
    return float(-(eigenvalues * np.log2(eigenvalues)).sum())


def auxiliary_labels(density_matrix: Qobj, parameters: Mapping[str, object]) -> dict[str, float | int]:
    pair_state = ptrace(density_matrix, [0, 1])
    bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit().proj()
    phase_score = (
        float(parameters["Jx"])
        + float(parameters["Jy"])
        + float(parameters["Jz"])
        + float(parameters["D"])
        + float(parameters["h1"])
        + float(parameters["h2"])
        + float(parameters["hz1"])
        + float(parameters["hz2"])
    )
    phase_label = 2 if phase_score > 1.5 else 1 if phase_score > 0.0 else 0
    return {
        "entropies": entropy_base2(density_matrix),
        "inter_entropies": entropy_base2(ptrace(pair_state, [0])) + entropy_base2(ptrace(pair_state, [1])),
        "target_fidelities": float(fidelity(pair_state, bell)),
        "phase_labels": phase_label,
    }
