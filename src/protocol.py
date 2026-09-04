"""Validation and submission gates for the frozen experiment protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


EXPECTED_BASIS_ORDER = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")
EXPECTED_PAULI15_ORDER = (
    "X0", "Y0", "Z0", "X1", "Y1", "Z1",
    "X0X1", "X0Y1", "X0Z1", "Y0X1", "Y0Y1", "Y0Z1", "Z0X1", "Z0Y1", "Z0Z1",
)
EXPECTED_XOBS_ORDER = ("X0", "Y0", "Z0", "X0X1", "Y0Y1", "Z0Z1")
EXPECTED_XOBS_INDICES = (0, 1, 2, 6, 10, 14)
EXPECTED_TARGETS = ("h1", "h2", "Jz")
EXPECTED_TRAINING_LOSS_WEIGHTS = {"h1": 1.0, "h2": 1.0, "Jz": 80.0}
BACKEND_ROLES = ("primary_noisy_simulator", "ideal_reference_simulator", "hardware")


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_contract(protocol: Mapping[str, object], backends: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    measurement = protocol.get("measurement", {})
    data_contract = protocol.get("model_data_contract", {})
    system = protocol.get("system", {})
    hamiltonian = protocol.get("hamiltonian", {})
    roles = backends.get("roles", {})

    if system.get("num_qubits") != 6:
        errors.append("system.num_qubits must equal 6")
    if system.get("initial_state") != "neel_010101_v1":
        errors.append("initial state must be neel_010101_v1")
    if system.get("boundary_condition") != "open":
        errors.append("boundary condition must be open")
    if tuple(measurement.get("basis_order", ())) != EXPECTED_BASIS_ORDER:
        errors.append("measurement basis order changed")
    if measurement.get("shots_per_basis") != 1024:
        errors.append("shots_per_basis must equal 1024")
    if tuple(measurement.get("pauli15_qa_order", ())) != EXPECTED_PAULI15_ORDER:
        errors.append("pauli15_qa order changed")
    if tuple(measurement.get("xobs_model_order", ())) != EXPECTED_XOBS_ORDER:
        errors.append("xobs_model order changed")
    if tuple(measurement.get("xobs_model_pauli15_indices", ())) != EXPECTED_XOBS_INDICES:
        errors.append("xobs_model selection indices changed")
    derived_xobs = tuple(EXPECTED_PAULI15_ORDER[index] for index in EXPECTED_XOBS_INDICES)
    if derived_xobs != EXPECTED_XOBS_ORDER:
        errors.append("xobs_model order is inconsistent with pauli15 indices")

    if tuple(data_contract.get("input_keys", ())) != ("xobs_model", "t"):
        errors.append("model input keys must be xobs_model and t")
    if tuple(data_contract.get("supervised_targets", ())) != EXPECTED_TARGETS:
        errors.append("supervised targets must be h1, h2, Jz")
    if data_contract.get("training_loss_weights") != EXPECTED_TRAINING_LOSS_WEIGHTS:
        errors.append("training loss weights must be h1=1, h2=1, Jz=80")
    if set(data_contract.get("input_keys", ())).intersection(EXPECTED_TARGETS):
        errors.append("supervised targets leaked into model inputs")
    if not data_contract.get("hardware_test_is_training_forbidden", False):
        errors.append("hardware TEST must be forbidden from training")
    if tuple(hamiltonian.get("primary_targets", ())) != EXPECTED_TARGETS:
        errors.append("Hamiltonian primary targets changed")

    for role in BACKEND_ROLES:
        if role not in roles:
            errors.append(f"missing backend role: {role}")
    hardware = roles.get("hardware", {})
    if hardware.get("display_name") != "tianyan-287":
        errors.append("hardware backend must be tianyan-287")
    primary = roles.get("primary_noisy_simulator", {})
    if primary.get("display_name") != "密度矩阵带噪声":
        errors.append("primary simulator must be noisy density matrix")
    reference = roles.get("ideal_reference_simulator", {})
    if reference.get("display_name") != "全振幅":
        errors.append("reference simulator must be full amplitude")
    return errors


def submission_blockers(
    protocol: Mapping[str, object],
    backends: Mapping[str, object],
    g0: Mapping[str, object] | None = None,
) -> list[str]:
    blockers = validate_contract(protocol, backends)
    roles = backends.get("roles", {})
    for role in BACKEND_ROLES:
        value = roles.get(role, {})
        if not value.get("machine_code"):
            blockers.append(f"{role}.machine_code was not confirmed by authenticated CQlib")
        if not value.get("submit_enabled", False):
            blockers.append(f"{role}.submit_enabled is false")
    if protocol.get("status") != "locked_at_g0":
        blockers.append("protocol is not locked_at_g0")
    digitalization = protocol.get("digitalization", {})
    if digitalization.get("selected_order") is None or digitalization.get("selected_steps") is None:
        blockers.append("Trotter order/steps are not selected")
    if not g0 or not g0.get("approved_by") or not g0.get("approved_at"):
        blockers.append("G0 approval is missing")
    return blockers


def assert_submission_ready(
    protocol: Mapping[str, object],
    backends: Mapping[str, object],
    g0: Mapping[str, object] | None = None,
) -> None:
    blockers = submission_blockers(protocol, backends, g0)
    if blockers:
        details = "\n- ".join(blockers)
        raise RuntimeError(f"Submission is blocked:\n- {details}")
