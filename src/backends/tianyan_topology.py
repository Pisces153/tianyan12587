"""Select a calibrated connected physical chain from a TianYan machine configuration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PhysicalChain:
    physical_labels: tuple[str, ...]
    circuit_qubits: tuple[int, ...]
    couplers: tuple[str, ...]
    mean_cz_error_percent: float
    mean_readout_error_percent: float


def _csv_set(value: object) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _metric_map(section: Mapping[str, object], metric: str) -> dict[str, float]:
    payload = section.get(metric, {})
    if not isinstance(payload, Mapping):
        return {}
    names = payload.get("qubit_used", [])
    values = payload.get("param_list", [])
    if not isinstance(names, Sequence) or not isinstance(values, Sequence):
        return {}
    return {str(name): float(value) for name, value in zip(names, values)}


def _circuit_qubit(label: str, *, zero_indexed: bool) -> int:
    if not label.startswith("Q") or not label[1:].isdigit():
        raise ValueError(f"Unexpected TianYan physical-qubit label: {label!r}")
    index = int(label[1:])
    return index if zero_indexed else index - 1


def _uses_zero_indexed_labels(overview: Mapping[str, object]) -> bool:
    labels = overview.get("qubits", [])
    if isinstance(labels, Sequence) and "Q0" in labels:
        return True
    coupler_map = overview.get("coupler_map", {})
    if not isinstance(coupler_map, Mapping):
        return False
    return any("Q0" in endpoints for endpoints in coupler_map.values() if isinstance(endpoints, Sequence))


def select_six_qubit_chain(machine_config: Mapping[str, object]) -> PhysicalChain:
    """Choose the lowest-error simple six-qubit path from current calibration data."""
    overview = machine_config.get("overview", {})
    if not isinstance(overview, Mapping):
        raise ValueError("Machine configuration lacks overview metadata.")
    coupler_map = overview.get("coupler_map", {})
    if not isinstance(coupler_map, Mapping):
        raise ValueError("Machine configuration lacks a coupler map.")
    zero_indexed = _uses_zero_indexed_labels(overview)
    disabled_qubits = _csv_set(machine_config.get("disabledQubits"))
    disabled_couplers = _csv_set(machine_config.get("disabledCouplers"))
    two_qubit = machine_config.get("twoQubitGate", {})
    readout = machine_config.get("readout", {})
    if not isinstance(two_qubit, Mapping) or not isinstance(readout, Mapping):
        raise ValueError("Machine configuration lacks calibration error data.")
    cz_gate = two_qubit.get("czGate", {})
    readout_array = readout.get("readoutArray", {})
    if not isinstance(cz_gate, Mapping) or not isinstance(readout_array, Mapping):
        raise ValueError("Machine configuration has invalid calibration sections.")
    cz_error = _metric_map(cz_gate, "gate error")
    readout_error = _metric_map(readout_array, "Readout Error")

    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for coupler, endpoints in coupler_map.items():
        if str(coupler) in disabled_couplers or str(coupler) not in cz_error:
            continue
        if not isinstance(endpoints, Sequence) or len(endpoints) != 2:
            continue
        first, second = (str(endpoint) for endpoint in endpoints)
        if first in disabled_qubits or second in disabled_qubits:
            continue
        graph[first].append((second, str(coupler)))
        graph[second].append((first, str(coupler)))

    candidates: dict[tuple[str, ...], PhysicalChain] = {}

    def visit(nodes: list[str], couplers: list[str]) -> None:
        if len(nodes) == 6:
            canonical = min(tuple(nodes), tuple(reversed(nodes)))
            chain = PhysicalChain(
                physical_labels=tuple(nodes),
                circuit_qubits=tuple(_circuit_qubit(node, zero_indexed=zero_indexed) for node in nodes),
                couplers=tuple(couplers),
                mean_cz_error_percent=sum(cz_error[coupler] for coupler in couplers) / len(couplers),
                mean_readout_error_percent=sum(readout_error.get(node, 100.0) for node in nodes) / len(nodes),
            )
            previous = candidates.get(canonical)
            score = chain.mean_cz_error_percent + chain.mean_readout_error_percent / 10.0
            if previous is None or score < previous.mean_cz_error_percent + previous.mean_readout_error_percent / 10.0:
                candidates[canonical] = chain
            return
        for neighbor, coupler in graph[nodes[-1]]:
            if neighbor not in nodes:
                visit([*nodes, neighbor], [*couplers, coupler])

    for start in graph:
        visit([start], [])
    if not candidates:
        raise RuntimeError("No active connected six-qubit chain is available on this calibration.")
    return min(
        candidates.values(),
        key=lambda chain: chain.mean_cz_error_percent + chain.mean_readout_error_percent / 10.0,
    )
