"""Bounded nonlinear nominal-model inversion for the TianYan V8 baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.backends.tianyan_v8_entangling import LOCAL6, pauli15_expectation
from src.features.pauli import select_pauli_features


def forward_local6(parameters: Mapping[str, float], times: Sequence[float]) -> np.ndarray:
    """Nominal V8 local6 trajectory in exactly the AEMTN observation order."""
    return np.stack(
        [select_pauli_features(pauli15_expectation(float(parameters["h1"]), float(parameters["h2"]), float(time)), LOCAL6) for time in times]
    )


def fit_parameters(observed_local6: np.ndarray, times: Sequence[float], *, lower: float, upper: float, max_nfev: int) -> tuple[dict[str, float], dict[str, object]]:
    """Fit bounded ``h1,h2`` from local6 only; no sealed truth is accepted."""
    observed = np.asarray(observed_local6, dtype=np.float64)
    time_array = np.asarray(times, dtype=np.float64)
    if observed.shape != (len(time_array), 6):
        raise ValueError(f"Expected local6 shape ({len(time_array)}, 6), got {observed.shape}")
    if not lower < upper or max_nfev < 1:
        raise ValueError("Invalid bounded nonlinear baseline settings")

    def residual(vector: np.ndarray) -> np.ndarray:
        prediction = forward_local6({"h1": float(vector[0]), "h2": float(vector[1])}, time_array)
        return (prediction - observed).reshape(-1)

    grid = np.linspace(lower * 0.6, upper * 0.6, 3, dtype=np.float64)
    candidates = [
        least_squares(
            residual,
            x0=np.asarray((first, second)),
            bounds=(np.full(2, lower), np.full(2, upper)),
            max_nfev=max_nfev,
            method="trf",
        )
        for first in grid
        for second in grid
    ]
    successful = [candidate for candidate in candidates if candidate.success]
    if not successful:
        raise RuntimeError("Nominal V8 inversion failed for every frozen starting point")
    result = min(successful, key=lambda candidate: float(candidate.cost))
    estimate = {"h1": float(result.x[0]), "h2": float(result.x[1])}
    metadata = {"method": "bounded_nonlinear_least_squares_nominal_V8", "fixed_multistart_grid": 3, "max_nfev_per_start": int(max_nfev), "nfev_selected": int(result.nfev), "cost": float(result.cost), "status": int(result.status)}
    return estimate, metadata


def estimate_offset(reference_local6: np.ndarray, observed_local6: np.ndarray, times: Sequence[float], *, lower: float, upper: float, max_nfev: int) -> tuple[dict[str, float], dict[str, object]]:
    """Infer before-minus-reference offset under the frozen nominal task model."""
    reference, ref_info = fit_parameters(reference_local6, times, lower=lower, upper=upper, max_nfev=max_nfev)
    observed, obs_info = fit_parameters(observed_local6, times, lower=lower, upper=upper, max_nfev=max_nfev)
    return ({name: float(observed[name] - reference[name]) for name in ("h1", "h2")}, {"reference_fit": ref_info, "before_fit": obs_info})
