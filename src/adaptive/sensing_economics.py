"""Statistical core for B-4 calibration-sensing economics.

All decisions consume observations, timestamps, shot counts, and acquisition
metadata only.  Simulation labels and latent parameters are deliberately absent
from the public analysis API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import chi2, norm


@dataclass(frozen=True)
class VarianceGate:
    degrees_of_freedom: int
    total_variance: float
    mean_shot_variance: float
    process_variance: float
    process_variance_ci_lower: float
    process_variance_ci_upper: float
    p_value: float
    passed: bool


@dataclass(frozen=True)
class OUFit:
    ok: bool
    process_variance: float | None
    process_variance_ci_lower: float | None
    process_variance_ci_upper: float | None
    tau_seconds: float | None
    tau_ci_lower_seconds: float | None
    tau_ci_upper_seconds: float | None
    n_bins: int


def binomial_variance_unbiased(values: Sequence[float], shots: Sequence[int] | int) -> np.ndarray:
    """Unbiased estimate of Var(p-hat) from binomial counts.

    p-hat(1-p-hat) * N/(N-1) / N = p-hat(1-p-hat)/(N-1).
    """
    probability = np.asarray(values, dtype=np.float64)
    sample_size = np.broadcast_to(np.asarray(shots, dtype=np.float64), probability.shape)
    if np.any(sample_size <= 1):
        raise ValueError("shot counts must exceed one")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("observed probabilities must lie in [0, 1]")
    return probability * (1.0 - probability) / (sample_size - 1.0)


def _arrays(
    values: Sequence[float],
    times_seconds: Sequence[float],
    shots: Sequence[int] | int,
    regime_ids: Sequence[str | int] | None,
    burst_flags: Sequence[bool] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_seconds, dtype=np.float64)
    sample_size = np.broadcast_to(np.asarray(shots, dtype=np.int64), observed.shape).copy()
    regimes = np.asarray(regime_ids if regime_ids is not None else np.zeros(len(observed), dtype=int))
    bursts = np.asarray(burst_flags if burst_flags is not None else np.zeros(len(observed), dtype=bool), dtype=bool)
    if not (observed.ndim == times.ndim == sample_size.ndim == regimes.ndim == bursts.ndim == 1):
        raise ValueError("analysis inputs must be one-dimensional")
    if not (len(observed) == len(times) == len(sample_size) == len(regimes) == len(bursts)):
        raise ValueError("analysis inputs must have equal length")
    if len(observed) < 3 or np.any(np.diff(times) < 0):
        raise ValueError("at least three time-ordered observations are required")
    return observed, times, sample_size, regimes, bursts


def variance_component_gate(
    values: Sequence[float],
    shots: Sequence[int] | int,
    regime_ids: Sequence[str | int] | None = None,
    burst_flags: Sequence[bool] | None = None,
    instrument_ids: Sequence[str | int] | None = None,
    *,
    alpha: float = 0.05,
) -> VarianceGate:
    """One-sided chi-square gate against analytically known shot variance.

    Each regime is centered independently, so calibration steps cannot create a
    continuous-drift claim.  Burst observations are excluded from the primary
    estimator by construction.
    """
    observed = np.asarray(values, dtype=np.float64)
    sample_size = np.broadcast_to(np.asarray(shots, dtype=np.int64), observed.shape)
    regimes = np.asarray(regime_ids if regime_ids is not None else np.zeros(len(observed), dtype=int))
    bursts = np.asarray(burst_flags if burst_flags is not None else np.zeros(len(observed), dtype=bool), dtype=bool)
    instruments = np.asarray(instrument_ids if instrument_ids is not None else np.zeros(len(observed), dtype=int))
    if not (len(observed) == len(sample_size) == len(regimes) == len(bursts) == len(instruments)):
        raise ValueError("variance gate inputs must have equal length")
    keep = ~bursts
    observed = observed[keep]
    sample_size = sample_size[keep]
    regimes = regimes[keep]
    instruments = instruments[keep]
    residual_parts: list[np.ndarray] = []
    degrees_of_freedom = 0
    groups = np.asarray([f"{regime!r}|{instrument!r}" for regime, instrument in zip(regimes, instruments, strict=True)])
    for group in np.unique(groups):
        block = observed[groups == group]
        if len(block) < 2:
            continue
        residual_parts.append(block - block.mean())
        degrees_of_freedom += len(block) - 1
    if degrees_of_freedom < 2:
        return VarianceGate(degrees_of_freedom, float("nan"), float("nan"), float("nan"), 0.0, float("inf"), 1.0, False)
    residual = np.concatenate(residual_parts)
    sum_squares = float(residual @ residual)
    total_variance = sum_squares / degrees_of_freedom
    shot_variance = binomial_variance_unbiased(observed, sample_size)
    mean_shot_variance = float(np.mean(shot_variance))
    statistic = sum_squares / max(mean_shot_variance, np.finfo(float).tiny)
    lower_total = sum_squares / chi2.ppf(1.0 - alpha, degrees_of_freedom)
    upper_total = sum_squares / chi2.ppf(alpha / 2.0, degrees_of_freedom)
    lower_process = float(lower_total - mean_shot_variance)
    upper_process = float(upper_total - mean_shot_variance)
    process_variance = float(total_variance - mean_shot_variance)
    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    return VarianceGate(
        degrees_of_freedom=degrees_of_freedom,
        total_variance=total_variance,
        mean_shot_variance=mean_shot_variance,
        process_variance=process_variance,
        process_variance_ci_lower=lower_process,
        process_variance_ci_upper=upper_process,
        p_value=p_value,
        passed=bool(lower_process > 0.0),
    )


def structure_function(
    values: Sequence[float],
    times_seconds: Sequence[float],
    shots: Sequence[int] | int,
    regime_ids: Sequence[str | int] | None,
    burst_flags: Sequence[bool] | None,
    lag_edges_seconds: Sequence[float],
    instrument_ids: Sequence[str | int] | None = None,
    *,
    alpha: float = 0.05,
) -> list[dict[str, float | int]]:
    """Exact shot-noise-debiased SF, excluding bursts and cross-regime pairs."""
    observed, times, sample_size, regimes, bursts = _arrays(values, times_seconds, shots, regime_ids, burst_flags)
    shot_variance = binomial_variance_unbiased(observed, sample_size)
    instruments = np.asarray(instrument_ids if instrument_ids is not None else np.zeros(len(observed), dtype=int))
    if instruments.ndim != 1 or len(instruments) != len(observed):
        raise ValueError("instrument_ids must be one-dimensional and match observations")
    left, right = np.triu_indices(len(observed), k=1)
    eligible = (
        (~bursts[left])
        & (~bursts[right])
        & (regimes[left] == regimes[right])
        & (instruments[left] == instruments[right])
    )
    left = left[eligible]
    right = right[eligible]
    lag = times[right] - times[left]
    corrected = (observed[right] - observed[left]) ** 2 - shot_variance[left] - shot_variance[right]
    raw = (observed[right] - observed[left]) ** 2
    floor = shot_variance[left] + shot_variance[right]
    edges = np.asarray(lag_edges_seconds, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("lag edges must be a strictly increasing vector")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    rows: list[dict[str, float | int]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (lag >= lo) & (lag < hi)
        pair_count = int(selected.sum())
        if pair_count < 2:
            continue
        distinct_points = int(np.unique(np.concatenate([left[selected], right[selected]])).size)
        effective_count = max(2, distinct_points - 1)
        estimate = float(np.mean(corrected[selected]))
        standard_error = float(np.std(corrected[selected], ddof=1) / np.sqrt(effective_count))
        rows.append({
            "lag_lo_seconds": float(lo),
            "lag_hi_seconds": float(hi),
            "lag_mid_seconds": float(np.mean(lag[selected])),
            "n_pairs": pair_count,
            "n_effective_points": distinct_points,
            "sf_raw": float(np.mean(raw[selected])),
            "shot_floor": float(np.mean(floor[selected])),
            "sf_debiased": estimate,
            "sf_standard_error": standard_error,
            "sf_ci_lower": estimate - z * standard_error,
            "sf_ci_upper": estimate + z * standard_error,
        })
    return rows


def fit_ou_structure(rows: Sequence[dict[str, float | int]], *, alpha: float = 0.05) -> OUFit:
    """Fit SF(t)=2*Var_proc*(1-exp(-t/tau)) using observed SF rows only."""
    usable = [row for row in rows if np.isfinite(float(row["sf_debiased"])) and float(row["lag_mid_seconds"]) > 0.0]
    if len(usable) < 3:
        return OUFit(False, None, None, None, None, None, None, len(usable))
    lag = np.asarray([float(row["lag_mid_seconds"]) for row in usable])
    estimate = np.asarray([float(row["sf_debiased"]) for row in usable])
    standard_error = np.asarray([max(float(row["sf_standard_error"]), np.finfo(float).eps) for row in usable])
    positive_scale = max(float(np.median(np.maximum(estimate, 0.0))), float(np.mean(standard_error)), np.finfo(float).eps)
    variance_start = max(positive_scale / 2.0, np.finfo(float).eps)
    tau_start = float(np.median(lag))

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        process_variance, tau = np.exp(log_parameters)
        model = 2.0 * process_variance * (1.0 - np.exp(-lag / tau))
        return (model - estimate) / standard_error

    lower_log_bounds = np.log([np.finfo(float).tiny, max(float(np.min(lag)) / 1000.0, np.finfo(float).tiny)])
    upper_log_bounds = np.log([1.0, max(float(np.max(lag)) * 1000.0, 1.0)])
    fitted = least_squares(
        residual,
        np.log([variance_start, tau_start]),
        bounds=(lower_log_bounds, upper_log_bounds),
        max_nfev=200,
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return OUFit(False, None, None, None, None, None, None, len(usable))
    parameters = np.exp(fitted.x)
    dof = max(len(usable) - 2, 1)
    try:
        covariance = np.linalg.inv(fitted.jac.T @ fitted.jac) * float((fitted.fun @ fitted.fun) / dof)
        log_standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    except np.linalg.LinAlgError:
        log_standard_error = np.asarray([float("inf"), float("inf")])
    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower = np.exp(np.maximum(fitted.x - z * log_standard_error, lower_log_bounds))
    upper = np.exp(np.minimum(fitted.x + z * log_standard_error, upper_log_bounds))
    return OUFit(
        ok=True,
        process_variance=float(parameters[0]),
        process_variance_ci_lower=float(lower[0]),
        process_variance_ci_upper=float(upper[0]),
        tau_seconds=float(parameters[1]),
        tau_ci_lower_seconds=float(lower[1]),
        tau_ci_upper_seconds=float(upper[1]),
        n_bins=len(usable),
    )


def ou_structure_value(interval_seconds: np.ndarray | float, process_variance: float, tau_seconds: float) -> np.ndarray:
    interval = np.asarray(interval_seconds, dtype=np.float64)
    return 2.0 * process_variance * (1.0 - np.exp(-interval / tau_seconds))


def ou_residual_variance(
    interval_seconds: np.ndarray | float,
    mean_probability: float,
    effective_shots_per_second: float,
    process_variance: float,
    tau_seconds: float,
) -> np.ndarray:
    interval = np.asarray(interval_seconds, dtype=np.float64)
    if np.any(interval <= 0.0) or effective_shots_per_second <= 0.0 or tau_seconds <= 0.0:
        raise ValueError("interval, throughput, and correlation time must be positive")
    shot_term = mean_probability * (1.0 - mean_probability) / (effective_shots_per_second * interval)
    integral_term = 2.0 * process_variance * (
        1.0 - (tau_seconds / interval) * (1.0 - np.exp(-interval / tau_seconds))
    )
    return shot_term + integral_term


def monotone_structure_curve(rows: Sequence[dict[str, float | int]]) -> dict[str, list[float]]:
    """Weighted isotonic projection of debiased SF with the physical SF(0)=0 anchor."""
    usable = sorted(
        (row for row in rows if np.isfinite(float(row["sf_debiased"])) and float(row["lag_mid_seconds"]) > 0.0),
        key=lambda row: float(row["lag_mid_seconds"]),
    )
    if not usable:
        return {"lag_seconds": [0.0], "sf": [0.0]}
    values = [max(float(row["sf_debiased"]), 0.0) for row in usable]
    weights = [max(float(row.get("n_effective_points", 1)), 1.0) for row in usable]
    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append({"start": index, "end": index, "weight": weight, "mean": value})
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(blocks[-1]["mean"]):
            right = blocks.pop()
            left = blocks.pop()
            combined_weight = float(left["weight"]) + float(right["weight"])
            combined_mean = (
                float(left["mean"]) * float(left["weight"])
                + float(right["mean"]) * float(right["weight"])
            ) / combined_weight
            blocks.append({
                "start": int(left["start"]),
                "end": int(right["end"]),
                "weight": combined_weight,
                "mean": combined_mean,
            })
    fitted = np.empty(len(usable), dtype=np.float64)
    for block in blocks:
        fitted[int(block["start"]): int(block["end"]) + 1] = float(block["mean"])
    return {
        "lag_seconds": [0.0, *[float(row["lag_mid_seconds"]) for row in usable]],
        "sf": [0.0, *[float(value) for value in fitted]],
    }


def nonparametric_residual_variance(
    interval_seconds: np.ndarray | float,
    mean_probability: float,
    effective_shots_per_second: float,
    rows: Sequence[dict[str, float | int]],
) -> np.ndarray:
    """Integrate monotone piecewise-linear SF; hold final plateau beyond data."""
    interval = np.asarray(interval_seconds, dtype=np.float64)
    if np.any(interval <= 0.0) or effective_shots_per_second <= 0.0:
        raise ValueError("interval and throughput must be positive")
    curve = monotone_structure_curve(rows)
    lags = np.asarray(curve["lag_seconds"], dtype=np.float64)
    values = np.asarray(curve["sf"], dtype=np.float64)

    def integral_one(limit: float) -> float:
        interior = lags[(lags > 0.0) & (lags < limit)]
        grid = np.concatenate(([0.0], interior, [limit]))
        height = np.interp(grid, lags, values, left=0.0, right=float(values[-1]))
        return float(np.trapezoid(height, grid))

    integrals = np.asarray([integral_one(float(value)) for value in interval.reshape(-1)]).reshape(interval.shape)
    shot_term = mean_probability * (1.0 - mean_probability) / (effective_shots_per_second * interval)
    return shot_term + integrals / interval


def optimum_nonparametric_interval(
    *,
    rows: Sequence[dict[str, float | int]],
    mean_probability: float,
    effective_shots_per_second: float,
    maximum_interval_seconds: float,
    minimum_interval_seconds: float | None = None,
) -> dict[str, float | bool]:
    lower = max(1.0 / effective_shots_per_second, float(minimum_interval_seconds or 0.0), np.finfo(float).eps)
    upper = float(maximum_interval_seconds)
    if upper <= lower:
        raise ValueError("maximum interval must exceed minimum interval")
    result = minimize_scalar(
        lambda value: float(nonparametric_residual_variance(value, mean_probability, effective_shots_per_second, rows)),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": max(lower * 1e-6, 1e-7)},
    )
    interval = float(result.x)
    tolerance = max((upper - lower) * 1e-4, 1e-7)
    return {
        "interval_seconds": interval,
        "minimum_residual_variance": float(result.fun),
        "interior": bool(result.success and interval > lower + tolerance and interval < upper - tolerance),
        "lower_bound_seconds": lower,
        "upper_bound_seconds": upper,
    }


def _simulate_fitted_ou_observations(
    observed: np.ndarray,
    times: np.ndarray,
    shots: np.ndarray,
    regimes: np.ndarray,
    *,
    process_variance: float,
    tau_seconds: float,
    generator: np.random.Generator,
) -> np.ndarray:
    probability = np.empty(len(observed), dtype=np.float64)
    for regime in np.unique(regimes):
        indices = np.flatnonzero(regimes == regime)
        indices = indices[np.argsort(times[indices])]
        level = float(np.mean(observed[indices]))
        process = np.empty(len(indices), dtype=np.float64)
        process[0] = generator.normal(0.0, np.sqrt(max(process_variance, 0.0)))
        for local_index in range(1, len(indices)):
            elapsed = max(float(times[indices[local_index]] - times[indices[local_index - 1]]), 0.0)
            coefficient = float(np.exp(-elapsed / max(tau_seconds, np.finfo(float).eps)))
            process[local_index] = (
                coefficient * process[local_index - 1]
                + np.sqrt(max(process_variance * (1.0 - coefficient**2), 0.0)) * generator.normal()
            )
        probability[indices] = np.clip(level + process, 1e-7, 1.0 - 1e-7)
    return generator.binomial(shots, probability) / shots


def parametric_ou_bootstrap(
    *,
    values: Sequence[float],
    times_seconds: Sequence[float],
    shots: Sequence[int] | int,
    regime_ids: Sequence[str | int] | None,
    burst_flags: Sequence[bool] | None,
    instrument_ids: Sequence[str | int] | None,
    lag_edges_seconds: Sequence[float],
    fit: OUFit,
    mean_probability: float,
    effective_shots_per_second: float,
    maximum_interval_seconds: float,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Two-layer parametric bootstrap: fitted OU process then binomial readout."""
    if resamples < 20 or not fit.ok or fit.process_variance is None or fit.tau_seconds is None:
        return {"available": False, "successful_resamples": 0, "requested_resamples": int(resamples)}
    observed, times, sample_size, regimes, bursts = _arrays(values, times_seconds, shots, regime_ids, burst_flags)
    generator = np.random.default_rng(seed)
    process_variances: list[float] = []
    taus: list[float] = []
    intervals: list[float] = []
    sf_samples: dict[tuple[float, float], list[float]] = {}
    for _ in range(resamples):
        simulated = _simulate_fitted_ou_observations(
            observed,
            times,
            sample_size,
            regimes,
            process_variance=float(fit.process_variance),
            tau_seconds=float(fit.tau_seconds),
            generator=generator,
        )
        rows = structure_function(
            simulated,
            times,
            sample_size,
            regimes,
            bursts,
            lag_edges_seconds,
            instrument_ids=instrument_ids,
            alpha=alpha,
        )
        for row in rows:
            key = (float(row["lag_lo_seconds"]), float(row["lag_hi_seconds"]))
            sf_samples.setdefault(key, []).append(float(row["sf_debiased"]))
        bootstrap_fit = fit_ou_structure(rows, alpha=alpha)
        if not bootstrap_fit.ok or bootstrap_fit.process_variance is None or bootstrap_fit.tau_seconds is None:
            continue
        process_variances.append(float(bootstrap_fit.process_variance))
        taus.append(float(bootstrap_fit.tau_seconds))
        optimum = optimum_ou_interval(
            mean_probability=mean_probability,
            effective_shots_per_second=effective_shots_per_second,
            process_variance=float(bootstrap_fit.process_variance),
            tau_seconds=float(bootstrap_fit.tau_seconds),
            maximum_interval_seconds=maximum_interval_seconds,
        )
        intervals.append(float(optimum["interval_seconds"]))
    minimum_success = max(20, resamples // 2)
    if len(intervals) < minimum_success:
        return {
            "available": False,
            "successful_resamples": len(intervals),
            "requested_resamples": int(resamples),
            "minimum_successful_resamples": minimum_success,
        }
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    sf_intervals = {
        f"{lo:.17g}:{hi:.17g}": [float(value) for value in np.quantile(samples, quantiles)]
        for (lo, hi), samples in sf_samples.items()
        if len(samples) >= minimum_success
    }
    return {
        "available": True,
        "method": "fitted OU process layer plus binomial observation layer",
        "requested_resamples": int(resamples),
        "successful_resamples": len(intervals),
        "seed": int(seed),
        "process_variance_interval": [float(value) for value in np.quantile(process_variances, quantiles)],
        "tau_seconds_interval": [float(value) for value in np.quantile(taus, quantiles)],
        "t_star_seconds_interval": [float(value) for value in np.quantile(intervals, quantiles)],
        "sf_intervals_by_edge": sf_intervals,
    }


def optimum_ou_interval(
    *,
    mean_probability: float,
    effective_shots_per_second: float,
    process_variance: float,
    tau_seconds: float,
    maximum_interval_seconds: float,
    minimum_interval_seconds: float | None = None,
) -> dict[str, float | bool]:
    lower = max(1.0 / effective_shots_per_second, float(minimum_interval_seconds or 0.0), np.finfo(float).eps)
    upper = float(maximum_interval_seconds)
    if upper <= lower:
        raise ValueError("maximum interval must exceed minimum interval")
    result = minimize_scalar(
        lambda value: float(ou_residual_variance(value, mean_probability, effective_shots_per_second, process_variance, tau_seconds)),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": max(lower * 1e-6, 1e-7)},
    )
    interval = float(result.x)
    tolerance = max((upper - lower) * 1e-4, 1e-7)
    return {
        "interval_seconds": interval,
        "minimum_residual_variance": float(result.fun),
        "interior": bool(result.success and interval > lower + tolerance and interval < upper - tolerance),
        "lower_bound_seconds": lower,
        "upper_bound_seconds": upper,
    }


def cadence_ratio_gate(fast_squared_error: Sequence[float], slow_squared_error: Sequence[float], *, alpha: float = 0.05) -> dict[str, float | bool]:
    """Paired delta-method CI for mean(fast loss)/mean(slow loss)."""
    fast = np.asarray(fast_squared_error, dtype=np.float64)
    slow = np.asarray(slow_squared_error, dtype=np.float64)
    if fast.shape != slow.shape or fast.ndim != 1 or len(fast) < 3:
        raise ValueError("paired cadence losses need equal one-dimensional samples")
    mean_fast = float(np.mean(fast))
    mean_slow = float(np.mean(slow))
    if mean_slow <= 0.0:
        return {"ratio": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "passed": False}
    ratio = mean_fast / mean_slow
    covariance = np.cov(np.column_stack([fast, slow]), rowvar=False, ddof=1) / len(fast)
    gradient = np.asarray([1.0 / mean_slow, -mean_fast / (mean_slow**2)])
    standard_error = float(np.sqrt(max(float(gradient @ covariance @ gradient), 0.0)))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return {
        "ratio": ratio,
        "ci_lower": ratio - z * standard_error,
        "ci_upper": ratio + z * standard_error,
        "passed": bool(ratio + z * standard_error < 1.0),
    }


def analyze_ou_sensing(
    *,
    values: Sequence[float],
    times_seconds: Sequence[float],
    shots: Sequence[int] | int,
    regime_ids: Sequence[str | int] | None,
    burst_flags: Sequence[bool] | None,
    instrument_ids: Sequence[str | int] | None = None,
    lag_edges_seconds: Sequence[float],
    effective_shots_per_second: float,
    maximum_interval_seconds: float,
    interface_floor_seconds: float,
    alpha: float = 0.05,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 20260804,
) -> dict[str, Any]:
    """Observed-data-only B-4 gate used by both hardware analysis and B1."""
    observed = np.asarray(values, dtype=np.float64)
    variance_gate = variance_component_gate(
        observed,
        shots,
        regime_ids,
        burst_flags,
        instrument_ids,
        alpha=alpha,
    )
    sf_rows = structure_function(
        observed,
        times_seconds,
        shots,
        regime_ids,
        burst_flags,
        lag_edges_seconds,
        instrument_ids=instrument_ids,
        alpha=alpha,
    )
    fit = fit_ou_structure(sf_rows, alpha=alpha) if variance_gate.passed else OUFit(False, None, None, None, None, None, None, len(sf_rows))
    result: dict[str, Any] = {
        "variance_gate": asdict(variance_gate),
        "structure_function": sf_rows,
        "ou_fit": asdict(fit),
        "interior_optimum_claim": False,
        "worth_sensing": False,
        "t_star_seconds": None,
        "t_star_ci_lower_seconds": None,
        "t_star_ci_upper_seconds": None,
        "minimum_residual_variance": None,
        "economic_separation": False,
        "economic_separation_margin": None,
        "worst_corner_residual_variance": None,
        "process_variance_ci_lower_for_economic_gate": None,
        "nonparametric_sensitivity": None,
        "parametric_bootstrap": {"available": False, "requested_resamples": int(bootstrap_resamples)},
    }
    if sf_rows:
        mean_probability = float(np.mean(observed))
        nonparametric_intrinsic = optimum_nonparametric_interval(
            rows=sf_rows,
            mean_probability=mean_probability,
            effective_shots_per_second=effective_shots_per_second,
            maximum_interval_seconds=maximum_interval_seconds,
        )
        nonparametric_constrained = optimum_nonparametric_interval(
            rows=sf_rows,
            mean_probability=mean_probability,
            effective_shots_per_second=effective_shots_per_second,
            maximum_interval_seconds=maximum_interval_seconds,
            minimum_interval_seconds=interface_floor_seconds,
        )
        result["nonparametric_sensitivity"] = {
            "role": "sensitivity_only",
            "monotone_curve": monotone_structure_curve(sf_rows),
            "intrinsic_optimum": nonparametric_intrinsic,
            "constrained_optimum": nonparametric_constrained,
        }
    if not variance_gate.passed or not fit.ok:
        return result
    assert fit.process_variance is not None and fit.tau_seconds is not None
    assert fit.process_variance_ci_lower is not None and fit.process_variance_ci_upper is not None
    assert fit.tau_ci_lower_seconds is not None and fit.tau_ci_upper_seconds is not None
    mean_probability = float(np.mean(observed))
    bootstrap = parametric_ou_bootstrap(
        values=observed,
        times_seconds=times_seconds,
        shots=shots,
        regime_ids=regime_ids,
        burst_flags=burst_flags,
        instrument_ids=instrument_ids,
        lag_edges_seconds=lag_edges_seconds,
        fit=fit,
        mean_probability=mean_probability,
        effective_shots_per_second=effective_shots_per_second,
        maximum_interval_seconds=maximum_interval_seconds,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        alpha=alpha,
    ) if bootstrap_resamples else result["parametric_bootstrap"]
    result["parametric_bootstrap"] = bootstrap
    process_variance_bounds = (fit.process_variance_ci_lower, fit.process_variance_ci_upper)
    tau_bounds = (fit.tau_ci_lower_seconds, fit.tau_ci_upper_seconds)
    if bool(bootstrap.get("available")):
        process_variance_bounds = tuple(float(value) for value in bootstrap["process_variance_interval"])
        tau_bounds = tuple(float(value) for value in bootstrap["tau_seconds_interval"])
        for row in sf_rows:
            key = f"{float(row['lag_lo_seconds']):.17g}:{float(row['lag_hi_seconds']):.17g}"
            if key in bootstrap["sf_intervals_by_edge"]:
                row["sf_bootstrap_ci_lower"], row["sf_bootstrap_ci_upper"] = bootstrap["sf_intervals_by_edge"][key]
        result["t_star_ci_lower_seconds"], result["t_star_ci_upper_seconds"] = bootstrap["t_star_seconds_interval"]
    optimum = optimum_ou_interval(
        mean_probability=mean_probability,
        effective_shots_per_second=effective_shots_per_second,
        process_variance=fit.process_variance,
        tau_seconds=fit.tau_seconds,
        maximum_interval_seconds=maximum_interval_seconds,
    )
    corner_intervals: list[float] = []
    corner_residuals: list[float] = []
    for process_variance in process_variance_bounds:
        for tau_seconds in tau_bounds:
            intrinsic_corner = optimum_ou_interval(
                mean_probability=mean_probability,
                effective_shots_per_second=effective_shots_per_second,
                process_variance=max(process_variance, np.finfo(float).tiny),
                tau_seconds=max(tau_seconds, np.finfo(float).tiny),
                maximum_interval_seconds=maximum_interval_seconds,
            )
            constrained_corner = optimum_ou_interval(
                mean_probability=mean_probability,
                effective_shots_per_second=effective_shots_per_second,
                process_variance=max(process_variance, np.finfo(float).tiny),
                tau_seconds=max(tau_seconds, np.finfo(float).tiny),
                maximum_interval_seconds=maximum_interval_seconds,
                minimum_interval_seconds=interface_floor_seconds,
            )
            corner_intervals.append(float(intrinsic_corner["interval_seconds"]))
            corner_residuals.append(float(constrained_corner["minimum_residual_variance"]))
    constrained = optimum_ou_interval(
        mean_probability=mean_probability,
        effective_shots_per_second=effective_shots_per_second,
        process_variance=fit.process_variance,
        tau_seconds=fit.tau_seconds,
        maximum_interval_seconds=maximum_interval_seconds,
        minimum_interval_seconds=interface_floor_seconds,
    )
    economic_separation = max(corner_residuals) < variance_gate.process_variance_ci_lower
    worst_corner_residual = max(corner_residuals)
    economic_margin = variance_gate.process_variance_ci_lower - worst_corner_residual
    result.update({
        "interior_optimum_claim": bool(optimum["interior"]),
        # T_floor may lie to the right of the intrinsic optimum.  That changes
        # the attainable cadence, not the go/no-go definition: sensing remains
        # worthwhile when the best attainable residual beats no sensing.
        "worth_sensing": bool(economic_separation),
        "t_star_seconds": float(optimum["interval_seconds"]),
        "t_star_ci_lower_seconds": float(bootstrap["t_star_seconds_interval"][0]) if bool(bootstrap.get("available")) else float(min(corner_intervals)),
        "t_star_ci_upper_seconds": float(bootstrap["t_star_seconds_interval"][1]) if bool(bootstrap.get("available")) else float(max(corner_intervals)),
        "minimum_residual_variance": float(constrained["minimum_residual_variance"]),
        "economic_separation": bool(economic_separation),
        "economic_separation_margin": float(economic_margin),
        "worst_corner_residual_variance": float(worst_corner_residual),
        "process_variance_ci_lower_for_economic_gate": float(variance_gate.process_variance_ci_lower),
    })
    return result
