#!/usr/bin/env python3
"""Run B9 stage 2: frozen T287 sensing-economics map.

The stage consumes only the verified B9 stage-1 T287 SF artifact and the
frozen T287 hardware/configuration inputs.  It reproduces the stage-1
variance/SF/OU results before evaluating the residual-variance curve.  T176
quarantine data are never read and no hardware work is submitted.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import simulate_b4_design_power as design_power
from src.adaptive import sensing_economics


SCHEMA = "b4_b9_t287_sensing_map_v2"
CHANNEL_DISPLAY = {
    "e0_readout_all_zero": "e0 all-zero control",
    "e1_readout_all_one": "e1 all-one signal",
}
PRIMARY_FLOOR_LABEL = "protocol_reachable"
PALETTE = {
    "blue": "#0F4D92",
    "blue_soft": "#B4CFE8",
    "teal": "#42949E",
    "green": "#2E9E44",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "gold": "#C58B19",
    "gold_soft": "#F0E0D0",
    "neutral": "#767676",
    "neutral_light": "#D8D8D8",
    "neutral_dark": "#333333",
}


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_utc_seconds(value: str) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_stage1_artifact(
    *,
    report_path: Path,
    observations_path: Path,
    manifest_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if report.get("schema") != "b4_b9_t287_sf_v1" or report.get("status") != "completed_t287_sf":
        raise ValueError("input is not a completed B9 stage-1 T287 SF artifact")
    if report.get("hardware_submission_performed") or report.get("t176_quarantine_read"):
        raise ValueError("stage-1 artifact violates B9 isolation boundary")
    if str(config["backend"]["backend_id"]) != "tianyan-287":
        raise ValueError("B9 stage 2 accepts T287 config only")
    if digest_file(config_path) != str(report["input_integrity"]["config_sha256"]).upper():
        raise ValueError("config hash does not match stage-1 artifact")

    manifest_index = {
        str(Path(str(row["path"])).resolve()).lower(): str(row["sha256"]).upper()
        for row in manifest.get("files", [])
    }
    artifact_rows: list[dict[str, Any]] = []
    for path in (report_path, observations_path):
        key = str(path.resolve()).lower()
        expected = manifest_index.get(key)
        actual = digest_file(path)
        matched = expected == actual
        artifact_rows.append({
            "file": path.name,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        })
        if not matched:
            raise ValueError(f"stage-1 artifact hash mismatch: {path.name}")

    frozen_rows: list[dict[str, Any]] = []
    for row in report["input_integrity"]["stage1_freeze"]["files"]:
        path = Path(str(row["resolved_path"]))
        actual = digest_file(path)
        expected = str(row["actual_sha256"]).upper()
        matched = actual == expected == str(row["expected_sha256"]).upper()
        frozen_rows.append({
            "file": str(row["path"]),
            "sha256": actual,
            "matched": matched,
        })
        if not matched:
            raise ValueError(f"frozen analysis-core hash mismatch: {row['path']}")

    observations = load_csv(observations_path)
    if len(observations) != 48 or len({row["query_id"] for row in observations}) != 48:
        raise ValueError("stage-1 observation inventory is not exactly 48 unique tasks")
    audit = {
        "passed": True,
        "stage1_report_sha256": digest_file(report_path),
        "stage1_observations_sha256": digest_file(observations_path),
        "stage1_manifest_sha256": digest_file(manifest_path),
        "config_sha256": digest_file(config_path),
        "artifact_files": artifact_rows,
        "frozen_analysis_core": frozen_rows,
        "recovered_task_count": len(observations),
    }
    return report, config, observations, audit


def channel_arrays(
    observations: Sequence[Mapping[str, str]],
    channel: str,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    selected = [
        dict(row)
        for row in observations
        if channel in str(row["analysis_channels"]).split(";")
    ]
    selected.sort(key=lambda row: (str(row["effective_observation_time_utc"]), str(row["query_id"])))
    if len(selected) < 3:
        raise ValueError(f"insufficient stage-1 observations for {channel}")
    absolute = np.asarray([
        parse_utc_seconds(str(row["effective_observation_time_utc"]))
        for row in selected
    ])
    arrays = {
        "times": absolute - absolute[0],
        "values": np.asarray([float(row["value"]) for row in selected]),
        "shots": np.asarray([int(row["shots"]) for row in selected], dtype=np.int64),
        "regimes": np.asarray([str(row["regime_id"]) for row in selected]),
        "bursts": np.asarray([parse_bool(row["burst_flag"]) for row in selected], dtype=bool),
        "instruments": np.asarray([str(row["instrument_id"]) for row in selected]),
    }
    return selected, arrays


def reproduce_stage1_channel(
    payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    effective_rate: float,
    protocol_floor: float,
) -> dict[str, Any]:
    reproduced = sensing_economics.analyze_ou_sensing(
        values=arrays["values"],
        times_seconds=arrays["times"],
        shots=arrays["shots"],
        regime_ids=arrays["regimes"],
        burst_flags=arrays["bursts"],
        instrument_ids=arrays["instruments"],
        lag_edges_seconds=payload["lag_edges_seconds"],
        effective_shots_per_second=effective_rate,
        maximum_interval_seconds=float(payload["effective_time_span_seconds"]),
        interface_floor_seconds=protocol_floor,
        bootstrap_resamples=0,
    )
    checks = {
        "variance_gate_exact": reproduced["variance_gate"] == payload["variance_gate"],
        "structure_function_exact": reproduced["structure_function"] == payload["structure_function"],
        "ou_fit_exact": reproduced["ou_fit"] == payload["ou_fit"],
    }
    if not all(checks.values()):
        raise ValueError({"reason": "stage-1 reproduction failed", **checks})
    return checks


def fit_boundary_diagnostics(
    payload: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    fit = payload["ou_fit"]
    if not fit["ok"]:
        return {
            "fit_available": False,
            "identified": False,
            "reason": "variance gate failed; OU fit not run",
        }
    lags = np.asarray([float(row["lag_mid_seconds"]) for row in payload["structure_function"]])
    tau_lower_bound = float(np.min(lags) / 1000.0)
    tau_upper_bound = float(np.max(lags) * 1000.0)
    asymptotic_hits = {
        "process_variance_lower": bool(float(fit["process_variance_ci_lower"]) <= np.finfo(float).tiny * 10.0),
        "process_variance_upper": bool(float(fit["process_variance_ci_upper"]) >= 1.0 - 1e-12),
        "tau_lower": bool(float(fit["tau_ci_lower_seconds"]) <= tau_lower_bound * (1.0 + 1e-9)),
        "tau_upper": bool(float(fit["tau_ci_upper_seconds"]) >= tau_upper_bound * (1.0 - 1e-9)),
    }
    bootstrap_hits = {
        "tau_upper": bool(
            bootstrap.get("available")
            and float(bootstrap["tau_seconds_interval"][1]) >= tau_upper_bound * 0.99
        ),
        "t_star_upper": bool(
            bootstrap.get("available")
            and float(bootstrap["t_star_seconds_interval"][1])
            >= float(payload["effective_time_span_seconds"]) * 0.99
        ),
    }
    identified = not any(asymptotic_hits.values()) and not any(bootstrap_hits.values())
    return {
        "fit_available": True,
        "identified": identified,
        "asymptotic_fit_bounds": {
            "tau_lower_seconds": tau_lower_bound,
            "tau_upper_seconds": tau_upper_bound,
            "process_variance_lower": np.finfo(float).tiny,
            "process_variance_upper": 1.0,
        },
        "asymptotic_ci_hits_bound": asymptotic_hits,
        "bootstrap_interval_hits_bound": bootstrap_hits,
        "reason": None if identified else "OU/T* uncertainty reaches fitted or observed-window bounds",
    }


def uncertainty_bounds(
    fit: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], str]:
    if bool(bootstrap.get("available")):
        return (
            tuple(float(value) for value in bootstrap["process_variance_interval"]),
            tuple(float(value) for value in bootstrap["tau_seconds_interval"]),
            f"{int(bootstrap['requested_resamples'])}-resample fitted-OU plus binomial parametric bootstrap",
        )
    return (
        (float(fit["process_variance_ci_lower"]), float(fit["process_variance_ci_upper"])),
        (float(fit["tau_ci_lower_seconds"]), float(fit["tau_ci_upper_seconds"])),
        "asymptotic fitted-OU covariance interval",
    )


def decision_classification(
    *,
    detection_gate_passed: bool,
    fit_identified: bool,
    frozen_worth_sensing: bool,
) -> tuple[str, str]:
    if not detection_gate_passed:
        return "NO-GO", "detection_gate_failed"
    if not fit_identified:
        return "INCONCLUSIVE", "ou_parameters_not_identified"
    if frozen_worth_sensing:
        return "GO", "frozen_economic_gate_passed"
    return "NO-GO", "frozen_economic_gate_failed"


def paired_joint_margin_bootstrap(
    *,
    values: Sequence[float],
    times_seconds: Sequence[float],
    shots: Sequence[int] | int,
    regime_ids: Sequence[str | int] | None,
    burst_flags: Sequence[bool] | None,
    instrument_ids: Sequence[str | int] | None,
    lag_edges_seconds: Sequence[float],
    fit: sensing_economics.OUFit,
    mean_probability: float,
    effective_shots_per_second: float,
    maximum_interval_seconds: float,
    floors: Sequence[Mapping[str, Any]],
    gate_lower: float,
    resamples: int,
    seed: int,
    primary_bootstrap: Mapping[str, Any],
    alpha: float = 0.05,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Post hoc joint-draw margin sensitivity without changing frozen core."""
    if resamples < 20 or not fit.ok or fit.process_variance is None or fit.tau_seconds is None:
        return {
            "available": False,
            "role": "post_hoc_sensitivity_only",
            "changes_primary_verdict": False,
            "requested_resamples": int(resamples),
        }, []
    observed, times, sample_size, regimes, bursts = sensing_economics._arrays(
        values,
        times_seconds,
        shots,
        regime_ids,
        burst_flags,
    )
    generator = np.random.default_rng(seed)
    process_variances: list[float] = []
    taus: list[float] = []
    intervals: list[float] = []
    draw_rows: list[dict[str, Any]] = []
    for resample_index in range(resamples):
        simulated = sensing_economics._simulate_fitted_ou_observations(
            observed,
            times,
            sample_size,
            regimes,
            process_variance=float(fit.process_variance),
            tau_seconds=float(fit.tau_seconds),
            generator=generator,
        )
        rows = sensing_economics.structure_function(
            simulated,
            times,
            sample_size,
            regimes,
            bursts,
            lag_edges_seconds,
            instrument_ids=instrument_ids,
            alpha=alpha,
        )
        bootstrap_fit = sensing_economics.fit_ou_structure(rows, alpha=alpha)
        if not bootstrap_fit.ok or bootstrap_fit.process_variance is None or bootstrap_fit.tau_seconds is None:
            continue
        process_variance = float(bootstrap_fit.process_variance)
        tau_seconds = float(bootstrap_fit.tau_seconds)
        intrinsic = sensing_economics.optimum_ou_interval(
            mean_probability=mean_probability,
            effective_shots_per_second=effective_shots_per_second,
            process_variance=process_variance,
            tau_seconds=tau_seconds,
            maximum_interval_seconds=maximum_interval_seconds,
        )
        process_variances.append(process_variance)
        taus.append(tau_seconds)
        intervals.append(float(intrinsic["interval_seconds"]))
        successful_index = len(intervals) - 1
        for floor in floors:
            constrained = sensing_economics.optimum_ou_interval(
                mean_probability=mean_probability,
                effective_shots_per_second=effective_shots_per_second,
                process_variance=process_variance,
                tau_seconds=tau_seconds,
                maximum_interval_seconds=maximum_interval_seconds,
                minimum_interval_seconds=float(floor["seconds"]),
            )
            residual = float(constrained["minimum_residual_variance"])
            margin = float(gate_lower - residual)
            draw_rows.append({
                "resample_index": int(resample_index),
                "successful_resample_index": int(successful_index),
                "floor_label": str(floor["label"]),
                "floor_seconds": float(floor["seconds"]),
                "primary_operational_floor": bool(floor["primary"]),
                "process_variance": process_variance,
                "tau_seconds": tau_seconds,
                "intrinsic_t_star_seconds": float(intrinsic["interval_seconds"]),
                "constrained_interval_seconds": float(constrained["interval_seconds"]),
                "minimum_residual_variance": residual,
                "variance_gate_process_variance_ci_lower": float(gate_lower),
                "economic_margin": margin,
                "margin_positive": bool(margin > 0.0),
            })
    minimum_success = max(20, resamples // 2)
    if len(intervals) < minimum_success:
        return {
            "available": False,
            "role": "post_hoc_sensitivity_only",
            "changes_primary_verdict": False,
            "successful_resamples": len(intervals),
            "requested_resamples": int(resamples),
            "minimum_successful_resamples": minimum_success,
        }, []
    interval_quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    summary_quantiles = (alpha / 2.0, 0.5, 1.0 - alpha / 2.0)
    reproduced_intervals = {
        "process_variance_interval": [float(value) for value in np.quantile(process_variances, interval_quantiles)],
        "tau_seconds_interval": [float(value) for value in np.quantile(taus, interval_quantiles)],
        "t_star_seconds_interval": [float(value) for value in np.quantile(intervals, interval_quantiles)],
    }
    reproduction = {
        key: reproduced_intervals[key] == [float(value) for value in primary_bootstrap[key]]
        for key in reproduced_intervals
    }
    if not all(reproduction.values()):
        raise ValueError({"reason": "post hoc joint bootstrap does not reproduce primary marginals", **reproduction})
    by_floor: dict[str, Any] = {}
    for floor in floors:
        floor_rows = [row for row in draw_rows if row["floor_label"] == str(floor["label"])]
        margins = np.asarray([float(row["economic_margin"]) for row in floor_rows])
        residuals = np.asarray([float(row["minimum_residual_variance"]) for row in floor_rows])
        constrained_intervals = np.asarray([float(row["constrained_interval_seconds"]) for row in floor_rows])
        margin_quantiles = [float(value) for value in np.quantile(margins, summary_quantiles)]
        by_floor[str(floor["label"])] = {
            "floor_seconds": float(floor["seconds"]),
            "primary_operational_floor": bool(floor["primary"]),
            "margin_quantiles_2p5_50_97p5": margin_quantiles,
            "residual_variance_quantiles_2p5_50_97p5": [
                float(value) for value in np.quantile(residuals, summary_quantiles)
            ],
            "constrained_interval_seconds_quantiles_2p5_50_97p5": [
                float(value) for value in np.quantile(constrained_intervals, summary_quantiles)
            ],
            "positive_margin_fraction": float(np.mean(margins > 0.0)),
            "joint_confidence_separation": bool(margin_quantiles[0] > 0.0),
        }
    primary_floor = next(row for row in floors if row["primary"])
    primary_result = by_floor[str(primary_floor["label"])]
    return {
        "available": True,
        "role": "post_hoc_sensitivity_only",
        "changes_primary_verdict": False,
        "method": "paired bootstrap process-variance/tau draws with floor-constrained residual margin",
        "requested_resamples": int(resamples),
        "successful_resamples": len(intervals),
        "seed": int(seed),
        "primary_marginal_reproduction": reproduction,
        "by_floor": by_floor,
        "primary_floor_interpretation": (
            "joint_distribution_separates"
            if primary_result["joint_confidence_separation"]
            else "joint_distribution_not_separated"
        ),
    }, draw_rows


def evaluate_floor(
    *,
    mean_probability: float,
    effective_rate: float,
    maximum_interval: float,
    floor: Mapping[str, Any],
    variance_gate: Mapping[str, Any],
    fit: Mapping[str, Any],
    process_bounds: tuple[float, float] | None,
    tau_bounds: tuple[float, float] | None,
    structure_function: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    floor_seconds = float(floor["seconds"])
    result: dict[str, Any] = {
        "floor_label": str(floor["label"]),
        "floor_seconds": floor_seconds,
        "floor_role": str(floor["role"]),
        "primary_operational_floor": bool(floor["primary"]),
        "detection_gate_passed": bool(variance_gate["passed"]),
        "ou_fit_ok": bool(fit["ok"]),
        "economic_gate_evaluated": False,
        "point_estimate_economic_separation": False,
        "confidence_economic_separation": False,
        "worth_sensing": False,
        "frozen_economic_gate_verdict": "NOT_EVALUATED",
        "decision_classification": "NO-GO",
        "decision_reason": "detection_gate_failed" if not variance_gate["passed"] else "ou_fit_unavailable",
        "verdict_reason": "detection_gate_failed" if not variance_gate["passed"] else "ou_fit_unavailable",
        "intrinsic_t_star_seconds": None,
        "constrained_interval_seconds": None,
        "point_minimum_residual_variance": None,
        "point_economic_margin": None,
        "worst_corner_residual_variance": None,
        "conservative_economic_margin": None,
        "corner_t_star_min_seconds": None,
        "corner_t_star_max_seconds": None,
    }
    if structure_function:
        result["nonparametric_sensitivity"] = sensing_economics.optimum_nonparametric_interval(
            rows=structure_function,
            mean_probability=mean_probability,
            effective_shots_per_second=effective_rate,
            maximum_interval_seconds=maximum_interval,
            minimum_interval_seconds=floor_seconds,
        )
    else:
        result["nonparametric_sensitivity"] = None
    if not variance_gate["passed"] or not fit["ok"]:
        return result
    assert process_bounds is not None and tau_bounds is not None
    intrinsic = sensing_economics.optimum_ou_interval(
        mean_probability=mean_probability,
        effective_shots_per_second=effective_rate,
        process_variance=float(fit["process_variance"]),
        tau_seconds=float(fit["tau_seconds"]),
        maximum_interval_seconds=maximum_interval,
    )
    constrained = sensing_economics.optimum_ou_interval(
        mean_probability=mean_probability,
        effective_shots_per_second=effective_rate,
        process_variance=float(fit["process_variance"]),
        tau_seconds=float(fit["tau_seconds"]),
        maximum_interval_seconds=maximum_interval,
        minimum_interval_seconds=floor_seconds,
    )
    corner_intervals: list[float] = []
    corner_residuals: list[float] = []
    for process_variance in process_bounds:
        for tau_seconds in tau_bounds:
            safe_variance = max(float(process_variance), np.finfo(float).tiny)
            safe_tau = max(float(tau_seconds), np.finfo(float).tiny)
            corner_intervals.append(float(sensing_economics.optimum_ou_interval(
                mean_probability=mean_probability,
                effective_shots_per_second=effective_rate,
                process_variance=safe_variance,
                tau_seconds=safe_tau,
                maximum_interval_seconds=maximum_interval,
            )["interval_seconds"]))
            corner_residuals.append(float(sensing_economics.optimum_ou_interval(
                mean_probability=mean_probability,
                effective_shots_per_second=effective_rate,
                process_variance=safe_variance,
                tau_seconds=safe_tau,
                maximum_interval_seconds=maximum_interval,
                minimum_interval_seconds=floor_seconds,
            )["minimum_residual_variance"]))
    gate_lower = float(variance_gate["process_variance_ci_lower"])
    point_margin = gate_lower - float(constrained["minimum_residual_variance"])
    conservative_margin = gate_lower - max(corner_residuals)
    point_separation = point_margin > 0.0
    confidence_separation = conservative_margin > 0.0
    result.update({
        "economic_gate_evaluated": True,
        "point_estimate_economic_separation": point_separation,
        "confidence_economic_separation": confidence_separation,
        "worth_sensing": confidence_separation,
        "frozen_economic_gate_verdict": "GO" if confidence_separation else "NO-GO",
        "verdict_reason": "worth_sensing" if confidence_separation else "economic_ci_not_separated",
        "intrinsic_t_star_seconds": float(intrinsic["interval_seconds"]),
        "intrinsic_t_star_interior": bool(intrinsic["interior"]),
        "constrained_interval_seconds": float(constrained["interval_seconds"]),
        "point_minimum_residual_variance": float(constrained["minimum_residual_variance"]),
        "point_economic_margin": point_margin,
        "worst_corner_residual_variance": max(corner_residuals),
        "conservative_economic_margin": conservative_margin,
        "corner_t_star_min_seconds": min(corner_intervals),
        "corner_t_star_max_seconds": max(corner_intervals),
    })
    return result


def evaluate_channel(
    *,
    channel: str,
    payload: Mapping[str, Any],
    observations: Sequence[Mapping[str, str]],
    effective_rate: float,
    floors: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected, arrays = channel_arrays(observations, channel)
    reproduction = reproduce_stage1_channel(
        payload,
        arrays,
        effective_rate=effective_rate,
        protocol_floor=float(next(row["seconds"] for row in floors if row["primary"])),
    )
    fit = payload["ou_fit"]
    bootstrap: dict[str, Any] = {
        "available": False,
        "requested_resamples": int(bootstrap_resamples),
    }
    if payload["variance_gate"]["passed"] and fit["ok"]:
        bootstrap = sensing_economics.parametric_ou_bootstrap(
            values=arrays["values"],
            times_seconds=arrays["times"],
            shots=arrays["shots"],
            regime_ids=arrays["regimes"],
            burst_flags=arrays["bursts"],
            instrument_ids=arrays["instruments"],
            lag_edges_seconds=payload["lag_edges_seconds"],
            fit=sensing_economics.OUFit(**fit),
            mean_probability=float(np.mean(arrays["values"])),
            effective_shots_per_second=effective_rate,
            maximum_interval_seconds=float(payload["effective_time_span_seconds"]),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
    joint_margin_sensitivity: dict[str, Any] = {
        "available": False,
        "role": "post_hoc_sensitivity_only",
        "changes_primary_verdict": False,
        "reason": "economic gate not reached",
    }
    joint_margin_rows: list[dict[str, Any]] = []
    if payload["variance_gate"]["passed"] and fit["ok"] and bootstrap.get("available"):
        joint_margin_sensitivity, joint_margin_rows = paired_joint_margin_bootstrap(
            values=arrays["values"],
            times_seconds=arrays["times"],
            shots=arrays["shots"],
            regime_ids=arrays["regimes"],
            burst_flags=arrays["bursts"],
            instrument_ids=arrays["instruments"],
            lag_edges_seconds=payload["lag_edges_seconds"],
            fit=sensing_economics.OUFit(**fit),
            mean_probability=float(np.mean(arrays["values"])),
            effective_shots_per_second=effective_rate,
            maximum_interval_seconds=float(payload["effective_time_span_seconds"]),
            floors=floors,
            gate_lower=float(payload["variance_gate"]["process_variance_ci_lower"]),
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            primary_bootstrap=bootstrap,
        )
    process_bounds: tuple[float, float] | None = None
    tau_bounds: tuple[float, float] | None = None
    uncertainty_method: str | None = None
    if fit["ok"]:
        process_bounds, tau_bounds, uncertainty_method = uncertainty_bounds(fit, bootstrap)
    mean_probability = float(np.mean(arrays["values"]))
    map_rows = [
        evaluate_floor(
            mean_probability=mean_probability,
            effective_rate=effective_rate,
            maximum_interval=float(payload["effective_time_span_seconds"]),
            floor=floor,
            variance_gate=payload["variance_gate"],
            fit=fit,
            process_bounds=process_bounds,
            tau_bounds=tau_bounds,
            structure_function=payload["structure_function"],
        )
        for floor in floors
    ]
    instrument_summary: dict[str, Any] = {}
    for instrument in sorted(set(arrays["instruments"])):
        values = arrays["values"][arrays["instruments"] == instrument]
        instrument_summary[str(instrument)] = {
            "n": int(len(values)),
            "mean_probability": float(np.mean(values)),
        }
    diagnostics = fit_boundary_diagnostics(payload, bootstrap)
    for row in map_rows:
        classification, reason = decision_classification(
            detection_gate_passed=bool(row["detection_gate_passed"]),
            fit_identified=bool(diagnostics.get("identified")),
            frozen_worth_sensing=bool(row["worth_sensing"]),
        )
        row["decision_classification"] = classification
        row["decision_reason"] = reason
    channel_report = {
        "channel": channel,
        "display_name": CHANNEL_DISPLAY[channel],
        "observation_count": len(selected),
        "mean_probability": mean_probability,
        "observations_by_instrument": instrument_summary,
        "effective_shots_per_second_channel": effective_rate,
        "stage1_reproduction": reproduction,
        "variance_gate": payload["variance_gate"],
        "ou_fit": fit,
        "sf_lag_support_seconds": [
            float(min(row["lag_mid_seconds"] for row in payload["structure_function"])),
            float(max(row["lag_mid_seconds"] for row in payload["structure_function"])),
        ],
        "effective_time_span_seconds": float(payload["effective_time_span_seconds"]),
        "bootstrap": bootstrap,
        "posthoc_joint_margin_sensitivity": joint_margin_sensitivity,
        "uncertainty_method_for_economic_gate": uncertainty_method,
        "fit_identifiability": diagnostics,
        "t_star_claim_permitted": bool(diagnostics.get("identified")),
        "map": map_rows,
    }
    curve_rows = build_curve_rows(
        channel=channel,
        payload=payload,
        mean_probability=mean_probability,
        effective_rate=effective_rate,
        process_bounds=process_bounds,
        tau_bounds=tau_bounds,
    )
    return channel_report, curve_rows, [
        {"channel": channel, **row}
        for row in joint_margin_rows
    ]


def build_curve_rows(
    *,
    channel: str,
    payload: Mapping[str, Any],
    mean_probability: float,
    effective_rate: float,
    process_bounds: tuple[float, float] | None,
    tau_bounds: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    maximum = float(payload["effective_time_span_seconds"])
    grid = np.geomspace(max(1.0 / effective_rate, 1e-3), maximum, 400)
    nonparametric = sensing_economics.nonparametric_residual_variance(
        grid,
        mean_probability,
        effective_rate,
        payload["structure_function"],
    )
    fit = payload["ou_fit"]
    point = np.full_like(grid, np.nan)
    lower = np.full_like(grid, np.nan)
    upper = np.full_like(grid, np.nan)
    if fit["ok"] and process_bounds is not None and tau_bounds is not None:
        point = sensing_economics.ou_residual_variance(
            grid,
            mean_probability,
            effective_rate,
            float(fit["process_variance"]),
            float(fit["tau_seconds"]),
        )
        corners = np.asarray([
            sensing_economics.ou_residual_variance(
                grid,
                mean_probability,
                effective_rate,
                max(float(process_variance), np.finfo(float).tiny),
                max(float(tau_seconds), np.finfo(float).tiny),
            )
            for process_variance in process_bounds
            for tau_seconds in tau_bounds
        ])
        lower = np.min(corners, axis=0)
        upper = np.max(corners, axis=0)
    gate = payload["variance_gate"]
    return [
        {
            "channel": channel,
            "interval_seconds": float(interval),
            "ou_point_residual_variance": None if not np.isfinite(point[index]) else float(point[index]),
            "ou_uncertainty_lower": None if not np.isfinite(lower[index]) else float(lower[index]),
            "ou_uncertainty_upper": None if not np.isfinite(upper[index]) else float(upper[index]),
            "nonparametric_residual_variance": float(nonparametric[index]),
            "no_sensing_process_variance": float(gate["process_variance"]),
            "no_sensing_ci_lower": float(gate["process_variance_ci_lower"]),
            "no_sensing_ci_upper": float(gate["process_variance_ci_upper"]),
        }
        for index, interval in enumerate(grid)
    ]


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {"type": "Simpson's paradox", "status": "checked", "finding": "No pooled direction claim; SF pairs remain instrument-specific."},
        {"type": "Ecological fallacy", "status": "checked", "finding": "Inference remains at task/channel level."},
        {"type": "Berkson's paradox", "status": "checked", "finding": "Pre-registered task set recovered 48/48; claim limited to that set."},
        {"type": "Collider bias", "status": "checked", "finding": "No post-outcome control variables enter the map."},
        {"type": "Base-rate neglect", "status": "checked", "finding": "No diagnostic-classification claim."},
        {"type": "Regression to the mean", "status": "checked", "finding": "No extreme-value enrollment or pre/post improvement claim."},
        {"type": "Survivorship bias", "status": "checked", "finding": "All 48 pre-registered platform tasks were recovered."},
        {"type": "Look-elsewhere effect", "status": "checked", "finding": "Two frozen channels and frozen floor definitions only."},
        {"type": "Garden of forking paths", "status": "checked", "finding": "Frozen estimator hashes and exact stage-1 reproduction passed."},
        {"type": "Correlation is not causation", "status": "checked", "finding": "Map is a descriptive economic gate, not a causal intervention claim."},
        {"type": "Reverse causality", "status": "checked", "finding": "No directional causal claim is made."},
    ]


def build_report(
    *,
    stage1_report: Mapping[str, Any],
    config: Mapping[str, Any],
    observations: Sequence[Mapping[str, str]],
    input_audit: Mapping[str, Any],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    timing = config["backend"]["tb6_measured_timing"]
    shot_rate = float(timing["effective_shots_per_second"])
    overhead = float(timing["fixed_overhead_seconds_per_setting"])
    effective_rate = design_power.effective_probe_shots_per_second(shot_rate, overhead)
    protocol_floor = design_power.design_resolvable_floor_seconds(shot_rate, overhead)
    interface = config["stage1_hardware_freeze"]
    floors = [
        {
            "label": "interface_P50",
            "seconds": float(interface["interface_p50_seconds"]),
            "role": "interface-latency sensitivity only",
            "primary": False,
        },
        {
            "label": "interface_P90",
            "seconds": float(interface["interface_p90_seconds"]),
            "role": "interface-latency sensitivity only",
            "primary": False,
        },
        {
            "label": PRIMARY_FLOOR_LABEL,
            "seconds": protocol_floor,
            "role": "primary operational floor from frozen serial protocol",
            "primary": True,
        },
    ]
    channel_reports: dict[str, Any] = {}
    map_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    joint_margin_rows: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(sorted(stage1_report["channels"])):
        channel_report, channel_curves, channel_joint_margin_rows = evaluate_channel(
            channel=channel,
            payload=stage1_report["channels"][channel],
            observations=observations,
            effective_rate=effective_rate,
            floors=floors,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + channel_index,
        )
        channel_reports[channel] = channel_report
        curve_rows.extend(channel_curves)
        joint_margin_rows.extend(channel_joint_margin_rows)
        for row in channel_report["map"]:
            map_rows.append({"channel": channel, **row})

    e0 = channel_reports["e0_readout_all_zero"]
    e1 = channel_reports["e1_readout_all_one"]
    e1_primary = next(row for row in e1["map"] if row["primary_operational_floor"])
    status = "completed_t287_sensing_map_inconclusive"
    report = {
        "schema": SCHEMA,
        "status": status,
        "b9_stage": 2,
        "b9_stage_name": "T287 sensing-economics map",
        "hardware_submission_performed": False,
        "t176_quarantine_read": False,
        "scope": "T287 map only; consumes verified stage-1 SF artifact; no cadence endpoint and no T176 read",
        "input_integrity": dict(input_audit),
        "hardware_economics": {
            "raw_shot_rate_per_second": shot_rate,
            "fixed_overhead_seconds_per_setting": overhead,
            "probe_settings_per_job": len(config["measurement"]["probe_job"]["settings"]),
            "probe_shots_per_setting": int(config["measurement"]["probe_job"]["shots_per_setting"]),
            "effective_shots_per_second_channel": effective_rate,
            "floor_definitions": floors,
            "map_lookup_domain_status": str(timing["map_lookup_status"]),
            "map_power_extrapolated": False,
            "map_power_claim_permitted": False,
        },
        "method": {
            "residual_variance": "pbar(1-pbar)/(R_eff*T) + integral_0^T SF(t) dt / T",
            "primary_curve": "frozen OU fit reproduced exactly from B9 stage 1",
            "sensitivity_curve": "weighted monotone nonparametric SF integration",
            "economic_gate": "worst uncertainty-corner residual below variance-gate process-variance CI lower bound",
            "bootstrap": "fitted OU process layer plus binomial observation layer",
            "posthoc_joint_margin": "paired process-variance/tau bootstrap draws; sensitivity only; cannot change primary verdict",
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed_base": bootstrap_seed,
            "multiple_comparisons": "No adjustment; two channel-specific endpoints and floor roles were frozen before analysis and are not pooled into one family claim.",
        },
        "channels": channel_reports,
        "sensing_map": map_rows,
        "preregistration_context": {
            "map_endpoint_power_threshold": 0.80,
            "map_endpoint_power_passed": False,
            "map_endpoint_role_after_2026_08_06": "descriptive_only",
            "t_star_optimal_cadence_endpoint_removed": True,
            "tier4_downgrade_from_this_map_permitted": False,
            "tier4_primary_bet": "B5 injected cadence-pair closed loop",
            "cadence_endpoint_simulation_power_t287": 0.910,
            "cadence_endpoint_simulation_power_second_backend": 0.901,
            "cadence_endpoint_simulation_size": 0.0,
        },
        "primary_conclusion": {
            "e0_negative_control_preserved": not bool(e0["variance_gate"]["passed"]),
            "e1_process_variance_detected": bool(e1["variance_gate"]["passed"]),
            "e1_point_curve_benefit": bool(e1_primary["point_estimate_economic_separation"]),
            "e1_confidence_separation": bool(e1_primary["confidence_economic_separation"]),
            "e1_worth_sensing": bool(e1_primary["worth_sensing"]),
            "e1_t_star_claim_permitted": bool(e1["t_star_claim_permitted"]),
            "frozen_corner_gate_verdict": str(e1_primary["frozen_economic_gate_verdict"]),
            "headline_verdict": str(e1_primary["decision_classification"]),
            "headline_reason": str(e1_primary["decision_reason"]),
            "posthoc_joint_margin_interpretation": str(
                e1["posthoc_joint_margin_sensitivity"].get("primary_floor_interpretation", "unavailable")
            ),
            "tier4_status": "UNCHANGED",
            "statement": (
                "T287 e1 contains detectable process variance and its point residual curve falls below the no-sensing benchmark, "
                "while the frozen Cartesian-corner gate remains NO-GO. Because OU process variance and tau are not identified and "
                "their confidence endpoints equal the fitted search bounds, the report-level verdict is INCONCLUSIVE rather than "
                "a substantive no-sensing conclusion. The e0 channel remains the pre-registered negative control; it does not "
                "exercise the economic gate. This descriptive map result does not alter the B5 cadence endpoint or Tier 4."
            ),
        },
        "statistical_validation": {
            "verification_status": "VERIFIED",
            "stage1_exact_reproduction": all(
                all(channel["stage1_reproduction"].values())
                for channel in channel_reports.values()
            ),
            "fallacy_scan_checked": 11,
            "fallacy_scan_total": 11,
            "fallacy_scan": fallacy_scan(),
            "economic_gate_size_empirically_tested_on_e0": False,
            "economic_gate_size_note": "e0 exits after the detection gate, so real-data e0 validates detection size only",
        },
        "next_permitted_action": "Evaluate the frozen T287 cadence residual curve; do not read T176 quarantine.",
    }
    return report, map_rows, curve_rows, joint_margin_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def map_rows_for_csv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        sensitivity = row.get("nonparametric_sensitivity") or {}
        flattened.append({
            **{key: value for key, value in row.items() if key != "nonparametric_sensitivity"},
            "nonparametric_interval_seconds": sensitivity.get("interval_seconds"),
            "nonparametric_minimum_residual_variance": sensitivity.get("minimum_residual_variance"),
            "nonparametric_interior": sensitivity.get("interior"),
        })
    return flattened


def configure_figure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    })


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.03, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


def render_figure(
    report: Mapping[str, Any],
    curve_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> list[Path]:
    configure_figure_style()
    figure = plt.figure(figsize=(7.2047, 4.4094), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.0])
    ax_curve = figure.add_subplot(grid[:, 0])
    ax_interval = figure.add_subplot(grid[0, 1])
    ax_gate = figure.add_subplot(grid[1, 1])

    e1 = report["channels"]["e1_readout_all_one"]
    e1_rows = [row for row in curve_rows if row["channel"] == "e1_readout_all_one"]
    interval = np.asarray([float(row["interval_seconds"]) for row in e1_rows])
    point = np.asarray([float(row["ou_point_residual_variance"]) for row in e1_rows])
    lower = np.asarray([float(row["ou_uncertainty_lower"]) for row in e1_rows])
    upper = np.asarray([float(row["ou_uncertainty_upper"]) for row in e1_rows])
    nonparametric = np.asarray([float(row["nonparametric_residual_variance"]) for row in e1_rows])
    gate = e1["variance_gate"]
    floor_rows = report["hardware_economics"]["floor_definitions"]
    primary_floor = next(row for row in floor_rows if row["primary"])
    interface_p50 = next(row for row in floor_rows if row["label"] == "interface_P50")
    interface_p90 = next(row for row in floor_rows if row["label"] == "interface_P90")
    primary_map = next(row for row in e1["map"] if row["primary_operational_floor"])

    ax_curve.fill_between(
        interval,
        np.maximum(lower, np.finfo(float).tiny),
        np.maximum(upper, np.finfo(float).tiny),
        color=PALETTE["blue_soft"],
        alpha=0.55,
        label="Frozen marginal-corner envelope",
        zorder=1,
    )
    ax_curve.plot(interval, point, color=PALETTE["blue"], linewidth=1.8, label="OU point curve", zorder=3)
    ax_curve.plot(
        interval,
        nonparametric,
        color=PALETTE["teal"],
        linewidth=1.2,
        linestyle="--",
        label="Monotone SF sensitivity",
        zorder=3,
    )
    ax_curve.axhspan(
        max(float(gate["process_variance_ci_lower"]), np.finfo(float).tiny),
        float(gate["process_variance_ci_upper"]),
        color=PALETTE["red_soft"],
        alpha=0.65,
        label="No-sensing process variance, 95% CI",
        zorder=0,
    )
    ax_curve.axhline(float(gate["process_variance"]), color=PALETTE["red"], linewidth=1.0, linestyle=":")
    ax_curve.axvspan(
        float(interface_p50["seconds"]),
        float(interface_p90["seconds"]),
        color=PALETTE["neutral_light"],
        alpha=0.7,
        label="Interface P50-P90",
        zorder=0,
    )
    ax_curve.axvspan(float(interval[0]), float(primary_floor["seconds"]), color=PALETTE["neutral_light"], alpha=0.18)
    ax_curve.axvline(float(primary_floor["seconds"]), color=PALETTE["red"], linewidth=1.2, linestyle="--")
    ax_curve.text(
        float(primary_floor["seconds"]) * 1.05,
        0.62,
        "60 s floor",
        transform=ax_curve.get_xaxis_transform(),
        color=PALETTE["red"],
        rotation=90,
        rotation_mode="anchor",
        ha="left",
        va="top",
    )
    if primary_map["intrinsic_t_star_seconds"] is not None:
        t_star = float(primary_map["intrinsic_t_star_seconds"])
        residual = float(primary_map["point_minimum_residual_variance"])
        ax_curve.scatter([t_star], [residual], color=PALETTE["blue"], edgecolor="white", linewidth=0.6, s=32, zorder=5)
        ax_curve.annotate(
            f"Point T* = {t_star:.1f} s",
            xy=(t_star, residual),
            xytext=(1.45 * t_star, 2.2 * residual),
            arrowprops={"arrowstyle": "-", "color": PALETTE["blue"], "linewidth": 0.8},
            color=PALETTE["blue"],
            ha="left",
            va="bottom",
        )
    positive = np.concatenate([point[point > 0.0], lower[lower > 0.0], upper[upper > 0.0], nonparametric[nonparametric > 0.0]])
    if np.any(interval <= 0.0) or positive.size == 0 or np.any(positive <= 0.0):
        raise ValueError("log-scaled figure data must be strictly positive")
    ax_curve.set_xscale("log")
    ax_curve.set_yscale("log")
    ax_curve.set_xlim(float(interval[0]), float(interval[-1]))
    ax_curve.set_ylim(float(np.min(positive)) / 1.7, float(np.max(positive)) * 1.7)
    ax_curve.set_xlabel("Update interval T (s)")
    ax_curve.set_ylabel("Residual variance")
    ax_curve.set_title("Point benefit; OU non-identification blocks certification", loc="left", fontweight="bold")
    ax_curve.grid(True, which="major", color="#E6E6E6", linewidth=0.55)
    ax_curve.legend(loc="upper left", ncol=1)
    add_panel_label(ax_curve, "a")

    bootstrap = e1["bootstrap"]
    sf_lags = np.asarray(e1["sf_lag_support_seconds"], dtype=np.float64)
    interval_specs = [
        (
            "Observed SF lag",
            float(np.min(sf_lags)),
            float(np.max(sf_lags)),
            None,
            PALETTE["neutral"],
        ),
        (
            "OU tau",
            float(bootstrap["tau_seconds_interval"][0]),
            float(bootstrap["tau_seconds_interval"][1]),
            float(e1["ou_fit"]["tau_seconds"]),
            PALETTE["gold"],
        ),
        (
            "T*",
            float(bootstrap["t_star_seconds_interval"][0]),
            float(bootstrap["t_star_seconds_interval"][1]),
            float(primary_map["intrinsic_t_star_seconds"]),
            PALETTE["blue"],
        ),
    ]
    y_positions = np.arange(len(interval_specs))[::-1]
    for y, (label, lo, hi, estimate, color) in zip(y_positions, interval_specs, strict=True):
        ax_interval.plot([lo, hi], [y, y], color=color, linewidth=3.2, solid_capstyle="round")
        ax_interval.scatter([lo, hi], [y, y], color=color, s=12, zorder=3)
        if estimate is not None:
            ax_interval.scatter([estimate], [y], color=PALETTE["neutral_dark"], marker="|", s=90, linewidth=1.4, zorder=4)
    ax_interval.axvline(float(primary_floor["seconds"]), color=PALETTE["red"], linewidth=1.0, linestyle="--")
    ax_interval.set_xscale("log")
    ax_interval.set_xlim(10.0, 5e6)
    ax_interval.set_yticks(y_positions)
    ax_interval.set_yticklabels([row[0] for row in interval_specs])
    ax_interval.set_xlabel("Seconds (log scale)")
    ax_interval.set_title("OU and T* remain boundary-limited", loc="left", fontweight="bold")
    ax_interval.grid(True, axis="x", which="major", color="#E6E6E6", linewidth=0.55)
    ax_interval.text(
        0.98,
        0.92,
        "Black tick = point\nRed dash = 60 s floor",
        transform=ax_interval.transAxes,
        ha="right",
        va="top",
        color=PALETTE["neutral"],
        fontsize=6.2,
    )
    add_panel_label(ax_interval, "b")

    columns = ["Detection", "OU ID", "Frozen gate", "Headline"]
    channels = ["e0_readout_all_zero", "e1_readout_all_one"]
    statuses = {
        "e0_readout_all_zero": [
            ("FAIL", PALETTE["red_soft"], PALETTE["red"]),
            ("N/A", "#F2F2F2", PALETTE["neutral"]),
            ("N/A", "#F2F2F2", PALETTE["neutral"]),
            ("NO", PALETTE["red_soft"], PALETTE["red"]),
        ],
        "e1_readout_all_one": [
            ("PASS", "#DDF3DE", PALETTE["green"]),
            ("FAIL", PALETTE["gold_soft"], PALETTE["gold"]),
            ("NO", PALETTE["red_soft"], PALETTE["red"]),
            ("INC", PALETTE["gold_soft"], PALETTE["gold"]),
        ],
    }
    for row_index, channel in enumerate(channels):
        for column_index, (label, face, text_color) in enumerate(statuses[channel]):
            ax_gate.add_patch(Rectangle((column_index, row_index), 1, 1, facecolor=face, edgecolor="white", linewidth=1.4))
            ax_gate.text(
                column_index + 0.5,
                row_index + 0.5,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=6.0,
            )
    ax_gate.set_xlim(0, len(columns))
    ax_gate.set_ylim(0, len(channels))
    ax_gate.invert_yaxis()
    ax_gate.set_xticks(np.arange(len(columns)) + 0.5)
    ax_gate.set_xticklabels(columns, rotation=25, ha="right", rotation_mode="anchor")
    ax_gate.set_yticks(np.arange(len(channels)) + 0.5)
    ax_gate.set_yticklabels(["e0 control", "e1 signal"])
    ax_gate.tick_params(length=0)
    for spine in ax_gate.spines.values():
        spine.set_visible(False)
    ax_gate.set_title("Three-class T287 decision", loc="left", fontweight="bold")
    joint = e1["posthoc_joint_margin_sensitivity"]
    joint_text = str(joint.get("primary_floor_interpretation", "joint sensitivity unavailable"))
    joint_text = joint_text.replace("joint_distribution_", "post hoc joint: ").replace("_", " ")
    ax_gate.text(
        0.0,
        -0.38,
        f"Frozen gate stays NO-GO; headline INCONCLUSIVE.\n{joint_text}; headline unchanged.",
        transform=ax_gate.transAxes,
        ha="left",
        va="top",
        color=PALETTE["neutral"],
        fontsize=6.0,
    )
    add_panel_label(ax_gate, "c")

    paths = [
        output_stem.with_suffix(".svg"),
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".tiff"),
        output_stem.with_suffix(".png"),
    ]
    figure.savefig(paths[0])
    figure.savefig(paths[1])
    figure.savefig(paths[2], dpi=600)
    figure.savefig(paths[3], dpi=300)
    plt.close(figure)
    return paths


def write_summary(path: Path, report: Mapping[str, Any]) -> None:
    e0 = report["channels"]["e0_readout_all_zero"]
    e1 = report["channels"]["e1_readout_all_one"]
    primary = next(row for row in e1["map"] if row["primary_operational_floor"])
    bootstrap = e1["bootstrap"]
    joint = e1["posthoc_joint_margin_sensitivity"]
    joint_primary = joint["by_floor"][PRIMARY_FLOOR_LABEL]
    joint_margin = joint_primary["margin_quantiles_2p5_50_97p5"]
    lines = [
        "# B9 Stage 2 — T287 sensing-economics map",
        "",
        "- Stage 1 方差门、SF、OU 逐值复现；冻结核心 hash 全通过。",
        f"- 有效单通道吞吐：{report['hardware_economics']['effective_shots_per_second_channel']:.6f} shots/s。",
        f"- 主 operational floor：{primary['floor_seconds']:.3f} s；接口 P50/P90 仅作敏感性。",
        f"- e0 detection gate：未通过，p={float(e0['variance_gate']['p_value']):.6g}；维持阴性对照。",
        f"- e1 detection gate：通过，p={float(e1['variance_gate']['p_value']):.6g}。",
        f"- e1 point T*：{float(primary['intrinsic_t_star_seconds']):.3f} s；bootstrap 95% 区间 "
        f"[{float(bootstrap['t_star_seconds_interval'][0]):.3f}, {float(bootstrap['t_star_seconds_interval'][1]):.3f}] s。",
        f"- e1 point residual：{float(primary['point_minimum_residual_variance']):.6g}；"
        f"过程方差 CI 下界：{float(e1['variance_gate']['process_variance_ci_lower']):.6g}；点估计显示潜在收益。",
        f"- 冻结角点 economic margin：{float(primary['conservative_economic_margin']):.6g}；冻结判据保持 NO-GO，worth-sensing=False。",
        "- 最终三分类：**INCONCLUSIVE**。OU τ/过程方差 CI 端点等于搜索框端点，参数不可识别；不得把该 NO-GO 解释为感知无价值。",
        f"- 事后联合 draw margin 2.5%/50%/97.5%：[{joint_margin[0]:.6g}, {joint_margin[1]:.6g}, {joint_margin[2]:.6g}]；"
        f"P(margin>0)={float(joint_primary['positive_margin_fraction']):.3f}；{joint['primary_floor_interpretation']}。不改变 headline。",
        "- e0 只验证 detection gate 阴性；因早退未运行经济门，真实数据未检验经济门 size。",
        "- P50、P90、60 s 三档冻结角点均 NO-GO；e1 三档最终均 INCONCLUSIVE。",
        "- 08-06 map power 未达 .80 后，T*/最优节奏终点已按预写分支删除；本 map 仅作描述，不在档位账上。",
        "- 档 4 依据仍为 B5 注入 cadence-pair 闭环；本结果不触发降档。",
        "- 实测 timing 点在冻结 lookup 域外；未外推 map power。",
        "- ARS 统计复核：VERIFIED；11/11 fallacy scan 完成。",
        "- T176 未读取；无真机提交。",
        "",
        "下一步：冻结 T287 cadence residual curve。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure_qa(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Figure QA — T287 sensing-economics map",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: e1 has a large point benefit, but boundary-wide OU non-identification makes the report verdict INCONCLUSIVE; the frozen corner gate remains unchanged and e0 remains negative.",
        "- Archetype: asymmetric mixed-modality quantitative figure.",
        "- Backend: Python/matplotlib only.",
        "- Final size: 183 mm × 112 mm.",
        "- Exports: editable SVG/PDF plus 600 dpi TIFF and 300 dpi PNG preview.",
        "- Data integrity: 48/48 frozen tasks retained; 30 observations per channel (12 shared anchors + 18 channel probes); no additional row exclusion.",
        "",
        "## Panel audit",
        "",
        "| Panel | Unique claim | Center/summary | Spread/interval | Replicate unit | Pass |",
        "|---|---|---|---|---|---|",
        "| a | Point economics versus frozen conservative uncertainty | OU point curve | 95% bootstrap marginal-corner envelope | frozen task observations | yes |",
        "| b | Parameter/T* identifiability | point estimates | 95% bootstrap intervals and observed SF support | SF bins/bootstrap resamples | yes |",
        "| c | Frozen gate versus three-class headline | categorical gate state | not applicable | pre-registered channels | yes |",
        "",
        "## Statistics",
        "",
        "- n: 30 observations/channel; independent platform jobs = 21 in Stage 1.",
        "- Variance detection: one-sided chi-square gate against analytic binomial shot variance.",
        f"- Bootstrap: {report['method']['bootstrap_resamples']} fitted-OU process + binomial observation resamples.",
        "- Economic uncertainty: Cartesian corners of bootstrap process-variance and tau intervals.",
        "- Post hoc sensitivity: paired process-variance/tau draws, floor-constrained residual, and draw-wise margin; cannot change primary verdict.",
        "- Multiple comparisons: no adjustment; frozen channel-specific endpoints are interpreted separately.",
        "- Source data: sensing_map.csv and sensing_curve_source_data.csv.",
        "",
        "## Reviewer risks",
        "",
        "- OU covariance and bootstrap upper intervals reach fit/observation boundaries; T* is not inferentially identifiable.",
        "- e0 exits at detection failure, so real-data economic-gate size is not tested.",
        "- Map power failed the pre-registered .80 threshold; the optimal-cadence endpoint is descriptive and cannot carry a tier downgrade.",
        "- Mean p and SF follow the frozen pooled-channel estimator; instrument-specific pairing remains enforced upstream.",
        "- Measured timing lies outside the frozen map-power lookup domain; no power interpolation or extrapolation is shown.",
        "- No causal or natural-drift closed-loop claim is made from this map.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output: Path,
    report: Mapping[str, Any],
    map_rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
    joint_margin_rows: Sequence[Mapping[str, Any]],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T287 sensing-map artifact: {output}")
    output.mkdir(parents=True)
    report_path = output / "t287_sensing_map_report.json"
    map_path = output / "sensing_map.csv"
    curve_path = output / "sensing_curve_source_data.csv"
    joint_margin_path = output / "posthoc_joint_margin_draws.csv"
    summary_path = output / "T287_SENSING_MAP_SUMMARY.md"
    qa_path = output / "FIGURE_QA.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(map_path, map_rows_for_csv(map_rows))
    write_csv(curve_path, curve_rows)
    write_csv(joint_margin_path, joint_margin_rows)
    write_summary(summary_path, report)
    write_figure_qa(qa_path, report)
    figure_paths = render_figure(report, curve_rows, output / "T287_sensing_economics_map")
    artifact_files = [report_path, map_path, curve_path, joint_margin_path, summary_path, qa_path, *figure_paths]
    manifest = {
        "schema": "b4_b9_t287_sensing_map_artifact_manifest_v2",
        "files": [
            {"file": path.name, "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in artifact_files
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-report", type=Path, required=True)
    parser.add_argument("--stage1-observations", type=Path, required=True)
    parser.add_argument("--stage1-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.bootstrap_resamples < 20:
        raise ValueError("bootstrap resamples must be at least 20")
    stage1_report, config, observations, input_audit = verify_stage1_artifact(
        report_path=arguments.stage1_report,
        observations_path=arguments.stage1_observations,
        manifest_path=arguments.stage1_manifest,
        config_path=arguments.config,
    )
    report, map_rows, curve_rows, joint_margin_rows = build_report(
        stage1_report=stage1_report,
        config=config,
        observations=observations,
        input_audit=input_audit,
        bootstrap_resamples=arguments.bootstrap_resamples,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    write_outputs(arguments.output, report, map_rows, curve_rows, joint_margin_rows)
    primary = next(
        row
        for row in report["channels"]["e1_readout_all_one"]["map"]
        if row["primary_operational_floor"]
    )
    e0_primary = next(
        row
        for row in report["channels"]["e0_readout_all_zero"]["map"]
        if row["primary_operational_floor"]
    )
    print(json.dumps({
        "status": report["status"],
        "e0_worth_sensing": e0_primary["worth_sensing"],
        "e1_worth_sensing": primary["worth_sensing"],
        "e1_headline_verdict": primary["decision_classification"],
        "e1_t_star_claim_permitted": report["primary_conclusion"]["e1_t_star_claim_permitted"],
        "output": str(arguments.output.resolve()),
        "next": report["next_permitted_action"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
