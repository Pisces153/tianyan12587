"""Leakage-audited distributional rolling-origin forecast for natural drift."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


MIN_TRAIN_SAMPLES = 40
TEST_SAMPLES = 10
MIN_FORWARD_FOLDS = 3
RIDGE = 1.0
STATE_NAMES = ("h1_effective", "h2_effective", "readout_all_zero_error", "readout_all_one_error")
JOINT_90_CHI_SQUARED = 7.779440339734858


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _self_hash(value: Mapping[str, Any]) -> str:
    copied = dict(value)
    copied.pop("self_sha256", None)
    canonical = json.dumps(copied, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(canonical).hexdigest().upper()


def _sign_report(report: dict[str, Any], *, feature_corpus_sha256: str) -> dict[str, Any]:
    report["analysis_task"] = "T7_forecast_head"
    report["feature_corpus_sha256"] = feature_corpus_sha256
    report["self_hash_scope"] = "canonical JSON excluding self_sha256"
    report["self_sha256"] = _self_hash(report)
    return report


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _brier(target: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((target - np.clip(probability, 0.0, 1.0)) ** 2))


def _bss(target: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> float | None:
    denominator = _brier(target, baseline)
    if denominator <= 1e-15:
        return None
    return float(1.0 - _brier(target, model) / denominator)


def _bootstrap_bss(target: np.ndarray, model: np.ndarray, baseline: np.ndarray, *, seed: int = 202609, samples: int = 1000) -> list[float | None]:
    if len(target) < 2:
        return [None, None]
    generator = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, len(target), len(target))
        score = _bss(target[indices], model[indices], baseline[indices])
        if score is not None and np.isfinite(score):
            scores.append(score)
    if not scores:
        return [None, None]
    return [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probability >= lower) & ((probability < upper) if index < bins - 1 else (probability <= upper))
        if not np.any(mask):
            continue
        confidence = float(probability[mask].mean())
        observed = float(target[mask].mean())
        fraction = float(mask.mean())
        ece += fraction * abs(confidence - observed)
        rows.append({"bin": index, "lower": lower, "upper": upper, "count": int(mask.sum()), "mean_prediction": confidence, "event_rate": observed})
    return float(ece), rows


def _state_matrix(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values: list[list[float]] = []
    for record in records:
        try:
            state = record["effective_field_state"]
            proxy = record["observable_environment_proxy"]
            row = [
                float(state["h1"]["value"]),
                float(state["h2"]["value"]),
                float(proxy["readout_all_zero_error"]["value"]),
                float(proxy["readout_all_one_error"]["value"]),
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("T7 requires T6 v2 effective-field and readout state") from error
        if not np.all(np.isfinite(row)):
            raise ValueError("T7 state contains a non-finite value")
        values.append(row)
    return np.asarray(values, dtype=np.float64)


def _calendar_matrix(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for record in records:
        try:
            timestamp = datetime.fromisoformat(str(record["scheduled_utc"]).replace("Z", "+00:00")).timestamp()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("T7 calendar baseline requires scheduled_utc") from error
        daily_phase = 2.0 * np.pi * ((timestamp % 86400.0) / 86400.0)
        rows.append([1.0, float(np.sin(daily_phase)), float(np.cos(daily_phase))])
    return np.asarray(rows, dtype=np.float64)


def _event(delta: np.ndarray, tolerance: float) -> float:
    return float(np.any(np.abs(delta) > tolerance))


def _make_samples(
    records: Sequence[Mapping[str, Any]], *, horizon: int, window: int, tolerance: float
) -> dict[str, Any]:
    state = _state_matrix(records)
    calendar = _calendar_matrix(records)
    features: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    events: list[float] = []
    persistence: list[float] = []
    sample_indices: list[int] = []
    prior_events = np.zeros(len(state), dtype=np.float64)
    if len(state) > 1:
        prior_events[1:] = [_event(state[index] - state[index - 1], tolerance) for index in range(1, len(state))]
    for current in range(window - 1, len(records) - horizon):
        history = state[current - window + 1:current + 1]
        delta = state[current + horizon] - state[current]
        features.append(history.ravel())
        histories.append(history)
        targets.append(delta)
        events.append(_event(delta, tolerance))
        persistence.append(float(prior_events[current]))
        sample_indices.append(current)
    return {
        "features": np.asarray(features, dtype=np.float64),
        "history": np.asarray(histories, dtype=np.float64),
        "target": np.asarray(targets, dtype=np.float64),
        "event": np.asarray(events, dtype=np.float64),
        "persistence_event": np.asarray(persistence, dtype=np.float64),
        "calendar": calendar[np.asarray(sample_indices, dtype=int)] if sample_indices else np.empty((0, 3)),
        "sample_indices": sample_indices,
    }


def _folds(sample_count: int, *, min_train: int, test_size: int, gap: int) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    train_end = min_train
    while train_end + gap + test_size <= sample_count:
        test_start = train_end + gap
        rows.append({"train": list(range(train_end)), "test": list(range(test_start, test_start + test_size))})
        train_end += test_size
    return rows


def _required_snapshots(*, horizon: int, window: int) -> int:
    required_samples = MIN_TRAIN_SAMPLES + horizon + MIN_FORWARD_FOLDS * TEST_SAMPLES
    return required_samples + window + horizon - 1


def _fit_ridge(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    penalty = np.diag(np.r_[0.0, np.full(design.shape[1] - 1, RIDGE)])
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return weights, mean, scale


def _predict_ridge(features: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean, scale = model
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    return design @ weights


def _regularize_covariance(residuals: np.ndarray) -> np.ndarray:
    if residuals.ndim != 2 or not len(residuals):
        raise ValueError("Cannot estimate a predictive covariance without residuals")
    covariance = residuals.T @ residuals / len(residuals)
    scale = max(float(np.max(np.diag(covariance))), 1e-8)
    return covariance + np.eye(covariance.shape[0]) * scale * 1e-6


def _gaussian_nll(target: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if target.ndim != 2 or mean.shape != target.shape:
        raise ValueError("Target and mean must be aligned two-dimensional arrays")
    if covariance.ndim == 2:
        covariance = np.broadcast_to(covariance, (len(target), *covariance.shape))
    if covariance.shape != (len(target), target.shape[1], target.shape[1]):
        raise ValueError("Predictive covariance must be one matrix or one matrix per sample")
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0.0) or np.any(~np.isfinite(logdet)):
        raise ValueError("Predictive covariance is not positive definite")
    residual = target - mean
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    return 0.5 * (target.shape[1] * np.log(2.0 * np.pi) + logdet + np.sum(residual * solved, axis=1))


def _joint_coverage(target: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim == 2:
        covariance = np.broadcast_to(covariance, (len(target), *covariance.shape))
    if covariance.shape != (len(target), target.shape[1], target.shape[1]):
        raise ValueError("Predictive covariance must be one matrix or one matrix per sample")
    residual = target - mean
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    squared_distance = np.sum(residual * solved, axis=1)
    return float(np.mean(squared_distance <= JOINT_90_CHI_SQUARED))


def _probe_label_disjoint(records: Sequence[Mapping[str, Any]]) -> bool:
    """Check T6 task split in exact corpus consumed by T7."""
    for record in records:
        proxy = record.get("proxy_task_labels")
        labels = record.get("label_task_labels")
        if not isinstance(proxy, Sequence) or isinstance(proxy, (str, bytes)):
            return False
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            return False
        if set(str(item) for item in proxy).intersection(str(item) for item in labels):
            return False
    return True


def _feature_lag_audit(records: Sequence[Mapping[str, Any]]) -> bool:
    expected = "label is contemporaneous and is only used at a future index by forecast evaluation"
    return bool(records) and all(
        isinstance(record.get("task_label"), Mapping)
        and record["task_label"].get("feature_lag_audit") == expected
        for record in records
    )


def _forecasting_skill_gate(gate: Mapping[str, bool], *, leakage_passed: bool) -> bool:
    """Evaluate registered gates without self-referencing final verdict."""
    return bool(
        leakage_passed
        and all(value for name, value in gate.items() if name != "forecasting_skill_claimed")
    )


def _event_probability(mean: np.ndarray, covariance: np.ndarray, *, tolerance: float, seed: int) -> float:
    generator = np.random.default_rng(seed)
    draws = generator.multivariate_normal(mean, covariance, size=4096, check_valid="raise")
    return float(np.mean(np.any(np.abs(draws) > tolerance, axis=1)))


def _probabilities(mean: np.ndarray, covariance: np.ndarray, *, tolerance: float, seed: int) -> np.ndarray:
    return np.asarray([
        _event_probability(row, covariance, tolerance=tolerance, seed=seed + index)
        for index, row in enumerate(mean)
    ])


def _linear_delta(history: np.ndarray, *, horizon: int) -> np.ndarray:
    if history.shape[1] < 2:
        return np.zeros((len(history), history.shape[2]), dtype=np.float64)
    return (history[:, -1, :] - history[:, 0, :]) * (horizon / (history.shape[1] - 1))


def _insufficient_report(
    *, corpus_rows: int, backend_id: str, reason: str, horizon: int, window: int,
    forward_samples: int, event_count: int, fold_count: int, minimum_forward_folds: bool,
    probe_label_disjoint: bool, feature_lag_audit_passed: bool, feature_corpus_sha256: str,
) -> dict[str, Any]:
    report = {
        "corpus": {"n_snapshots": corpus_rows, "backend_id": backend_id},
        "task": {"horizon_delta": horizon, "window_k": window, "state_names": list(STATE_NAMES)},
        "cv": {"scheme": "rolling_origin_forward_chain", "n_folds": fold_count, "min_required_folds": MIN_FORWARD_FOLDS, "gap": horizon, "shuffle_used": False},
        "feature_availability": {
            "state_names": list(STATE_NAMES), "n_forward_samples": forward_samples,
            "event_count": event_count, "non_event_count": forward_samples - event_count,
        },
        "gate": {
            "minimum_forward_folds": minimum_forward_folds, "bss_positive": False, "ci_lower_above_zero": False,
            "ece_ok": False, "distribution_metrics_complete": False,
            "l2_probe_label_disjoint": probe_label_disjoint,
            "l3_feature_lag_audit_passed": feature_lag_audit_passed,
            "forecasting_skill_claimed": False,
        },
        "verdict": "no forecasting skill at current corpus length or state availability",
        "if_no_skill": {
            "required_corpus_length_estimate": _required_snapshots(horizon=horizon, window=window),
            "reasoning": reason,
        },
    }
    return _sign_report(report, feature_corpus_sha256=feature_corpus_sha256)


def run_rolling_origin(
    corpus_path: Path,
    output_root: Path,
    *,
    backend_id: str,
    horizon: int = 1,
    window: int = 3,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    if horizon < 1 or window < 2 or tolerance <= 0.0:
        raise ValueError("horizon must be positive; window must be at least two; tolerance must be positive")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite T7 artifact: {output_root}")
    feature_corpus_sha256 = sha256(corpus_path.read_bytes()).hexdigest().upper()
    records = sorted((row for row in _read_json_lines(corpus_path) if row["backend_id"] == backend_id), key=lambda row: int(row["snapshot_index"]))
    specs = {row.get("feature_spec_sha256") for row in records}
    if len(specs) > 1:
        raise ValueError("Feature corpus mixes feature specifications")
    probe_label_disjoint = _probe_label_disjoint(records)
    feature_lag_audit_passed = _feature_lag_audit(records)
    output_root.mkdir(parents=True)
    try:
        samples = _make_samples(records, horizon=horizon, window=window, tolerance=tolerance)
    except ValueError as error:
        report = _insufficient_report(
            corpus_rows=len(records), backend_id=backend_id, reason=str(error), horizon=horizon, window=window,
            forward_samples=0, event_count=0, fold_count=0, minimum_forward_folds=False,
            probe_label_disjoint=probe_label_disjoint, feature_lag_audit_passed=feature_lag_audit_passed,
            feature_corpus_sha256=feature_corpus_sha256,
        )
        _write_json(output_root / "forecast_report.json", report)
        _write_json(output_root / "cv_folds.json", {"folds": [], "shuffle_used": False})
        _write_json(output_root / "leakage_tests.json", {"L1_passed": None, "L2_probe_label_disjoint": probe_label_disjoint, "L3_feature_lag_audit_passed": feature_lag_audit_passed})
        _write_json(output_root / "forecast_head_artifact.json", {"trained": False, "reason": str(error)})
        return report

    folds = _folds(len(samples["target"]), min_train=MIN_TRAIN_SAMPLES, test_size=TEST_SAMPLES, gap=horizon)
    event_count = int(samples["event"].sum())
    if len(folds) < MIN_FORWARD_FOLDS or np.unique(samples["event"]).size < 2:
        reason = "Corpus has fewer than the preregistered three forward-chain folds." if len(folds) < MIN_FORWARD_FOLDS else "Forward-chain samples have no event variation."
        report = _insufficient_report(
            corpus_rows=len(records), backend_id=backend_id, reason=reason, horizon=horizon, window=window,
            forward_samples=len(samples["target"]), event_count=event_count, fold_count=len(folds),
            minimum_forward_folds=len(folds) >= MIN_FORWARD_FOLDS,
            probe_label_disjoint=probe_label_disjoint, feature_lag_audit_passed=feature_lag_audit_passed,
            feature_corpus_sha256=feature_corpus_sha256,
        )
        _write_json(output_root / "forecast_report.json", report)
        _write_json(output_root / "cv_folds.json", {"folds": folds, "shuffle_used": False})
        _write_json(output_root / "leakage_tests.json", {"L1_passed": None, "L2_probe_label_disjoint": probe_label_disjoint, "L3_feature_lag_audit_passed": feature_lag_audit_passed})
        _write_json(output_root / "forecast_head_artifact.json", {"trained": False, "reason": reason})
        return report

    target_rows: list[np.ndarray] = []
    event_rows: list[float] = []
    model_probability: list[float] = []
    persistence_probability: list[float] = []
    calendar_probability: list[float] = []
    linear_probability: list[float] = []
    climatology_probability: list[float] = []
    permutation_probability: list[float] = []
    model_means: list[np.ndarray] = []
    calendar_means: list[np.ndarray] = []
    linear_means: list[np.ndarray] = []
    persistence_means: list[np.ndarray] = []
    model_covariances: list[np.ndarray] = []
    calendar_covariances: list[np.ndarray] = []
    linear_covariances: list[np.ndarray] = []
    persistence_covariances: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    last_model: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    generator = np.random.default_rng(202609)
    for fold_index, fold in enumerate(folds):
        train = np.asarray(fold["train"], dtype=int)
        test = np.asarray(fold["test"], dtype=int)
        target_train = samples["target"][train]
        model = _fit_ridge(samples["features"][train], target_train)
        calendar = _fit_ridge(samples["calendar"][train], target_train)
        last_model = model
        model_train = _predict_ridge(samples["features"][train], model)
        calendar_train = _predict_ridge(samples["calendar"][train], calendar)
        linear_train = _linear_delta(samples["history"][train], horizon=horizon)
        model_covariance = _regularize_covariance(target_train - model_train)
        calendar_covariance = _regularize_covariance(target_train - calendar_train)
        linear_covariance = _regularize_covariance(target_train - linear_train)
        persistence_covariance = _regularize_covariance(target_train)
        model_test = _predict_ridge(samples["features"][test], model)
        calendar_test = _predict_ridge(samples["calendar"][test], calendar)
        linear_test = _linear_delta(samples["history"][test], horizon=horizon)
        persistence_test = np.zeros_like(model_test)
        permuted = _fit_ridge(samples["features"][train], generator.permutation(target_train))
        permuted_test = _predict_ridge(samples["features"][test], permuted)
        permutation_covariance = _regularize_covariance(target_train - _predict_ridge(samples["features"][train], permuted))
        target_test = samples["target"][test]
        target_rows.extend(target_test)
        event_rows.extend(samples["event"][test])
        model_probability.extend(_probabilities(model_test, model_covariance, tolerance=tolerance, seed=202609 + 10000 * fold_index))
        calendar_probability.extend(_probabilities(calendar_test, calendar_covariance, tolerance=tolerance, seed=202609 + 20000 * fold_index))
        linear_probability.extend(_probabilities(linear_test, linear_covariance, tolerance=tolerance, seed=202609 + 30000 * fold_index))
        permutation_probability.extend(_probabilities(permuted_test, permutation_covariance, tolerance=tolerance, seed=202609 + 40000 * fold_index))
        persistence_probability.extend(samples["persistence_event"][test])
        climatology_probability.extend(np.full(len(test), samples["event"][train].mean()))
        model_means.extend(model_test)
        calendar_means.extend(calendar_test)
        linear_means.extend(linear_test)
        persistence_means.extend(persistence_test)
        model_covariances.extend([model_covariance] * len(test))
        calendar_covariances.extend([calendar_covariance] * len(test))
        linear_covariances.extend([linear_covariance] * len(test))
        persistence_covariances.extend([persistence_covariance] * len(test))
        fold_rows.append({
            "fold": fold_index, "train_sample_indices": fold["train"], "test_sample_indices": fold["test"],
            "train_snapshot_indices": [samples["sample_indices"][value] for value in fold["train"]],
            "test_snapshot_indices": [samples["sample_indices"][value] for value in fold["test"]],
        })

    target = np.asarray(target_rows)
    event = np.asarray(event_rows)
    model_p = np.asarray(model_probability)
    persistence_p = np.asarray(persistence_probability)
    calendar_p = np.asarray(calendar_probability)
    linear_p = np.asarray(linear_probability)
    climatology_p = np.asarray(climatology_probability)
    permutation_p = np.asarray(permutation_probability)
    model_mean = np.asarray(model_means)
    calendar_mean = np.asarray(calendar_means)
    linear_mean = np.asarray(linear_means)
    persistence_mean = np.asarray(persistence_means)
    # Keep fold-local covariance attached to each held-out row. Reusing fold 0
    # covariance would leak calibration assumptions across time and misstate NLL.
    model_covariance = np.asarray(model_covariances)
    calendar_covariance = np.asarray(calendar_covariances)
    linear_covariance = np.asarray(linear_covariances)
    persistence_covariance = np.asarray(persistence_covariances)
    bss_persist = _bss(event, model_p, persistence_p)
    bss_calendar = _bss(event, model_p, calendar_p)
    bss_linear = _bss(event, model_p, linear_p)
    bss_climatology = _bss(event, model_p, climatology_p)
    bss_ci = _bootstrap_bss(event, model_p, persistence_p)
    permutation_bss = _bss(event, permutation_p, persistence_p)
    leakage_passed = permutation_bss is not None and abs(permutation_bss) <= 0.10
    ece, reliability = _ece(event, model_p)
    distribution = {
        "target": "delta_state_over_horizon",
        "state_names": list(STATE_NAMES),
        "joint_coverage_nominal": 0.9,
        "joint_coverage": {
            "M1_state_ridge": _joint_coverage(target, model_mean, model_covariance),
            "B_persist": _joint_coverage(target, persistence_mean, persistence_covariance),
            "B_calendar_fixed": _joint_coverage(target, calendar_mean, calendar_covariance),
            "B_linear": _joint_coverage(target, linear_mean, linear_covariance),
        },
        "mean_nll": {
            "M1_state_ridge": float(_gaussian_nll(target, model_mean, model_covariance).mean()),
            "B_persist": float(_gaussian_nll(target, persistence_mean, persistence_covariance).mean()),
            "B_calendar_fixed": float(_gaussian_nll(target, calendar_mean, calendar_covariance).mean()),
            "B_linear": float(_gaussian_nll(target, linear_mean, linear_covariance).mean()),
        },
    }
    gate = {
        "minimum_forward_folds": len(folds) >= MIN_FORWARD_FOLDS,
        "bss_positive": bool(all(score is not None and score > 0.0 for score in (bss_persist, bss_calendar, bss_linear))),
        "ci_lower_above_zero": bool(bss_ci[0] is not None and bss_ci[0] > 0.0),
        "ece_ok": bool(ece < 0.1),
        "distribution_metrics_complete": bool(all(np.isfinite(value) for family in distribution.values() if isinstance(family, dict) for value in family.values())),
        "l2_probe_label_disjoint": probe_label_disjoint,
        "l3_feature_lag_audit_passed": feature_lag_audit_passed,
        "forecasting_skill_claimed": False,
    }
    gate["forecasting_skill_claimed"] = _forecasting_skill_gate(gate, leakage_passed=leakage_passed)
    report = {
        "corpus": {"n_snapshots": len(records), "backend_id": backend_id, "feature_spec_sha256": next(iter(specs)) if specs else None},
        "task": {"horizon_delta": horizon, "window_k": window, "tolerance_per_state": tolerance, "state_names": list(STATE_NAMES), "event_base_rate": float(event.mean())},
        "cv": {"scheme": "rolling_origin_forward_chain", "n_folds": len(folds), "min_required_folds": MIN_FORWARD_FOLDS, "gap": horizon, "shuffle_used": False},
        "models": {"M1_state_ridge_gaussian": {"bss_vs_persist": bss_persist, "bss_ci_95": bss_ci, "bss_vs_calendar_fixed": bss_calendar, "bss_vs_linear": bss_linear, "bss_vs_climatology": bss_climatology, "ece": ece}},
        "baselines": {
            "B_persist": {"brier": _brier(event, persistence_p)}, "B_calendar_fixed": {"brier": _brier(event, calendar_p)},
            "B_linear": {"brier": _brier(event, linear_p)}, "B_climatology": {"brier": _brier(event, climatology_p)},
        },
        "distribution": distribution,
        "leakage": {
            "L1_label_permutation_bss": permutation_bss,
            "L1_passed": leakage_passed,
            "L2_probe_label_disjoint": probe_label_disjoint,
            "L3_feature_lag_audit_passed": feature_lag_audit_passed,
        },
        "gate": gate,
        "verdict": "forecasting skill supported" if gate["forecasting_skill_claimed"] else "no forecasting skill at current corpus length or gate threshold",
        "if_no_skill": None if gate["forecasting_skill_claimed"] else {"required_corpus_length_estimate": _required_snapshots(horizon=horizon, window=window), "reasoning": "Continue fixed-protocol collection; do not relax forward-fold, baseline, CI, reliability, or leakage gates."},
    }
    _sign_report(report, feature_corpus_sha256=feature_corpus_sha256)
    _write_json(output_root / "forecast_report.json", report)
    _write_json(output_root / "cv_folds.json", {"folds": fold_rows, "shuffle_used": False})
    _write_json(output_root / "leakage_tests.json", report["leakage"])
    _write_json(output_root / "forecast_head_artifact.json", {
        "type": "ridge_gaussian_state_drift_forecast", "state_names": list(STATE_NAMES), "window": window,
        "horizon": horizon, "last_fold_coefficients": last_model[0].tolist() if last_model else [],
        "sha256": sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest(),
    })
    with (output_root / "baseline_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_event", "model", "persistence", "calendar_fixed", "linear", "climatology"])
        writer.writeheader()
        writer.writerows({"target_event": int(event[index]), "model": float(model_p[index]), "persistence": float(persistence_p[index]), "calendar_fixed": float(calendar_p[index]), "linear": float(linear_p[index]), "climatology": float(climatology_p[index])} for index in range(len(event)))
    with (output_root / "reliability_diagram.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin", "lower", "upper", "count", "mean_prediction", "event_rate"])
        writer.writeheader()
        writer.writerows(reliability)
    return report
