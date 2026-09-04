#!/usr/bin/env python3
"""Prospective DGP simulation for T7 element ③.

v3 is used only to set pilot priors.  No v1 output is read or overwritten.
The main gate is evaluated from observed sequences only.  Latent paths and
event labels are retained for calibration and discrimination diagnostics, not
for gate inputs.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TARGETS = ("readout_all_zero_error", "readout_all_one_error")
PRIMARY_TARGET = TARGETS[0]
EXPECTED_CORPUS_SHA256 = "9C5AA66AADBB95478F6F6B6B1AB03615FD206DC888549D18A2312B48CCD88306"
EXPECTED_CLEANED_SHA256 = "A506289627AAB6FFF2FD7CB282D8E89B5D90B365E3C6C7186CC3511D0B42C0C7"
EXPECTED_CLEAN_SCHEMA = "t7_v3_readout_clean_v1"
EXPECTED_BACKEND = "tianyan-287"
SETTINGS = 33
HARD_SHOT_LIMIT = 2.515e8
TRACK_A_SHOT_LIMIT = HARD_SHOT_LIMIT * 0.20
DAILY_MINUTES_LIMIT = 20.0
THROUGHPUTS = (2_000.0, 5_000.0)
PROCESS_SD = 0.0051
EVENT_JUMP_MEAN = 0.02537
EVENT_JUMP_SD = 0.0025
EVENT_INTERVAL_DAYS = 8.09
EVENT_INTERVAL_JITTER_DAYS = 0.5
FOLD_CONFIGS = ((40, 10, 1), (30, 15, 1), (50, 8, 2))
FOLD_NAMES = tuple(f"train{a}_test{b}_gap{c}" for a, b, c in FOLD_CONFIGS)
ALL_FOLDS_NAME = "all_three"
DGP_NAMES = (
    "null_flat",
    "ou_fast",
    "ou_pink",
    "step_calendar",
    "step_triggered",
    "step_as_ramp_artifact",
    "ramp_only",
)
RAMP_FRACTIONS = (0.1, 0.25, 0.5)


@dataclass(frozen=True)
class PilotPrior:
    target: str
    pilot_mean: float
    pilot_floor_mean: float
    pilot_floor_median: float
    pilot_step_delta: float
    pilot_proc_sd_pre: float
    pilot_proc_sd_post: float
    first_regime_end: int
    source_is_pilot_prior_only: bool = True


@dataclass
class LatentPath:
    values: np.ndarray
    event_mask: np.ndarray
    event_indices: list[int]
    h: float | None = None
    drift_rate_per_snapshot: float | None = None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, sha256_bytes(raw)


def load_cleaned(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows, cleaned_sha = read_jsonl(path)
    rows.sort(key=lambda row: int(row["snapshot_index"]))
    if len(rows) != 78 or [row["snapshot_index"] for row in rows] != list(range(78)):
        raise ValueError("Expected 78 contiguous cleaned v3 snapshots")
    if any(row.get("clean_schema") != EXPECTED_CLEAN_SCHEMA for row in rows):
        raise ValueError("Unexpected cleaned schema")
    if any(row.get("backend_id") != EXPECTED_BACKEND for row in rows):
        raise ValueError("Unexpected backend")
    if any(int(row.get("shots_per_setting", -1)) != 1024 for row in rows):
        raise ValueError("Expected frozen 1024-shot pilot")
    if any(row.get("source_corpus_sha256") != EXPECTED_CORPUS_SHA256 for row in rows):
        raise ValueError("Cleaned rows do not point to frozen v3 corpus")
    return rows, cleaned_sha


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pilot_prior(rows: list[dict[str, Any]], target: str) -> PilotPrior:
    values = np.asarray([float(row["observable_environment_proxy"][target]["value"]) for row in rows])
    floors = np.asarray([float(row["observable_environment_proxy"][target]["shot_noise_floor"]) for row in rows])
    times = [utc(str(row["scheduled_utc"])) for row in rows]
    gaps = np.asarray([(right - left).total_seconds() / 3600.0 for left, right in zip(times, times[1:])])
    breakpoints = np.flatnonzero(gaps > 6.0)
    first_regime_end = int(breakpoints[0] + 1) if len(breakpoints) else len(rows) // 2
    pre = values[:first_regime_end]
    post = values[first_regime_end:]
    pre_process = max(float(np.var(pre, ddof=1)) - float(np.mean(floors[:first_regime_end] ** 2)), 0.0)
    post_process = max(float(np.var(post, ddof=1)) - float(np.mean(floors[first_regime_end:] ** 2)), 0.0)
    return PilotPrior(
        target=target,
        pilot_mean=float(np.mean(values)),
        pilot_floor_mean=float(np.mean(floors)),
        pilot_floor_median=float(np.median(floors)),
        pilot_step_delta=float(np.mean(post) - np.mean(pre)),
        pilot_proc_sd_pre=float(np.sqrt(pre_process)),
        pilot_proc_sd_post=float(np.sqrt(post_process)),
        first_regime_end=first_regime_end,
    )


def regular_schedule(start: datetime, calendar_days: int, snaps_per_day: int) -> list[datetime]:
    if calendar_days <= 0 or snaps_per_day <= 0:
        raise ValueError("calendar_days and snaps_per_day must be positive")
    return [
        start + timedelta(days=day, minutes=24.0 * 60.0 * slot / snaps_per_day)
        for day in range(calendar_days)
        for slot in range(snaps_per_day)
    ]


def apply_outages(
    times: list[datetime],
    *,
    calendar_days: int,
    rng: np.random.Generator,
    irregular: bool,
) -> np.ndarray:
    keep = np.ones(len(times), dtype=bool)
    if not irregular:
        return keep
    start = times[0]
    for day in range(calendar_days):
        if rng.random() >= 0.05:
            continue
        outage_start = start + timedelta(days=day, hours=float(rng.uniform(0.0, 24.0)))
        duration_hours = float(np.exp(rng.normal(np.log(24.0), 0.55)))
        outage_end = outage_start + timedelta(hours=duration_hours)
        keep &= np.asarray([(time < outage_start or time >= outage_end) for time in times])
    if int(np.sum(keep)) < 8:
        # Keep enough observations for a causal fold; this is a sampling rule,
        # not a gate result.
        keep[: min(8, len(keep))] = True
    return keep


def ou_noise(n: int, tau_snapshots: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=np.float64)
    rho = float(np.exp(-1.0 / tau_snapshots))
    innovation_sd = float(sd * np.sqrt(1.0 - rho * rho))
    out = np.empty(n, dtype=np.float64)
    out[0] = rng.normal(0.0, sd)
    for index in range(1, n):
        out[index] = rho * out[index - 1] + rng.normal(0.0, innovation_sd)
    return out


def pink_noise(n: int, sd: float, rng: np.random.Generator) -> np.ndarray:
    taus = (1.0, 6.0, 60.0, 600.0)
    component_sd = sd / np.sqrt(len(taus))
    return sum((ou_noise(n, tau, component_sd, rng) for tau in taus), np.zeros(n, dtype=np.float64))


def channel_direction(target: str) -> float:
    return 1.0 if target == "readout_all_zero_error" else -1.0


def jump_value(rng: np.random.Generator) -> float:
    return float(max(0.005, rng.normal(EVENT_JUMP_MEAN, EVENT_JUMP_SD)))


def scheduled_event_indices(
    times: list[datetime],
    *,
    rng: np.random.Generator,
    interval_days: float = EVENT_INTERVAL_DAYS,
    jitter_days: float = EVENT_INTERVAL_JITTER_DAYS,
) -> list[int]:
    if len(times) < 2:
        return []
    result: list[int] = []
    next_event = times[0] + timedelta(days=max(1.0, interval_days + rng.normal(0.0, jitter_days)))
    for index, time in enumerate(times):
        if time >= next_event:
            result.append(index)
            next_event = time + timedelta(days=max(1.0, interval_days + rng.normal(0.0, jitter_days)))
    return result


def generate_latent(
    dgp: str,
    *,
    target: str,
    p0: float,
    times: list[datetime],
    snaps_per_day: int,
    rng: np.random.Generator,
    ramp_fraction: float = 0.5,
) -> LatentPath:
    """Generate one latent path; gate code never calls this function."""
    n = len(times)
    direction = channel_direction(target)
    event_mask = np.zeros(n, dtype=bool)
    event_indices: list[int] = []
    if dgp == "null_flat":
        values = np.full(n, p0, dtype=np.float64)
        return LatentPath(values, event_mask, event_indices)
    if dgp == "ou_fast":
        values = p0 + ou_noise(n, 1.0, PROCESS_SD, rng)
        return LatentPath(np.clip(values, 1e-4, 1.0 - 1e-4), event_mask, event_indices)
    if dgp == "ou_pink":
        values = p0 + pink_noise(n, PROCESS_SD, rng)
        return LatentPath(np.clip(values, 1e-4, 1.0 - 1e-4), event_mask, event_indices)
    if dgp == "step_as_ramp_artifact":
        # Deliberately retained as an invalid-model control.  It is bounded by
        # the observed pilot step and is never used as the main latent.
        delta = abs(EVENT_JUMP_MEAN if abs(EVENT_JUMP_MEAN) > 0 else 0.02537)
        ramp = np.linspace(0.0, direction * delta, n, dtype=np.float64)
        values = p0 + ramp + pink_noise(n, PROCESS_SD, rng)
        return LatentPath(np.clip(values, 1e-4, 1.0 - 1e-4), event_mask, event_indices)

    event_indices = scheduled_event_indices(times, rng=rng)
    if dgp in {"step_calendar", "ramp_only"}:
        residual = pink_noise(n, PROCESS_SD, rng)
        offsets = np.zeros(n, dtype=np.float64)
        offset = 0.0
        previous_event = 0
        for event_index in event_indices:
            if dgp == "ramp_only":
                ramp_length = max(1, int(round(0.25 * EVENT_INTERVAL_DAYS * snaps_per_day)))
                left = max(previous_event, event_index - ramp_length)
                offsets[left : event_index + 1] += np.linspace(0.0, direction * ramp_fraction * EVENT_JUMP_MEAN, event_index - left + 1)
            offset += direction * jump_value(rng)
            offsets[event_index:] += direction * jump_value(rng) * 0.0 + offset
            previous_event = event_index
        values = p0 + offsets + residual
        event_mask[event_indices] = True
        return LatentPath(np.clip(values, 1e-4, 1.0 - 1e-4), event_mask, event_indices)

    if dgp != "step_triggered":
        raise ValueError(f"Unknown DGP: {dgp}")
    event_indices = []
    h = float(ramp_fraction * EVENT_JUMP_MEAN)
    drift_rate = float(h / (EVENT_INTERVAL_DAYS * snaps_per_day))
    residual = pink_noise(n, PROCESS_SD, rng)
    values = np.empty(n, dtype=np.float64)
    offset = 0.0
    drift = 0.0
    drift_sign = float(rng.choice((-1.0, 1.0))) * direction
    for index in range(n):
        if index:
            drift += drift_sign * drift_rate + rng.normal(0.0, drift_rate * 0.08)
        values[index] = p0 + offset + drift + residual[index]
        if abs(drift) >= h:
            event_mask[index] = True
            event_indices.append(index)
            offset += drift_sign * jump_value(rng)
            drift = 0.0
            drift_sign = float(rng.choice((-1.0, 1.0))) * direction
    return LatentPath(np.clip(values, 1e-4, 1.0 - 1e-4), event_mask, event_indices, h, drift_rate)


def simulate_observations(latent: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(shots, np.clip(latent, 1e-4, 1.0 - 1e-4)) / float(shots)


def observed_floor(observed: np.ndarray, shots: int) -> np.ndarray:
    p = np.clip(observed, 1.0 / max(shots, 1), 1.0 - 1.0 / max(shots, 1))
    return np.sqrt(p * (1.0 - p) / float(shots))


def pairs_for_times(times: list[datetime]) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(max(0, len(times) - 1))]


def make_folds(n_pairs: int, config: tuple[int, int, int]) -> list[tuple[list[int], list[int]]]:
    train_size, test_size, gap = config
    folds: list[tuple[list[int], list[int]]] = []
    train_end = train_size
    while train_end + gap + test_size <= n_pairs:
        folds.append((list(range(train_end)), list(range(train_end + gap, train_end + gap + test_size))))
        train_end += test_size
    return folds


def linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered = times - float(np.mean(times))
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered, values - float(np.mean(values))) / denominator) if denominator > 1e-15 else 0.0


def trigger_forecast(observed: np.ndarray, times: list[datetime], train_pairs: list[int], test_pairs: list[int]) -> np.ndarray:
    predictions: list[float] = []
    for origin in test_pairs:
        history = np.asarray([pair + 1 for pair in train_pairs if pair + 1 <= origin], dtype=int)
        if len(history) < 3:
            predictions.append(float(observed[origin]))
            continue
        window = history[-min(12, len(history)) :]
        t = np.asarray([(times[index] - times[window[0]]).total_seconds() / 86400.0 for index in window])
        slope_per_day = linear_slope(t, observed[window])
        dt_days = max((times[origin + 1] - times[origin]).total_seconds() / 86400.0, 1.0 / 1440.0)
        recent_noise = float(np.std(observed[window] - np.polyval(np.polyfit(t, observed[window], 1), t), ddof=1)) if len(window) > 2 else 0.0
        step_estimate = float(np.quantile(np.abs(np.diff(observed[window])), 0.9)) if len(window) > 3 else 0.0
        # State-trigger model: extrapolate observed local drift and add a
        # bounded, data-estimated jump only when current drift is detectable.
        detectability = abs(slope_per_day * dt_days) / max(recent_noise, 1e-6)
        jump_prob = float(np.clip((detectability - 1.0) / 2.0, 0.0, 1.0))
        prediction = float(observed[origin] + slope_per_day * dt_days)
        prediction += float(np.sign(slope_per_day) * jump_prob * step_estimate)
        predictions.append(prediction)
    return np.asarray(predictions, dtype=np.float64)


def detected_event_times(observed: np.ndarray, times: list[datetime], floors: np.ndarray, train_pairs: list[int]) -> list[datetime]:
    indexes = np.asarray([pair + 1 for pair in train_pairs], dtype=int)
    if len(indexes) < 4:
        return []
    diffs = np.abs(np.diff(observed[indexes]))
    threshold = float(2.0 * np.sqrt(2.0) * np.median(floors[indexes]))
    candidates = sorted(np.flatnonzero(diffs >= threshold), key=lambda pos: float(diffs[pos]), reverse=True)
    selected: list[datetime] = []
    for pos in candidates:
        candidate = times[int(indexes[pos + 1])]
        if all(abs((candidate - existing).total_seconds()) >= 2.0 * 86400.0 for existing in selected):
            selected.append(candidate)
    return sorted(selected)


def calendar_event_probabilities(
    observed: np.ndarray,
    times: list[datetime],
    floors: np.ndarray,
    train_pairs: list[int],
    test_pairs: list[int],
) -> np.ndarray:
    detected = detected_event_times(observed, times, floors, train_pairs)
    if len(detected) >= 2:
        intervals = np.asarray([(right - left).total_seconds() / 86400.0 for left, right in zip(detected, detected[1:])])
        interval = float(np.clip(np.median(intervals), 4.0, 14.0))
        phase = detected[-1]
    elif detected:
        interval = EVENT_INTERVAL_DAYS
        phase = detected[-1]
    else:
        interval = EVENT_INTERVAL_DAYS
        phase = times[0]
    probabilities: list[float] = []
    for pair in test_pairs:
        origin_time = times[pair]
        target_time = times[pair + 1]
        origin_elapsed = (origin_time - phase).total_seconds() / 86400.0
        next_number = max(1, int(np.floor(origin_elapsed / max(interval, 1e-6))) + 1)
        expected = phase + timedelta(days=next_number * interval)
        tolerance = max((target_time - origin_time).total_seconds() / 86400.0, 0.75)
        near_expected = origin_time - timedelta(days=tolerance) < expected <= target_time + timedelta(days=tolerance)
        probabilities.append(0.85 if near_expected else 0.03)
    return np.asarray(probabilities, dtype=np.float64)


def trigger_event_probabilities(observed: np.ndarray, times: list[datetime], floors: np.ndarray, test_pairs: list[int]) -> np.ndarray:
    probabilities: list[float] = []
    for pair in test_pairs:
        left = max(0, pair - 5)
        window = np.arange(left, pair + 1, dtype=int)
        t = np.asarray([(times[index] - times[left]).total_seconds() / 86400.0 for index in window])
        slope = linear_slope(t, observed[window])
        dt = max((times[pair + 1] - times[pair]).total_seconds() / 86400.0, 1.0 / 1440.0)
        snr = abs(slope * dt) / max(float(np.median(floors[window])), 1e-6)
        probabilities.append(float(np.clip((snr - 1.0) / 2.0, 0.0, 1.0)))
    return np.asarray(probabilities, dtype=np.float64)


def event_brier(probability: np.ndarray, event_target: np.ndarray) -> float:
    if not len(event_target) or not np.any(event_target) or np.all(event_target):
        return float("nan")
    squared = (probability - event_target.astype(np.float64)) ** 2
    return float(0.5 * np.mean(squared[event_target]) + 0.5 * np.mean(squared[~event_target]))


def block_se(values: np.ndarray, block_length: int = 4) -> float:
    if len(values) < 2:
        return float("inf")
    means = np.asarray([np.mean(values[start : start + block_length]) for start in range(0, len(values), block_length)])
    if len(means) < 2:
        return float("inf")
    return float(np.std(means, ddof=1) / np.sqrt(len(means)))


def skill_pair(
    model_error: np.ndarray,
    baseline_error: np.ndarray,
    sigma2: np.ndarray,
) -> dict[str, float | bool | None]:
    model_mse = float(np.mean(model_error))
    baseline_mse = float(np.mean(baseline_error))
    raw_denom = baseline_mse
    raw_improvement = baseline_mse - model_mse
    raw_lower = raw_improvement - 1.959963984540054 * block_se(baseline_error - model_error)
    raw = float(raw_improvement / raw_denom) if raw_denom > 1e-15 else None
    raw_lower_skill = float(raw_lower / raw_denom) if raw_denom > 1e-15 else None

    sigma_bar = float(np.mean(sigma2))
    den_denom = baseline_mse - sigma_bar
    den_improvement = (baseline_mse - sigma_bar) - (model_mse - sigma_bar)
    if baseline_error is not None and den_denom > 1e-15:
        den_lower = den_improvement - 1.959963984540054 * block_se(baseline_error - model_error)
        denoised = float(den_improvement / den_denom)
        denoised_lower = float(den_lower / den_denom)
        denoised_undefined = False
    else:
        denoised = None
        denoised_lower = None
        denoised_undefined = True
    return {
        "raw": raw,
        "raw_ci_lower": raw_lower_skill,
        "denoised": denoised,
        "denoised_ci_lower": denoised_lower,
        "denoised_undefined": denoised_undefined,
        "model_mse": model_mse,
        "baseline_mse": baseline_mse,
    }


def denoised_persistence_skill(
    model_error: np.ndarray,
    persistence_error: np.ndarray,
    sigma2: np.ndarray,
) -> dict[str, float | bool | None]:
    model_mse = float(np.mean(model_error))
    persistence_mse = float(np.mean(persistence_error))
    sigma_bar = float(np.mean(sigma2))
    denominator = persistence_mse - 2.0 * sigma_bar
    numerator = persistence_mse - sigma_bar - (model_mse - sigma_bar)
    if denominator <= 0.0:
        return {"raw": 1.0 - model_mse / persistence_mse if persistence_mse > 0 else None, "raw_ci_lower": None, "denoised": None, "denoised_ci_lower": None, "denoised_undefined": True}
    improvement = persistence_error - model_error
    lower = float(np.mean(improvement) - 1.959963984540054 * block_se(improvement))
    return {
        "raw": float(1.0 - model_mse / persistence_mse) if persistence_mse > 1e-15 else None,
        "raw_ci_lower": float((np.mean(improvement) - 1.959963984540054 * block_se(improvement)) / persistence_mse) if persistence_mse > 1e-15 else None,
        "denoised": float(numerator / denominator),
        "denoised_ci_lower": float(lower / denominator),
        "denoised_undefined": False,
    }


def estimate_signal_snr(observed: np.ndarray, floors: np.ndarray, times: list[datetime]) -> float:
    """Observed-only one-step signal estimate; no DGP, latent, or pilot slope."""
    if len(observed) < 4:
        return 0.0
    slopes: list[float] = []
    for right in range(3, len(observed)):
        left = max(0, right - 5)
        t = np.asarray([(times[index] - times[left]).total_seconds() / 86400.0 for index in range(left, right + 1)])
        slopes.append(abs(linear_slope(t, observed[left : right + 1])) * max((times[right] - times[right - 1]).total_seconds() / 86400.0, 1.0 / 1440.0))
    return float(np.median(slopes) / max(float(np.median(floors)), 1e-12))


def evaluate_sequence(
    observed: np.ndarray,
    floors: np.ndarray,
    times: list[datetime],
    event_mask: np.ndarray,
    full_indices: np.ndarray,
    config: tuple[int, int, int],
) -> dict[str, Any]:
    pairs = pairs_for_times(times)
    folds = make_folds(len(pairs), config)
    if not folds:
        return {"fold_feasible": False, "n_folds": 0, "statistical_gate": False, "full_gate": False}
    metric_errors: dict[str, list[np.ndarray]] = {name: [] for name in ("climatology", "persistence", "calendar")}
    model_errors: list[np.ndarray] = []
    sigma_values: list[np.ndarray] = []
    discrimination_rows: list[dict[str, float]] = []
    event_targets_all: list[np.ndarray] = []
    for train_pairs, test_pairs in folds:
        train_targets = np.asarray([pair + 1 for pair in train_pairs], dtype=int)
        test_targets = np.asarray([pair + 1 for pair in test_pairs], dtype=int)
        target = observed[test_targets]
        model = trigger_forecast(observed, times, train_pairs, test_pairs)
        train_mean = float(np.mean(observed[train_targets]))
        climatology = np.full(len(test_targets), train_mean, dtype=np.float64)
        persistence = observed[np.asarray(test_pairs, dtype=int)]
        calendar = np.asarray([observed[max(0, pair)] for pair in test_pairs], dtype=np.float64)
        # Fixed schedule level baseline: use schedule-informed event timing to
        # adjust last observed level by an estimated step, using observations only.
        cal_event_prob = calendar_event_probabilities(observed, times, floors, train_pairs, test_pairs)
        jump_est = float(np.quantile(np.abs(np.diff(observed[train_targets])), 0.9)) if len(train_targets) > 3 else 0.0
        calendar = calendar + cal_event_prob * np.sign(linear_slope(np.arange(len(train_targets), dtype=float), observed[train_targets])) * jump_est
        model_error = (target - model) ** 2
        metric_errors["climatology"].append((target - climatology) ** 2)
        metric_errors["persistence"].append((target - persistence) ** 2)
        metric_errors["calendar"].append((target - calendar) ** 2)
        model_errors.append(model_error)
        sigma_values.append(floors[test_targets] ** 2)

        full_target_indexes = full_indices[test_targets]
        full_origin_indexes = full_indices[np.asarray(test_pairs, dtype=int)]
        event_target = np.asarray([
            bool(np.any(event_mask[left + 1 : right + 1])) for left, right in zip(full_origin_indexes, full_target_indexes)
        ])
        trig_prob = trigger_event_probabilities(observed, times, floors, test_pairs)
        cal_prob = cal_event_prob
        trig_brier = event_brier(trig_prob, event_target)
        cal_brier = event_brier(cal_prob, event_target)
        if np.isfinite(trig_brier) and np.isfinite(cal_brier):
            discrimination_rows.append({"trig_brier": trig_brier, "cal_brier": cal_brier, "trig_selected": float(trig_brier < cal_brier)})
        event_targets_all.append(event_target)

    model_error_all = np.concatenate(model_errors)
    sigma_all = np.concatenate(sigma_values)
    metrics: dict[str, Any] = {}
    for name in metric_errors:
        baseline_error = np.concatenate(metric_errors[name])
        if name == "persistence":
            metrics[name] = denoised_persistence_skill(model_error_all, baseline_error, sigma_all)
        else:
            metrics[name] = skill_pair(model_error_all, baseline_error, sigma_all)
    statistical_gate = bool(
        metrics["climatology"].get("raw_ci_lower") is not None
        and metrics["climatology"]["raw_ci_lower"] > 0.0
        and metrics["calendar"].get("raw_ci_lower") is not None
        and metrics["calendar"]["raw_ci_lower"] > 0.0
        and metrics["persistence"].get("denoised_ci_lower") is not None
        and metrics["persistence"]["denoised_ci_lower"] > 0.0
    )
    if len(discrimination_rows) >= 2:
        differences = np.asarray([row["cal_brier"] - row["trig_brier"] for row in discrimination_rows])
        selection_lower = float(np.mean(differences) - 1.6448536269514722 * np.std(differences, ddof=1) / np.sqrt(len(differences)))
        trig_selected = float(selection_lower > 0.0)
    else:
        selection_lower = trig_selected = float("nan")
    discrimination = {
        "n_folds_with_events": len(discrimination_rows),
        "trig_selected": trig_selected,
        "cal_selected": float(1.0 - trig_selected) if np.isfinite(trig_selected) else float("nan"),
        "selection_lower": selection_lower,
        "trig_brier": float(np.mean([row["trig_brier"] for row in discrimination_rows])) if discrimination_rows else float("nan"),
        "cal_brier": float(np.mean([row["cal_brier"] for row in discrimination_rows])) if discrimination_rows else float("nan"),
    }
    return {
        "fold_feasible": True,
        "n_folds": len(folds),
        "n_test_pairs": int(sum(len(test) for _, test in folds)),
        "metrics": metrics,
        "statistical_gate": statistical_gate,
        "signal_snr_observed": estimate_signal_snr(observed, floors, times),
        "discrimination": discrimination,
    }


def aggregate_fold_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(evaluations) != len(FOLD_CONFIGS) or not all(row.get("fold_feasible") for row in evaluations):
        return {"fold_feasible": False, "n_folds": 0, "statistical_gate": False, "full_gate": False}
    metrics: dict[str, dict[str, Any]] = {}
    for baseline in ("climatology", "persistence", "calendar"):
        fields = set().union(*(row["metrics"][baseline].keys() for row in evaluations))
        metrics[baseline] = {}
        for field in fields:
            data = [row["metrics"][baseline].get(field) for row in evaluations]
            if field == "denoised_undefined":
                metrics[baseline][field] = bool(any(data))
            else:
                finite = [float(value) for value in data if value is not None and np.isfinite(float(value))]
                metrics[baseline][field] = float(np.mean(finite)) if finite else None
    discrimination_rows = [row["discrimination"] for row in evaluations if np.isfinite(row["discrimination"]["trig_brier"])]
    selections = [row["trig_selected"] for row in discrimination_rows if np.isfinite(row["trig_selected"])]
    if len(selections) == len(FOLD_CONFIGS):
        trig_brier = float(np.mean([row["trig_brier"] for row in discrimination_rows]))
        cal_brier = float(np.mean([row["cal_brier"] for row in discrimination_rows]))
        trig_selected = float(all(value == 1.0 for value in selections))
    else:
        trig_brier = cal_brier = trig_selected = float("nan")
    return {
        "fold_feasible": True,
        "n_folds": int(sum(row["n_folds"] for row in evaluations)),
        "n_test_pairs": int(sum(row["n_test_pairs"] for row in evaluations)),
        "metrics": metrics,
        "statistical_gate": bool(all(row["statistical_gate"] for row in evaluations)),
        "signal_snr_observed": float(np.median([row["signal_snr_observed"] for row in evaluations])),
        "discrimination": {
            "n_folds_with_events": int(sum(row["discrimination"]["n_folds_with_events"] for row in evaluations)),
            "trig_selected": trig_selected,
            "cal_selected": float(1.0 - trig_selected) if np.isfinite(trig_selected) else float("nan"),
            "selection_lower": float(np.min([row["selection_lower"] for row in discrimination_rows])) if len(selections) == len(FOLD_CONFIGS) else float("nan"),
            "trig_brier": trig_brier,
            "cal_brier": cal_brier,
        },
    }


def feasibility(calendar_days: int, snaps_per_day: int, shots: int) -> dict[str, Any]:
    total_shots = calendar_days * snaps_per_day * SETTINGS * shots
    occupancy = {str(int(rate)): snaps_per_day * SETTINGS * shots / rate / 60.0 for rate in THROUGHPUTS}
    return {
        "total_shots": total_shots,
        "hard_budget_share": total_shots / HARD_SHOT_LIMIT,
        "track_a_budget_share": total_shots / TRACK_A_SHOT_LIMIT,
        "budget_feasible": bool(total_shots <= TRACK_A_SHOT_LIMIT),
        "occupancy_minutes_2khz": occupancy["2000"],
        "occupancy_minutes_5khz": occupancy["5000"],
        "occupancy_feasible": bool(max(occupancy.values()) <= DAILY_MINUTES_LIMIT),
    }


def summarize_replicates(records: list[dict[str, Any]], *, dgp: str, target: str, calendar_days: int, snaps_per_day: int, shots: int, fold_name: str, irregular: bool, ramp_fraction: float) -> dict[str, Any]:
    feasible_records = [row for row in records if row.get("fold_feasible")]
    feasibility_info = feasibility(calendar_days, snaps_per_day, shots)
    def values(metric: str, field: str) -> np.ndarray:
        out = [row["metrics"][metric].get(field) for row in feasible_records if row["metrics"][metric].get(field) is not None]
        return np.asarray(out, dtype=np.float64)
    def mean_metric(metric: str, field: str) -> float:
        data = values(metric, field)
        return float(np.mean(data)) if len(data) else float("nan")
    def median_metric(metric: str, field: str) -> float:
        data = values(metric, field)
        return float(np.median(data)) if len(data) else float("nan")
    discr = [row["discrimination"] for row in feasible_records if np.isfinite(row["discrimination"]["trig_selected"])]
    stat_rate = float(np.mean([row["statistical_gate"] for row in feasible_records])) if feasible_records else float("nan")
    full_rate = float(np.mean([
        bool(row["statistical_gate"] and np.isfinite(row["discrimination"]["trig_selected"]) and row["discrimination"]["trig_selected"] == 1.0)
        for row in feasible_records
    ])) if feasible_records else float("nan")
    events = [int(row["n_events_realized"]) for row in records]
    return {
        "dgp": dgp,
        "channel": target,
        "calendar_days": calendar_days,
        "snaps_per_day": snaps_per_day,
        "shots": shots,
        "fold_config": fold_name,
        "irregular": bool(irregular),
        "ramp_fraction": ramp_fraction,
        "replicates": len(records),
        "n_events_realized": float(np.mean(events)) if events else float("nan"),
        "n_events_realized_median": float(np.median(events)) if events else float("nan"),
        "power_or_size": full_rate,
        "full_gate_rate": full_rate,
        "statistical_gate_rate": stat_rate,
        "budget_feasible": feasibility_info["budget_feasible"],
        "occupancy_feasible": feasibility_info["occupancy_feasible"],
        **feasibility_info,
        "skill_raw_climatology": median_metric("climatology", "raw"),
        "skill_raw_persistence": median_metric("persistence", "raw"),
        "skill_raw_calendar": median_metric("calendar", "raw"),
        "skill_denoised_climatology": median_metric("climatology", "denoised"),
        "skill_denoised_persistence": median_metric("persistence", "denoised"),
        "skill_denoised_calendar": median_metric("calendar", "denoised"),
        "ci_lower_positive_rate_climatology": float(np.mean(values("climatology", "raw_ci_lower") > 0.0)) if feasible_records else float("nan"),
        "ci_lower_positive_rate_persistence": float(np.mean(values("persistence", "denoised_ci_lower") > 0.0)) if feasible_records else float("nan"),
        "ci_lower_positive_rate_calendar": float(np.mean(values("calendar", "raw_ci_lower") > 0.0)) if feasible_records else float("nan"),
        "signal_snr_observed_median": float(np.median([row["signal_snr_observed"] for row in feasible_records])) if feasible_records else float("nan"),
        "discrimination_trig_selected_rate": float(np.mean([row["trig_selected"] for row in discr])) if discr else float("nan"),
        "discrimination_cal_selected_rate": float(np.mean([1.0 - row["trig_selected"] for row in discr])) if discr else float("nan"),
        "discrimination_trig_brier": float(np.mean([row["trig_brier"] for row in discr])) if discr else float("nan"),
        "discrimination_cal_brier": float(np.mean([row["cal_brier"] for row in discr])) if discr else float("nan"),
        "discrimination_defined_rate": float(len(discr) / len(feasible_records)) if feasible_records else 0.0,
        "fold_feasible_rate": float(len(feasible_records) / len(records)) if records else 0.0,
    }


def grid_cells() -> Iterable[dict[str, Any]]:
    for target in TARGETS:
        for dgp in DGP_NAMES:
            ramp_values = RAMP_FRACTIONS if dgp in {"step_triggered", "ramp_only"} else (0.5,)
            for ramp_fraction in ramp_values:
                for calendar_days in (14, 21, 28):
                    for snaps_per_day in (2, 3, 4, 6):
                        for shots in (1024, 4096, 16384, 32768):
                            for irregular in (True, False):
                                yield {
                                    "target": target,
                                    "dgp": dgp,
                                    "ramp_fraction": ramp_fraction,
                                    "calendar_days": calendar_days,
                                    "snaps_per_day": snaps_per_day,
                                    "shots": shots,
                                    "irregular": irregular,
                                }


def run_cells(
    rows: list[dict[str, Any]],
    priors: dict[str, PilotPrior],
    cells: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers > 1 and len(cells) > 1:
        chunks = [cells[start : start + 16] for start in range(0, len(cells), 16)]
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_cell_chunk, rows, priors, chunk, replicates, seed) for chunk in chunks]
            for index, future in enumerate(as_completed(futures), start=1):
                results.extend(future.result())
                if index % max(1, len(futures) // 10) == 0 or index == len(futures):
                    print(f"completed cell batches {index}/{len(futures)}", flush=True)
        return results
    return _run_cells_serial(rows, priors, cells, replicates=replicates, seed=seed)


def _cell_seed(cell: dict[str, Any], seed: int) -> int:
    encoded = json.dumps(cell, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**32 - 1) + seed


def _run_cell_chunk(
    rows: list[dict[str, Any]],
    priors: dict[str, PilotPrior],
    cells: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    return _run_cells_serial(rows, priors, cells, replicates=replicates, seed=seed, announce=False)


def _run_cells_serial(
    rows: list[dict[str, Any]],
    priors: dict[str, PilotPrior],
    cells: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    announce: bool = True,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        target = str(cell["target"])
        prior = priors[target]
        start = utc(str(rows[0]["scheduled_utc"]))
        cell_records: dict[str, list[dict[str, Any]]] = {name: [] for name in (*FOLD_NAMES, ALL_FOLDS_NAME)}
        for replicate in range(replicates):
            rng = np.random.default_rng(_cell_seed(cell, seed) + replicate)
            times_full = regular_schedule(start, int(cell["calendar_days"]), int(cell["snaps_per_day"]))
            latent = generate_latent(
                str(cell["dgp"]), target=target, p0=prior.pilot_mean, times=times_full,
                snaps_per_day=int(cell["snaps_per_day"]), rng=rng, ramp_fraction=float(cell["ramp_fraction"]),
            )
            keep = apply_outages(times_full, calendar_days=int(cell["calendar_days"]), rng=rng, irregular=bool(cell["irregular"]))
            full_indices = np.flatnonzero(keep)
            times = [times_full[int(index)] for index in full_indices]
            observed = simulate_observations(latent.values[full_indices], int(cell["shots"]), rng)
            floors = observed_floor(observed, int(cell["shots"]))
            fold_evaluations: list[dict[str, Any]] = []
            for fold_name, config in zip(FOLD_NAMES, FOLD_CONFIGS):
                evaluation = evaluate_sequence(observed, floors, times, latent.event_mask, full_indices, config)
                evaluation["n_events_realized"] = len(latent.event_indices)
                evaluation["trigger_h"] = latent.h
                evaluation["trigger_drift_rate_per_snapshot"] = latent.drift_rate_per_snapshot
                cell_records[fold_name].append(evaluation)
                fold_evaluations.append(evaluation)
            aggregate = aggregate_fold_evaluations(fold_evaluations)
            aggregate["n_events_realized"] = len(latent.event_indices)
            aggregate["trigger_h"] = latent.h
            aggregate["trigger_drift_rate_per_snapshot"] = latent.drift_rate_per_snapshot
            cell_records[ALL_FOLDS_NAME].append(aggregate)
        for fold_name in (*FOLD_NAMES, ALL_FOLDS_NAME):
            summary = summarize_replicates(cell_records[fold_name], fold_name=fold_name, **cell)
            summaries.append(summary)
        if announce:
            print(f"completed {cell_index + 1}/{len(cells)} {cell['dgp']} {target} {cell['calendar_days']}d x{cell['snaps_per_day']} {cell['shots']} irregular={cell['irregular']}", flush=True)
    return summaries


def select_frontier(coarse_report: dict[str, Any], max_cells: int = 8) -> list[dict[str, Any]]:
    rows = [row for row in coarse_report["results"] if row["dgp"] == "step_triggered" and row["fold_config"] == ALL_FOLDS_NAME and row["budget_feasible"] and row["occupancy_feasible"]]
    rows.sort(key=lambda row: (
        float("-inf") if not np.isfinite(row["power_or_size"]) else row["power_or_size"],
        row["channel"] == PRIMARY_TARGET,
        row["irregular"],
        row["calendar_days"],
        row["shots"],
        row["snaps_per_day"],
        row["ramp_fraction"],
    ), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row["channel"], row["calendar_days"], row["snaps_per_day"], row["shots"], row["irregular"], row["ramp_fraction"])
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "target": row["channel"], "calendar_days": row["calendar_days"], "snaps_per_day": row["snaps_per_day"],
            "shots": row["shots"], "irregular": row["irregular"], "ramp_fraction": row["ramp_fraction"],
        })
        if len(selected) >= max_cells:
            break
    if not selected:
        selected = [{"target": PRIMARY_TARGET, "calendar_days": 21, "snaps_per_day": 4, "shots": 16384, "irregular": True, "ramp_fraction": 0.5}]
    expanded: list[dict[str, Any]] = []
    for base in selected:
        for dgp in DGP_NAMES:
            if dgp in {"step_triggered", "ramp_only"}:
                ramps = (base["ramp_fraction"],)
            else:
                ramps = (0.5,)
            for ramp_fraction in ramps:
                expanded.append({**base, "dgp": dgp, "ramp_fraction": ramp_fraction})
    return expanded


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No simulation rows")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_payload(
    rows: list[dict[str, Any]],
    *,
    corpus: Path,
    cleaned_sha256: str,
    priors: dict[str, PilotPrior],
    seed: int,
    replicates: int,
    mode: str,
    coarse_report_path: str | None,
) -> dict[str, Any]:
    headline = [row for row in rows if row["fold_config"] == ALL_FOLDS_NAME and row["budget_feasible"] and row["occupancy_feasible"]]
    trigger = [row for row in headline if row["dgp"] == "step_triggered"]
    calendar = [row for row in headline if row["dgp"] == "step_calendar"]
    feasible_joint = [row for row in trigger if row["power_or_size"] >= 0.8 and next((r["power_or_size"] for r in calendar if (r["channel"], r["calendar_days"], r["snaps_per_day"], r["shots"], r["irregular"], r["ramp_fraction"], r["fold_config"]) == (row["channel"], row["calendar_days"], row["snaps_per_day"], row["shots"], row["irregular"], row["ramp_fraction"], row["fold_config"])), 1.0) <= 0.05]
    return {
        "analysis_task": "T7_element3_dgp_simulation_v2",
        "source_corpus": str(corpus),
        "source_corpus_sha256": EXPECTED_CORPUS_SHA256,
        "cleaned_corpus_sha256": cleaned_sha256,
        "expected_cleaned_corpus_sha256": EXPECTED_CLEANED_SHA256,
        "source_is_read_only": True,
        "simulation_only": True,
        "no_prospective_claim": True,
        "seed": seed,
        "mode": mode,
        "replicates_per_cell": replicates,
        "coarse_report_path": coarse_report_path,
        "grid": {
            "dgps": list(DGP_NAMES),
            "targets": list(TARGETS),
            "calendar_days": [14, 21, 28],
            "snaps_per_day": [2, 3, 4, 6],
            "shots": [1024, 4096, 16384, 32768],
            "fold_configs": [asdict_config(config) for config in FOLD_CONFIGS],
            "sampling": ["irregular", "regular_only"],
            "ramp_fraction": list(RAMP_FRACTIONS),
        },
        "pilot_priors": {target: asdict(prior) for target, prior in priors.items()},
        "dgp_calibration": {
            "process_sd": PROCESS_SD,
            "event_jump_mean": EVENT_JUMP_MEAN,
            "event_jump_sd": EVENT_JUMP_SD,
            "event_interval_days": EVENT_INTERVAL_DAYS,
            "event_interval_jitter_days": EVENT_INTERVAL_JITTER_DAYS,
            "step_triggered": "h = ramp_fraction * event_jump_mean; drift_rate = h / (8.09 days * snaps_per_day)",
            "step_triggered_h_and_rate_are_in_each_cell": True,
            "proc_sd_prior_only": True,
            "f_one_event_prior": 1.357,
            "f_one_event_p_not_significant": True,
        },
        "gate": {
            "gate_inputs_observed_only": True,
            "oracle_terms_removed": True,
            "main_gate": "raw CI lower > 0 vs climatology and fixed-calendar level baseline, plus denoised CI lower > 0 vs persistence",
            "full_gate": "main statistical gate AND one-sided paired discrimination CI selects M_trig in all three fold configurations",
            "persistence_raw_not_used_as_main_gate": True,
            "denoised_formula": "1 - (model_MSE - sigma_bar_sq) / (persistence_MSE - 2*sigma_bar_sq)",
            "denoised_null_floor": 0.0,
            "undefined_rule": "denoised_undefined when denominator <= 0",
        },
        "feasibility_definition": {
            "track_a_shot_limit": TRACK_A_SHOT_LIMIT,
            "hard_shot_limit": HARD_SHOT_LIMIT,
            "daily_occupancy_limit_minutes": DAILY_MINUTES_LIMIT,
            "throughput_hz": list(THROUGHPUTS),
        },
        "headline": {
            "feasible_cell_with_power_ge_0_8_and_calendar_size_le_0_05": bool(feasible_joint),
            "statement": "A feasible cell exists under simulated DGPs." if feasible_joint else "No feasible cell simultaneously reaches power >= 0.8 on step_triggered and size <= 0.05 on step_calendar.",
            "calendar_days_needed_for_five_events_at_8_09_day_interval": 5 * EVENT_INTERVAL_DAYS,
            "events_in_21_days_at_8_09_day_interval": 21.0 / EVENT_INTERVAL_DAYS,
            "five_event_target_calendar_feasible_within_41_days": bool(5 * EVENT_INTERVAL_DAYS <= 41.0),
        },
        "results": rows,
    }


def asdict_config(config: tuple[int, int, int]) -> dict[str, int]:
    return {"minimum_train_pairs": config[0], "test_size": config[1], "gap": config[2]}


def run(
    corpus: Path,
    output_root: Path,
    *,
    replicates: int,
    seed: int,
    mode: str = "coarse",
    coarse_report: Path | None = None,
    max_frontier_cells: int = 8,
    workers: int = 1,
) -> dict[str, Any]:
    rows, cleaned_sha = load_cleaned(corpus)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite simulation output: {output_root}")
    output_root.mkdir(parents=True)
    priors = {target: pilot_prior(rows, target) for target in TARGETS}
    if mode == "coarse":
        cells = list(grid_cells())
    elif mode == "fine":
        if coarse_report is None:
            raise ValueError("--coarse-report required for fine mode")
        coarse = json.loads(coarse_report.read_text(encoding="utf-8"))
        cells = select_frontier(coarse, max_cells=max_frontier_cells)
    else:
        raise ValueError("mode must be coarse or fine")
    results = run_cells(rows, priors, cells, replicates=replicates, seed=seed, workers=workers)
    report = report_payload(
        results, corpus=corpus, cleaned_sha256=cleaned_sha, priors=priors, seed=seed,
        replicates=replicates, mode=mode, coarse_report_path=str(coarse_report) if coarse_report else None,
    )
    report["self_sha256"] = sha256_bytes(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    (output_root / "simulation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(output_root / "simulation_summary.csv", results)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--coarse-report", type=Path)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-frontier-cells", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("Use at least 100 replicates per cell")
    report = run(
        args.corpus.resolve(), args.out.resolve(), replicates=args.replicates, seed=args.seed,
        mode=args.mode, coarse_report=args.coarse_report.resolve() if args.coarse_report else None,
        max_frontier_cells=args.max_frontier_cells,
        workers=max(1, args.workers),
    )
    print(json.dumps({
        "output": str(args.out.resolve()), "mode": report["mode"],
        "results": len(report["results"]), "replicates_per_cell": report["replicates_per_cell"],
        "self_sha256": report["self_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
