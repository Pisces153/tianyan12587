#!/usr/bin/env python3
"""B-4 SF, sensing-economics map, and frozen-v3 regression analysis."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_crossover_feasibility as crossover
from src.adaptive import sensing_economics


DEFAULT_CLEANED = Path(r"E:\TianYan\XA-202609\artifacts\analysis\T7_v3_readout_cleaned_20260803_v2\cleaned_features.jsonl")
DEFAULT_RAW_CORPUS = Path(r"E:\TianYan\XA-202609\artifacts\hardware\xa202609_tianyan_287_readout_natural_drift_v3")
DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_sensing_economics_v3_regression_20260804_v3")
CHANNELS = ("readout_all_zero_error", "readout_all_one_error")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cleaned_channels(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records = sorted(_read_jsonl(path), key=lambda row: (str(row["backend_id"]), int(row["snapshot_index"])))
    grouped: dict[tuple[str, str], dict[str, list[Any]]] = {}
    for record in records:
        backend = str(record["backend_id"])
        stamp = datetime.fromisoformat(str(record["scheduled_utc"]).replace("Z", "+00:00")).timestamp()
        regime = str(record["quality"]["calibration_regime_id"])
        for channel in CHANNELS:
            bucket = grouped.setdefault((backend, channel), {
                "times": [], "values": [], "shots": [], "regimes": [], "bursts": [], "snapshot_ids": [],
            })
            bucket["times"].append(stamp)
            bucket["values"].append(float(record["observable_environment_proxy"][channel]["value"]))
            bucket["shots"].append(int(record["shots_per_setting"]))
            bucket["regimes"].append(regime)
            bucket["bursts"].append(False)
            bucket["snapshot_ids"].append(str(record["snapshot_id"]))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, bucket in grouped.items():
        times = np.asarray(bucket["times"], dtype=np.float64)
        order = np.argsort(times)
        result[key] = {
            "times_seconds": times[order] - times[order][0],
            "values": np.asarray(bucket["values"], dtype=np.float64)[order],
            "shots": np.asarray(bucket["shots"], dtype=np.int64)[order],
            "regime_ids": np.asarray(bucket["regimes"])[order],
            "burst_flags": np.asarray(bucket["bursts"], dtype=bool)[order],
            "snapshot_ids": np.asarray(bucket["snapshot_ids"])[order],
            "timestamp_source": "scheduled_utc from frozen cleaned v3 corpus; v4 analysis must use recorded platform timestamps",
        }
    return result


def lag_edges(times_seconds: Sequence[float], bins: int = 12) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64)
    positive = np.diff(np.unique(times))
    positive = positive[positive > 0.0]
    if len(positive) == 0:
        raise ValueError("channel has no positive time gaps")
    lower = max(float(np.median(positive)) * 0.6, 1e-6)
    upper = float((times[-1] - times[0]) / 2.0)
    if upper <= lower:
        upper = lower * 2.0
    return np.geomspace(lower, upper, bins)


def fit_power_law(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | bool | None]:
    positive = [row for row in rows if float(row["sf_debiased"]) > 0.0]
    if len(positive) < 3:
        return {"ok": False, "n_used": len(positive), "alpha": None, "se_alpha": None}
    x = np.log(np.asarray([float(row["lag_mid_seconds"]) for row in positive]))
    y = np.log(np.asarray([float(row["sf_debiased"]) for row in positive]))
    alpha, intercept = np.polyfit(x, y, 1)
    residual = y - (alpha * x + intercept)
    degrees_of_freedom = max(len(positive) - 2, 1)
    residual_variance = float(residual @ residual) / degrees_of_freedom
    standard_error = float(np.sqrt(residual_variance / np.sum((x - x.mean()) ** 2)))
    return {
        "ok": True,
        "n_used": len(positive),
        "alpha": float(alpha),
        "se_alpha": standard_error,
        "coefficient": float(np.exp(intercept)),
    }


def _ablation_structure_function(
    *,
    values: Sequence[float],
    times_seconds: Sequence[float],
    shots: Sequence[int] | int,
    regimes: Sequence[str | int],
    lag_edges_seconds: Sequence[float],
    unbiased_shot_noise: bool,
    enforce_pairing_discipline: bool,
    minimum_pairs_per_bin: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    observed = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_seconds, dtype=np.float64)
    sample_size = np.broadcast_to(np.asarray(shots, dtype=np.int64), observed.shape)
    regime_ids = np.asarray(regimes)
    left, right = np.triu_indices(len(observed), k=1)
    total_pairs = len(left)
    if enforce_pairing_discipline:
        eligible = regime_ids[left] == regime_ids[right]
        left = left[eligible]
        right = right[eligible]
    lag = times[right] - times[left]
    squared = (observed[right] - observed[left]) ** 2
    if unbiased_shot_noise:
        variances = sensing_economics.binomial_variance_unbiased(observed, sample_size)
    else:
        variances = observed * (1.0 - observed) / sample_size
    floor = variances[left] + variances[right]
    edges = np.asarray(lag_edges_seconds, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (lag >= lower) & (lag < upper)
        pair_count = int(selected.sum())
        if pair_count < minimum_pairs_per_bin:
            continue
        rows.append({
            "lag_mid_seconds": float(np.mean(lag[selected])),
            "sf_debiased": float(np.mean(squared[selected] - floor[selected])),
            "n_pairs": pair_count,
        })
    return rows, {
        "all_candidate_pairs": total_pairs,
        "eligible_pairs": len(left),
        "excluded_cross_regime_pairs": total_pairs - len(left),
        "retained_bins": len(rows),
    }


def alpha_ablation(cleaned_path: Path, raw_corpus: Path) -> dict[str, Any]:
    """Turn on the three known path changes one at a time from the exact legacy path."""
    cleaned_records = {
        str(row["snapshot_id"]): row
        for row in _read_jsonl(cleaned_path)
        if str(row["backend_id"]) == "tianyan-287"
    }
    journal = _read_jsonl(raw_corpus / "snapshots.jsonl")
    submitted = {str(row["snapshot_id"]): row for row in journal if row["event"] == "submitted"}
    joined: list[dict[str, Any]] = []
    for collected in (row for row in journal if row["event"] == "collected"):
        snapshot_id = str(collected["snapshot_id"])
        cleaned = cleaned_records.get(snapshot_id)
        counts_path = raw_corpus / "raw" / f"{snapshot_id}_counts.npz"
        if cleaned is None or not counts_path.exists():
            continue
        blob = np.load(counts_path, allow_pickle=True)
        labels = [str(value) for value in blob["labels"]]
        label_index = labels.index("readout_all_one")
        shots = int(blob["shots"])
        value = 1.0 - float(blob["counts"][label_index, crossover.PROBE_SUCCESS_INDEX["readout_all_one"]]) / shots
        raw_stamp = submitted.get(snapshot_id, collected).get("wallclock_submit_utc", collected["recorded_at_utc"])
        joined.append({
            "snapshot_id": snapshot_id,
            "raw_time": datetime.fromisoformat(str(raw_stamp)).timestamp(),
            "scheduled_time": datetime.fromisoformat(str(cleaned["scheduled_utc"]).replace("Z", "+00:00")).timestamp(),
            "value": value,
            "cleaned_value": float(cleaned["observable_environment_proxy"]["readout_all_one_error"]["value"]),
            "shots": shots,
            "regime": str(cleaned["quality"]["calibration_regime_id"]),
        })
    if len(joined) < 3:
        raise ValueError("insufficient joined v3 rows for alpha ablation")
    joined.sort(key=lambda row: float(row["raw_time"]))
    values = np.asarray([row["value"] for row in joined], dtype=np.float64)
    cleaned_values = np.asarray([row["cleaned_value"] for row in joined], dtype=np.float64)
    raw_times = np.asarray([row["raw_time"] for row in joined], dtype=np.float64)
    scheduled_times = np.asarray([row["scheduled_time"] for row in joined], dtype=np.float64)
    raw_times -= raw_times[0]
    scheduled_times -= scheduled_times[0]
    sample_sizes = np.asarray([row["shots"] for row in joined], dtype=np.int64)
    regimes = np.asarray([row["regime"] for row in joined])
    raw_edges = np.geomspace(float(np.median(np.diff(raw_times))) * 0.6, float(raw_times[-1]) / 2.0, 12)
    scheduled_edges = lag_edges(scheduled_times)
    specifications = [
        ("legacy_exact", "legacy p-hat(1-p-hat)/N floor; all pairs; legacy wallclock lag bins", values, raw_times, raw_edges, False, False, 8,
         "exact regression scaffold; calibration-step pairs and client submission jitter remain present"),
        ("plus_unbiased_shot_noise", "turn on p-hat(1-p-hat)/(N-1) shot-noise correction only", values, raw_times, raw_edges, True, False, 8,
         "subtracts the finite-N upward correction to the analytic shot floor; no pair membership changes"),
        ("plus_pairing_discipline", "exclude cross-regime pairs; role axis is single-valued in v3", values, raw_times, raw_edges, True, True, 8,
         "removes calibration-step contrasts from continuous-drift SF; v3 contains one probe role, so role splitting contributes zero here"),
        ("plus_lag_binning_primary", "use frozen scheduled timestamps and primary minimum-bin support", cleaned_values, scheduled_times, scheduled_edges, True, True, 2,
         "moves pairs onto the acquisition schedule axis and retains sparse eligible bins used by the primary estimator"),
    ]
    stages: list[dict[str, Any]] = []
    previous_alpha: float | None = None
    for name, change, stage_values, stage_times, edges, unbiased, pairing, minimum_pairs, interpretation in specifications:
        sf_rows, audit = _ablation_structure_function(
            values=stage_values,
            times_seconds=stage_times,
            shots=sample_sizes,
            regimes=regimes,
            lag_edges_seconds=edges,
            unbiased_shot_noise=unbiased,
            enforce_pairing_discipline=pairing,
            minimum_pairs_per_bin=minimum_pairs,
        )
        fit = fit_power_law(sf_rows)
        alpha = float(fit["alpha"])
        stages.append({
            "stage": name,
            "change_activated": change,
            "alpha": alpha,
            "se_alpha": float(fit["se_alpha"]),
            "delta_alpha_from_previous": None if previous_alpha is None else alpha - previous_alpha,
            "direction": None if previous_alpha is None else ("toward_zero" if abs(alpha) < abs(previous_alpha) else "away_from_zero"),
            "physical_interpretation": interpretation,
            **audit,
        })
        previous_alpha = alpha
    legacy_alpha = float(stages[0]["alpha"])
    primary_alpha = float(stages[-1]["alpha"])
    total_delta = primary_alpha - legacy_alpha
    summed_delta = float(sum(float(stage["delta_alpha_from_previous"]) for stage in stages[1:]))
    random_walk_sigma = (1.0 - primary_alpha) / float(stages[-1]["se_alpha"])
    accounting_closes = bool(abs(total_delta - summed_delta) <= 1e-12)
    legacy_matches = bool(abs(legacy_alpha - (-0.07793072513299217)) <= 1e-12)
    primary_matches = bool(abs(primary_alpha - (-0.00907345094607346)) <= 1e-12)
    random_walk_preserved = bool(random_walk_sigma >= 10.0)
    return {
        "factor_order": ["unbiased_shot_noise", "pairing_discipline", "lag_binning_primary"],
        "n_joined_snapshots": len(joined),
        "maximum_raw_vs_cleaned_value_difference": float(np.max(np.abs(values - cleaned_values))),
        "submission_vs_scheduled_lag_offset_seconds": {
            "median": float(np.median(raw_times - scheduled_times)),
            "minimum": float(np.min(raw_times - scheduled_times)),
            "maximum": float(np.max(raw_times - scheduled_times)),
        },
        "stages": stages,
        "total_alpha_shift": total_delta,
        "summed_step_shifts": summed_delta,
        "shift_accounting_closes": accounting_closes,
        "legacy_regression_matches": legacy_matches,
        "primary_path_matches": primary_matches,
        "random_walk_alpha_one_exclusion_sigma": random_walk_sigma,
        "random_walk_exclusion_preserved": random_walk_preserved,
        "T_B4_freeze_ready": bool(legacy_matches and primary_matches and accounting_closes and random_walk_preserved),
    }


def event_characterization(series: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = np.asarray(series["values"], dtype=np.float64)
    shots = np.asarray(series["shots"], dtype=np.int64)
    regimes = np.asarray(series["regime_ids"])
    ordered_regimes = list(dict.fromkeys(str(value) for value in regimes))
    rows: list[dict[str, Any]] = []
    for before_name, after_name in zip(ordered_regimes[:-1], ordered_regimes[1:]):
        before = values[regimes == before_name]
        after = values[regimes == after_name]
        pooled = np.concatenate([before, after])
        pooled_shots = np.concatenate([shots[regimes == before_name], shots[regimes == after_name]])
        floor = float(np.sqrt(np.mean(sensing_economics.binomial_variance_unbiased(pooled, pooled_shots))))
        step = float(np.mean(after) - np.mean(before))
        rows.append({
            "before_regime": before_name,
            "after_regime": after_name,
            "before_count": len(before),
            "after_count": len(after),
            "before_mean": float(np.mean(before)),
            "after_mean": float(np.mean(after)),
            "step": step,
            "mean_single_observation_shot_floor": floor,
            "step_to_shot_floor_ratio": step / floor,
            "role": "supportive event characterization; cross-event pairs excluded from primary SF",
        })
    return rows


def _map_row(
    backend: str,
    channel: str,
    floor_label: str,
    floor_seconds: float,
    analysis: Mapping[str, Any],
    throughput: float,
) -> dict[str, Any]:
    fit = analysis["ou_fit"]
    nonparametric = analysis.get("nonparametric_sensitivity") or {}
    nonparametric_intrinsic = nonparametric.get("intrinsic_optimum") or {}
    nonparametric_constrained = nonparametric.get("constrained_optimum") or {}
    return {
        "backend_id": backend,
        "channel": channel,
        "interface_floor_label": floor_label,
        "interface_floor_seconds": floor_seconds,
        "throughput_shots_per_second_total": throughput,
        "effective_shots_per_second_channel": throughput / 2.0,
        "variance_gate_passed": bool(analysis["variance_gate"]["passed"]),
        "process_variance": analysis["variance_gate"]["process_variance"],
        "process_variance_ci_lower": analysis["variance_gate"]["process_variance_ci_lower"],
        "process_variance_ci_upper": analysis["variance_gate"]["process_variance_ci_upper"],
        "ou_fit_ok": bool(fit["ok"]),
        "ou_process_variance": fit["process_variance"],
        "ou_tau_seconds": fit["tau_seconds"],
        "ou_t_star_seconds": analysis["t_star_seconds"],
        "ou_t_star_ci_lower_seconds": analysis["t_star_ci_lower_seconds"],
        "ou_t_star_ci_upper_seconds": analysis["t_star_ci_upper_seconds"],
        "ou_minimum_residual_variance": analysis["minimum_residual_variance"],
        "worth_sensing": bool(analysis["worth_sensing"]),
        "nonparametric_role": nonparametric.get("role"),
        "nonparametric_t_star_seconds": nonparametric_intrinsic.get("interval_seconds"),
        "nonparametric_constrained_interval_seconds": nonparametric_constrained.get("interval_seconds"),
        "nonparametric_minimum_residual_variance": nonparametric_constrained.get("minimum_residual_variance"),
        "bootstrap_available": bool(analysis.get("parametric_bootstrap", {}).get("available")),
    }


def run(
    *,
    cleaned_path: Path,
    raw_corpus: Path,
    output: Path,
    throughput: float,
    interface_floor_p50: float,
    interface_floor_p90: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B4 analysis: {output}")
    channels = load_cleaned_channels(cleaned_path)
    map_rows: list[dict[str, Any]] = []
    sf_payload: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    channel_reports: dict[str, Any] = {}
    for channel_index, ((backend, channel), series) in enumerate(sorted(channels.items())):
        edges = lag_edges(series["times_seconds"])
        floor_reports: dict[str, Any] = {}
        for floor_label, floor_seconds in (("P50", interface_floor_p50), ("P90", interface_floor_p90)):
            analysis = sensing_economics.analyze_ou_sensing(
                values=series["values"],
                times_seconds=series["times_seconds"],
                shots=series["shots"],
                regime_ids=series["regime_ids"],
                burst_flags=series["burst_flags"],
                lag_edges_seconds=edges,
                effective_shots_per_second=throughput / 2.0,
                maximum_interval_seconds=float(series["times_seconds"][-1] - series["times_seconds"][0]),
                interface_floor_seconds=floor_seconds,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed + channel_index,
            )
            floor_reports[floor_label] = analysis
            map_rows.append(_map_row(backend, channel, floor_label, floor_seconds, analysis, throughput))
        primary = floor_reports["P50"]
        key = f"{backend}:{channel}"
        sf_payload[key] = primary["structure_function"]
        event_rows = event_characterization(series)
        for row in event_rows:
            events.append({"backend_id": backend, "channel": channel, **row})
        channel_reports[key] = {
            "n_observations": len(series["values"]),
            "timestamp_source": series["timestamp_source"],
            "regime_ids": list(dict.fromkeys(str(value) for value in series["regime_ids"])),
            "power_law_fit_primary_sf": fit_power_law(primary["structure_function"]),
            "P50": primary,
            "P90": floor_reports["P90"],
        }

    reference = crossover.variance_decomposition(crossover.load_reference_replicates(str(raw_corpus)))
    legacy_probe = crossover.load_probe_series(str(raw_corpus))["readout_all_one"]
    legacy_edges = np.geomspace(float(np.median(np.diff(legacy_probe.t))) * 0.6, float(legacy_probe.t[-1]) / 2.0, 12)
    legacy_all_pairs_power = crossover.fit_power_law(crossover.structure_function(legacy_probe, legacy_edges))
    ablation = alpha_ablation(cleaned_path, raw_corpus)
    all_one_key = next(key for key in channel_reports if key.endswith(":readout_all_one_error"))
    all_zero_key = next(key for key in channel_reports if key.endswith(":readout_all_zero_error"))
    all_one_power = channel_reports[all_one_key]["power_law_fit_primary_sf"]
    all_zero_event = next(row for row in events if row["channel"] == "readout_all_zero_error")
    regression = {
        "high_lag_power_law": {
            "primary_regime_safe_alpha": all_one_power["alpha"],
            "primary_regime_safe_se_alpha": all_one_power["se_alpha"],
            "legacy_all_pairs_alpha": legacy_all_pairs_power["alpha"],
            "legacy_all_pairs_se_alpha": legacy_all_pairs_power["se_alpha"],
            "legacy_target_alpha": -0.079,
            "legacy_target_se": 0.088,
            "legacy_reproduced": bool(
                abs(float(legacy_all_pairs_power["alpha"]) - (-0.079)) <= 0.01
                and abs(float(legacy_all_pairs_power["se_alpha"]) - 0.088) <= 0.01
            ),
            "note": "primary SF excludes cross-regime pairs; legacy all-pairs value is regression-only and cannot enter the primary estimate",
        },
        "reference_variance_decomposition": {
            "within_ratio": reference["within_ratio"],
            "between_ratio": reference["between_ratio"],
            "positions": reference["positions"],
        },
        "control_and_positive_channels": {
            "e0_variance_gate_passed": channel_reports[all_zero_key]["P50"]["variance_gate"]["passed"],
            "e0_worth_sensing": channel_reports[all_zero_key]["P50"]["worth_sensing"],
            "e1_variance_gate_passed": channel_reports[all_one_key]["P50"]["variance_gate"]["passed"],
            "e1_worth_sensing": channel_reports[all_one_key]["P50"]["worth_sensing"],
        },
        "event_signature": {
            "all_zero_step": all_zero_event["step"],
            "step_to_shot_floor_ratio": all_zero_event["step_to_shot_floor_ratio"],
            "cross_event_pairs_excluded": True,
            "event_role": "supportive_only",
        },
        "alpha_ablation": ablation,
    }
    report = {
        "schema": "b4_sensing_economics_v1",
        "simulation_only": False,
        "evidence_scope": "frozen T287 v3 regression; not new B4 hardware collection",
        "inputs": {
            "cleaned_path": str(cleaned_path.resolve()),
            "raw_corpus": str(raw_corpus.resolve()),
            "throughput_status": "planning input; T-B6 dry-run must replace before Stage-1",
            "throughput_shots_per_second": throughput,
            "interface_floor_status": "planning input; T-B6 P50/P90 must replace before Stage-1",
            "interface_floor_p50_seconds": interface_floor_p50,
            "interface_floor_p90_seconds": interface_floor_p90,
        },
        "method": {
            "primary_sf": "exact binomial shot-noise debias; burst and cross-regime pairs excluded",
            "primary_curve": "OU parametric fit",
            "sensitivity_curve": "weighted monotone nonparametric interpolation",
            "uncertainty": "fitted OU process layer plus binomial observation layer parametric bootstrap",
            "gate_inputs": "observations, timestamps, shots, regime_id, burst_flag, throughput, interface floor only",
        },
        "channels": channel_reports,
        "sensing_map": map_rows,
        "event_characterization": events,
        "v3_regression": regression,
    }
    output.mkdir(parents=True)
    (output / "sensing_economics_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "structure_functions.json").write_text(json.dumps(sf_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "sensing_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(map_rows[0]))
        writer.writeheader()
        writer.writerows(map_rows)
    with (output / "event_characterization.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    ablation_lines = [
        "# T-B4 α 逐项消融",
        "",
        "一次只打开一个变化源；未改容差。",
        "",
        "| 步 | α ± SE | Δα | 方向 | 配对数 | 物理解读 |",
        "|---|---:|---:|---|---:|---|",
    ]
    for stage in ablation["stages"]:
        delta = "—" if stage["delta_alpha_from_previous"] is None else f"{float(stage['delta_alpha_from_previous']):+.8f}"
        ablation_lines.append(
            f"| {stage['stage']} | {float(stage['alpha']):+.8f} ± {float(stage['se_alpha']):.8f} | "
            f"{delta} | {stage['direction'] or '基线'} | {stage['eligible_pairs']} | {stage['physical_interpretation']} |"
        )
    ablation_lines.extend([
        "",
        f"- 位移闭合：{ablation['shift_accounting_closes']}；总 Δα={float(ablation['total_alpha_shift']):+.8f}。",
        f"- 随机行走 α=1 排除：{float(ablation['random_walk_alpha_one_exclusion_sigma']):.2f}σ。",
        f"- T-B4 可冻结：{ablation['T_B4_freeze_ready']}。",
        "",
    ])
    (output / "B4_TB4_ALPHA_ABLATION.md").write_text("\n".join(ablation_lines), encoding="utf-8")
    summary = [
        "# B-4 T-B4 sensing economics v3 regression",
        "",
        "本产物复算冻结 T287 v3 语料，不是新 B4 真机证据。事件仅作支持性表征；跨事件观测对不进主 SF。",
        "",
        f"- e0 detection/worth: {regression['control_and_positive_channels']['e0_variance_gate_passed']} / {regression['control_and_positive_channels']['e0_worth_sensing']}",
        f"- e1 detection/worth: {regression['control_and_positive_channels']['e1_variance_gate_passed']} / {regression['control_and_positive_channels']['e1_worth_sensing']}",
        f"- reference within/between variance ratio: {reference['within_ratio']:.4f} / {reference['between_ratio']:.4f}",
        f"- all-zero calibration-window step: {all_zero_event['step']:+.8f} ({all_zero_event['step_to_shot_floor_ratio']:.3f}× single-observation shot floor)",
        f"- e1 primary regime-safe SF alpha: {float(all_one_power['alpha']):+.4f} ± {float(all_one_power['se_alpha']):.4f}",
        f"- legacy all-pairs regression alpha: {float(legacy_all_pairs_power['alpha']):+.4f} ± {float(legacy_all_pairs_power['se_alpha']):.4f} (regression-only)",
        f"- alpha ablation closes: {ablation['shift_accounting_closes']}; random-walk alpha=1 exclusion: {float(ablation['random_walk_alpha_one_exclusion_sigma']):.2f} sigma.",
        "- T-B4 freeze is allowed only after the alpha-ablation artifact is reviewed; no tolerance was changed.",
        "- OU is primary; monotone nonparametric curve is sensitivity only.",
        "- Throughput and interface floors remain planning inputs until T-B6 dry-run.",
        "",
    ]
    (output / "B4_TB4_V3_REGRESSION.md").write_text("\n".join(summary), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--raw-corpus", type=Path, default=DEFAULT_RAW_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--throughput-shots-per-second", type=float, default=490.0)
    parser.add_argument("--interface-floor-p50-seconds", type=float, default=23.0)
    parser.add_argument("--interface-floor-p90-seconds", type=float, default=23.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    arguments = parser.parse_args()
    report = run(
        cleaned_path=arguments.cleaned,
        raw_corpus=arguments.raw_corpus,
        output=arguments.output,
        throughput=arguments.throughput_shots_per_second,
        interface_floor_p50=arguments.interface_floor_p50_seconds,
        interface_floor_p90=arguments.interface_floor_p90_seconds,
        bootstrap_resamples=arguments.bootstrap_resamples,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    print(json.dumps(report["v3_regression"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
