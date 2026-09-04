#!/usr/bin/env python3
"""Contrast the B-4 map gate from an old peak through measured high-c points."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diagnose_b4_regrid_mechanism as mechanism
from scripts import regrid_b4_endpoints as regrid
from scripts import simulate_b4_design_power as simulation
from src.adaptive import sensing_economics


SEED = 20260804
REPLICATES = 1000
PROFILES = (
    ("old_peak", 850.0, 0.0),
    ("old_boundary", 850.0, 0.7),
    ("tianyan-287_measured", 1486.0918995653424, 6.990118313333854),
    ("tianyan176_measured", 3792.6059369503105, 12.951896450002824),
)


def profiles() -> list[dict[str, Any]]:
    return [
        {
            "backend_id": label,
            "profile_role": "root_cause_contrast",
            "shot_rate_per_second": rate,
            "fixed_overhead_seconds_per_setting": overhead,
            "anchor_shots_per_setting": regrid.ANCHOR_BASE_SHOTS,
        }
        for label, rate, overhead in PROFILES
    ]


def _nullable_median(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return None if not values else float(np.median(values))


def analytic_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    rate = float(profile["shot_rate_per_second"])
    overhead = float(profile["fixed_overhead_seconds_per_setting"])
    effective_rate = simulation.effective_probe_shots_per_second(rate, overhead)
    floor = simulation.design_resolvable_floor_seconds(rate, overhead)
    schedule = simulation.analysis_schedule(rate, overhead)
    maximum = float(schedule["times_seconds"][-1] - schedule["times_seconds"][0])
    rows: list[dict[str, float]] = []
    for tau_minutes in (15.0, 30.0):
        optimum = sensing_economics.optimum_ou_interval(
            mean_probability=simulation.PRIMARY_PROBABILITY,
            effective_shots_per_second=effective_rate,
            process_variance=6.1e-4,
            tau_seconds=tau_minutes * 60.0,
            maximum_interval_seconds=maximum,
        )
        t_star = float(optimum["interval_seconds"])
        minimum = float(optimum["minimum_residual_variance"])
        near = [
            float(sensing_economics.ou_residual_variance(
                t_star * multiplier,
                simulation.PRIMARY_PROBABILITY,
                effective_rate,
                6.1e-4,
                tau_minutes * 60.0,
            ))
            for multiplier in (0.8, 1.25)
        ]
        constrained = sensing_economics.optimum_ou_interval(
            mean_probability=simulation.PRIMARY_PROBABILITY,
            effective_shots_per_second=effective_rate,
            process_variance=6.1e-4,
            tau_seconds=tau_minutes * 60.0,
            maximum_interval_seconds=maximum,
            minimum_interval_seconds=floor,
        )
        rows.append({
            "tau_minutes": tau_minutes,
            "t_star_seconds": t_star,
            "design_floor_seconds": floor,
            "floor_over_t_star": floor / t_star,
            "minimum_residual_variance": minimum,
            "local_absolute_penalty": max(near) - minimum,
            "constrained_residual_variance": float(constrained["minimum_residual_variance"]),
            "constraint_penalty": float(constrained["minimum_residual_variance"]) - minimum,
        })
    return {
        "backend_id": profile["backend_id"],
        "shot_rate_per_second": rate,
        "fixed_overhead_seconds_per_setting": overhead,
        "effective_probe_shots_per_second": effective_rate,
        "design_floor_seconds": floor,
        "tau_contrasts": rows,
    }


def classify(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_backend_gap = {
        (str(row["backend_id"]), int(row["session_gap_days"])): row
        for row in summaries
    }
    old = [by_backend_gap[("old_peak", gap)] for gap in regrid.SESSION_GAP_DAY_NUISANCE]
    measured = [
        row for row in summaries
        if str(row["backend_id"]) in {"tianyan-287_measured", "tianyan176_measured"}
    ]
    old_pass = all(float(row["power"]) >= regrid.POWER_TARGET for row in old)
    detection_fit_clean = max(
        float(row["detection_failure_rate"]) + float(row["ou_fit_failure_rate"])
        for row in summaries
    ) <= 0.01
    economic_worsens = min(float(row["economic_separation_failure_rate"]) for row in measured) > max(
        float(row["economic_separation_failure_rate"]) for row in old
    )
    floor_moves_right = min(float(row["floor_right_of_t_star_rate"]) for row in measured) > max(
        float(row["floor_right_of_t_star_rate"]) for row in old
    )
    if old_pass and detection_fit_clean and economic_worsens and floor_moves_right:
        verdict = "high_fixed_overhead_pushes_floor_past_tstar_and_breaks_economic_separation"
    else:
        verdict = "mechanism_not_closed"
    return {
        "old_peak_all_gaps_pass": old_pass,
        "detection_and_ou_fit_clean": detection_fit_clean,
        "economic_failure_strictly_worse_at_both_measured_points": economic_worsens,
        "floor_right_of_tstar_strictly_worse_at_both_measured_points": floor_moves_right,
        "verdict": verdict,
    }


def run(*, output: Path, replicates: int, seed: int, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mechanism contrast: {output}")
    if seed != SEED or replicates != REPLICATES:
        raise ValueError(f"mechanism contrast is frozen at seed={SEED}, replicates={REPLICATES}")
    selected = profiles()
    jobs = [
        (profile, gap, replicates, seed)
        for profile in selected
        for gap in regrid.SESSION_GAP_DAY_NUISANCE
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            reports = list(executor.map(mechanism._run_one, jobs))
    else:
        reports = [mechanism._run_one(job) for job in jobs]
    rows = [row for report in reports for row in report["rows"]]
    summaries: list[dict[str, Any]] = []
    for report in reports:
        summary = dict(report["summary"])
        summary.update({
            "worst_corner_residual_variance_median": _nullable_median(report["rows"], "worst_corner_residual_variance"),
            "process_variance_ci_lower_median": _nullable_median(report["rows"], "process_variance_ci_lower"),
        })
        summaries.append(summary)
    rows.sort(key=lambda row: (str(row["backend_id"]), int(row["session_gap_days"]), int(row["replicate"])))
    summaries.sort(key=lambda row: (str(row["backend_id"]), int(row["session_gap_days"])))
    decision = classify(summaries)
    payload = {
        "schema": "b4_map_root_cause_mechanism_contrast_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "seed": seed,
        "replicates_per_profile": replicates,
        "session_count": regrid.SESSION_COUNT,
        "session_gap_days": list(regrid.SESSION_GAP_DAY_NUISANCE),
        "profiles": selected,
        "analytic_profiles": [analytic_profile(profile) for profile in selected],
        "summaries": summaries,
        "decision": decision,
    }
    output.mkdir(parents=True)
    (output / "mechanism_contrast.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "mechanism_summaries.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "replicate_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        output=arguments.output,
        replicates=arguments.replicates,
        seed=arguments.seed,
        workers=max(1, arguments.workers),
    )
    print(json.dumps(report["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
