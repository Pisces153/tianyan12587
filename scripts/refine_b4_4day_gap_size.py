#!/usr/bin/env python3
"""Refine 4-day session-gap e0 sizes selected by the 1,000-run Wilson interval."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import regrid_b4_endpoints as regrid
from scripts import simulate_b4_design_power as simulation


SIZE_TARGET = 0.05
SESSION_GAP_DAYS = 4
ENDPOINT_FAMILY = {
    "interior_optimum_null": "interior_optimum_null",
    "tstar_regret_c1p25_conditional": "interior_optimum_null",
    "event_not_misread_as_continuous": "event_not_misread_as_continuous",
    "worth_sensing_map": "worth_sensing_map",
    "cadence_pair_ratio": "cadence_pair_ratio",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def interval_crosses_target(interval: list[float] | tuple[float, float], target: float = SIZE_TARGET) -> bool:
    return float(interval[0]) <= target <= float(interval[1])


def select_refinements(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source_row in report["rows"]:
        if int(source_row.get("session_gap_days", -1)) != SESSION_GAP_DAYS:
            continue
        endpoint = str(source_row["endpoint"])
        family = ENDPOINT_FAMILY[endpoint]
        if not interval_crosses_target(source_row["size_mc_ci95"]):
            continue
        key = (
            str(source_row["backend_id"]),
            str(source_row["cell_role"]),
            float(source_row["shot_rate_per_second"]),
            float(source_row["fixed_overhead_seconds_per_setting"]),
            int(source_row["anchor_shots_per_setting"]),
            family,
        )
        row = selected.setdefault(key, {
            "backend_id": source_row["backend_id"],
            "cell_role": source_row["cell_role"],
            "shot_rate_per_second": float(source_row["shot_rate_per_second"]),
            "fixed_overhead_seconds_per_setting": float(source_row["fixed_overhead_seconds_per_setting"]),
            "anchor_shots_per_setting": int(source_row["anchor_shots_per_setting"]),
            "endpoint_family": family,
            "source_endpoints": [],
            "source_size": float(source_row["size"]),
            "source_size_mc_ci95": list(source_row["size_mc_ci95"]),
        })
        row["source_endpoints"].append(endpoint)
    rows = list(selected.values())
    for row in rows:
        row["source_endpoints"].sort()
    rows.sort(key=lambda row: (
        str(row["backend_id"]),
        str(row["cell_role"]),
        float(row["shot_rate_per_second"]),
        float(row["fixed_overhead_seconds_per_setting"]),
        str(row["endpoint_family"]),
    ))
    return rows


def _cell(dgp: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dgp": dgp,
        "shot_rate_per_second": float(profile["shot_rate_per_second"]),
        "fixed_overhead_seconds_per_setting": float(profile["fixed_overhead_seconds_per_setting"]),
        "tau_minutes": None,
        "process_variance": None,
    }


def _run_one(arguments: tuple[Mapping[str, Any], int, int]) -> dict[str, Any]:
    profile, replicates, seed = arguments
    previous_days = simulation.CALENDAR_DAYS
    previous_gap = simulation.SESSION_GAP_DAYS
    previous_anchor = simulation.ANCHOR_SHOTS_PER_SETTING
    simulation.CALENDAR_DAYS = regrid.SESSION_COUNT
    simulation.SESSION_GAP_DAYS = SESSION_GAP_DAYS
    simulation.ANCHOR_SHOTS_PER_SETTING = int(profile["anchor_shots_per_setting"])
    try:
        family = str(profile["endpoint_family"])
        component_sizes: dict[str, float]
        if family == "interior_optimum_null":
            result = simulation.run_cell(_cell("null_flat", profile), replicates, seed)
            component_sizes = {"null_flat": float(result["interior_optimum_claim_rate"])}
        elif family == "event_not_misread_as_continuous":
            component_sizes = {
                dgp: float(simulation.run_cell(_cell(dgp, profile), replicates, seed)["event_continuous_false_positive_rate"])
                for dgp in ("step_calendar", "step_triggered", "step_as_ramp_artifact")
            }
        elif family == "worth_sensing_map":
            result = simulation.run_map_endpoint(
                float(profile["shot_rate_per_second"]),
                replicates,
                seed,
                float(profile["fixed_overhead_seconds_per_setting"]),
            )
            component_sizes = {"map_e0": float(result["size"])}
        elif family == "cadence_pair_ratio":
            result = simulation.run_cadence_endpoint(
                float(profile["shot_rate_per_second"]),
                replicates,
                seed,
                float(profile["fixed_overhead_seconds_per_setting"]),
            )
            component_sizes = {"cadence_e0": float(result["size"])}
        else:
            raise ValueError(f"unsupported endpoint family: {family}")
        selected_component = max(component_sizes, key=component_sizes.get)
        size = component_sizes[selected_component]
        successes = int(round(size * replicates))
        interval = regrid.wilson_interval(successes, replicates)
        if interval[1] <= SIZE_TARGET:
            ci_decision = "clean_below_0.05"
        elif interval[0] > SIZE_TARGET:
            ci_decision = "exceeds_0.05"
        else:
            ci_decision = "unresolved_at_40000"
        return {
            **dict(profile),
            "session_count": regrid.SESSION_COUNT,
            "session_gap_days": SESSION_GAP_DAYS,
            "replicates_e0": replicates,
            "seed": seed,
            "selected_component": selected_component,
            "component_sizes": json.dumps(component_sizes, ensure_ascii=False, sort_keys=True),
            "size_successes": successes,
            "size": size,
            "power_complement": 1.0 - size,
            "size_mc_ci95_lower": interval[0],
            "size_mc_ci95_upper": interval[1],
            "size_point_pass": size <= SIZE_TARGET,
            "size_ci_decision": ci_decision,
        }
    finally:
        simulation.CALENDAR_DAYS = previous_days
        simulation.SESSION_GAP_DAYS = previous_gap
        simulation.ANCHOR_SHOTS_PER_SETTING = previous_anchor


def run(*, source: Path, output: Path, replicates: int, seed: int, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite size refinement: {output}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("schema") != "b4_regrid_measured_points_five_endpoints_v2":
        raise ValueError("source must be gap-corrected regrid v2")
    profiles = select_refinements(report)
    jobs = [(profile, replicates, seed) for profile in profiles]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            rows = list(executor.map(_run_one, jobs))
    else:
        rows = [_run_one(job) for job in jobs]
    payload = {
        "schema": "b4_regrid_4day_gap_size_refinement_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "source_regrid_report": str(source.resolve()),
        "source_regrid_report_sha256": digest(source),
        "selection_rule": "session_gap_days=4 and 1000-run Wilson 95% size interval includes 0.05; T* and interior-null share one null refinement",
        "replicates_per_selected_e0": replicates,
        "seed": seed,
        "selected_profile_count": len(profiles),
        "rows": rows,
    }
    output.mkdir(parents=True)
    (output / "size_refinement.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output / "size_refinement.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        source=arguments.source,
        output=arguments.output,
        replicates=arguments.replicates,
        seed=arguments.seed,
        workers=max(1, arguments.workers),
    )
    print(json.dumps({"selected_profiles": report["selected_profile_count"], "rows": len(report["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
