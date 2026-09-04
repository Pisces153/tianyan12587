#!/usr/bin/env python3
"""Regrid all five B-4 endpoints around measured hardware timing points.

This is a simulation-only sensitivity ledger.  It evaluates the frozen
analysis code at the two measured ``(R, c)`` points, a three-by-three local
neighbourhood around each point, a fixed three-session design with inter-session
gap nuisance values ``1, 2, 4`` days, and one anchor-shot leverage cell whose
anchor duration scales with R.
Every endpoint keeps paired e0/size and e1/power fields.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import simulate_b4_design_power as simulation


MEASURED_POINTS = (
    {"backend_id": "tianyan-287", "shot_rate_per_second": 1486.09, "fixed_overhead_seconds_per_setting": 6.990},
    {"backend_id": "tianyan176", "shot_rate_per_second": 3792.61, "fixed_overhead_seconds_per_setting": 12.952},
)
R_NEIGHBOUR_MULTIPLIERS = (0.9, 1.0, 1.1)
C_NEIGHBOUR_OFFSETS = (-1.0, 0.0, 1.0)
SESSION_COUNT = 3
SESSION_GAP_DAY_NUISANCE = (1, 2, 4)
ANCHOR_BASE_SHOTS = 1024
ANCHOR_BASE_RATE = 490.0
POWER_TARGET = 0.8
SIZE_TARGET = 0.05
ENDPOINT_NAMES = (
    "interior_optimum_null",
    "tstar_regret_c1p25_conditional",
    "worth_sensing_map",
    "cadence_pair_ratio",
    "event_not_misread_as_continuous",
)


def normalize_row_fields(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = sorted({key for row in rows for key in row})
    return fields, [{field: row.get(field) for field in fields} for row in rows]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def anchor_shots_for_rate(shot_rate: float) -> int:
    """Keep anchor setting wallclock near baseline by scaling shots with R."""
    return max(1, int(round(ANCHOR_BASE_SHOTS * float(shot_rate) / ANCHOR_BASE_RATE)))


def timing_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for point in MEASURED_POINTS:
        for multiplier in R_NEIGHBOUR_MULTIPLIERS:
            for c_offset in C_NEIGHBOUR_OFFSETS:
                rate = round(float(point["shot_rate_per_second"]) * multiplier, 2)
                overhead = round(float(point["fixed_overhead_seconds_per_setting"]) + c_offset, 3)
                if overhead < 0.0:
                    continue
                cells.append({
                    "backend_id": point["backend_id"],
                    "point_label": point["backend_id"],
                    "neighbour_multiplier": multiplier,
                    "neighbour_c_offset_seconds": c_offset,
                    "shot_rate_per_second": rate,
                    "fixed_overhead_seconds_per_setting": overhead,
                    "cell_role": "measured_point" if multiplier == 1.0 and c_offset == 0.0 else "local_neighbour",
                    "anchor_shot_leverage": False,
                    "anchor_shots_per_setting": ANCHOR_BASE_SHOTS,
                })
        leverage_rate = float(point["shot_rate_per_second"])
        cells.append({
            "backend_id": point["backend_id"],
            "point_label": point["backend_id"],
            "neighbour_multiplier": 1.0,
            "neighbour_c_offset_seconds": 0.0,
            "shot_rate_per_second": round(leverage_rate, 2),
            "fixed_overhead_seconds_per_setting": round(float(point["fixed_overhead_seconds_per_setting"]), 3),
            "cell_role": "anchor_shot_leverage",
            "anchor_shot_leverage": True,
            "anchor_shots_per_setting": anchor_shots_for_rate(leverage_rate),
        })
    return cells


def _cell(dgp: str, rate: float, overhead: float, tau: float | None = None, variance: float | None = None) -> dict[str, Any]:
    return {
        "dgp": dgp,
        "shot_rate_per_second": rate,
        "fixed_overhead_seconds_per_setting": overhead,
        "tau_minutes": tau,
        "process_variance": variance,
    }


def _run_profile(arguments: tuple[Mapping[str, Any], int, int, int]) -> dict[str, Any]:
    cell, session_gap_days, replicates, seed = arguments
    rate = float(cell["shot_rate_per_second"])
    overhead = float(cell["fixed_overhead_seconds_per_setting"])
    previous_days = simulation.CALENDAR_DAYS
    previous_gap_days = simulation.SESSION_GAP_DAYS
    previous_anchor = simulation.ANCHOR_SHOTS_PER_SETTING
    simulation.CALENDAR_DAYS = SESSION_COUNT
    simulation.SESSION_GAP_DAYS = int(session_gap_days)
    simulation.ANCHOR_SHOTS_PER_SETTING = int(cell["anchor_shots_per_setting"])
    try:
        timing = simulation.timing_fields(rate, overhead)
        null = simulation.run_cell(_cell("null_flat", rate, overhead), replicates, seed)
        null_size = float(null["interior_optimum_claim_rate"])
        ou_rows = [
            simulation.run_cell(_cell("ou", rate, overhead, tau, 6.1e-4), replicates, seed)
            for tau in (15.0, 30.0)
        ]
        regret_power = float(np.mean([float(row["tstar_regret_conditional_power"] or 0.0) for row in ou_rows]))
        regret_unconditional = float(np.mean([float(row["tstar_regret_unconditional_power"] or 0.0) for row in ou_rows]))
        gate_rate = float(np.mean([float(row["regret_gate_condition_rate"] or 0.0) for row in ou_rows]))
        event_rows = [
            simulation.run_cell(_cell(dgp, rate, overhead), replicates, seed)
            for dgp in ("step_calendar", "step_triggered", "step_as_ramp_artifact")
        ]
        event_size = max(float(row["event_continuous_false_positive_rate"]) for row in event_rows)
        endpoint_rows = [
            {
                "endpoint": "interior_optimum_null",
                **timing,
                "size": null_size,
                "power": 1.0 - null_size,
                "size_pass": null_size <= SIZE_TARGET,
                "power_pass": 1.0 - null_size >= POWER_TARGET,
                "unconditional_power": None,
                "gate_condition_rate": None,
            },
            {
                "endpoint": "tstar_regret_c1p25_conditional",
                **timing,
                "size": null_size,
                "power": regret_power,
                "size_pass": null_size <= SIZE_TARGET,
                "power_pass": regret_power >= POWER_TARGET,
                "unconditional_power": regret_unconditional,
                "gate_condition_rate": gate_rate,
            },
            {
                **simulation.run_map_endpoint(rate, replicates, seed, overhead),
                "unconditional_power": None,
                "gate_condition_rate": None,
            },
            {
                **simulation.run_cadence_endpoint(rate, replicates, seed, overhead),
                "unconditional_power": None,
                "gate_condition_rate": None,
            },
            {
                "endpoint": "event_not_misread_as_continuous",
                **timing,
                "size": event_size,
                "power": 1.0 - event_size,
                "size_pass": event_size <= SIZE_TARGET,
                "power_pass": 1.0 - event_size >= POWER_TARGET,
                "unconditional_power": None,
                "gate_condition_rate": None,
            },
        ]
        for row in endpoint_rows:
            row.update({
                "joint_pass": bool(row["size_pass"] and row["power_pass"]),
                "n_sequences_e0_or_null": replicates,
                "n_sequences_e1_or_injected": replicates,
                **dict(cell),
                "session_count": SESSION_COUNT,
                "session_gap_days": session_gap_days,
            })
        return {
            "cell": dict(cell),
            "session_count": SESSION_COUNT,
            "session_gap_days": session_gap_days,
            "rows": endpoint_rows,
            "diagnostics": {
                "null_interior_claim_rate": null_size,
                "regret_gate_condition_rate": gate_rate,
                "regret_conditional_power": regret_power,
                "regret_unconditional_power": regret_unconditional,
                "event_false_positive_rate": event_size,
                "anchor_setting_seconds": simulation.setting_duration_seconds(
                    int(cell["anchor_shots_per_setting"]), rate, overhead
                ),
            },
        }
    finally:
        simulation.CALENDAR_DAYS = previous_days
        simulation.SESSION_GAP_DAYS = previous_gap_days
        simulation.ANCHOR_SHOTS_PER_SETTING = previous_anchor


def run(*, output: Path, replicates: int, seed: int, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite regrid output: {output}")
    cells = timing_cells()
    jobs = [
        (cell, gap_days, replicates, seed)
        for cell in cells
        for gap_days in SESSION_GAP_DAY_NUISANCE
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            reports = list(executor.map(_run_profile, jobs))
    else:
        reports = [_run_profile(job) for job in jobs]
    rows = [row for report in reports for row in report["rows"]]
    rows.sort(key=lambda row: (
        str(row["backend_id"]),
        str(row["cell_role"]),
        float(row["shot_rate_per_second"]),
        float(row["fixed_overhead_seconds_per_setting"]),
        int(row["session_gap_days"]),
        str(row["endpoint"]),
    ))
    for row in rows:
        size_successes = int(round(float(row["size"]) * replicates))
        power_successes = int(round(float(row["power"]) * replicates))
        row.update({
            "size_successes_e0": size_successes,
            "power_successes_e1": power_successes,
            "size_mc_ci95": list(wilson_interval(size_successes, replicates)),
            "power_mc_ci95": list(wilson_interval(power_successes, replicates)),
        })
    profile_pass = {}
    for report in reports:
        cell = report["cell"]
        key = f"{cell['backend_id']}|{cell['cell_role']}|R={cell['shot_rate_per_second']:.2f}|c={cell['fixed_overhead_seconds_per_setting']:.3f}|session_gap_days={report['session_gap_days']}"
        profile_pass[key] = bool(all(row["joint_pass"] for row in report["rows"]))
    fields, rows = normalize_row_fields(rows)
    payload = {
        "schema": "b4_regrid_measured_points_five_endpoints_v2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "replicates_per_endpoint_side": replicates,
        "seed": seed,
        "endpoint_names": list(ENDPOINT_NAMES),
        "measured_points": list(MEASURED_POINTS),
        "neighbour_definition": {
            "rate_multipliers": list(R_NEIGHBOUR_MULTIPLIERS),
            "overhead_offsets_seconds_per_setting": list(C_NEIGHBOUR_OFFSETS),
            "cell_role_rule": "exact measured point plus eight local neighbours; leverage cell repeats measured point with R-scaled anchor shots",
        },
        "session_count": SESSION_COUNT,
        "session_gap_day_nuisance": list(SESSION_GAP_DAY_NUISANCE),
        "anchor_shot_leverage": {
            "base_shots_per_setting": ANCHOR_BASE_SHOTS,
            "base_rate_per_second": ANCHOR_BASE_RATE,
            "formula": "round(1024 * R / 490)",
            "purpose": "backup cell only; does not alter registered collection unless power loss is observed",
        },
        "targets": {"size_max": SIZE_TARGET, "power_min": POWER_TARGET},
        "cell_count": len(cells),
        "profile_count": len(jobs),
        "profile_pass": profile_pass,
        "rows": rows,
        "diagnostics": [
            {
                "cell": report["cell"],
                "session_count": report["session_count"],
                "session_gap_days": report["session_gap_days"],
                **report["diagnostics"],
            }
            for report in reports
        ],
        "outside_domain_policy": "measured points are outside frozen lookup domain; this is sensitivity evidence only, no extrapolation and no on-site burst change",
    }
    output.mkdir(parents=True)
    (output / "regrid_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "regrid_endpoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "regrid_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        diagnostics = payload["diagnostics"]
        fields = sorted({key for row in diagnostics for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    lines = [
        "# B-4 measured-point regrid",
        "",
        f"{len(cells)} timing cells × {len(SESSION_GAP_DAY_NUISANCE)} session-gap nuisance values × five endpoints; {replicates} e0/e1 replicates per endpoint side.",
        "",
        "Measured points outside frozen lookup domain remain sensitivity evidence only. Collection design and Stage-1 branch unchanged.",
        "",
        "| backend | role | R | c | session gap days | endpoint | size/e0 | power/e1 | joint |",
        "|---|---|---:|---:|---:|---|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend_id']} | {row['cell_role']} | {row['shot_rate_per_second']:.2f} | "
            f"{row['fixed_overhead_seconds_per_setting']:.3f} | {row['session_gap_days']} | {row['endpoint']} | "
            f"{row['size']:.4f} | {row['power']:.4f} | {'yes' if row['joint_pass'] else 'no'} |"
        )
    (output / "REGRID_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def normalize_existing(output: Path) -> dict[str, Any]:
    report_path = output / "regrid_report.json"
    csv_path = output / "regrid_endpoints.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fields, rows = normalize_row_fields([dict(row) for row in report["rows"]])
    report["rows"] = rows
    report["field_set_audit"] = {
        "json_row_fields": fields,
        "csv_fields": fields,
        "difference": [],
        "passed": True,
    }
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_report.replace(report_path)
    temporary_csv.replace(csv_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--normalize-existing", action="store_true")
    args = parser.parse_args()
    if args.normalize_existing:
        report = normalize_existing(args.output)
        print(json.dumps({"normalized_rows": len(report["rows"]), "field_set_audit": report["field_set_audit"]}))
        return 0
    report = run(output=args.output, replicates=args.replicates, seed=args.seed, workers=max(1, args.workers))
    print(json.dumps({"cell_count": report["cell_count"], "profile_count": report["profile_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
