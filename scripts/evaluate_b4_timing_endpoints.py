#!/usr/bin/env python3
"""Run all five B-4 endpoints for every two-parameter timing-grid cell."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import simulate_b4_design_power as simulation


DEFAULT_OUTPUT = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_all_endpoints_timing_grid_20260804"
)


def _cell(dgp: str, shot_rate: int, overhead: float, tau: float | None = None, variance: float | None = None) -> dict[str, Any]:
    return {
        "dgp": dgp,
        "shot_rate_per_second": shot_rate,
        "fixed_overhead_seconds_per_setting": overhead,
        "tau_minutes": tau,
        "process_variance": variance,
    }


def _run_profile(arguments: tuple[int, float, int, int]) -> list[dict[str, Any]]:
    shot_rate, overhead, replicates, seed = arguments
    timing = simulation.timing_fields(shot_rate, overhead)
    null = simulation.run_cell(_cell("null_flat", shot_rate, overhead), replicates, seed)
    null_size = float(null["interior_optimum_claim_rate"])
    ou_rows = [
        simulation.run_cell(_cell("ou", shot_rate, overhead, tau, 6.1e-4), replicates, seed)
        for tau in (15.0, 30.0)
    ]
    regret_power = float(np.mean([float(row["tstar_regret_conditional_power"] or 0.0) for row in ou_rows]))
    regret_unconditional = float(np.mean([float(row["tstar_regret_unconditional_power"] or 0.0) for row in ou_rows]))
    gate_rate = float(np.mean([float(row["regret_gate_condition_rate"] or 0.0) for row in ou_rows]))
    event_rows = [
        simulation.run_cell(_cell(dgp, shot_rate, overhead), replicates, seed)
        for dgp in ("step_calendar", "step_triggered", "step_as_ramp_artifact")
    ]
    event_size = max(float(row["event_continuous_false_positive_rate"]) for row in event_rows)
    rows = [
        {
            "endpoint": "interior_optimum_null",
            **timing,
            "size": null_size,
            "power": 1.0 - null_size,
            "size_pass": null_size <= 0.05,
            "power_pass": 1.0 - null_size >= 0.8,
            "unconditional_power": None,
            "gate_condition_rate": None,
        },
        {
            "endpoint": "tstar_regret_c1p25_conditional",
            **timing,
            "size": null_size,
            "power": regret_power,
            "size_pass": null_size <= 0.05,
            "power_pass": regret_power >= 0.8,
            "unconditional_power": regret_unconditional,
            "gate_condition_rate": gate_rate,
        },
        {
            **simulation.run_map_endpoint(shot_rate, replicates, seed, overhead),
            "unconditional_power": None,
            "gate_condition_rate": None,
        },
        {
            **simulation.run_cadence_endpoint(shot_rate, replicates, seed, overhead),
            "unconditional_power": None,
            "gate_condition_rate": None,
        },
        {
            "endpoint": "event_not_misread_as_continuous",
            **timing,
            "size": event_size,
            "power": 1.0 - event_size,
            "size_pass": event_size <= 0.05,
            "power_pass": 1.0 - event_size >= 0.8,
            "unconditional_power": None,
            "gate_condition_rate": None,
        },
    ]
    for row in rows:
        row["joint_pass"] = bool(row["size_pass"] and row["power_pass"])
        row["n_sequences_e0_or_null"] = replicates
        row["n_sequences_e1_or_injected"] = replicates
    return rows


def run(output: Path, replicates: int, seed: int, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite endpoint-grid output: {output}")
    profiles = simulation.TIMING_PROFILES
    arguments = [(rate, overhead, replicates, seed) for rate, overhead in profiles]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
            nested = list(executor.map(_run_profile, arguments))
    else:
        nested = [_run_profile(item) for item in arguments]
    rows = [row for profile_rows in nested for row in profile_rows]
    rows.sort(key=lambda row: (
        int(row["shot_rate_per_second"]),
        float(row["fixed_overhead_seconds_per_setting"]),
        str(row["endpoint"]),
    ))
    profile_pass: dict[str, bool] = {}
    endpoint_count: dict[str, int] = {}
    for rate, overhead in profiles:
        selected = [
            row for row in rows
            if int(row["shot_rate_per_second"]) == rate
            and float(row["fixed_overhead_seconds_per_setting"]) == overhead
        ]
        key = f"R={rate},c={overhead:.1f}"
        endpoint_count[key] = len(selected)
        profile_pass[key] = bool(len(selected) == 5 and all(row["joint_pass"] for row in selected))
    intermediate_complete = all(
        endpoint_count[f"R={rate},c={overhead:.1f}"] == 5
        for rate in (600, 750, 850)
        for overhead in simulation.FIXED_OVERHEADS_SECONDS_PER_SETTING
    )
    report = {
        "schema": "b4_all_endpoints_timing_grid_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "replicates_per_endpoint_side": replicates,
        "seed": seed,
        "endpoint_names": sorted({str(row["endpoint"]) for row in rows}),
        "intermediate_rates_600_750_850_all_five_endpoints_complete": intermediate_complete,
        "endpoint_count_by_profile": endpoint_count,
        "profile_pass": profile_pass,
        "rows": rows,
    }
    output.mkdir(parents=True)
    (output / "all_endpoints_timing_grid.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with (output / "all_endpoints_timing_grid.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# B-4 双参数网格五终点账",
        "",
        f"每个 endpoint side {replicates} 条序列。600/750/850 每个 overhead 格均有五终点：`{intermediate_complete}`。",
        "",
        "| R | c | endpoint | size | power | joint pass |",
        "|---:|---:|---|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shot_rate_per_second']} | {row['fixed_overhead_seconds_per_setting']:.1f} | "
            f"{row['endpoint']} | {row['size']:.4f} | {row['power']:.4f} | {'是' if row['joint_pass'] else '否'} |"
        )
    lines.append("")
    (output / "B4_ALL_ENDPOINTS_TIMING_GRID.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    report = run(arguments.output, arguments.replicates, arguments.seed, max(1, arguments.workers))
    print(json.dumps({
        "intermediate_complete": report["intermediate_rates_600_750_850_all_five_endpoints_complete"],
        "profile_pass": report["profile_pass"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
