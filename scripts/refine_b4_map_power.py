#!/usr/bin/env python3
"""High-replicate B-4 map endpoint on the frozen two-parameter timing grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import simulate_b4_design_power as simulation


DEFAULT_OUTPUT = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_power_timing_grid_20260804_v2"
)
DEFAULT_SHOT_RATES = simulation.SHOT_RATES_PER_SECOND
DEFAULT_SETTING_OVERHEADS = simulation.FIXED_OVERHEADS_SECONDS_PER_SETTING
POWER_TARGET = 0.8
RATE_DOMAIN = (435.0, 1075.0)
OVERHEAD_DOMAIN = (0.0, 1.4)
RATE_BINS = (
    (435.0, 545.0, 490),
    (545.0, 675.0, 600),
    (675.0, 800.0, 750),
    (800.0, 925.0, 850),
    (925.0, 1075.0, 1000),
)
OVERHEAD_BINS = (
    (0.0, 0.25, 0.0),
    (0.25, 0.8, 0.5),
    (0.8, 1.4, 1.1),
)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        probability * (1.0 - probability) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _bin_center(value: float, bins: Sequence[tuple[float, float, float | int]]) -> float | int | None:
    for index, (lower, upper, center) in enumerate(bins):
        if lower <= value < upper or (index == len(bins) - 1 and value == upper):
            return center
    return None


def select_lookup_cell(
    rows: Sequence[Mapping[str, Any]],
    measured_shot_rate_per_second: float,
    measured_fixed_overhead_seconds_per_setting: float,
) -> dict[str, Any] | None:
    """Map B6 point estimates to one exact preregistered rectangular cell."""
    rate_center = _bin_center(float(measured_shot_rate_per_second), RATE_BINS)
    overhead_center = _bin_center(float(measured_fixed_overhead_seconds_per_setting), OVERHEAD_BINS)
    if rate_center is None or overhead_center is None:
        return None
    for row in rows:
        if (
            int(row["shot_rate_per_second"]) == int(rate_center)
            and float(row["fixed_overhead_seconds_per_setting"]) == float(overhead_center)
        ):
            return dict(row)
    raise ValueError("lookup grid is missing a preregistered timing cell")


def decision_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_profile = {
        (int(row["shot_rate_per_second"]), float(row["fixed_overhead_seconds_per_setting"])): row
        for row in rows
    }
    cells: list[dict[str, Any]] = []
    for rate_index, (rate_lower, rate_upper, rate_center) in enumerate(RATE_BINS):
        for overhead_index, (overhead_lower, overhead_upper, overhead_center) in enumerate(OVERHEAD_BINS):
            row = by_profile[(int(rate_center), float(overhead_center))]
            rate_right = "]" if rate_index == len(RATE_BINS) - 1 else ")"
            overhead_right = "]" if overhead_index == len(OVERHEAD_BINS) - 1 else ")"
            cells.append({
                "rate_interval": f"[{rate_lower:g}, {rate_upper:g}{rate_right}",
                "overhead_interval": f"[{overhead_lower:g}, {overhead_upper:g}{overhead_right}",
                "grid_shot_rate_per_second": int(rate_center),
                "grid_fixed_overhead_seconds_per_setting": float(overhead_center),
                "size": float(row["size"]),
                "power": float(row["power"]),
                "full_map_preregistration": bool(row["size_pass"] and row["power_pass"]),
            })
    return cells


def _run_one(arguments: tuple[int, float, int, int]) -> dict[str, Any]:
    shot_rate, overhead, replicates, seed = arguments
    return simulation.run_map_endpoint(shot_rate, replicates, seed, overhead)


def run(
    *,
    output: Path,
    shot_rates: Sequence[int],
    setting_overheads: Sequence[float],
    replicates: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite map refinement output: {output}")
    rates = tuple(sorted(set(int(value) for value in shot_rates)))
    overheads = tuple(sorted(set(float(value) for value in setting_overheads)))
    if rates != tuple(DEFAULT_SHOT_RATES) or overheads != tuple(DEFAULT_SETTING_OVERHEADS):
        raise ValueError("freeze run requires the complete preregistered 5 x 3 timing grid")
    if replicates < 10_000:
        raise ValueError("at least 10000 e0 and 10000 e1 sequences per timing cell are required")
    arguments = [
        (rate, overhead, replicates, seed)
        for rate in rates
        for overhead in overheads
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
            rows = list(executor.map(_run_one, arguments))
    else:
        rows = [_run_one(item) for item in arguments]
    rows.sort(key=lambda row: (
        int(row["shot_rate_per_second"]),
        float(row["fixed_overhead_seconds_per_setting"]),
    ))
    for row in rows:
        e0_successes = int(round(float(row["size"]) * replicates))
        e1_successes = int(round(float(row["power"]) * replicates))
        size_lower, size_upper = wilson_interval(e0_successes, replicates)
        power_lower, power_upper = wilson_interval(e1_successes, replicates)
        row.update({
            "n_sequences_e0": replicates,
            "n_sequences_e1": replicates,
            "size_successes": e0_successes,
            "power_successes": e1_successes,
            "size_mc_ci95_lower": size_lower,
            "size_mc_ci95_upper": size_upper,
            "power_mc_ci95_lower": power_lower,
            "power_mc_ci95_upper": power_upper,
            "target_power": POWER_TARGET,
        })
    report = {
        "schema": "b4_map_power_timing_grid_v2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "endpoint": "worth_sensing_map",
        "replicates_per_e0_cell": replicates,
        "replicates_per_e1_cell": replicates,
        "seed": seed,
        "power_target": POWER_TARGET,
        "shot_rate_grid_per_second": list(rates),
        "fixed_overhead_grid_seconds_per_setting": list(overheads),
        "b6_point_estimator": {
            "time_statistic": "median roundtrip seconds at each of 1024 and 16384 shots/setting",
            "shot_rate_formula": "(total_shots_high-total_shots_low)/(median_time_high-median_time_low)",
            "fixed_overhead_formula": "(median_time_low-total_shots_low/shot_rate)/settings_per_job",
            "decision_uses": "point estimates; confidence bounds are reported but do not choose the branch",
        },
        "lookup_domain": {
            "shot_rate_per_second_closed_interval": list(RATE_DOMAIN),
            "fixed_overhead_seconds_per_setting_closed_interval": list(OVERHEAD_DOMAIN),
            "internal_boundary_rule": "left-closed, right-open; final interval right-closed",
        },
        "outside_domain_behavior": "map endpoint is not fully preregistered; collection stays unchanged; report no extrapolated power; add no on-site burst pair",
        "rows": rows,
        "decision_cells": decision_cells(rows),
        "on_site_discretion": False,
    }
    output.mkdir(parents=True)
    (output / "map_power_timing_grid.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "map_power_timing_grid.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# B-4 地图终点双参数冻结表",
        "",
        f"每格 e0/e1 各 {replicates} 条序列。选择变量为 B6 点估计 `(R_hat, c_hat)`；不使用‘约’。",
        "",
        "- `R_hat=(N_high-N_low)/(median(t_high)-median(t_low))`。",
        "- `c_hat=(median(t_low)-N_low/R_hat)/settings_per_job`，单位 s/setting。",
        "- 有效域：`R_hat ∈ [435,1075]` 且 `c_hat ∈ [0,1.4]`。内部区间左闭右开；最后一格右闭。",
        "- 域外：地图终点不全额纳入；采集不变；不外推 power；不现场加 burst 配对。",
        "",
        "| R grid | c grid | anchor setting s | probe setting s | floor s | e0 size | e1 power | e1 MC 95% CI | full |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['shot_rate_per_second'])} | {float(row['fixed_overhead_seconds_per_setting']):.1f} | "
            f"{float(row['anchor_setting_duration_seconds']):.4f} | {float(row['probe_setting_duration_seconds']):.4f} | "
            f"{float(row['design_resolvable_floor_seconds']):.4f} | {float(row['size']):.4f} | {float(row['power']):.4f} | "
            f"[{float(row['power_mc_ci95_lower']):.4f}, {float(row['power_mc_ci95_upper']):.4f}] | "
            f"{'是' if bool(row['size_pass'] and row['power_pass']) else '否'} |"
        )
    lines.extend(["", "## 精确分支矩形", "", "| R_hat interval | c_hat interval | selected grid | full |", "|---|---|---|:---:|"])
    for cell in report["decision_cells"]:
        lines.append(
            f"| `{cell['rate_interval']}` | `{cell['overhead_interval']}` | "
            f"`({cell['grid_shot_rate_per_second']}, {cell['grid_fixed_overhead_seconds_per_setting']:.1f})` | "
            f"{'是' if cell['full_map_preregistration'] else '否'} |"
        )
    lines.extend(["", "低功效分支：采集设计不变；如实写本格 size/power；不现场增配。", ""])
    (output / "B4_MAP_TIMING_LOOKUP.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shot-rates", type=int, nargs="+", default=list(DEFAULT_SHOT_RATES))
    parser.add_argument("--setting-overheads", type=float, nargs="+", default=list(DEFAULT_SETTING_OVERHEADS))
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    report = run(
        output=arguments.output,
        shot_rates=arguments.shot_rates,
        setting_overheads=arguments.setting_overheads,
        replicates=arguments.replicates,
        seed=arguments.seed,
        workers=max(1, arguments.workers),
    )
    print(json.dumps({"rows": report["rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
