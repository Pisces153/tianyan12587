#!/usr/bin/env python3
"""Merge 10k grid and adaptive 40k refinements into the frozen B6 lookup."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import refine_b4_map_power as refinement


DEFAULT_GRID = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_power_timing_grid_20260804_v2\map_power_timing_grid.json")
DEFAULT_DIAGNOSTIC = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_timing_diagnostic_20260804_v2\diagnostic_summary.json")
DEFAULT_BORDERLINE = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_borderline_timing_cells_20260804\borderline_summary.json")
DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_timing_lookup_frozen_20260804")


def _summary_map(reports: Sequence[Mapping[str, Any]]) -> dict[tuple[int, float], Mapping[str, Any]]:
    mapped: dict[tuple[int, float], Mapping[str, Any]] = {}
    for report in reports:
        for row in report["summaries"]:
            key = (int(row["shot_rate_per_second"]), float(row["fixed_overhead_seconds_per_setting"]))
            mapped[key] = row
    return mapped


def run(grid_path: Path, diagnostic_path: Path, borderline_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen lookup: {output}")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    borderline = json.loads(borderline_path.read_text(encoding="utf-8"))
    refinements = _summary_map((diagnostic, borderline))
    rows = [dict(row) for row in grid["rows"]]
    for row in rows:
        key = (int(row["shot_rate_per_second"]), float(row["fixed_overhead_seconds_per_setting"]))
        if key in refinements:
            summary = refinements[key]
            trials = int(summary["n_replicates"])
            successes = int(round(float(summary["power"]) * trials))
            lower, upper = refinement.wilson_interval(successes, trials)
            row.update({
                "power": float(summary["power"]),
                "power_successes": successes,
                "n_sequences_e1": trials,
                "power_mc_ci95_lower": lower,
                "power_mc_ci95_upper": upper,
                "power_pass": bool(float(summary["power"]) >= refinement.POWER_TARGET),
                "power_source": "adaptive_40000_replicate_refinement",
            })
        else:
            row["power_source"] = "complete_grid_10000_replicates"
    cells = refinement.decision_cells(rows)
    report = {
        "schema": "b4_map_timing_lookup_frozen_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "source_grid": str(grid_path.resolve()),
        "source_diagnostic": str(diagnostic_path.resolve()),
        "source_borderline_refinement": str(borderline_path.resolve()),
        "adaptive_refinement_rule": "increase to 40000 when the 10000-replicate Wilson 95% interval crosses 0.8, plus frozen mechanism profiles 850/c0 and 1000/c0,c0.5,c1.1",
        "power_target_uses_point_estimate": True,
        "b6_point_estimator": grid["b6_point_estimator"],
        "lookup_domain": grid["lookup_domain"],
        "outside_domain_behavior": grid["outside_domain_behavior"],
        "decision_cells": cells,
        "rows": rows,
        "on_site_discretion": False,
    }
    output.mkdir(parents=True)
    (output / "frozen_lookup.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "frozen_lookup.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# B-4 地图终点最终冻结查表",
        "",
        "B6 只代入点估计 `(R_hat,c_hat)`；无‘约’、无现场裁量。power 分支用 MC 点估计；95% CI 同报。",
        "",
        "- `R_hat=(N_high-N_low)/(median(t_high)-median(t_low))`。",
        "- `c_hat=(median(t_low)-N_low/R_hat)/settings_per_job`，单位 s/setting。",
        "- 有效域：`R_hat ∈ [435,1075]` 且 `c_hat ∈ [0,1.4]`。内部区间左闭右开；最后一格右闭。",
        "- 域外：地图终点不全额纳入；采集不变；不外推 power；不现场加 burst 配对。",
        "",
        "| R grid | c grid | e0 n/size | e1 n/power | e1 95% CI | source | full |",
        "|---:|---:|---:|---:|---:|---|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shot_rate_per_second']} | {row['fixed_overhead_seconds_per_setting']:.1f} | "
            f"{row['n_sequences_e0']}/{row['size']:.4f} | {row['n_sequences_e1']}/{row['power']:.4f} | "
            f"[{row['power_mc_ci95_lower']:.4f}, {row['power_mc_ci95_upper']:.4f}] | {row['power_source']} | "
            f"{'是' if row['size_pass'] and row['power_pass'] else '否'} |"
        )
    lines.extend(["", "## 精确矩形", "", "| R_hat | c_hat | grid | full |", "|---|---|---|:---:|"])
    for cell in cells:
        lines.append(
            f"| `{cell['rate_interval']}` | `{cell['overhead_interval']}` | "
            f"`({cell['grid_shot_rate_per_second']},{cell['grid_fixed_overhead_seconds_per_setting']:.1f})` | "
            f"{'是' if cell['full_map_preregistration'] else '否'} |"
        )
    lines.extend(["", "否分支：采集不变；如实写格点 size/power；不增配。", ""])
    (output / "B4_MAP_TIMING_LOOKUP_FROZEN.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--borderline", type=Path, default=DEFAULT_BORDERLINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = run(arguments.grid, arguments.diagnostic, arguments.borderline, arguments.output)
    print(json.dumps({
        "full_cells": sum(cell["full_map_preregistration"] for cell in report["decision_cells"]),
        "rows": [{"R": row["shot_rate_per_second"], "c": row["fixed_overhead_seconds_per_setting"], "power": row["power"]} for row in report["rows"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
