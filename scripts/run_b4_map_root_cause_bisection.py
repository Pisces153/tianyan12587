#!/usr/bin/env python3
"""Bisect the failed old-domain B-4 map control into overhead versus pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import regrid_b4_endpoints as regrid
from scripts import simulate_b4_design_power as simulation


RATE = 850.0
PEAK_OVERHEAD = 0.0
BOUNDARY_OVERHEAD = 0.7
SESSION_GAPS = (1, 2, 4)
SEED = 20260804
REPLICATES = 10000
ANCHOR_SHOTS = 1024
AB_REPORT = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_root_cause_two_cut_ab_20260805_v2"
    r"\map_root_cause_ab.json"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def _run_peak(arguments: tuple[int, int, int]) -> dict[str, Any]:
    gap, replicates, seed = arguments
    previous_days = simulation.CALENDAR_DAYS
    previous_gap = simulation.SESSION_GAP_DAYS
    previous_anchor = simulation.ANCHOR_SHOTS_PER_SETTING
    simulation.CALENDAR_DAYS = regrid.SESSION_COUNT
    simulation.SESSION_GAP_DAYS = gap
    simulation.ANCHOR_SHOTS_PER_SETTING = ANCHOR_SHOTS
    try:
        row = simulation.run_map_endpoint(RATE, replicates, seed, PEAK_OVERHEAD)
        size_successes = int(round(float(row["size"]) * replicates))
        power_successes = int(round(float(row["power"]) * replicates))
        return {
            **row,
            "session_count": regrid.SESSION_COUNT,
            "session_gap_days": gap,
            "anchor_shots_per_setting": ANCHOR_SHOTS,
            "replicates_per_side": replicates,
            "size_mc_ci95": list(regrid.wilson_interval(size_successes, replicates)),
            "power_mc_ci95": list(regrid.wilson_interval(power_successes, replicates)),
            "joint_pass": bool(row["size_pass"] and row["power_pass"]),
        }
    finally:
        simulation.CALENDAR_DAYS = previous_days
        simulation.SESSION_GAP_DAYS = previous_gap
        simulation.ANCHOR_SHOTS_PER_SETTING = previous_anchor


def classify(peak_rows: list[dict[str, Any]], boundary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    peak_by_gap = {int(row["session_gap_days"]): row for row in peak_rows}
    boundary_by_gap = {int(row["session_gap_days"]): row for row in boundary_rows}
    peak_all_pass = set(peak_by_gap) == set(SESSION_GAPS) and all(bool(row["joint_pass"]) for row in peak_by_gap.values())
    boundary_all_pass = set(boundary_by_gap) == set(SESSION_GAPS) and all(bool(row["joint_pass"]) for row in boundary_by_gap.values())
    if peak_all_pass and not boundary_all_pass:
        verdict = "pipeline_intact_overhead_moves_old_domain_control_to_power_boundary"
        next_action = "test high-R/high-c economic-separation mechanism; do not change endpoint or threshold"
    elif not bool(peak_by_gap.get(1, {}).get("joint_pass")):
        verdict = "runner_or_endpoint_regression"
        next_action = "bisect wrapper numeric normalization and endpoint implementation"
    elif not peak_all_pass:
        verdict = "session_gap_semantics_change_power"
        next_action = "audit gap-specific schedule and lag-bin semantics"
    else:
        verdict = "pipeline_intact_both_controls_pass"
        next_action = "attribute new-domain loss to the high-R/high-c regime and test mechanism"
    return {
        "corrected_peak_c0_all_gaps_pass": peak_all_pass,
        "corrected_boundary_c0p7_all_gaps_pass": boundary_all_pass,
        "verdict": verdict,
        "next_action": next_action,
    }


def run(*, output: Path, replicates: int, seed: int, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bisection output: {output}")
    if seed != SEED or replicates != REPLICATES:
        raise ValueError(f"bisection is frozen at seed={SEED}, replicates={REPLICATES}")
    ab = json.loads(AB_REPORT.read_text(encoding="utf-8"))
    boundary_rows = list(ab["adaptive_map_refinement"]["rows"])
    jobs = [(gap, replicates, seed) for gap in SESSION_GAPS]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            peak_rows = list(executor.map(_run_peak, jobs))
    else:
        peak_rows = [_run_peak(job) for job in jobs]
    peak_rows.sort(key=lambda row: int(row["session_gap_days"]))
    decision = classify(peak_rows, boundary_rows)
    payload = {
        "schema": "b4_map_root_cause_bisection_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "seed": seed,
        "replicates_per_side": replicates,
        "fixed": {
            "R": RATE,
            "session_count": regrid.SESSION_COUNT,
            "session_gap_days": list(SESSION_GAPS),
            "anchor_shots_per_setting": ANCHOR_SHOTS,
            "power_target": regrid.POWER_TARGET,
            "size_target": regrid.SIZE_TARGET,
            "runner_numeric_type": "float",
        },
        "split": {"peak_overhead": PEAK_OVERHEAD, "boundary_overhead": BOUNDARY_OVERHEAD},
        "source_ab_report": str(AB_REPORT),
        "source_ab_report_sha256": digest(AB_REPORT),
        "peak_c0_rows": peak_rows,
        "boundary_c0p7_rows": boundary_rows,
        "decision": decision,
    }
    output.mkdir(parents=True)
    (output / "map_root_cause_bisection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=3)
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
