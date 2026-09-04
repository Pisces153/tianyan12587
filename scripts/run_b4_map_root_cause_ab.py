#!/usr/bin/env python3
"""Run the preregistered two-cut B-4 map root-cause A/B.

Cut 1 replays one legacy all-pass grid cell with the frozen seed.  Cut 2 uses
the gap-corrected three-session pipeline at an old-domain point.  No threshold,
DGP, endpoint, or seed is tuned from the observed result.
"""

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
from scripts import evaluate_b4_timing_endpoints as legacy_evaluation
from scripts import simulate_b4_design_power as simulation


LEGACY_ARTIFACT = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_all_endpoints_timing_grid_20260804"
    r"\all_endpoints_timing_grid.json"
)
FROZEN_MANIFEST = ROOT / "docs" / "B4_TB1_TB4_MODULE_FREEZE_20260804.json"
SIMULATION_SOURCE = ROOT / "scripts" / "simulate_b4_design_power.py"
FROZEN_SIMULATION_SHA256 = "94894ED879B6C85AFF7BFE8E5C37FF55F8D7E93F160DF069F3AE8829A270176D"
LEGACY_RATE = 850.0
LEGACY_OVERHEAD = 0.0
CORRECTED_RATE = 850.0
CORRECTED_OVERHEAD = 0.7
SESSION_GAPS = (1, 2, 4)
ANCHOR_SHOTS = 1024
SEED = 20260804
REPLICATES = 1000
REFINEMENT_REPLICATES = 10000


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def cell(label: str, rate: float, overhead: float) -> dict[str, Any]:
    return {
        "backend_id": label,
        "point_label": label,
        "neighbour_multiplier": 1.0,
        "neighbour_c_offset_seconds": 0.0,
        "shot_rate_per_second": float(rate),
        "fixed_overhead_seconds_per_setting": float(overhead),
        "cell_role": label,
        "anchor_shot_leverage": False,
        "anchor_shots_per_setting": ANCHOR_SHOTS,
    }


def legacy_artifact_rows(path: Path = LEGACY_ARTIFACT) -> list[dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "b4_all_endpoints_timing_grid_v1":
        raise ValueError("legacy control must use the frozen all-endpoints grid")
    rows = [
        row
        for row in report["rows"]
        if float(row["shot_rate_per_second"]) == LEGACY_RATE
        and float(row["fixed_overhead_seconds_per_setting"]) == LEGACY_OVERHEAD
    ]
    rows.sort(key=lambda row: str(row["endpoint"]))
    if len(rows) != len(regrid.ENDPOINT_NAMES) or not all(bool(row["joint_pass"]) for row in rows):
        raise ValueError("legacy control cell is not the expected five-endpoint all-pass profile")
    return rows


def compare_legacy_rows(
    replay_rows: list[Mapping[str, Any]],
    artifact_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    replay = {str(row["endpoint"]): row for row in replay_rows}
    artifact = {str(row["endpoint"]): row for row in artifact_rows}
    if set(replay) != set(artifact):
        raise ValueError("legacy replay endpoint set differs from frozen artifact")
    comparisons: list[dict[str, Any]] = []
    for endpoint in sorted(artifact):
        expected = artifact[endpoint]
        observed = replay[endpoint]
        comparisons.append({
            "endpoint": endpoint,
            "artifact_size": float(expected["size"]),
            "replay_size": float(observed["size"]),
            "artifact_power": float(expected["power"]),
            "replay_power": float(observed["power"]),
            "size_exact_match": float(expected["size"]) == float(observed["size"]),
            "power_exact_match": float(expected["power"]) == float(observed["power"]),
            "artifact_joint_pass": bool(expected["joint_pass"]),
            "replay_joint_pass": bool(observed["joint_pass"]),
        })
    return comparisons


def _run_job(arguments: tuple[dict[str, Any], int, int, int]) -> dict[str, Any]:
    return regrid._run_profile(arguments)


def _run_legacy_job(arguments: tuple[int, int]) -> dict[str, Any]:
    replicates, seed = arguments
    previous_days = simulation.CALENDAR_DAYS
    previous_gap = simulation.SESSION_GAP_DAYS
    previous_anchor = simulation.ANCHOR_SHOTS_PER_SETTING
    simulation.CALENDAR_DAYS = regrid.SESSION_COUNT
    simulation.SESSION_GAP_DAYS = 1
    simulation.ANCHOR_SHOTS_PER_SETTING = ANCHOR_SHOTS
    try:
        rows = legacy_evaluation._run_profile((int(LEGACY_RATE), LEGACY_OVERHEAD, replicates, seed))
        control = cell("legacy_all_pass_control", LEGACY_RATE, LEGACY_OVERHEAD)
        for row in rows:
            row.update({
                **control,
                "session_count": regrid.SESSION_COUNT,
                "session_gap_days": 1,
            })
        return {"cell": control, "session_count": regrid.SESSION_COUNT, "session_gap_days": 1, "rows": rows}
    finally:
        simulation.CALENDAR_DAYS = previous_days
        simulation.SESSION_GAP_DAYS = previous_gap
        simulation.ANCHOR_SHOTS_PER_SETTING = previous_anchor


def _run_map_refinement(arguments: tuple[int, int, int]) -> dict[str, Any]:
    gap, replicates, seed = arguments
    previous_days = simulation.CALENDAR_DAYS
    previous_gap = simulation.SESSION_GAP_DAYS
    previous_anchor = simulation.ANCHOR_SHOTS_PER_SETTING
    simulation.CALENDAR_DAYS = regrid.SESSION_COUNT
    simulation.SESSION_GAP_DAYS = int(gap)
    simulation.ANCHOR_SHOTS_PER_SETTING = ANCHOR_SHOTS
    try:
        row = simulation.run_map_endpoint(CORRECTED_RATE, replicates, seed, CORRECTED_OVERHEAD)
        size_successes = int(round(float(row["size"]) * replicates))
        power_successes = int(round(float(row["power"]) * replicates))
        return {
            **row,
            "session_count": regrid.SESSION_COUNT,
            "session_gap_days": gap,
            "anchor_shots_per_setting": ANCHOR_SHOTS,
            "replicates_per_side": replicates,
            "size_successes": size_successes,
            "power_successes": power_successes,
            "size_mc_ci95": list(regrid.wilson_interval(size_successes, replicates)),
            "power_mc_ci95": list(regrid.wilson_interval(power_successes, replicates)),
            "joint_pass": bool(row["size_pass"] and row["power_pass"]),
        }
    finally:
        simulation.CALENDAR_DAYS = previous_days
        simulation.SESSION_GAP_DAYS = previous_gap
        simulation.ANCHOR_SHOTS_PER_SETTING = previous_anchor


def classify(
    legacy_comparison: list[Mapping[str, Any]],
    corrected_rows: list[Mapping[str, Any]],
    refined_map_rows: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    legacy_exact = all(
        bool(row["size_exact_match"])
        and bool(row["power_exact_match"])
        and bool(row["replay_joint_pass"])
        for row in legacy_comparison
    )
    map_rows = list(refined_map_rows or [row for row in corrected_rows if row["endpoint"] == "worth_sensing_map"])
    corrected_map_pass = len(map_rows) == len(SESSION_GAPS) and all(bool(row["joint_pass"]) for row in map_rows)
    if not legacy_exact:
        verdict = "legacy_baseline_not_reproduced_pipeline_regression"
    elif corrected_map_pass:
        verdict = "pipeline_intact_new_R_c_regime_effect_supported"
    else:
        verdict = "old_domain_not_passed_pipeline_regression_requires_bisection"
    return {
        "legacy_five_endpoint_exact_reproduction": legacy_exact,
        "corrected_old_domain_map_all_gaps_pass": corrected_map_pass,
        "verdict": verdict,
        "next_action": (
            "test the high-R/high-c economic-separation mechanism without changing endpoints"
            if verdict == "pipeline_intact_new_R_c_regime_effect_supported"
            else "bisect gap semantics, runner wrapping, and endpoint implementation before interpreting new-domain power"
        ),
    }


def run(
    *,
    output: Path,
    replicates: int,
    refinement_replicates: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite A/B output: {output}")
    if seed != SEED:
        raise ValueError(f"root-cause A/B seed is frozen at {SEED}")
    artifact_rows = legacy_artifact_rows()
    corrected_jobs = [
        (cell("corrected_old_domain", CORRECTED_RATE, CORRECTED_OVERHEAD), gap, replicates, seed)
        for gap in SESSION_GAPS
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(corrected_jobs) + 1)) as executor:
            legacy_future = executor.submit(_run_legacy_job, (replicates, seed))
            corrected_reports = list(executor.map(_run_job, corrected_jobs))
            legacy_report = legacy_future.result()
    else:
        legacy_report = _run_legacy_job((replicates, seed))
        corrected_reports = [_run_job(job) for job in corrected_jobs]
    legacy_comparison = compare_legacy_rows(legacy_report["rows"], artifact_rows)
    reports = [legacy_report, *corrected_reports]
    rows = [row for report in reports for row in report["rows"]]
    rows.sort(key=lambda row: (str(row["cell_role"]), int(row["session_gap_days"]), str(row["endpoint"])))
    corrected_rows = [row for row in rows if row["cell_role"] == "corrected_old_domain"]
    initial_map_rows = [row for row in corrected_rows if row["endpoint"] == "worth_sensing_map"]
    refinement_gaps: list[int] = []
    for row in initial_map_rows:
        successes = int(round(float(row["power"]) * replicates))
        interval = regrid.wilson_interval(successes, replicates)
        if interval[0] <= regrid.POWER_TARGET <= interval[1]:
            refinement_gaps.append(int(row["session_gap_days"]))
    refined_map_rows: list[dict[str, Any]] = []
    if refinement_gaps:
        refinement_jobs = [(gap, refinement_replicates, seed) for gap in refinement_gaps]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(refinement_jobs))) as executor:
                refined_map_rows = list(executor.map(_run_map_refinement, refinement_jobs))
        else:
            refined_map_rows = [_run_map_refinement(job) for job in refinement_jobs]
    final_map_by_gap = {int(row["session_gap_days"]): row for row in initial_map_rows}
    final_map_by_gap.update({int(row["session_gap_days"]): row for row in refined_map_rows})
    final_map_rows = [final_map_by_gap[gap] for gap in SESSION_GAPS]
    decision = classify(legacy_comparison, corrected_rows, final_map_rows)
    frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    frozen_entry = next(row for row in frozen["files"] if row["path"] == "scripts/simulate_b4_design_power.py")
    if str(frozen_entry["sha256"]).upper() != FROZEN_SIMULATION_SHA256:
        raise ValueError("frozen source lineage manifest changed")
    payload = {
        "schema": "b4_map_root_cause_two_cut_ab_v2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "thresholds_unchanged": {"size": regrid.SIZE_TARGET, "power": regrid.POWER_TARGET},
        "seed": seed,
        "replicates_per_endpoint_side": replicates,
        "adaptive_map_refinement": {
            "rule": "refine corrected old-domain map to 10000 when the 1000-run Wilson 95% power interval crosses 0.80",
            "replicates_per_side": refinement_replicates,
            "selected_session_gap_days": refinement_gaps,
            "rows": refined_map_rows,
        },
        "anchor_shots_per_setting": ANCHOR_SHOTS,
        "legacy_control": {
            "R": LEGACY_RATE,
            "c": LEGACY_OVERHEAD,
            "session_count": regrid.SESSION_COUNT,
            "session_gap_days": 1,
            "artifact": str(LEGACY_ARTIFACT),
            "artifact_sha256": digest(LEGACY_ARTIFACT),
            "artifact_source_sha256": FROZEN_SIMULATION_SHA256,
            "current_source_sha256": digest(SIMULATION_SOURCE),
            "source_note": (
                "Frozen source bytes are no longer present in the worktree. The captured inverse patch restores "
                "the same 3-session/1-day schedule semantics; exact five-endpoint artifact reproduction is the control."
            ),
            "comparison": legacy_comparison,
        },
        "corrected_old_domain": {
            "R": CORRECTED_RATE,
            "c": CORRECTED_OVERHEAD,
            "session_count": regrid.SESSION_COUNT,
            "session_gap_days": list(SESSION_GAPS),
        },
        "decision": decision,
        "rows": rows,
    }
    output.mkdir(parents=True)
    (output / "map_root_cause_ab.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields, normalized = regrid.normalize_row_fields(rows)
    with (output / "map_root_cause_ab.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--refinement-replicates", type=int, default=REFINEMENT_REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    report = run(
        output=arguments.output,
        replicates=arguments.replicates,
        refinement_replicates=arguments.refinement_replicates,
        seed=arguments.seed,
        workers=max(1, arguments.workers),
    )
    print(json.dumps(report["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
