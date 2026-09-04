#!/usr/bin/env python3
"""Per-replicate map mechanism diagnostics at measured B-4 timing points."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diagnose_b4_map_timing as diagnostic
from scripts import regrid_b4_endpoints as regrid
from scripts import run_b4_exact_anchor_leverage as exact_leverage
from scripts import simulate_b4_design_power as simulation


def profiles() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in regrid.MEASURED_POINTS:
        rate = float(point["shot_rate_per_second"])
        overhead = float(point["fixed_overhead_seconds_per_setting"])
        rows.extend([
            {
                "backend_id": point["backend_id"],
                "profile_role": "measured_point",
                "shot_rate_per_second": rate,
                "fixed_overhead_seconds_per_setting": overhead,
                "anchor_shots_per_setting": regrid.ANCHOR_BASE_SHOTS,
            },
            {
                "backend_id": point["backend_id"],
                "profile_role": "exact_anchor_shot_leverage",
                "shot_rate_per_second": rate,
                "fixed_overhead_seconds_per_setting": overhead,
                "anchor_shots_per_setting": exact_leverage.exact_shots_for_rate(rate),
            },
        ])
    return rows


def _run_one(arguments: tuple[Mapping[str, Any], int, int, int]) -> dict[str, Any]:
    profile, session_gap_days, replicates, seed = arguments
    previous_days = simulation.CALENDAR_DAYS
    previous_gap_days = simulation.SESSION_GAP_DAYS
    previous_anchor = simulation.ANCHOR_SHOTS_PER_SETTING
    simulation.CALENDAR_DAYS = regrid.SESSION_COUNT
    simulation.SESSION_GAP_DAYS = int(session_gap_days)
    simulation.ANCHOR_SHOTS_PER_SETTING = int(profile["anchor_shots_per_setting"])
    try:
        result = diagnostic._run_profile((
            float(profile["shot_rate_per_second"]),
            float(profile["fixed_overhead_seconds_per_setting"]),
            replicates,
            seed,
        ))
        for row in result["rows"]:
            row.update({
                "backend_id": profile["backend_id"],
                "profile_role": profile["profile_role"],
                "session_count": regrid.SESSION_COUNT,
                "session_gap_days": session_gap_days,
                "anchor_shots_per_setting": int(profile["anchor_shots_per_setting"]),
            })
        summary = diagnostic.summarize(result)
        summary.update({
            "backend_id": profile["backend_id"],
            "profile_role": profile["profile_role"],
            "session_count": regrid.SESSION_COUNT,
            "session_gap_days": session_gap_days,
            "anchor_shots_per_setting": int(profile["anchor_shots_per_setting"]),
        })
        return {"rows": result["rows"], "summary": summary}
    finally:
        simulation.CALENDAR_DAYS = previous_days
        simulation.SESSION_GAP_DAYS = previous_gap_days
        simulation.ANCHOR_SHOTS_PER_SETTING = previous_anchor


def run(*, output: Path, replicates: int, seed: int, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mechanism diagnostics: {output}")
    jobs = [
        (profile, days, replicates, seed)
        for profile in profiles()
        for days in regrid.SESSION_GAP_DAY_NUISANCE
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            reports = list(executor.map(_run_one, jobs))
    else:
        reports = [_run_one(job) for job in jobs]
    rows = [row for report in reports for row in report["rows"]]
    rows.sort(key=lambda row: (
        str(row["backend_id"]),
        str(row["profile_role"]),
        int(row["session_gap_days"]),
        int(row["replicate"]),
    ))
    summaries = [report["summary"] for report in reports]
    summaries.sort(key=lambda row: (str(row["backend_id"]), str(row["profile_role"]), int(row["session_gap_days"])))
    payload = {
        "schema": "b4_regrid_mechanism_per_replicate_v2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "replicates_per_profile": replicates,
        "seed": seed,
        "profile_count": len(jobs),
        "replicate_row_count": len(rows),
        "diagnostic_scope": "worth_sensing_map e1 OU(variance=6.1e-4, tau alternating 15/30 min); e0 size remains in regrid_report.json",
        "summaries": summaries,
    }
    output.mkdir(parents=True)
    with (output / "replicate_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "diagnostic_summaries.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output / "diagnostic_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B-4 regrid mechanism diagnostics",
        "",
        f"{len(jobs)} profiles, {replicates} e1 replicates/profile. Raw per-replicate gate rows are preserved in `replicate_diagnostics.csv`.",
        "",
        "| backend | role | session gap days | power | detection fail | OU fit fail | economic fail | tau CI width median s | T* median s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['backend_id']} | {row['profile_role']} | {row['session_gap_days']} | {row['power']:.4f} | "
            f"{row['detection_failure_rate']:.4f} | {row['ou_fit_failure_rate']:.4f} | "
            f"{row['economic_separation_failure_rate']:.4f} | {row['tau_ci_width_seconds_median']:.4f} | "
            f"{row['t_star_seconds_median']:.4f} |"
        )
    (output / "MECHANISM_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    report = run(output=args.output, replicates=args.replicates, seed=args.seed, workers=max(1, args.workers))
    print(json.dumps({"profiles": report["profile_count"], "rows": report["replicate_row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
