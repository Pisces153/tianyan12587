#!/usr/bin/env python3
"""Run the exact anchor-shot leverage requested for the B-4 tag decision."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import regrid_b4_endpoints as regrid
from scripts import simulate_b4_design_power as simulation


REFERENCE_RATE = 1486.09
REFERENCE_SHOTS_PER_SETTING = 3072


def exact_shots_for_rate(rate: float) -> int:
    return max(1, int(round(REFERENCE_SHOTS_PER_SETTING * float(rate) / REFERENCE_RATE)))


def exact_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for point in regrid.MEASURED_POINTS:
        rate = float(point["shot_rate_per_second"])
        overhead = float(point["fixed_overhead_seconds_per_setting"])
        shots = exact_shots_for_rate(rate)
        setting_seconds = simulation.setting_duration_seconds(shots, rate, overhead)
        cells.append({
            "backend_id": point["backend_id"],
            "point_label": point["backend_id"],
            "neighbour_multiplier": 1.0,
            "neighbour_c_offset_seconds": 0.0,
            "shot_rate_per_second": rate,
            "fixed_overhead_seconds_per_setting": overhead,
            "cell_role": "exact_anchor_shot_leverage",
            "anchor_shot_leverage": True,
            "anchor_shots_per_setting": shots,
            "anchor_setting_seconds": setting_seconds,
            "reference_lags_seconds_positions_11_22_32": [
                multiplier * setting_seconds for multiplier in (11, 22, 32)
            ],
        })
    return cells


def run(*, output: Path, replicates: int, seed: int, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exact leverage output: {output}")
    cells = exact_cells()
    jobs = [
        (cell, gap_days, replicates, seed)
        for cell in cells
        for gap_days in regrid.SESSION_GAP_DAY_NUISANCE
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            reports = list(executor.map(regrid._run_profile, jobs))
    else:
        reports = [regrid._run_profile(job) for job in jobs]
    rows = [row for report in reports for row in report["rows"]]
    rows.sort(key=lambda row: (str(row["backend_id"]), int(row["session_gap_days"]), str(row["endpoint"])))
    for row in rows:
        size_successes = int(round(float(row["size"]) * replicates))
        power_successes = int(round(float(row["power"]) * replicates))
        row.update({
            "size_successes_e0": size_successes,
            "power_successes_e1": power_successes,
            "size_mc_ci95": list(regrid.wilson_interval(size_successes, replicates)),
            "power_mc_ci95": list(regrid.wilson_interval(power_successes, replicates)),
        })
    fields, rows = regrid.normalize_row_fields(rows)
    payload = {
        "schema": "b4_exact_anchor_shot_leverage_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "reference_rate_per_second": REFERENCE_RATE,
        "reference_shots_per_setting": REFERENCE_SHOTS_PER_SETTING,
        "formula": "round(3072 * R / 1486.09)",
        "session_count": regrid.SESSION_COUNT,
        "session_gap_day_nuisance": list(regrid.SESSION_GAP_DAY_NUISANCE),
        "replicates_per_endpoint_side": replicates,
        "seed": seed,
        "cell_count": len(cells),
        "profile_count": len(jobs),
        "cells": cells,
        "lag_mechanism_audit": (
            "At measured c, positive anchor shots cannot restore v3 23/46/67 s lags; "
            "the exact T287 3072-shot cell yields the reported longer wallclock lags."
        ),
        "rows": rows,
    }
    output.mkdir(parents=True)
    (output / "exact_anchor_leverage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "exact_anchor_leverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    report = run(
        output=arguments.output,
        replicates=arguments.replicates,
        seed=arguments.seed,
        workers=max(1, arguments.workers),
    )
    print(json.dumps({"profiles": report["profile_count"], "rows": len(report["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
