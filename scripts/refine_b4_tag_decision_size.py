#!/usr/bin/env python3
"""Run 40,000 e0 refinements only for measured/exact-leverage tag cells."""

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

from scripts import refine_b4_4day_gap_size as refinement


def selected_profiles(regrid_report: dict[str, Any], exact_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row for row in regrid_report["rows"]
        if row["cell_role"] == "measured_point"
    ] + list(exact_report["rows"])
    return refinement.select_refinements({"rows": rows})


def run(
    *,
    regrid_source: Path,
    exact_source: Path,
    output: Path,
    replicates: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tag size refinement: {output}")
    regrid_report = json.loads(regrid_source.read_text(encoding="utf-8"))
    exact_report = json.loads(exact_source.read_text(encoding="utf-8"))
    if regrid_report.get("schema") != "b4_regrid_measured_points_five_endpoints_v2":
        raise ValueError("regrid source must use gap-corrected v2 schema")
    if exact_report.get("schema") != "b4_exact_anchor_shot_leverage_v1":
        raise ValueError("exact source must use exact anchor leverage v1 schema")
    profiles = selected_profiles(regrid_report, exact_report)
    jobs = [(profile, replicates, seed) for profile in profiles]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            rows = list(executor.map(refinement._run_one, jobs))
    else:
        rows = [refinement._run_one(job) for job in jobs]
    payload = {
        "schema": "b4_tag_decision_4day_gap_size_refinement_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "simulation_only": True,
        "regrid_source": str(regrid_source.resolve()),
        "regrid_source_sha256": refinement.digest(regrid_source),
        "exact_leverage_source": str(exact_source.resolve()),
        "exact_leverage_source_sha256": refinement.digest(exact_source),
        "selection_rule": (
            "session_gap_days=4; measured_point or exact_anchor_shot_leverage only; "
            "1000-run Wilson 95% size interval includes 0.05; T* and interior-null share one null refinement"
        ),
        "replicates_per_selected_e0": replicates,
        "seed": seed,
        "selected_profile_count": len(profiles),
        "selected_profiles": profiles,
        "rows": rows,
    }
    output.mkdir(parents=True)
    (output / "tag_size_refinement.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = sorted({key for row in rows for key in row})
    with (output / "tag_size_refinement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regrid-source", type=Path, required=True)
    parser.add_argument("--exact-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    report = run(
        regrid_source=arguments.regrid_source,
        exact_source=arguments.exact_source,
        output=arguments.output,
        replicates=arguments.replicates,
        seed=arguments.seed,
        workers=max(1, arguments.workers),
    )
    print(json.dumps({"selected_profiles": report["selected_profile_count"], "rows": len(report["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
