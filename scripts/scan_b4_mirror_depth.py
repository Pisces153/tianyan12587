#!/usr/bin/env python3
"""T-B6.4: raw-count mirror-depth ladder and deterministic depth selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import b4_dry_run_common as common
from src.adaptive import task_metric_mirror


DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\hardware\B4_TB6\mirror_depth.json")


def select_depth(rows: Sequence[Mapping[str, Any]]) -> int | None:
    candidates = [row for row in rows if 0.3 <= float(row["median_success_probability"]) <= 0.7]
    if not candidates:
        return None
    return int(min(candidates, key=lambda row: (abs(float(row["median_success_probability"]) - 0.5), int(row["depth"])))["depth"])


def execute(config_path: Path, output: Path, *, confirm_hardware: bool, depths: Sequence[int] = (2, 4, 8, 16, 32),
            seeds_per_depth: int = 4, shots: int = 1024,
            platform_factory: Callable[[Mapping[str, Any]], Any] = common.platform_from_config) -> dict[str, Any]:
    if not confirm_hardware:
        raise RuntimeError("mirror-depth scan requires --confirm-hardware")
    config = common.load_config(config_path)
    manifest = task_metric_mirror.build_depth_ladder_manifest(
        config["backend"]["physical_qubits"], depths, seeds_per_depth=seeds_per_depth, seed=20260804
    )
    platform = platform_factory(config)
    job, results = common.run_job(
        platform=platform,
        config=config,
        circuits=[str(row["qcis"]) for row in manifest["tasks"]],
        shots_per_setting=shots,
        name=f"XA202609_B4_TB6_MIRROR_DEPTH_{config['backend']['backend_id']}",
        max_wait_seconds=900,
        poll_seconds=5,
    )
    by_id = {str(row["experimentTaskId"]): row for row in results}
    observations: list[dict[str, Any]] = []
    for query_id, task in zip(job["query_ids"], manifest["tasks"], strict=True):
        counts = common.raw_counts(by_id[query_id], task["physical_qubits"], shots)
        score = task_metric_mirror.success_probability_from_raw_counts(
            counts,
            task["ideal_bitstring"],
            shots=shots,
        )
        observations.append({
            "task_id": task["task_id"],
            "depth": task["depth"],
            "seed": task["seed"],
            "success_probability": score["success_probability"],
            "success_count": score["success_count"],
            "shots": shots,
        })
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in observations:
        grouped[int(row["depth"])].append(float(row["success_probability"]))
    depth_rows = [
        {
            "depth": depth,
            "median_success_probability": statistics.median(values),
            "minimum_success_probability": min(values),
            "maximum_success_probability": max(values),
            "replicates": len(values),
        }
        for depth, values in sorted(grouped.items())
    ]
    selected = select_depth(depth_rows)
    report = {
        "schema": "b4_tb6_mirror_depth_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend_id": config["backend"]["backend_id"],
        "job": job,
        "observations": observations,
        "depth_summary": depth_rows,
        "selected_depth": selected,
        "selection_passed": selected is not None,
        "selection_rule": "among depths with median raw-count success in [0.3,0.7], choose closest to 0.5; tie chooses shallower depth",
        "minimum_actual_improvement_rule": "freeze as a multiplier of the analytic paired-binomial noise floor at selected depth and frozen shots; no absolute probability threshold",
    }
    common.write_new(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--seeds-per-depth", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()
    report = execute(arguments.config, arguments.output, confirm_hardware=arguments.confirm_hardware, depths=arguments.depths, seeds_per_depth=arguments.seeds_per_depth, shots=arguments.shots)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["selection_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
