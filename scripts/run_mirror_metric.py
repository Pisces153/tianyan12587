#!/usr/bin/env python3
"""Score B-4 mirror-task raw counts and paired strategy performance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive import task_metric_mirror


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(input_path: Path, config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mirror metric output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "b4_mirror_metric_v1":
        raise ValueError("unexpected mirror metric config schema")
    bootstrap = config["bootstrap"]
    report = task_metric_mirror.compare_strategies(
        _read_jsonl(input_path),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
        confidence_level=float(bootstrap["confidence_level"]),
        minimum_actual_improvement=float(config["endpoint"]["minimum_actual_improvement"]),
    )
    report["schema"] = "b4_mirror_metric_report_v1"
    report["config"] = {
        "path": str(config_path.resolve()),
        "primary_metric": config["endpoint"]["primary_metric"],
        "primary_source": config["endpoint"]["primary_source"],
        "minimum_actual_improvement_status": config["endpoint"]["minimum_actual_improvement_status"],
        "selected_depth": config["depth_selection"]["selected_depth"],
        "physical_qubits_tianyan_287": config["hardware_anchor"]["physical_qubits_tianyan_287"],
    }
    output.mkdir(parents=True)
    (output / "mirror_metric_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = report["paired_rows"]
    if rows:
        with (output / "paired_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def build_manifest(config_path: Path, output: Path, depths: list[int], seeds_per_depth: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mirror manifest: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = task_metric_mirror.build_depth_ladder_manifest(
        config["hardware_anchor"]["physical_qubits_tianyan_287"],
        depths,
        seeds_per_depth=seeds_per_depth,
        seed=int(config["bootstrap"]["seed"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--config", type=Path, default=ROOT / "config" / "b4_mirror_metric_v1.json")
    score.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser("build-depth-ladder")
    build.add_argument("--config", type=Path, default=ROOT / "config" / "b4_mirror_metric_v1.json")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    build.add_argument("--seeds-per-depth", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.command == "score":
        payload = run(arguments.input, arguments.config, arguments.output)["endpoint"]
    else:
        payload = build_manifest(arguments.config, arguments.output, arguments.depths, arguments.seeds_per_depth)
        payload = {"task_count": len(payload["tasks"]), "depths": payload["depths"], "output": str(arguments.output)}
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
