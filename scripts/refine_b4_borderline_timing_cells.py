#!/usr/bin/env python3
"""Refine timing-grid cells whose 10k map-power interval crosses 0.8."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diagnose_b4_map_timing as diagnostic


DEFAULT_OUTPUT = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_borderline_timing_cells_20260804"
)
PROFILES = ((600, 0.0), (750, 0.5))


def run(output: Path, replicates: int, seed: int, workers: int) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite borderline refinement: {output}")
    arguments = [(rate, overhead, replicates, seed) for rate, overhead in PROFILES]
    with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
        results = list(executor.map(diagnostic._run_profile, arguments))
    summaries = [diagnostic.summarize(result) for result in results]
    report = {
        "schema": "b4_map_borderline_timing_cells_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_rule": "10k Wilson 95% interval crossed power target 0.8",
        "replicates_per_profile": replicates,
        "seed": seed,
        "summaries": summaries,
        "replicate_sources": [],
    }
    output.mkdir(parents=True)
    for result in results:
        profile = result["profile"]
        name = f"replicates_R{profile['shot_rate_per_second']}_c{profile['fixed_overhead_seconds_per_setting']:.1f}.csv"
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
        report["replicate_sources"].append(str((output / name).resolve()))
    (output / "borderline_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    report = run(arguments.output, arguments.replicates, arguments.seed, max(1, arguments.workers))
    print(json.dumps({"summaries": report["summaries"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
