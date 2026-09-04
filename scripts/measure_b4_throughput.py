#!/usr/bin/env python3
"""T-B6.1: same-structure 1024/16384-shot differential on both backends."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import b4_dry_run_common as common
from scripts import drift_campaign_v4


DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\hardware\B4_TB6\throughput.json")


def execute(
    config_paths: list[Path],
    output: Path,
    *,
    confirm_hardware: bool,
    platform_factory: Callable[[Mapping[str, Any]], Any] = common.platform_from_config,
) -> dict[str, Any]:
    if not confirm_hardware:
        raise RuntimeError("hardware throughput measurement requires --confirm-hardware")
    measurements: list[dict[str, Any]] = []
    estimates: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = common.load_config(config_path)
        programs = drift_campaign_v4.build_probe_programs(config)
        platform = platform_factory(config)
        backend_rows: list[dict[str, Any]] = []
        for index, shots in enumerate((1024, 16384, 16384, 1024)):
            record, _ = common.run_job(
                platform=platform,
                config=config,
                circuits=[str(row["qcis"]) for row in programs],
                shots_per_setting=shots,
                name=f"XA202609_B4_TB6_RATE_{config['backend']['backend_id']}_{shots}_{index}",
                max_wait_seconds=900,
                poll_seconds=5,
            )
            record["repeat_index"] = index
            measurements.append(record)
            backend_rows.append(record)
        estimates.append({"backend_id": config["backend"]["backend_id"], **common.estimate_rate_and_overhead(backend_rows)})
    report = {
        "schema": "b4_tb6_throughput_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "same_structure": "two frozen readout probe circuits at both shot levels",
        "measurements": measurements,
        "estimates": estimates,
    }
    common.write_new(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+", default=[ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json", ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()
    print(json.dumps(execute(arguments.configs, arguments.output, confirm_hardware=arguments.confirm_hardware), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
