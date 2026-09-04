#!/usr/bin/env python3
"""T-B6.2: ten minimal probe-job submit-to-result roundtrips."""

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


DEFAULT_OUTPUT = Path(r"E:\TianYan\XA-202609\artifacts\hardware\B4_TB6\interface_floor.json")


def execute(config_path: Path, output: Path, *, confirm_hardware: bool, jobs: int = 10, shots: int = 1024,
            platform_factory: Callable[[Mapping[str, Any]], Any] = common.platform_from_config) -> dict[str, Any]:
    if not confirm_hardware or jobs != 10:
        raise RuntimeError("interface-floor measurement requires --confirm-hardware and exactly 10 jobs")
    config = common.load_config(config_path)
    programs = drift_campaign_v4.build_probe_programs(config)
    platform = platform_factory(config)
    rows: list[dict[str, Any]] = []
    for index in range(jobs):
        record, _ = common.run_job(
            platform=platform,
            config=config,
            circuits=[str(row["qcis"]) for row in programs],
            shots_per_setting=shots,
            name=f"XA202609_B4_TB6_FLOOR_{config['backend']['backend_id']}_{index:02d}",
            max_wait_seconds=600,
            poll_seconds=2,
        )
        rows.append(record)
    durations = [float(row["roundtrip_seconds"]) for row in rows]
    report = {
        "schema": "b4_tb6_interface_floor_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend_id": config["backend"]["backend_id"],
        "jobs": rows,
        "P50_seconds": common.percentile(durations, 0.50),
        "P90_seconds": common.percentile(durations, 0.90),
        "T_fast_freeze_rule": "T_fast >= 2 * measured interface-floor P90",
    }
    common.write_new(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()
    print(json.dumps(execute(arguments.config, arguments.output, confirm_hardware=arguments.confirm_hardware), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
