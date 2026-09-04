#!/usr/bin/env python3
"""Refresh T6/T7/T8 only after a complete dual-backend collection pair exists.

This script only reads campaign artifacts and writes fresh analysis artifacts.
It has no platform client and cannot submit, query, or modify hardware work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive.environment_proxy import extract_campaign
from src.adaptive.forecast import run_rolling_origin


def complete_pair_snapshot_ids(campaign_root: Path) -> tuple[list[str], list[str]]:
    """Return collected IDs from schedule slots complete for every registered backend."""
    rows = [json.loads(line) for line in (campaign_root / "snapshots.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((campaign_root / "campaign_manifest.json").read_text(encoding="utf-8"))
    expected = sorted(str(row["backend_id"]) for row in manifest["backend_probe_manifests"])
    submitted = {str(row["snapshot_id"]): row for row in rows if row.get("event") == "submitted"}
    collected = {str(row["snapshot_id"]) for row in rows if row.get("event") == "collected"}
    by_schedule: dict[str, dict[str, str]] = {}
    for snapshot_id in collected:
        submission = submitted.get(snapshot_id)
        if submission is None:
            continue
        by_schedule.setdefault(str(submission["scheduled_utc"]), {})[str(submission["backend_id"])] = snapshot_id
    completed = [by_schedule[slot] for slot in sorted(by_schedule) if sorted(by_schedule[slot]) == expected]
    return expected, [snapshot_id for pair in completed for snapshot_id in (pair[backend] for backend in expected)]


def _run_preflight(arguments: argparse.Namespace, *, t6_report: Path, forecast_reports: list[Path], out: Path) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "analyze_natural_drift.py"), "bandit-preflight", "--t6-report", str(t6_report), "--out", str(out), "--t3-artifact", str(arguments.t3_artifact), "--t3-conservative-prior", str(arguments.t3_conservative_prior), "--t4-fidelity-gate", str(arguments.t4_fidelity_gate)]
    for report in forecast_reports:
        command.extend(("--forecast-report", str(report)))
    subprocess.run(command, check=True)


def refresh(arguments: argparse.Namespace) -> dict[str, object]:
    backends, snapshot_ids = complete_pair_snapshot_ids(arguments.campaign_root)
    pair_count = len(snapshot_ids) // len(backends)
    if pair_count == 0:
        return {"status": "waiting_for_first_complete_dual_backend_pair", "hardware_submission_performed": False}
    suffix = f"auto_complete_{len(snapshot_ids)}snapshots_{pair_count}pairs"
    t6_root = arguments.analysis_root / f"T6_{suffix}"
    t7_root = arguments.analysis_root / f"T7_{suffix}"
    t8_root = arguments.analysis_root / f"T8_{suffix}"
    if t6_root.exists() or t7_root.exists() or t8_root.exists():
        return {"status": "already_refreshed_without_overwrite", "complete_pair_count": pair_count, "snapshot_count": len(snapshot_ids), "hardware_submission_performed": False}
    extracted = extract_campaign(arguments.campaign_root, t6_root, included_snapshot_ids=set(snapshot_ids))
    corpus = Path(str(extracted["corpus_path"]))
    reports: list[Path] = []
    for backend in backends:
        out = t7_root / backend
        run_rolling_origin(corpus, out, backend_id=backend)
        reports.append(out / "forecast_report.json")
    _run_preflight(arguments, t6_report=t6_root / "feature_extraction_report.json", forecast_reports=reports, out=t8_root)
    return {"status": "refreshed", "complete_pair_count": pair_count, "snapshot_count": len(snapshot_ids), "backends": backends, "t6": str(t6_root), "t7": str(t7_root), "t8": str(t8_root), "hardware_submission_performed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--t3-artifact", type=Path, required=True)
    parser.add_argument("--t3-conservative-prior", type=Path, required=True)
    parser.add_argument("--t4-fidelity-gate", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refresh(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
