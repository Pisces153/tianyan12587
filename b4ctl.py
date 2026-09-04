#!/usr/bin/env python3
"""Unified, safety-gated facade over the frozen AEMTN/B4 scripts."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

CORE_TESTS = [
    "tests/test_counts_features.py",
    "tests/test_training_pipeline.py",
    "tests/test_sensing_economics.py",
    "tests/test_cadence_permutation.py",
    "tests/test_task_metric_mirror.py",
    "tests/test_drift_campaign_v4.py",
    "tests/test_cadence_pair_loop.py",
    "tests/test_run_b4_cadence_pair_hardware.py",
    "tests/test_b4_session1_simulation_contingency.py",
    "tests/test_analyze_b4_t176_hybrid_final.py",
]


def run(command: list[str]) -> int:
    return subprocess.call(command, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    sub.add_parser("pipeline")
    sub.add_parser("verify")
    sub.add_parser("reproduce-final")
    test = sub.add_parser("test")
    test.add_argument("--tier", choices=("core", "full"), default="core")
    train = sub.add_parser("train")
    train.add_argument("args", nargs=argparse.REMAINDER)
    generate = sub.add_parser("generate-data")
    generate.add_argument("args", nargs=argparse.REMAINDER)
    hardware = sub.add_parser("hardware-run")
    hardware.add_argument("--allow-hardware", action="store_true")
    hardware.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command == "inventory":
        print((ROOT / "CODE_INVENTORY.md").read_text(encoding="utf-8"))
        return 0
    if args.command == "pipeline":
        print((ROOT / "PIPELINE.md").read_text(encoding="utf-8"))
        return 0
    if args.command == "verify":
        return run([sys.executable, "tools/verify_package.py"])
    if args.command == "reproduce-final":
        return run([sys.executable, "tools/reproduce_public_final.py"])
    if args.command == "test":
        targets = CORE_TESTS if args.tier == "core" else ["tests"]
        return run([sys.executable, "-m", "pytest", "-q", *targets])
    if args.command == "train":
        return run([sys.executable, "scripts/train_sim.py", *args.args])
    if args.command == "generate-data":
        return run([sys.executable, "scripts/generate_sim_dataset.py", *args.args])
    if args.command == "hardware-run":
        if not args.allow_hardware:
            print("REFUSED: hardware submission requires explicit --allow-hardware", file=sys.stderr)
            return 2
        return run([sys.executable, "scripts/run_b4_cadence_pair_hardware.py", *args.args])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
