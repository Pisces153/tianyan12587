#!/usr/bin/env python3
"""Recompute the public, derived B4 final ratios and permutation diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive.cadence_permutation import cadence_ratio_permutation_gate


def close(left: float, right: float, tolerance: float = 1e-14) -> bool:
    return bool(abs(left - right) <= tolerance)


def gate(rows):
    fast = np.asarray([float(row["fast_endpoint_squared_residual"]) for row in rows])
    slow = np.asarray([float(row["slow_endpoint_squared_residual"]) for row in rows])
    return cadence_ratio_permutation_gate(fast, slow, permutations=20000, seed=20260815)


def main() -> int:
    evidence = ROOT / "evidence" / "B4_T176_HYBRID_FINAL_20260829"
    rows = list(csv.DictReader((evidence / "hybrid_pair_rows.csv").open("r", encoding="utf-8-sig", newline="")))
    report = json.loads((evidence / "hybrid_final_report.json").read_text(encoding="utf-8"))
    groups = {
        "hardware_session0_diagnostic": [row for row in rows if row["evidence_origin"] == "hardware_session0"],
        "simulation_session1_diagnostic": [row for row in rows if row["evidence_origin"] == "simulation_session1"],
        "hybrid_primary": rows,
    }
    results = {name: gate(group) for name, group in groups.items()}
    expected = report["statistical_findings"]
    failures = []
    for name, result in results.items():
        for key in ("ratio", "p_value", "critical_ratio"):
            if not close(float(result[key]), float(expected[name][key])):
                failures.append(f"{name}.{key}")
    payload = {
        name: {
            "pairs": int(result["pair_count"]),
            "ratio": float(result["ratio"]),
            "relative_reduction": 1.0 - float(result["ratio"]),
            "critical_ratio": float(result["critical_ratio"]),
            "p_value": float(result["p_value"]),
            "passed": bool(result["passed"]),
        }
        for name, result in results.items()
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        print("PUBLIC FINAL REPRODUCTION FAILED: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("PUBLIC FINAL REPRODUCTION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
