"""在线复算冻结终测结果，不触碰任何私有证据。

与 `tools/reproduce_public_final.py` 逻辑一致，但这里把它做成可复用的
纯函数（供评审在线上按钮触发，也供 CLI ``aemtn reproduce-final`` 调用）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .paths import evidence_dir


def _load_rows(base: Path) -> list[dict[str, str]]:
    ev = base / "B4_T176_HYBRID_FINAL_20260829"
    return list(csv.DictReader((ev / "hybrid_pair_rows.csv").open(encoding="utf-8-sig", newline="")))


def _gate(rows: list[dict[str, str]]):
    from src.adaptive.cadence_permutation import cadence_ratio_permutation_gate

    fast = [float(r["fast_endpoint_squared_residual"]) for r in rows]
    slow = [float(r["slow_endpoint_squared_residual"]) for r in rows]
    return cadence_ratio_permutation_gate(fast, slow, permutations=20000, seed=20260815)


def reproduce_final(base: Path | None = None) -> dict[str, object]:
    """重算 hybrid 三个证据层，返回结构化结果，并与冻结报告对比。

    Returns:
        形如 ``{"hardware_session0_diagnostic": {...}, ...}`` 的字典；每个条目含
        ``pairs``、``ratio``、``relative_reduction``、``critical_ratio``、
        ``p_value``、``passed``，以及 ``matches_frozen``（是否与冻结值一致）。
    """
    base = base or evidence_dir()
    rows = _load_rows(base)
    report = json.loads(
        (base / "B4_T176_HYBRID_FINAL_20260829" / "hybrid_final_report.json").read_text(encoding="utf-8")
    )
    groups = {
        "hardware_session0_diagnostic": [r for r in rows if r["evidence_origin"] == "hardware_session0"],
        "simulation_session1_diagnostic": [r for r in rows if r["evidence_origin"] == "simulation_session1"],
        "hybrid_primary": rows,
    }
    results: dict[str, object] = {}
    for name, group in groups.items():
        gate = _gate(group)
        matches = True
        for key in ("ratio", "p_value", "critical_ratio"):
            exp = float(report["statistical_findings"][name][key])
            if abs(float(gate[key]) - exp) > 1e-14:
                matches = False
        results[name] = {
            "pairs": int(gate["pair_count"]),
            "ratio": float(gate["ratio"]),
            "relative_reduction": 1.0 - float(gate["ratio"]),
            "critical_ratio": float(gate["critical_ratio"]),
            "p_value": float(gate["p_value"]),
            "passed": bool(gate["passed"]),
            "matches_frozen": matches,
        }
    return results
