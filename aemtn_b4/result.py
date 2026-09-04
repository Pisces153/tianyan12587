"""加载冻结的终测证据，返回结构化对象。

只读取随包发布的**公开派生** JSON/CSV（本身不含私有 raw counts/NPZ），
并附上不可省略的结论边界。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .paths import PROJECT_ROOT, evidence_dir

# 随包分发的 12 张主图目录
_FIGURES = PROJECT_ROOT / "figures" / "single_figures_v2_diverse_20260831"


@dataclass(frozen=True)
class HybridResult:
    """单个证据层的冻结统计量（ratio / p / critical / reduction）。"""

    layer: str
    pairs: int
    ratio: float
    relative_reduction: float
    critical_ratio: float
    p_value: float
    passed: bool

    def to_row(self) -> dict[str, object]:
        return {
            "证据层": self.layer,
            "pairs": self.pairs,
            "ratio": f"{self.ratio:.9f}",
            "相对降低": f"{100.0 * self.relative_reduction:.2f}%",
            "critical_ratio": f"{self.critical_ratio:.9f}",
            "p 值": f"{self.p_value:.6f}",
            "通过": self.passed,
        }


@dataclass(frozen=True)
class FinalReport:
    """B4/T176 hybrid final 的完整只读视图。"""

    analysis_label: str
    project_status: str
    registered_hardware_status: str
    confidence: str
    decision: dict[str, object]
    pair_composition: dict[str, object]
    findings: dict[str, object]
    rows: list[dict[str, object]]
    claim_boundary: dict[str, object]

    @property
    def layers(self) -> list[HybridResult]:
        _LAYER_LABEL = {
            "hardware_session0_diagnostic": "真机 Session 0（诊断）",
            "hybrid_primary": "混合闭环（primary）",
            "simulation_session1_diagnostic": "模拟 Session 1（反事实）",
        }
        out: list[HybridResult] = []
        for key, val in self.findings.items():
            if "pair_count" not in val:
                continue  # 非 pair 统计层（如 multiple_comparisons / prediction_check）
            out.append(
                HybridResult(
                    layer=str(_LAYER_LABEL.get(key, val.get("schema", key))),
                    pairs=int(val["pair_count"]),
                    ratio=float(val["ratio"]),
                    relative_reduction=float(val["relative_reduction"]),
                    critical_ratio=float(val["critical_ratio"]),
                    p_value=float(val["p_value"]),
                    passed=bool(val["passed"]),
                )
            )
        return out


def load_final_report(root: Path | None = None) -> FinalReport:
    """从冻结 JSON 载入终测报告与 pair 行。"""
    base = root or evidence_dir()
    ev = base / "B4_T176_HYBRID_FINAL_20260829"
    report = json.loads((ev / "hybrid_final_report.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((ev / "hybrid_pair_rows.csv").open(encoding="utf-8-sig", newline="")))
    clean_rows: list[dict[str, object]] = []
    for row in rows:
        clean_rows.append(
            {
                "pair": row.get("analysis_pair_id", ""),
                "origin": row.get("evidence_origin", ""),
                "block_order": row.get("block_order", ""),
                "fast": float(row["fast_endpoint_squared_residual"]),
                "slow": float(row["slow_endpoint_squared_residual"]),
            }
        )
    decision = report.get("decision", {})
    return FinalReport(
        analysis_label=str(report.get("analysis_label", "")),
        project_status=str(decision.get("simulation_assisted_status", "")),
        registered_hardware_status=str(
            decision.get("registered_hardware_endpoint_status", "")
        ),
        confidence=str(decision.get("hardware_session0_evidence_grade", "")),
        decision=decision,
        pair_composition=report.get("pair_composition", {}),
        findings=report.get("statistical_findings", {}),
        rows=clean_rows,
        claim_boundary=report.get("claim_boundary", {}),
    )


def load_figure_source(name: str) -> list[dict[str, object]]:
    """读取单张主图对应的 source CSV（带 UTF-8 BOM 兼容）。"""
    path = _FIGURES / f"{name}.source.csv"
    if not path.is_file():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    return [dict(r) for r in rows]
