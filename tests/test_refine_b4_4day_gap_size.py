from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refine_b4_4day_gap_size", ROOT / "scripts" / "refine_b4_4day_gap_size.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def source_row(endpoint: str, interval: list[float]) -> dict[str, object]:
    return {
        "backend_id": "tianyan-287",
        "cell_role": "measured_point",
        "shot_rate_per_second": 1486.09,
        "fixed_overhead_seconds_per_setting": 6.990,
        "anchor_shots_per_setting": 1024,
        "session_gap_days": 4,
        "endpoint": endpoint,
        "size": 0.051,
        "size_mc_ci95": interval,
    }


def test_selection_deduplicates_tstar_and_interior_null() -> None:
    report = {
        "rows": [
            source_row("interior_optimum_null", [0.04, 0.06]),
            source_row("tstar_regret_c1p25_conditional", [0.04, 0.06]),
            source_row("worth_sensing_map", [0.0, 0.01]),
        ]
    }
    selected = module.select_refinements(report)
    assert len(selected) == 1
    assert selected[0]["endpoint_family"] == "interior_optimum_null"
    assert selected[0]["source_endpoints"] == ["interior_optimum_null", "tstar_regret_c1p25_conditional"]


def test_smoke_writes_gap_corrected_refinement(tmp_path: Path) -> None:
    source = tmp_path / "regrid_report.json"
    source.write_text(json.dumps({
        "schema": "b4_regrid_measured_points_five_endpoints_v2",
        "rows": [source_row("interior_optimum_null", [0.04, 0.06])],
    }), encoding="utf-8")
    report = module.run(
        source=source,
        output=tmp_path / "refinement",
        replicates=2,
        seed=5,
        workers=1,
    )
    assert report["selected_profile_count"] == 1
    assert report["rows"][0]["session_count"] == 3
    assert report["rows"][0]["session_gap_days"] == 4
    assert (tmp_path / "refinement" / "size_refinement.json").exists()
