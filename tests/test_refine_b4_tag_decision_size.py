from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refine_b4_tag_decision_size", ROOT / "scripts" / "refine_b4_tag_decision_size.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def row(role: str, endpoint: str, interval: list[float]) -> dict[str, object]:
    return {
        "backend_id": "tianyan176",
        "cell_role": role,
        "shot_rate_per_second": 3792.61,
        "fixed_overhead_seconds_per_setting": 12.952,
        "anchor_shots_per_setting": 1024 if role == "measured_point" else 7840,
        "session_gap_days": 4,
        "endpoint": endpoint,
        "size": 0.043,
        "size_mc_ci95": interval,
    }


def test_selection_uses_only_measured_and_exact_tag_cells() -> None:
    regrid = {"rows": [
        row("measured_point", "interior_optimum_null", [0.03, 0.06]),
        row("local_neighbour", "interior_optimum_null", [0.03, 0.06]),
    ]}
    exact = {"rows": [row("exact_anchor_shot_leverage", "event_not_misread_as_continuous", [0.03, 0.06])]}
    selected = module.selected_profiles(regrid, exact)
    assert len(selected) == 2
    assert {profile["cell_role"] for profile in selected} == {"measured_point", "exact_anchor_shot_leverage"}
