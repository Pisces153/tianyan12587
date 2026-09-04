from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_b4_exact_anchor_leverage", ROOT / "scripts" / "run_b4_exact_anchor_leverage.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_exact_cells_use_3072_at_t287_and_scale_t176() -> None:
    cells = module.exact_cells()
    assert len(cells) == 2
    t287 = next(cell for cell in cells if cell["backend_id"] == "tianyan-287")
    t176 = next(cell for cell in cells if cell["backend_id"] == "tianyan176")
    assert t287["anchor_shots_per_setting"] == 3072
    assert t176["anchor_shots_per_setting"] == module.exact_shots_for_rate(3792.61)
    assert t287["reference_lags_seconds_positions_11_22_32"][0] > 23.0
    assert t176["reference_lags_seconds_positions_11_22_32"][0] > t287["reference_lags_seconds_positions_11_22_32"][0]


def test_smoke_writes_all_five_endpoints(tmp_path: Path) -> None:
    report = module.run(output=tmp_path / "exact", replicates=1, seed=5, workers=1)
    assert report["profile_count"] == 6
    assert len(report["rows"]) == 30
    assert {row["endpoint"] for row in report["rows"]} == set(module.regrid.ENDPOINT_NAMES)
    assert (tmp_path / "exact" / "exact_anchor_leverage.json").exists()
