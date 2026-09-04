from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("regrid_b4_endpoints", ROOT / "scripts" / "regrid_b4_endpoints.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_cells_include_measured_neighbours_and_anchor_leverage() -> None:
    cells = module.timing_cells()
    assert len(cells) == 20
    assert sum(cell["cell_role"] == "measured_point" for cell in cells) == 2
    assert sum(cell["cell_role"] == "anchor_shot_leverage" for cell in cells) == 2
    assert all(cell["anchor_shots_per_setting"] == 1024 for cell in cells if not cell["anchor_shot_leverage"])
    assert module.anchor_shots_for_rate(1486.09) > 1024
    assert module.anchor_shots_for_rate(3792.61) > module.anchor_shots_for_rate(1486.09)


def test_smoke_regrid_writes_paired_endpoint_ledger(tmp_path: Path) -> None:
    report = module.run(output=tmp_path / "regrid", replicates=2, seed=7, workers=1)
    assert report["cell_count"] == 20
    assert report["profile_count"] == 60
    assert report["session_count"] == 3
    assert report["session_gap_day_nuisance"] == [1, 2, 4]
    assert {row["endpoint"] for row in report["rows"]} == set(module.ENDPOINT_NAMES)
    assert {row["session_gap_days"] for row in report["rows"]} == {1, 2, 4}
    assert all("size" in row and "power" in row and "size_successes_e0" in row for row in report["rows"])
    parsed = json.loads((tmp_path / "regrid" / "regrid_report.json").read_text(encoding="utf-8"))
    assert parsed["outside_domain_policy"].startswith("measured points are outside")


def test_normalize_existing_makes_json_and_csv_field_sets_identical(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "regrid_report.json").write_text(
        json.dumps({"rows": [{"a": 1}, {"a": 2, "b": 3}]}) + "\n",
        encoding="utf-8",
    )
    (output / "regrid_endpoints.csv").write_text("a,b\n1,\n2,3\n", encoding="utf-8")
    report = module.normalize_existing(output)
    assert report["field_set_audit"]["passed"] is True
    assert all(set(row) == {"a", "b"} for row in report["rows"])
