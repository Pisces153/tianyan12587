from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_b4_regrid_mechanism", ROOT / "scripts" / "diagnose_b4_regrid_mechanism.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_profiles_cover_two_backends_and_anchor_leverage() -> None:
    rows = module.profiles()
    assert len(rows) == 4
    assert {row["backend_id"] for row in rows} == {"tianyan-287", "tianyan176"}
    assert {row["profile_role"] for row in rows} == {"measured_point", "exact_anchor_shot_leverage"}


def test_smoke_writes_true_per_replicate_rows(tmp_path: Path) -> None:
    report = module.run(output=tmp_path / "diagnostics", replicates=1, seed=5, workers=1)
    assert report["profile_count"] == 12
    assert report["replicate_row_count"] == 12
    assert (tmp_path / "diagnostics" / "replicate_diagnostics.csv").exists()
