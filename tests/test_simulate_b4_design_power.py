from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("simulate_b4_design_power", ROOT / "scripts" / "simulate_b4_design_power.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

REFINE_SPEC = importlib.util.spec_from_file_location(
    "refine_b4_map_power", ROOT / "scripts" / "refine_b4_map_power.py"
)
assert REFINE_SPEC and REFINE_SPEC.loader
refinement = importlib.util.module_from_spec(REFINE_SPEC)
sys.modules[REFINE_SPEC.name] = refinement
REFINE_SPEC.loader.exec_module(refinement)


def test_schedule_matches_b4_three_day_design_exactly() -> None:
    schedule = module.design_schedule(490, 0.0)
    readout = schedule["readout_probe_times_seconds"]
    reference = schedule["reference_probe_times_seconds"]
    assert len(readout) == 3 * 6
    assert len(reference) == 3 * 4
    probe_setting_seconds = module.PROBE_SHOTS_PER_SETTING / 490.0
    assert readout[0] == probe_setting_seconds
    assert np.all(np.diff(readout[:6]) > 0.0)
    expected = np.asarray(module.REFERENCE_POSITIONS_ZERO_INDEXED) * 1024 / 490.0
    assert np.allclose(reference[:4], expected)
    assert readout[6] - readout[0] == 86400.0
    assert module.ANCHOR_SHOTS_PER_SETTING == 1024
    assert module.PROBE_SHOTS_PER_SETTING == 16384
    assert module.SHOT_RATES_PER_SECOND == (490, 600, 750, 850, 1000)
    assert module.FIXED_OVERHEADS_SECONDS_PER_SETTING == (0.0, 0.5, 1.1)
    combined = module.analysis_schedule(490, 0.0)
    assert list(combined["instrument_ids"]).count("anchor_33") == 12
    assert list(combined["instrument_ids"]).count("probe_burst") == 18
    assert set(combined["shots"]) == {1024, 16384}


def test_grid_contains_all_ou_combinations_and_every_dgp() -> None:
    cells = module.cell_grid()
    assert len(cells) == 15 * (1 + 4 * 3 + 3 + 3)
    assert {cell["dgp"] for cell in cells} == set(module.DGP_NAMES)
    combinations = {
        (cell["tau_minutes"], cell["process_variance"])
        for cell in cells
        if cell["dgp"] == "ou"
        and cell["shot_rate_per_second"] == 490
        and cell["fixed_overhead_seconds_per_setting"] == 0.0
    }
    assert combinations == set((tau, variance) for tau in module.OU_TAU_MINUTES for variance in module.PROCESS_VARIANCES)


def test_step_latent_remains_step_and_artifact_is_separate() -> None:
    times = module.design_schedule()["readout_probe_times_seconds"]
    latent = module.generate_latent("step_as_ramp_artifact", times, np.random.default_rng(4))
    assert latent.calibration["latent_form"] == "piecewise_constant_step"
    assert len(np.unique(latent.probability)) == 2
    assert np.count_nonzero(np.diff(latent.probability)) == 1
    assert latent.artifact_projection is not None
    assert np.count_nonzero(np.diff(latent.artifact_projection)) > 1


def test_triggered_events_come_from_threshold_crossing_not_fixed_calendar() -> None:
    times = np.arange(0.0, 20.0 * 86400.0, 3600.0)
    first = module.generate_latent("step_triggered", times, np.random.default_rng(11))
    second = module.generate_latent("step_triggered", times, np.random.default_rng(12))
    assert first.event_indices
    assert second.event_indices
    assert first.event_indices != second.event_indices
    assert "trigger_threshold" in first.calibration
    assert np.count_nonzero(np.diff(first.probability)) == len(first.event_indices)


def test_primary_null_size_is_at_most_five_percent() -> None:
    row = module.run_cell(
        {
            "dgp": "null_flat",
            "shot_rate_per_second": 490,
            "fixed_overhead_seconds_per_setting": 0.0,
            "tau_minutes": None,
            "process_variance": None,
        },
        replicates=300,
        seed=20260804,
    )
    assert row["interior_optimum_claim_rate"] <= 0.05


def test_smoke_outputs_have_identical_json_csv_result_fields(tmp_path: Path) -> None:
    output = tmp_path / "b4_smoke"
    report = module.run(output=output, replicates=2, seed=8, workers=1, timing_profiles=((490, 0.0),))
    with (output / "simulation_summary.csv").open(encoding="utf-8", newline="") as handle:
        fields = set(next(csv.DictReader(handle)).keys())
    json_fields = set(json.loads((output / "simulation_report.json").read_text(encoding="utf-8"))["results"][0])
    assert fields == json_fields == set(report["results"][0])
    assert report["field_set_audit"] if "field_set_audit" in report else True
    assert (output / "B4_B1_CONCLUSION_20260804.md").exists()


def test_gate_module_has_no_scenario_or_truth_branch() -> None:
    source = (ROOT / "src" / "adaptive" / "sensing_economics.py").read_text(encoding="utf-8")
    assert "scenario ==" not in source
    public_path = source[source.index("def analyze_ou_sensing"):]
    assert "dgp" not in public_path
    assert "latent" not in public_path


def test_map_refinement_uses_exact_two_parameter_bins() -> None:
    rows = [
        {
            "shot_rate_per_second": rate,
            "fixed_overhead_seconds_per_setting": overhead,
            "size": 0.01,
            "power": 0.81,
            "size_pass": True,
            "power_pass": True,
        }
        for rate in module.SHOT_RATES_PER_SECOND
        for overhead in module.FIXED_OVERHEADS_SECONDS_PER_SETTING
    ]
    selected = refinement.select_lookup_cell(rows, 675.0, 0.8)
    assert selected is not None
    assert selected["shot_rate_per_second"] == 750
    assert selected["fixed_overhead_seconds_per_setting"] == 1.1
    assert refinement.select_lookup_cell(rows, 1075.0001, 1.0) is None
    assert refinement.select_lookup_cell(rows, 1000.0, -0.0001) is None
    lower, upper = refinement.wilson_interval(7920, 10000)
    assert lower < 0.792 < upper


def test_overhead_changes_anchor_lags_and_probe_duration_separately() -> None:
    slow = module.reference_offsets_seconds(490, 0.0)
    fast_with_overhead = module.reference_offsets_seconds(1000, 1.1)
    assert np.allclose(slow[1:], fast_with_overhead[1:], rtol=0.02)
    assert module.setting_duration_seconds(16384, 490, 0.0) > 30.0
    assert module.setting_duration_seconds(16384, 1000, 1.1) < 18.0


def test_session_gap_changes_spacing_without_changing_session_count() -> None:
    previous_days = module.CALENDAR_DAYS
    previous_gap = module.SESSION_GAP_DAYS
    try:
        module.CALENDAR_DAYS = 3
        module.SESSION_GAP_DAYS = 4
        schedule = module.design_schedule(490, 0.0)
        readout = schedule["readout_probe_times_seconds"]
        per_session = len(module.BURST_MINUTES)
        assert len(readout) == 3 * per_session
        assert np.isclose(readout[per_session] - readout[0], 4 * 86400.0)
        assert np.isclose(readout[2 * per_session] - readout[per_session], 4 * 86400.0)
    finally:
        module.CALENDAR_DAYS = previous_days
        module.SESSION_GAP_DAYS = previous_gap
