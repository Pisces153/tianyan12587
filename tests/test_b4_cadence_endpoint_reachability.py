"""Config-derived reachability of the registered cadence endpoint.

The previous reachability test applied a synthetic 0.45x effect on the mirror-loss scale,
which cannot detect a shots-per-setting choice that buries the cadence contrast under a
common estimator-noise floor.  These tests read the shots per setting out of the config,
convert them into the noise floor they actually produce and push the result through the
frozen gate, so a bad shot level fails before any hardware time is spent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import simulate_b4_cadence_endpoint_power as endpoint


ROOT = Path(__file__).resolve().parents[1]
DEFECTIVE_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v2.json"
AMENDED_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v3.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_shot_floor_matches_both_calibration_arms() -> None:
    for arm in endpoint.CALIBRATION_ARMS:
        modelled = endpoint.estimator_shot_floor(int(arm["shots_per_setting"]))
        observed = float(arm["observed_shot_floor"])
        assert abs(modelled - observed) / observed < 0.01


def test_amended_config_reaches_the_registered_endpoint() -> None:
    report = endpoint.evaluate_config(load(AMENDED_CONFIG), replicates=8000, seed=20260815)
    assert report["shots_per_setting"] == 14336
    assert report["minimum_adjudicated_cycle_pairs"] == 80
    assert report["reachable"] is True
    binding = next(row for row in report["rows"] if row["pair_count"] == 80)
    assert binding["power"] >= 0.8
    assert binding["boundary_size"] <= 0.05


def test_amended_config_survives_a_prudent_unmodelled_floor() -> None:
    report = endpoint.evaluate_config(
        load(AMENDED_CONFIG),
        extra_floor=2.0e-4,
        replicates=8000,
        seed=20260815,
    )
    binding = next(row for row in report["rows"] if row["pair_count"] == 80)
    assert binding["power"] >= 0.8


def test_defective_three_thousand_shot_config_is_rejected() -> None:
    """The v2 plan is unreachable and must be caught by this test, not by hardware time."""
    config = load(DEFECTIVE_CONFIG)
    correction = config["collection_correction"]
    assert int(correction["sensing_shots_per_setting"]) == 3072
    # v2 predates the reachability criteria, so supply the registered thresholds it was
    # implicitly claiming: the frozen T-B1 cell reported power 0.910 and size 0.000.
    correction["minimum_power"] = 0.8
    correction["maximum_boundary_size"] = 0.05
    report = endpoint.evaluate_config(config, replicates=8000, seed=20260815)
    binding = next(row for row in report["rows"] if row["pair_count"] == 24)
    assert report["reachable"] is False
    assert binding["power"] < 0.6
    assert report["expected_ratio"] > 0.6


def test_boundary_null_is_not_the_frozen_simulation_null() -> None:
    """The frozen T-B1 cadence cell evaluates its size four times away from the boundary."""
    fast_shot_variance = 0.25 / (490.0 * 90.0)
    slow_shot_variance = 0.25 / (490.0 * 360.0)
    assert fast_shot_variance / slow_shot_variance == pytest.approx(4.0)
    # Fixing the shots per setting removes that asymmetry, so the corrected collection's
    # null sits at ratio 1.0 and its size has to be established on the boundary.
    config = load(AMENDED_CONFIG)
    report = endpoint.evaluate_config(config, replicates=8000, seed=20260815)
    assert report["pure_drift_ratio"] == pytest.approx(0.3708924335436652, rel=1e-9)
    assert 0.0 < report["expected_ratio"] < 0.55
