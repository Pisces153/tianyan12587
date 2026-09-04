"""The forty-minute collection design has to fail here, not on the machine.

The v2 plan reached the point of being a submission entry with a realised power of 0.462
under a headline claim of 0.910, because nothing in the test suite recomputed its
operating characteristic from its own numbers.  This file closes that: every quantity the
v4 config asserts about itself is recomputed from the config, and the design is refused if
it does not hold.  A bad shot level, a mis-sized session or an over-budget schedule fails
before any machine time is spent.

Monte Carlo note.  Several assertions below are simulation estimates, so they are stated
against the config's declared figure plus a Monte Carlo allowance rather than against a
bare bound.  Where a design choice is being pinned -- the pair count, the registered
minimum -- the test checks the *neighbouring* value fails, which is what makes the choice
falsifiable rather than merely satisfied.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.run_cadence_pair_loop import validate_config
from src.adaptive.cadence_permutation import (
    permutation_claim_rate,
    permutation_claim_rate_under_block_offsets,
)
from src.adaptive.shared_baseline_sensing import DRIFT_SHAPES, endpoint_shot_floor

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json"


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def correction(config: dict) -> dict:
    return config["collection_correction"]


def _drift(cadence_seconds: float, config: dict) -> float:
    ou = config["controlled_ou"]
    return 4.0 * ou["stationary_process_variance"] * (
        1.0 - math.exp(-cadence_seconds / ou["tau_seconds"])
    )


def _sensing_shots(correction: dict) -> int:
    return (
        2
        * correction["cycles_per_cadence_per_session"]
        * correction["sensing_settings_per_cycle"]
        * correction["sensing_shots_per_setting"]
    )


def _baseline_shots(correction: dict) -> int:
    """Start and end measurements are sized independently; both are two settings."""
    return correction["baseline_settings_per_measurement"] * (
        correction["baseline_shots_per_setting"]
        + correction["baseline_end_shots_per_setting"]
    )


def _session_shots(correction: dict) -> int:
    """mirror_shots_per_qc_cycle covers the fixed and adaptive settings together.

    Charging it per setting is what the frozen design did before 2026-08-15, which
    over-budgeted the mirror by 16384 shots a session.
    """
    mirror = correction["mirror_qc_jobs_per_session"] * correction["mirror_shots_per_qc_cycle"]
    return _sensing_shots(correction) + _baseline_shots(correction) + mirror


def _session_jobs(correction: dict) -> int:
    return (
        2 * correction["cycles_per_cadence_per_session"]
        + correction["baseline_measurements_per_session"]
        + correction["mirror_qc_jobs_per_session"]
    )


def _session_busy(correction: dict) -> float:
    if correction.get("timing_budget_model") == "role_envelope_sum_task_runtime":
        return sum(
            correction["role_jobs_per_session"][role]
            * correction["role_settings_per_job"][role]
            * correction["role_task_runtime_seconds"][role]
            for role in ("baseline", "sense", "mirror")
        )
    return (
        _session_shots(correction) / correction["shot_rate_per_second_used"]
        + _session_jobs(correction) * correction["seconds_per_job_used"]
    )


def test_the_frozen_simulation_contract_is_untouched(config: dict) -> None:
    """The hardware schedule lives in collection_correction, not in the T-B5 contract."""
    validate_config(config)
    assert config["simulation_days"] == 3
    assert len(config["block_order_by_day"]) == 3
    assert len(config["gate_amplitude_ladder"]["amplitudes"]) == 3
    assert config["cadence"]["fast_seconds"] == 90.0
    assert config["cadence"]["slow_seconds"] == 360.0


def test_every_declared_count_is_recomputed_from_the_design(correction: dict) -> None:
    assert _session_shots(correction) == correction["shots_per_session"]
    assert _session_jobs(correction) == correction["jobs_per_session"]
    assert correction["sessions"] * _session_shots(correction) == correction["shots_total"]
    assert (
        correction["sessions"] * correction["cycles_per_cadence_per_session"]
        == correction["registered_cycle_pairs_total"]
    )
    assert (
        correction["sensing_settings_per_cycle"] * correction["sensing_shots_per_setting"]
        == correction["sensing_shots_per_cycle"]
    )
    settings = (
        2 * correction["cycles_per_cadence_per_session"] * correction["sensing_settings_per_cycle"]
        + correction["baseline_measurements_per_session"] * correction["baseline_settings_per_measurement"]
        + correction["mirror_qc_jobs_per_session"] * correction["mirror_settings_per_repetition"]
    )
    assert settings == correction["settings_per_session"]


def test_the_schedule_fits_the_forty_minute_ceiling_and_the_session_cap(
    config: dict, correction: dict
) -> None:
    busy = _session_busy(correction)
    assert busy == pytest.approx(correction["modelled_busy_seconds_per_session"], rel=1e-9)
    assert busy <= config["daily_window_seconds"]
    total = correction["sessions"] * busy
    assert total <= correction["machine_time_ceiling_seconds"]
    assert total == pytest.approx(correction["modelled_busy_seconds_total"], rel=1e-9)
    slack = correction["machine_time_ceiling_seconds"] - total
    assert slack == pytest.approx(correction["machine_time_ceiling_slack_seconds"], abs=1e-6)
    # The T176 role envelope leaves substantial slack.  It cannot be reclaimed after
    # registration by increasing shots: that would move the endpoint floor, ratio and
    # Monte Carlo operating characteristic.  The design therefore retains all 40 pairs
    # and deliberately under-spends instead of silently changing its statistics.
    assert slack == pytest.approx(1104.704)
    assert "not_reclaimed" in " ".join(
        key for key in correction if "slack" in key and "operational" not in key
    )
    assert "statistical registration" in correction[
        "machine_time_ceiling_slack_not_reclaimed_reason"
    ]


def test_the_mirror_qc_budget_is_what_the_runner_actually_submits(
    config: dict, correction: dict
) -> None:
    """The unit that was wrong once: per QC cycle, not per setting.

    The runner submits ``mirror.shots_per_task`` on each of the fixed and adaptive
    settings, so a QC cycle costs twice that, and charging the declared per-cycle figure
    per setting again double-counts it.  Tied to the loop config so the two cannot drift.
    """
    assert correction["mirror_shots_per_qc_cycle"] == (
        correction["mirror_settings_per_repetition"] * config["mirror"]["shots_per_task"]
    )


def test_the_budget_uses_the_t176_role_envelope_not_the_refuted_setting_model(
    correction: dict,
) -> None:
    assert correction["timing_backend_id"] == "tianyan176"
    assert correction["timing_budget_model"] == "role_envelope_sum_task_runtime"
    assert correction["shot_rate_per_second_used"] == pytest.approx(1414.98122405956)
    assert correction["seconds_per_job_used"] == 0.0
    assert correction["seconds_per_setting_used"] == 0.0
    assert correction["role_jobs_per_session"] == {
        "baseline": 2,
        "sense": 40,
        "mirror": 2,
    }
    assert correction["execution_wall_seconds_per_session"] == pytest.approx(323.824)
    assert _session_busy(correction) == pytest.approx(647.648)
    assert len(correction["timing_source_ledger_sha256"]) == 64


def test_a_sensing_job_completes_inside_the_fast_cadence_period(config: dict, correction: dict) -> None:
    # Same-job settings execute as one batch.  Cadence feasibility therefore uses the
    # observed job execution envelope, not the conservative quota sum across both tasks.
    job = correction["role_task_runtime_seconds"]["sense"]
    margin = job + config["cadence"]["measured_interface_p90_seconds"]
    assert margin < config["cadence"]["fast_seconds"]
    # The registered lower bound on the fast cadence is 2x the interface P90; the job has
    # to leave at least that much of the period unused or a stall spills into the next cycle.
    assert config["cadence"]["fast_seconds"] - job >= config["cadence"][
        "hardware_lower_bound_seconds"
    ]


def test_shot_allocation_follows_the_registered_minimisation_rule(correction: dict) -> None:
    """S_baseline = S_injected * sqrt(pairs_per_session) minimises the total floor.

    There is no factor of two under the square root: that factor belonged to the
    superseded design in which the estimate averaged the session's two baseline
    measurements, which the online compensation ordering makes impossible.
    """
    ratio = correction["baseline_shots_per_setting"] / correction["sensing_shots_per_setting"]
    assert ratio == pytest.approx(
        math.sqrt(correction["cycles_per_cadence_per_session"]), rel=2e-3
    )


def test_the_estimate_uses_only_the_baseline_that_exists_when_the_cycle_runs(
    correction: dict,
) -> None:
    assert correction["baseline_used_by_estimate"] == "session_start"
    assert correction["baseline_measurement_positions"] == ["session_start", "session_end"]
    floor = endpoint_shot_floor(
        injected_shots_per_setting=correction["sensing_shots_per_setting"],
        baseline_shots_per_setting=correction["baseline_shots_per_setting"],
    )
    assert floor["shared_baseline_floor"] == pytest.approx(
        correction["endpoint_shot_floor_shared"]
    )
    assert floor["per_cycle_floor"] == pytest.approx(correction["endpoint_shot_floor_per_cycle"])


def test_declared_endpoint_means_follow_from_the_shot_floor_and_the_ou_process(
    config: dict, correction: dict
) -> None:
    floor = endpoint_shot_floor(
        injected_shots_per_setting=correction["sensing_shots_per_setting"],
        baseline_shots_per_setting=correction["baseline_shots_per_setting"],
    )
    assert floor["total_floor"] == pytest.approx(correction["endpoint_shot_floor_total"])
    fast = floor["total_floor"] + _drift(config["cadence"]["fast_seconds"], config)
    slow = floor["total_floor"] + _drift(config["cadence"]["slow_seconds"], config)
    assert fast == pytest.approx(correction["expected_fast_endpoint_mean"])
    assert slow == pytest.approx(correction["expected_slow_endpoint_mean"])
    assert fast / slow == pytest.approx(correction["expected_ratio"])
    # The shot floor dilutes the pure-drift contrast toward 1.0, never away from it.
    assert correction["pure_drift_ratio_limit"] < correction["expected_ratio"] < 1.0


def test_amortising_the_baseline_is_what_makes_the_budget_reachable(correction: dict) -> None:
    """A four-setting design refitted into the same budget is below the power bar.

    This is the quantitative reason the amortisation is in the plan, and it has to be
    stated as a power comparison rather than a floor ratio: at the same shots per setting
    the four-setting floor is only 1.63x worse, which understates the cost, because a
    four-setting cycle also spends twice the shots and so cannot hold those shots.
    """
    floor = endpoint_shot_floor(
        injected_shots_per_setting=correction["sensing_shots_per_setting"],
        baseline_shots_per_setting=correction["baseline_shots_per_setting"],
    )
    assert floor["four_setting_floor_at_same_shots"] > floor["total_floor"]

    pairs = correction["cycles_per_cadence_per_session"]
    jobs = 2 * pairs + correction["mirror_qc_jobs_per_session"]
    rate = correction["shot_rate_per_second_used"]
    mirror = correction["mirror_qc_jobs_per_session"] * correction["mirror_shots_per_qc_cycle"]
    affordable = (
        correction["modelled_busy_seconds_per_session"] - jobs * correction["seconds_per_job_used"]
    ) * rate - mirror
    four_setting_shots = int(affordable // (2 * pairs * 4))
    assert four_setting_shots > 0
    four_setting_floor = 4.746 / four_setting_shots
    assert four_setting_floor > correction["endpoint_shot_floor_total"]

    # And it is not a marginal loss: the refitted design's expected ratio is far enough
    # toward the null that the pair count available cannot reach the registered power.
    fast = four_setting_floor + (
        correction["expected_fast_endpoint_mean"] - correction["endpoint_shot_floor_total"]
    )
    slow = four_setting_floor + (
        correction["expected_slow_endpoint_mean"] - correction["endpoint_shot_floor_total"]
    )
    power = permutation_claim_rate(
        pair_count=correction["registered_cycle_pairs_total"],
        fast_mean=fast,
        slow_mean=slow,
        replicates=6000,
        permutations=600,
        seed=20260815,
    )
    assert power < correction["minimum_power"]


def test_the_registered_rule_reaches_the_declared_power_and_holds_its_size(
    correction: dict,
) -> None:
    fast = correction["expected_fast_endpoint_mean"]
    slow = correction["expected_slow_endpoint_mean"]
    power = permutation_claim_rate(
        pair_count=correction["registered_cycle_pairs_total"],
        fast_mean=fast,
        slow_mean=slow,
        replicates=6000,
        permutations=600,
        seed=20260815,
    )
    assert power >= correction["minimum_power"]
    assert power == pytest.approx(correction["expected_power"], abs=0.02)
    size = permutation_claim_rate(
        pair_count=correction["registered_cycle_pairs_total"],
        fast_mean=fast,
        slow_mean=fast,
        replicates=6000,
        permutations=600,
        seed=777,
    )
    assert size <= correction["maximum_boundary_size"] + 0.008


def test_the_registered_minimum_pair_count_is_the_smallest_that_holds_power(
    correction: dict,
) -> None:
    """minimum_adjudicated_cycle_pairs is a lost-cycle tolerance, so it must hold power.

    Pinning it needs both directions: the registered value clears the bar and the next
    value down does not, which is what makes 30 a choice rather than a round number.
    """
    minimum = correction["minimum_adjudicated_cycle_pairs"]

    def power_at(pairs: int) -> float:
        return permutation_claim_rate(
            pair_count=pairs,
            fast_mean=correction["expected_fast_endpoint_mean"],
            slow_mean=correction["expected_slow_endpoint_mean"],
            replicates=20000,
            permutations=800,
            seed=20260815,
        )

    at_minimum = power_at(minimum)
    # Monte Carlo allowance: 20000 replicates give a standard error near 0.003 at p ~ 0.8.
    assert at_minimum >= correction["minimum_power"] - 0.009
    assert power_at(minimum - 2) < correction["minimum_power"]


def test_losing_a_whole_session_is_registered_as_inconclusive(correction: dict) -> None:
    """Half the pairs from one starting cadence is underpowered, and the config says so."""
    power = permutation_claim_rate(
        pair_count=correction["cycles_per_cadence_per_session"],
        fast_mean=correction["expected_fast_endpoint_mean"],
        slow_mean=correction["expected_slow_endpoint_mean"],
        replicates=6000,
        permutations=600,
        seed=20260815,
    )
    assert power < correction["minimum_power"]
    assert correction["cycles_per_cadence_per_session"] < correction[
        "minimum_adjudicated_cycle_pairs"
    ]
    assert "INCONCLUSIVE" in correction["whole_session_loss_policy"]


def test_starting_cadence_is_balanced_across_sessions(correction: dict) -> None:
    orders = [tuple(row) for row in correction["session_block_order"]]
    assert len(orders) == correction["sessions"]
    assert orders.count(("fast", "slow")) == orders.count(("slow", "fast"))


def test_baseline_drift_cannot_manufacture_a_pass(correction: dict) -> None:
    """The load-bearing safety property of the shared baseline.

    A shared baseline leaves an offset common to every cycle it serves, and the risk that
    matters is not that it costs power -- it is that an offset loaded unequally onto the two
    cadence blocks could inflate the false-positive rate.  Measured here at the drift level
    the design's own readout cannot exclude: under every registered shape the realised
    boundary size stays at or below nominal.
    """
    fast = correction["expected_fast_endpoint_mean"]
    offset = (1.645 * math.sqrt(2.0)) ** 2 * correction["endpoint_shot_floor_shared"]
    for name, (first, second) in DRIFT_SHAPES.items():
        size = permutation_claim_rate_under_block_offsets(
            pairs_per_session=correction["cycles_per_cadence_per_session"],
            sessions=correction["sessions"],
            fast_mean=fast,
            slow_mean=fast,
            block_one_offset=offset * first,
            block_two_offset=offset * second,
            replicates=8000,
            permutations=800,
            seed=777,
        )
        assert size <= correction["maximum_boundary_size"] + 0.008, name


def test_the_balanced_block_order_is_what_protects_the_size(correction: dict) -> None:
    """Same drift, both sessions starting on the same cadence: the size blows up.

    This is why session_block_order is registered as opposite starting cadences rather
    than left to convenience, and it is the control that makes the previous test's pass
    informative instead of vacuous.
    """
    fast = correction["expected_fast_endpoint_mean"]
    offset = 8.0 * correction["endpoint_shot_floor_shared"]
    first, second = DRIFT_SHAPES["linear_ramp"]
    balanced = permutation_claim_rate_under_block_offsets(
        pairs_per_session=correction["cycles_per_cadence_per_session"],
        sessions=2,
        fast_mean=fast,
        slow_mean=fast,
        block_one_offset=offset * first,
        block_two_offset=offset * second,
        replicates=8000,
        permutations=800,
        seed=777,
    )
    unbalanced = permutation_claim_rate_under_block_offsets(
        pairs_per_session=2 * correction["cycles_per_cadence_per_session"],
        sessions=1,
        fast_mean=fast,
        slow_mean=fast,
        block_one_offset=offset * first,
        block_two_offset=offset * second,
        replicates=8000,
        permutations=800,
        seed=777,
    )
    assert balanced <= correction["maximum_boundary_size"] + 0.008
    assert unbalanced > 2.0 * balanced


def test_the_drift_sensitivity_is_registered_with_its_cost_stated(correction: dict) -> None:
    """The design must not claim the drift is harmless, only that it cannot fake a pass."""
    rule = correction["baseline_drift_sensitivity_rule"]
    for shape in DRIFT_SHAPES:
        assert shape.split("_")[0] in rule
    assert "cannot manufacture a pass" in correction["baseline_drift_measured_effect"]
    assert "power" in correction["baseline_drift_measured_effect"]
    # A limitation with no bound has to be recorded as such rather than dressed up.
    assert "invisible" in correction["baseline_drift_undetectable_mode"]
    assert "no baseline drift" in correction["expected_power_note"]


def test_the_prediction_is_committed_and_brackets_the_expected_ratio(correction: dict) -> None:
    lower, upper = correction["preregistered_ratio_interval"]
    assert lower < correction["preregistered_ratio_prediction"] < upper
    assert lower < correction["expected_ratio"] < upper
    # A prediction that already contains the null would be unfalsifiable by hit or miss.
    assert upper < 1.0
    assert correction["simulation_pooling_permitted"] is False


def test_the_amendment_keeps_the_registered_statistic_and_bans_the_escape_hatches(
    correction: dict,
) -> None:
    assert correction["primary_adjudication"] == "cadence_ratio_permutation_gate"
    assert "cadence_ratio_gate" in correction["secondary_adjudication"]
    assert correction["gate_module_change_permitted"] is False
    assert correction["outlier_exclusion_permitted"] is False
    assert correction["optional_stopping_permitted"] is False
    assert correction["baseline_sharing_scope"] == "whole_session_both_cadence_blocks"
    assert correction["pair_discard_granularity"] == "cycle_pair"


def test_the_reachability_evidence_is_pinned_by_hash(correction: dict) -> None:
    assert correction["reachability_evidence"].endswith(
        "forty_minute_prediction_start_only_baseline.json"
    )
    assert len(correction["reachability_evidence_sha256"]) == 64
    int(correction["reachability_evidence_sha256"], 16)
