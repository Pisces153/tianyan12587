"""Amortising the baseline must not redefine the estimator.

The load-bearing claim of ``shared_baseline_sensing`` is that moving the zero-field
condition out to one shared measurement per session changes the *cost* of a cycle and the
*size* of its shot floor, and changes nothing about the registered arithmetic.  The first
test pins that by running both code paths over the same raw counts and demanding exact
agreement.  The rest pin the two quantitative claims the design rests on -- the floor
halves, and the shared residual is bounded by a quarter of the measured drift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.adaptive import shared_baseline_sensing as shared

import scripts.run_b4_cadence_pair_hardware as runner
import scripts.run_cadence_pair_hardware_smoke as hardware


PHASE_TIME = 0.47


def _counts(generator: np.random.Generator, shots: int) -> np.ndarray:
    """Four valid six-qubit raw-count rows, one per (condition, axis) setting."""
    rows = []
    for _ in range(4):
        weights = generator.random(64) + 0.05
        rows.append(generator.multinomial(shots, weights / weights.sum()))
    return np.asarray(rows, dtype=np.int64)


def _expectations(counts: np.ndarray, shots: int, offset: int) -> list[dict[str, float]]:
    return [
        {
            "observed_y": hardware.single_qubit_expectation(counts[offset], field_index, shots),
            "observed_z": hardware.single_qubit_expectation(counts[offset + 1], field_index, shots),
        }
        for field_index in range(2)
    ]


def test_shared_baseline_path_reproduces_the_registered_estimator_exactly() -> None:
    """Same counts, same shots, one shared baseline: identical estimates and sigmas."""
    generator = np.random.default_rng(7)
    shots = 4096
    for _ in range(5):
        counts = _counts(generator, shots)
        registered = runner.estimate_fields_from_counts(counts, shots, PHASE_TIME)
        baseline = shared.baseline_record(
            _expectations(counts, shots, 0), shots, session_index=0, position="session_start"
        )
        amortised = shared.differential_estimate(
            _expectations(counts, shots, 2), shots, baseline, PHASE_TIME
        )
        assert amortised["estimates"] == registered["estimates"]
        assert amortised["shot_sigmas"] == registered["shot_sigmas"]


def test_a_cycle_costs_two_settings_instead_of_four() -> None:
    generator = np.random.default_rng(8)
    shots = 2048
    counts = _counts(generator, shots)
    registered = runner.estimate_fields_from_counts(counts, shots, PHASE_TIME)
    baseline = shared.baseline_record(
        _expectations(counts, shots, 0), shots, session_index=0, position="session_start"
    )
    amortised = shared.differential_estimate(
        _expectations(counts, shots, 2), shots, baseline, PHASE_TIME
    )
    assert registered["shots_used_total"] == 4 * shots
    assert amortised["shots_used_total"] == 2 * shots
    assert amortised["settings_used"] == 2


def test_shot_floor_halves_against_the_calibrated_four_setting_constant() -> None:
    """The halving is an identity in sigma = sqrt(var_base + var_inj)/(2T), not a fit."""
    floor = shared.endpoint_shot_floor(
        injected_shots_per_setting=8192,
        baseline_shots_per_setting=10**9,
    )
    assert floor["per_cycle_floor"] == pytest.approx(
        floor["four_setting_floor_at_same_shots"] / 2.0
    )
    assert floor["shared_baseline_floor"] < 1e-8


def test_the_shared_term_is_not_divided_by_the_measurement_count() -> None:
    """Online compensation means no cycle can subtract a start/end average.

    The superseded design halved this term by averaging the session's two baseline
    measurements, which is unavailable: the compensation is computed at cycle time from
    the baseline that exists then, and the closing measurement does not exist yet.
    """
    floor = shared.endpoint_shot_floor(
        injected_shots_per_setting=6186, baseline_shots_per_setting=27664
    )
    assert floor["shared_baseline_floor"] == pytest.approx(2.373 / 27664)


def test_floor_matches_the_registered_v4_design_point() -> None:
    floor = shared.endpoint_shot_floor(
        injected_shots_per_setting=6186,
        baseline_shots_per_setting=27664,
    )
    assert floor["total_floor"] == pytest.approx(4.693875e-4, rel=1e-5)
    # The shared part is deliberately *not* negligible.  Minimising the total floor under a
    # shot budget puts it at 1/sqrt(pairs_per_session) of the per-cycle part; forcing it
    # smaller would raise the total.  It is safe to leave large because it is common to both
    # arms, and the measured boundary size confirms it cannot manufacture a pass.
    assert floor["shared_fraction"] == pytest.approx(1.0 / math.sqrt(20), rel=1e-3)


def test_averaging_two_baselines_halves_the_shared_variance() -> None:
    records = [
        {
            "session_index": 3,
            "position": position,
            "shots_per_setting": 33625,
            "fields": [{"phase": 0.4, "phase_variance": 2.0e-6} for _ in range(2)],
        }
        for position in ("session_start", "session_end")
    ]
    average = shared.average_baseline(records)
    assert average["fields"][0]["phase_variance"] == pytest.approx(1.0e-6)
    assert average["measurement_count"] == 2


def test_baseline_phases_are_averaged_on_the_circle() -> None:
    """Two measurements straddling pi must not average to zero."""
    records = [
        {
            "session_index": 1,
            "position": "session_start",
            "shots_per_setting": 1024,
            "fields": [{"phase": 3.10, "phase_variance": 1.0e-6} for _ in range(2)],
        },
        {
            "session_index": 1,
            "position": "session_end",
            "shots_per_setting": 1024,
            "fields": [{"phase": -3.10, "phase_variance": 1.0e-6} for _ in range(2)],
        },
    ]
    average = shared.average_baseline(records)
    assert abs(abs(average["fields"][0]["phase"]) - math.pi) < 0.05


def test_averaging_refuses_to_mix_sessions_or_repeat_a_position() -> None:
    """A baseline that mixed sessions would break the exchangeability argument."""
    template = {
        "position": "session_start",
        "shots_per_setting": 1024,
        "fields": [{"phase": 0.1, "phase_variance": 1.0e-6} for _ in range(2)],
    }
    with pytest.raises(ValueError):
        shared.average_baseline([
            {**template, "session_index": 0},
            {**template, "session_index": 1, "position": "session_end"},
        ])
    with pytest.raises(ValueError):
        shared.average_baseline([
            {**template, "session_index": 0},
            {**template, "session_index": 0},
        ])


def test_drift_qc_separates_shot_noise_from_a_real_drift() -> None:
    def record(position: str, phase: float) -> dict:
        return {
            "session_index": 0,
            "position": position,
            "shots_per_setting": 33625,
            "fields": [{"phase": phase, "phase_variance": 1.0e-6} for _ in range(2)],
        }

    quiet = shared.baseline_drift_qc(
        record("session_start", 0.20),
        record("session_end", 0.2010),
        phase_time_seconds=PHASE_TIME,
    )
    assert quiet["static_within_shot_noise"]

    moved = shared.baseline_drift_qc(
        record("session_start", 0.20),
        record("session_end", 0.25),
        phase_time_seconds=PHASE_TIME,
    )
    assert not moved["static_within_shot_noise"]
    assert moved["fields"][0]["field_units_drift"] == pytest.approx(0.05 / (2.0 * PHASE_TIME))


def test_drift_sensitivity_uses_an_upper_limit_and_not_a_threshold() -> None:
    """A quiet session still produces a non-zero sensitivity offset.

    This is the point of taking a confidence limit rather than thresholding: the start
    baseline's own noise puts a floor of three on the detectable z, which is above the
    drift level that costs power, so a threshold would only invent a criterion the design
    cannot meet.  A quiet readout therefore has to yield a bound, not a clean bill.
    """
    shared_floor = 2.373 / 27664
    quiet = {
        "session_index": 0,
        "fields": [{"phase_drift_z": 0.0}, {"phase_drift_z": 0.0}],
    }
    result = shared.drift_sensitivity_offsets(quiet, shared_baseline_floor=shared_floor)
    assert result["upper_limit_z"] == pytest.approx(1.645 * math.sqrt(2.0))
    assert result["endpoint_offset_at_upper_limit"] > 0.0
    # A measured drift adds to the limit rather than replacing it.
    moved = {
        "session_index": 0,
        "fields": [{"phase_drift_z": 2.0}, {"phase_drift_z": -0.5}],
    }
    grown = shared.drift_sensitivity_offsets(moved, shared_baseline_floor=shared_floor)
    assert grown["measured_worst_absolute_z"] == pytest.approx(2.0)
    assert grown["upper_limit_z"] == pytest.approx(2.0 + 1.645 * math.sqrt(2.0))
    assert grown["endpoint_offset_at_upper_limit"] > result["endpoint_offset_at_upper_limit"]


def test_the_three_registered_drift_shapes_bracket_the_linear_ramp() -> None:
    """The offset is squared, so a linear ramp is not the extreme case in either direction."""
    shared_floor = 2.373 / 27664
    qc = {"session_index": 1, "fields": [{"phase_drift_z": 1.0}, {"phase_drift_z": 1.0}]}
    result = shared.drift_sensitivity_offsets(qc, shared_baseline_floor=shared_floor)
    total = result["endpoint_offset_at_upper_limit"]
    shapes = result["shapes"]
    ramp = shapes["linear_ramp"]
    # Block midpoints at T/4 and 3T/4 of the session, squared: 1/16 and 9/16.
    assert ramp["block_one_offset"] == pytest.approx(total / 16.0)
    assert ramp["block_two_offset"] == pytest.approx(9.0 * total / 16.0)
    # The step carries the whole asymmetry; the transient carries none of it.
    step = shapes["step_at_block_boundary"]
    transient = shapes["early_saturating_transient"]
    assert step["block_two_offset"] - step["block_one_offset"] == pytest.approx(total)
    assert transient["block_two_offset"] == pytest.approx(transient["block_one_offset"])
    ramp_asymmetry = ramp["block_two_offset"] - ramp["block_one_offset"]
    assert 0.0 < ramp_asymmetry < total


def test_drift_qc_refuses_a_mislabelled_pair() -> None:
    record = {
        "session_index": 0,
        "position": "session_end",
        "shots_per_setting": 1024,
        "fields": [{"phase": 0.1, "phase_variance": 1.0e-6} for _ in range(2)],
    }
    with pytest.raises(ValueError):
        shared.baseline_drift_qc(record, record, phase_time_seconds=PHASE_TIME)
