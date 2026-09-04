"""Session-shared baseline for the two-axis differential field estimator.

The registered estimator is a *differential*: it measures the atan2 phase with no
injected field, measures it again with the field on, and divides the wrapped difference
by ``2 * phase_time``.  Four settings per cycle -- (baseline, injected) x (Y, Z) -- is
how the loop has measured it so far, which spends half of every cycle's shots
re-measuring a bias that does not change from cycle to cycle.

Two consequences of moving the baseline out to one shared measurement per session, both
of which follow from the estimator's own error formula
``sigma = sqrt(var_baseline + var_injected) / (2 * phase_time)``:

1. **The cycle costs two settings instead of four.**  Sensing shots per cycle halve at
   fixed shots per setting, and -- separately -- the fixed job overhead halves.
2. **The shot floor halves at fixed shots per setting.**  ``var_baseline`` is driven to
   the shared measurement's own (much larger) shot count, so the per-cycle floor drops
   from ``4.746 / S`` to ``2.373 / S`` plus a shared term.  This is an identity in
   the formula above, not a fitted constant: the two conditions enter the variance with
   equal weight, so removing one of them from the per-cycle budget removes exactly half.

**Why the estimate can only use the session-start measurement.**  The registered endpoint
is ``|mirror_fields + compensation|**2`` and ``compensation`` is computed *online* by the
shield from that cycle's own sensed estimate.  A cycle can therefore only subtract a
baseline that already exists when it runs.  The session-end measurement does not exist at
cycle time, so the start/end average is not available to the estimate and the shared
variance term is ``2.373 / S_start`` with no factor of two.  ``average_baseline`` below is
kept for the offline diagnostic attribution described under the drift QC; it is not on the
endpoint path.

**Why the sharing has to span both cadence blocks.**  A shared baseline leaves a
residual error that is *common* to every cycle it serves, and a common additive offset
does not average down over pairs.  If the sharing were per-block, the fast and slow arms
would carry two independent offsets, the registered cycle pairs would stop being
exchangeable under the boundary null, and the within-pair permutation calibration in
``cadence_permutation`` would lose its exactness.  One baseline serving *both* blocks
puts the identical offset in both arms of every pair, so swapping the cadence labels
within a pair still preserves the joint law and the permutation test stays exact.

The offset that survives is a common additive term on both arms, and a common additive
term pushes the ratio *up*, toward the null value of 1.0.  It costs power; it cannot
manufacture a pass.  What it can do is drift across the session, which would make the
two blocks see slightly different offsets after all -- so the baseline is measured at
session start and at session end, ``baseline_drift_qc`` reports the difference against
its own shot noise, and ``drift_sensitivity_offsets`` converts its upper confidence limit
into the block offsets that the pre-registered sensitivity analysis re-runs the verdict
against.  The assumption is measured and reported, not assumed.

**What the drift actually does, measured rather than argued.**  Injecting the offsets this
module produces into the registered permutation rule at the frozen design point, over all
three pre-registered shapes and drift levels up to ``z = 4``, never raises the realized
boundary size above 0.0512: the asymmetric shapes make the test *conservative* (down to
0.0325) and the symmetric shape leaves it exactly at nominal.  Baseline drift cannot
manufacture a pass.  The cost is power, and the binding shape is the symmetric one -- the
one that is perfectly exchangeable and therefore not an exchangeability problem at all,
just a larger shared floor diluting the contrast.  The protection against the asymmetric
shapes is the registered balanced session block order: session 0 runs fast first and
session 1 runs slow first, so a drift's unequal loading of the two blocks lands on
opposite arms and cancels to first order in the pooled statistic.  Running both sessions
in the same order instead raises the realized size from 0.0510 to 0.1968 at a drift of
eight times the shared floor.

One failure mode is bounded by nothing here: a transient that departs and returns between
the two baseline measurements is invisible to the drift readout at any shot count.  It is
recorded as a limitation, not as a controlled risk.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

SCHEMA = "session_shared_baseline_differential_sensing_v1"

# The four-setting endpoint shot floor was calibrated on two independent r2 hardware
# arms as F(S) = 4.746 / S (S=22050 gave F*S=4.7552, S=88200 gave F*S=4.7364, agreeing
# to 0.4% across a 4x change in shots).  Baseline and injected enter
# sigma = sqrt(var_baseline + var_injected) / (2*T) with equal weight, so each condition
# contributes exactly half of that.
PER_CONDITION_FLOOR_CONSTANT = 2.373
FOUR_SETTING_FLOOR_CONSTANT = 2.0 * PER_CONDITION_FLOOR_CONSTANT


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle onto (-pi, pi], the same way the registered estimator does."""
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def phase_from_expectations(observed_y: float, observed_z: float, shots: int) -> dict[str, float]:
    """The registered per-condition phase and its delta-method variance.

    Byte-identical to the arithmetic inlined in
    ``run_b4_cadence_pair_hardware.estimate_fields_from_counts``; a test asserts the
    agreement so that sharing the baseline does not redefine the estimator.
    """
    observed_y = float(observed_y)
    observed_z = float(observed_z)
    shots = int(shots)
    if shots <= 0:
        raise ValueError("phase estimation needs a positive shot count")
    radius_squared = max(observed_y * observed_y + observed_z * observed_z, 1e-12)
    phase = math.atan2(-observed_y, observed_z)
    variance_y = max((1.0 - observed_y * observed_y) / shots, 1.0 / (shots * shots))
    variance_z = max((1.0 - observed_z * observed_z) / shots, 1.0 / (shots * shots))
    derivative_y = -observed_z / radius_squared
    derivative_z = observed_y / radius_squared
    phase_variance = max(
        derivative_y * derivative_y * variance_y + derivative_z * derivative_z * variance_z,
        1e-15,
    )
    return {
        "observed_y": observed_y,
        "observed_z": observed_z,
        "phase": phase,
        "phase_variance": phase_variance,
    }


def baseline_record(
    axis_expectations: Sequence[Mapping[str, float]],
    shots: int,
    *,
    session_index: int,
    position: str,
) -> dict[str, Any]:
    """One zero-field baseline measurement, held for the whole session.

    ``axis_expectations`` carries one mapping per controlled field with keys
    ``observed_y`` and ``observed_z``.  ``position`` records whether this is the session's
    opening or closing measurement, which is what the drift QC pairs up.
    """
    if position not in ("session_start", "session_end"):
        raise ValueError("baseline position must be session_start or session_end")
    if len(axis_expectations) != 2:
        raise ValueError("the controlled injection carries exactly two fields")
    fields = [
        phase_from_expectations(row["observed_y"], row["observed_z"], shots)
        for row in axis_expectations
    ]
    return {
        "schema": SCHEMA,
        "role": "session_shared_zero_field_baseline",
        "session_index": int(session_index),
        "position": position,
        "shots_per_setting": int(shots),
        "settings": 2,
        "fields": fields,
    }


def average_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine a session's baseline measurements.  **Not on the endpoint path.**

    Averaging the opening and closing measurements would halve the shared variance and
    centre the residual offset on the middle of the session, but the compensation each
    cycle applies is computed online from the baseline that exists at that moment, so no
    cycle can subtract an average containing a later measurement.  This is retained for the
    offline diagnostic attribution -- reconstructing, after the fact, how large each cycle's
    compensation error is likely to have been -- which is a reported readout and never the
    registered endpoint.  Phases are averaged on the circle, not on the line.
    """
    if not records:
        raise ValueError("a session needs at least one baseline measurement")
    sessions = {int(record["session_index"]) for record in records}
    if len(sessions) != 1:
        raise ValueError("baseline records to be averaged must come from one session")
    positions = [record["position"] for record in records]
    if len(set(positions)) != len(positions):
        raise ValueError("baseline positions must be distinct")
    count = len(records)
    fields: list[dict[str, float]] = []
    for field_index in range(2):
        rows = [record["fields"][field_index] for record in records]
        anchor = float(rows[0]["phase"])
        offsets = [wrap_to_pi(float(row["phase"]) - anchor) for row in rows]
        phase = wrap_to_pi(anchor + sum(offsets) / count)
        variance = sum(float(row["phase_variance"]) for row in rows) / (count * count)
        fields.append({"phase": phase, "phase_variance": variance})
    return {
        "schema": SCHEMA,
        "role": "session_shared_zero_field_baseline_average",
        "session_index": sessions.pop(),
        "positions": positions,
        "measurement_count": count,
        "shots_per_setting_total": sum(int(record["shots_per_setting"]) for record in records),
        "fields": fields,
        "exchangeability_note": (
            "this average serves every cycle of both cadence blocks in the session, so "
            "its residual offset is common to the fast and slow arm of every registered "
            "pair and within-pair label swaps remain measure preserving under the "
            "boundary null"
        ),
    }


def differential_estimate(
    injected_expectations: Sequence[Mapping[str, float]],
    injected_shots: int,
    baseline: Mapping[str, Any],
    phase_time_seconds: float,
) -> dict[str, Any]:
    """The registered differential estimate against a stored session baseline.

    Same formula as the four-setting path -- wrapped phase difference over ``2 * T``, and
    the two conditions' phase variances added -- with the baseline's variance now coming
    from the shared measurement instead of from this cycle's own shots.
    """
    if len(injected_expectations) != 2:
        raise ValueError("the controlled injection carries exactly two fields")
    phase_time = float(phase_time_seconds)
    if phase_time <= 0.0:
        raise ValueError("phase time must be positive")
    estimates: list[float] = []
    sigmas: list[float] = []
    fields: list[dict[str, Any]] = []
    for field_index, row in enumerate(injected_expectations):
        injected = phase_from_expectations(row["observed_y"], row["observed_z"], injected_shots)
        baseline_field = baseline["fields"][field_index]
        phase_difference = wrap_to_pi(injected["phase"] - float(baseline_field["phase"]))
        estimate = phase_difference / (2.0 * phase_time)
        sigma = math.sqrt(
            injected["phase_variance"] + float(baseline_field["phase_variance"])
        ) / (2.0 * phase_time)
        estimates.append(estimate)
        sigmas.append(sigma)
        fields.append({
            "field": f"h{field_index + 1}",
            "baseline": dict(baseline_field),
            "injected": injected,
            "wrapped_phase_difference": phase_difference,
            "estimate": estimate,
            "shot_sigma": sigma,
        })
    return {
        "estimator": "session_shared_baseline_differential_two_axis_atan2_effective_field",
        "estimator_lineage": (
            "identical to baseline_differential_two_axis_atan2_effective_field with the "
            "zero-field condition amortised over the session"
        ),
        "estimates": estimates,
        "shot_sigmas": sigmas,
        "fields": fields,
        "shots_used_total": 2 * int(injected_shots),
        "settings_used": 2,
        "baseline_session_index": int(baseline["session_index"]),
    }


def endpoint_shot_floor(
    *,
    injected_shots_per_setting: int,
    baseline_shots_per_setting: int,
) -> dict[str, float]:
    """Expected estimator shot floor of the per-cycle endpoint under the shared baseline.

    ``per_cycle`` is the part that is independent across cycles and therefore averages
    down over pairs; ``shared`` is the part common to every cycle in the session, which
    does not.  Both arms carry both terms, so the shared part enters the ratio as a
    common additive offset -- diluting the contrast toward 1.0, never away from it.

    ``baseline_shots_per_setting`` is the *session-start* measurement alone.  There is no
    division by the number of baseline measurements: the compensation is computed online,
    so a cycle cannot subtract an average that includes a measurement taken after it ran.
    """
    injected = int(injected_shots_per_setting)
    baseline = int(baseline_shots_per_setting)
    if injected <= 0 or baseline <= 0:
        raise ValueError("shot counts must be positive")
    per_cycle = PER_CONDITION_FLOOR_CONSTANT / injected
    shared = PER_CONDITION_FLOOR_CONSTANT / baseline
    return {
        "per_cycle_floor": per_cycle,
        "shared_baseline_floor": shared,
        "total_floor": per_cycle + shared,
        "four_setting_floor_at_same_shots": FOUR_SETTING_FLOOR_CONSTANT / injected,
        "shared_fraction": shared / per_cycle,
    }


def baseline_drift_qc(
    first: Mapping[str, Any],
    last: Mapping[str, Any],
    *,
    phase_time_seconds: float,
    sigma_threshold: float = 3.0,
) -> dict[str, Any]:
    """Measure how far the shared bias moved across the session.

    This is the empirical check on the one assumption the amortisation adds.  It is a
    reported readout, not a gate on the endpoint: the endpoint verdict is re-run against
    the asymmetry this implies (see ``block_offset_asymmetry``) and both results are
    reported whatever the drift turns out to be.
    """
    if first["position"] != "session_start" or last["position"] != "session_end":
        raise ValueError("drift QC pairs the session's opening and closing baselines")
    phase_time = float(phase_time_seconds)
    fields: list[dict[str, float]] = []
    for field_index in range(2):
        start = first["fields"][field_index]
        end = last["fields"][field_index]
        drift = wrap_to_pi(float(end["phase"]) - float(start["phase"]))
        sigma = math.sqrt(float(start["phase_variance"]) + float(end["phase_variance"]))
        fields.append({
            "field": f"h{field_index + 1}",
            "phase_drift": drift,
            "phase_drift_sigma": sigma,
            "phase_drift_z": drift / sigma if sigma > 0.0 else float("nan"),
            "field_units_drift": drift / (2.0 * phase_time),
        })
    worst = max(abs(row["phase_drift_z"]) for row in fields)
    return {
        "schema": SCHEMA,
        "role": "session_shared_baseline_drift_qc",
        "session_index": int(first["session_index"]),
        "fields": fields,
        "worst_absolute_z": worst,
        "sigma_threshold": float(sigma_threshold),
        "static_within_shot_noise": bool(worst <= float(sigma_threshold)),
        "interpretation": (
            "a drift consistent with shot noise leaves the shared offset common to both "
            "cadence blocks and the permutation calibration exact; a larger drift is not "
            "a failure of the collection, it is an input to the pre-registered "
            "sensitivity re-run"
        ),
        "threshold_role": (
            "sigma_threshold is a reported readout only; the sensitivity re-run is driven "
            "by the one-sided upper confidence limit in drift_sensitivity_offsets, not by "
            "this flag, because no shot count in this design can resolve the drift level "
            "at which power degrades and a threshold would only invent a criterion the "
            "design cannot meet"
        ),
    }


# The pre-registered drift shapes, as the fraction of the full endpoint offset ``D`` that
# each cadence block carries.  The estimate subtracts the *session-start* baseline, so a
# block's offset is set by how much of the drift had already happened by the middle of that
# block.  A linear ramp puts the block midpoints at T/4 and 3T/4 of the session, and the
# endpoint is a squared residual, so the offsets go as (1/4)**2 and (3/4)**2.  The step and
# the early transient are the two extremes that bracket it: all of the drift between the
# blocks, and all of it before either.
DRIFT_SHAPES: dict[str, tuple[float, float]] = {
    "linear_ramp": (1.0 / 16.0, 9.0 / 16.0),
    "step_at_block_boundary": (0.0, 1.0),
    "early_saturating_transient": (1.0, 1.0),
}
DRIFT_UPPER_LIMIT_Z = 1.645  # one-sided 95%


def drift_sensitivity_offsets(
    drift_qc: Mapping[str, Any],
    *,
    shared_baseline_floor: float,
    confidence_z: float = DRIFT_UPPER_LIMIT_Z,
) -> dict[str, Any]:
    """Endpoint offsets the pre-registered sensitivity analysis re-runs the verdict under.

    The drift is used as an *upper confidence limit*, not as a pass/fail threshold.  There
    is no shot count at which a conventional three-sigma readout could resolve the drift
    level that matters -- the start baseline's own noise puts a floor of three on the
    detectable ``z`` -- so thresholding it would only manufacture a criterion the design
    cannot meet.  Taking the one-sided limit instead makes any shot count usable and makes
    a larger closing measurement simply give a tighter bound.

    ``z`` is the drift in units of the start baseline's own phase sigma, so the endpoint
    offset it implies is ``D = z**2 * shared_baseline_floor``: both quantities are squared
    per-field residuals summed over the two controlled fields, and their ratio is the
    squared ratio of the phases.  Each shape then distributes ``D`` over the two cadence
    blocks.  Because the registered session block order is balanced, block 1 is the fast
    arm in session 0 and the slow arm in session 1, so an asymmetric shape's loading
    cancels to first order in the pooled statistic; the sensitivity re-run measures what
    survives.
    """
    shared = float(shared_baseline_floor)
    if shared <= 0.0:
        raise ValueError("the shared baseline floor must be positive")
    worst = max(abs(float(row["phase_drift_z"])) for row in drift_qc["fields"])
    upper_z = worst + float(confidence_z) * math.sqrt(2.0)
    offset = upper_z * upper_z * shared
    return {
        "schema": SCHEMA,
        "role": "session_shared_baseline_drift_sensitivity_offsets",
        "session_index": int(drift_qc["session_index"]),
        "measured_worst_absolute_z": worst,
        "confidence_z": float(confidence_z),
        "upper_limit_z": upper_z,
        "upper_limit_z_note": (
            "the QC statistic's own sigma is sqrt(2) times the start baseline's when the "
            "closing measurement carries the same shots, which is what the sqrt(2) is"
        ),
        "endpoint_offset_at_upper_limit": offset,
        "shapes": {
            name: {"block_one_offset": offset * first, "block_two_offset": offset * second}
            for name, (first, second) in DRIFT_SHAPES.items()
        },
        "reporting_rule": (
            "re-run the registered verdict under every shape and report all of them "
            "alongside the primary verdict, whatever the measured drift turns out to be"
        ),
        "measured_effect_note": (
            "across all three shapes and drift levels up to z = 4 the realized boundary "
            "size never exceeds 0.0512, so drift costs power and cannot manufacture a pass"
        ),
    }
