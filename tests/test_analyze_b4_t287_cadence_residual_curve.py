from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import analyze_b4_t287_cadence_residual_curve as module


@pytest.mark.parametrize("backend_id", ["tianyan-287", "tianyan176"])
def test_backend_provenance_accepts_each_exact_platform_spelling(backend_id: str) -> None:
    assert module.validate_backend_provenance(
        hardware_report={"backend_id": backend_id},
        plan={"backend_id": backend_id},
        backend_config={"backend": {"backend_id": backend_id}},
        expected_backend_id=backend_id,
    ) == backend_id


def test_backend_provenance_requires_independent_expected_backend() -> None:
    with pytest.raises(ValueError, match="provenance mismatch"):
        module.validate_backend_provenance(
            hardware_report={"backend_id": "tianyan176"},
            plan={"backend_id": "tianyan176"},
            backend_config={"backend": {"backend_id": "tianyan176"}},
            expected_backend_id="tianyan-287",
        )


def mirror_score(replicate: int, strategy: str, success_probability: float) -> dict[str, object]:
    return {
        "replicate": replicate,
        "strategy": strategy,
        "success_probability": success_probability,
    }


def test_mirror_pair_aggregation_uses_adaptive_fast_slow_loss() -> None:
    cycles = []
    for session in range(3):
        for cadence, success in (("fast", 0.8), ("slow", 0.5)):
            cycles.append({
                "session_index": session,
                "cadence": cadence,
                "mirror_scores": [
                    mirror_score(0, "fixed", 0.4),
                    mirror_score(0, "adaptive", success),
                    mirror_score(1, "fixed", 0.4),
                    mirror_score(1, "adaptive", success),
                ],
            })
    rows, endpoint = module.aggregate_mirror_pairs(cycles, expected_pair_count=6)
    assert len(rows) == 6
    assert math.isclose(endpoint["fast_loss_mean"], 0.2)
    assert math.isclose(endpoint["slow_loss_mean"], 0.5)
    assert endpoint["ratio_gate"]["passed"] is True


def test_analytic_endpoint_residual_adds_two_field_ou_increment() -> None:
    value = module.analytic_endpoint_residual(
        cadence_seconds=90.0,
        shot_sigmas=[0.01, 0.02],
        process_variance=0.00061,
        tau_seconds=300.0,
    )
    expected = 0.01**2 + 0.02**2 + 4.0 * 0.00061 * (1.0 - math.exp(-90.0 / 300.0))
    assert math.isclose(value, expected)


def test_decision_does_not_promote_tracking_sensitivity_over_primary_gate() -> None:
    result = module.decision(
        {
            "available": False,
            "ratio_gate": None,
        },
        {"ratio_gate": {"passed": False}},
        {"session_paired_ratio_sensitivity": {"passed": True}},
    )
    assert result["headline_verdict"] == "INCONCLUSIVE"
    assert result["tier4_cadence_status"] == "PENDING_CORRECTED_COLLECTION"
    assert result["tracking_direction_sensitivity_passed"] is True


def test_registered_cycle_endpoint_rejects_six_to_one_session_collapse() -> None:
    rows = []
    for session in range(3):
        for cycle_index in range(6):
            rows.append({
                "cycle_id": f"session{session:02d}-fast-{cycle_index:02d}",
                "session_index": session,
                "cycle_index": cycle_index,
                "cadence": "fast",
                "observed_endpoint_residual_squared": 0.5,
            })
        rows.append({
            "cycle_id": f"session{session:02d}-slow-00",
            "session_index": session,
            "cycle_index": 0,
            "cadence": "slow",
            "observed_endpoint_residual_squared": 1.0,
        })
    pairs, endpoint = module.registered_cycle_residual_endpoint(rows)
    assert pairs == []
    assert endpoint["available"] is False
    assert endpoint["n_fast_cycles"] == 18
    assert endpoint["n_slow_cycles"] == 3


def test_registered_cycle_endpoint_uses_24_within_session_pairs() -> None:
    rows = []
    for session in range(3):
        for cadence, value in (("fast", 0.45), ("slow", 1.0)):
            for cycle_index in range(8):
                rows.append({
                    "cycle_id": f"session{session:02d}-{cadence}-{cycle_index:02d}",
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "cadence": cadence,
                    "observed_endpoint_residual_squared": value,
                })
    pairs, endpoint = module.registered_cycle_residual_endpoint(rows)
    assert len(pairs) == 24
    assert endpoint["available"] is True
    assert endpoint["ratio_gate"]["passed"] is True


def test_registered_cycle_endpoint_rejects_platform_cross_day_session() -> None:
    rows = []
    for session in range(3):
        for cadence, value in (("fast", 0.45), ("slow", 1.0)):
            for cycle_index in range(8):
                rows.append({
                    "cycle_id": f"session{session:02d}-{cadence}-{cycle_index:02d}",
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "cadence": cadence,
                    "observed_endpoint_residual_squared": value,
                })
    pairs, endpoint = module.registered_cycle_residual_endpoint(
        rows,
        eligible_sessions={1, 2},
        platform_session_integrity={"eligible_sessions": [1, 2]},
    )
    assert len(pairs) == 16
    assert endpoint["available"] is False
    assert endpoint["count_complete_session_pair_blocks"] == [0, 1, 2]
    assert endpoint["complete_session_pair_blocks"] == [1, 2]
    assert "platform-time" in endpoint["unavailable_reason"]


def test_registered_cycle_endpoint_discards_equal_but_incomplete_session_block() -> None:
    rows = []
    for session in range(3):
        cycle_count = 7 if session == 0 else 8
        for cadence, value in (("fast", 0.45), ("slow", 1.0)):
            for cycle_index in range(cycle_count):
                rows.append({
                    "cycle_id": f"session{session:02d}-{cadence}-{cycle_index:02d}",
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "cadence": cadence,
                    "observed_endpoint_residual_squared": value,
                })
    pairs, endpoint = module.registered_cycle_residual_endpoint(rows)
    assert len(pairs) == 16
    assert endpoint["complete_session_pair_blocks"] == [1, 2]
    assert endpoint["available"] is False


def test_exact_three_of_twenty_one_assignment_audit_cannot_reach_point_zero_five() -> None:
    rows = []
    for index in range(21):
        rows.append({
            "cadence": "slow" if index in {0, 1, 20} else "fast",
            "observed_endpoint_residual_squared": 0.009774 if index == 20 else 0.00045,
        })
    result = module.exact_slow_assignment_sensitivity(rows)
    assert result["available"] is True
    assert result["assignment_count"] == 1330
    assert result["assignments_at_least_as_extreme_without_global_maximum"] == 0
    assert result["one_sided_exact_p_value"] > 0.05
    assert result["valid_for_adjudication"] is False


def test_figure_tick_parent_size_protects_log_exponent_glyph_floor() -> None:
    module.configure_figure_style()
    assert float(module.mpl.rcParams["xtick.labelsize"]) >= 8.2
    assert float(module.mpl.rcParams["ytick.labelsize"]) >= 8.2


def test_platform_timing_keeps_queue_axis_separate() -> None:
    cycles = [{
        "cycle_id": "cycle-0",
        "session_index": 0,
        "cadence": "fast",
        "cadence_seconds": 90.0,
        "loop_query_ids": ["l1"],
        "mirror_query_ids": ["m1"],
    }]
    ledger = {
        "entries": [
            {
                "query_id": "l1",
                "execution_start_time_utc": "2026-08-12T00:00:00.000Z",
                "execution_end_time_utc": "2026-08-12T00:00:10.000Z",
            },
            {
                "query_id": "m1",
                "execution_start_time_utc": "2026-08-12T00:02:00.000Z",
                "execution_end_time_utc": "2026-08-12T00:02:10.000Z",
            },
        ]
    }
    _, cycle_rows, summary = module.platform_timing_rows(cycles, ledger, block_duration_seconds=600.0)
    assert cycle_rows[0]["controlled_injection_interval_seconds"] == 90.0
    assert cycle_rows[0]["platform_start_to_start_seconds"] == 120.0
    assert summary["exclusion_policy"].startswith("none")


def test_corrected_platform_session_integrity_uses_loop_run_start_span() -> None:
    residual_rows = []
    job_rows = []
    for session in range(3):
        for cadence_index, cadence in enumerate(("fast", "slow")):
            for cycle_index in range(8):
                cycle_id = f"session{session:02d}-{cadence}-{cycle_index:02d}"
                residual_rows.append({
                    "cycle_id": cycle_id,
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "cadence": cadence,
                })
                seconds = session * 86400 + cadence_index * 1000 + cycle_index * 90
                if session == 0 and cadence == "slow" and cycle_index == 7:
                    seconds += 86400
                job_rows.append({
                    "cycle_id": cycle_id,
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "cadence": cadence,
                    "job_role": "loop",
                    "execution_start_min_utc": (
                        np.datetime64("2026-08-16T00:00:00") + np.timedelta64(seconds, "s")
                    ).astype("datetime64[s]").astype(str) + "+00:00",
                })
                job_rows.append({
                    "cycle_id": cycle_id,
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "cadence": cadence,
                    "job_role": "mirror",
                    "execution_start_min_utc": (
                        np.datetime64("2026-08-16T00:00:30") + np.timedelta64(seconds, "s")
                    ).astype("datetime64[s]").astype(str) + "+00:00",
                })
    integrity = module.platform_session_pairing_integrity(
        residual_rows,
        job_rows,
        {"collection_correction": {
            "operational_session_wallclock_seconds": 3600.0,
            "operational_session_completion_window_seconds": 4500.0,
        }},
    )
    assert integrity["required_for_adjudication"] is True
    assert integrity["eligible_sessions"] == [1, 2]
    assert integrity["rows"][0]["same_session_time_complete"] is False


def test_paired_ratio_gate_fixture_is_null() -> None:
    fast = np.asarray([0.64, 0.65, 0.66])
    slow = np.asarray([0.64, 0.65, 0.66])
    gate = module.cadence_ratio_gate(fast, slow)
    assert gate["passed"] is False
    assert gate["ratio"] == 1.0


def _v4_rows(*, sessions: int = 2, pairs_per_session: int = 20, drop: set[tuple[int, int, str]] | None = None) -> list[dict[str, object]]:
    """Rows shaped like the forty-minute collection: two sessions, twenty pairs each."""
    drop = drop or set()
    rows: list[dict[str, object]] = []
    for session in range(sessions):
        for cadence, value in (("fast", 0.45), ("slow", 1.0)):
            for cycle_index in range(pairs_per_session):
                if (session, cycle_index, cadence) in drop:
                    continue
                rows.append({
                    "cycle_id": f"session{session:02d}-{cadence}-{cycle_index:02d}",
                    "session_index": session,
                    "cycle_index": cycle_index,
                    "cadence": cadence,
                    # A little spread so the permutation null is not degenerate.
                    "observed_endpoint_residual_squared": value * (1.0 + 0.05 * cycle_index),
                })
    return rows


def test_cycle_pair_granularity_costs_one_pair_not_the_whole_session() -> None:
    """The behavioural consequence of the shared baseline, stated as a contrast.

    Under the per-cycle zero-field design a lost cycle took its session's other 19 pairs
    with it, because the surviving cycles had no common reference.  The session-start
    measurement survives the loss, so the same collection loses one pair.  Both branches
    are exercised on identical rows so the difference is the granularity and nothing else.
    """
    rows = _v4_rows(drop={(0, 7, "slow")})
    _, block = module.registered_cycle_residual_endpoint(
        rows,
        expected_pair_count=40,
        expected_pairs_per_session=20,
        minimum_pair_count=30,
    )
    assert block["observed_pair_count"] == 20
    assert block["available"] is False

    pairs, endpoint = module.registered_cycle_residual_endpoint(
        rows,
        expected_pair_count=40,
        expected_pairs_per_session=20,
        minimum_pair_count=30,
        pair_discard_granularity="cycle_pair",
        sessions_with_baseline={0, 1},
    )
    assert len(pairs) == 39
    assert endpoint["available"] is True
    assert [row["cycle_index"] for row in endpoint["discarded_cycle_pairs"]] == [7]
    assert endpoint["discarded_session_pair_blocks"] == []
    assert "missing slow cycle" in endpoint["discarded_cycle_pairs"][0]["reason"]


def test_a_missing_session_baseline_discards_that_whole_block() -> None:
    """The one loss the amortisation does not survive, and it is scoped to its session."""
    pairs, endpoint = module.registered_cycle_residual_endpoint(
        _v4_rows(),
        expected_pair_count=40,
        expected_pairs_per_session=20,
        minimum_pair_count=30,
        pair_discard_granularity="cycle_pair",
        sessions_with_baseline={1},
    )
    assert len(pairs) == 20
    assert endpoint["available"] is False
    assert [row["session_index"] for row in endpoint["discarded_session_pair_blocks"]] == [0]
    assert "every cycle of both cadence blocks" in endpoint["discarded_session_pair_blocks"][0]["reason"]
    assert "30" in endpoint["unavailable_reason"]


def test_the_permutation_calibration_decides_and_the_delta_method_is_still_reported() -> None:
    """Primary is read from the registration, not chosen after seeing either verdict."""
    pairs, endpoint = module.registered_cycle_residual_endpoint(
        _v4_rows(),
        expected_pair_count=40,
        expected_pairs_per_session=20,
        minimum_pair_count=30,
        pair_discard_granularity="cycle_pair",
        sessions_with_baseline={0, 1},
        primary_adjudication="cadence_ratio_permutation_gate",
    )
    assert len(pairs) == 40
    permutation = endpoint["permutation_gate"]
    assert permutation is not None
    # Same point statistic under both calibrations: only the critical value differs.
    assert permutation["ratio"] == endpoint["ratio_gate"]["ratio"]
    assert permutation["frozen_delta_method"]["passed"] == endpoint["ratio_gate"]["passed"]

    verdict = module.decision(endpoint, {"ratio_gate": {"passed": True}}, {
        "session_paired_ratio_sensitivity": {"passed": True}
    })
    assert verdict["primary_adjudication"] == "cadence_ratio_permutation_gate"
    assert verdict["primary_p_value"] == permutation["p_value"]
    assert verdict["secondary_frozen_delta_method_passed"] is endpoint["ratio_gate"]["passed"]
    assert verdict["registered_cycle_residual_endpoint_passed"] is permutation["passed"]
    assert "permutation" in verdict["statement"]


def test_a_disagreement_between_calibrations_is_reported_not_resolved() -> None:
    """The anti-conservative case: delta-method passes, the exact calibration does not.

    The frozen gate is the one that can pass here, so a rule that took whichever verdict
    was available would systematically inherit its inflated size.  The decision must keep
    the permutation verdict and record the disagreement.
    """
    endpoint = {
        "available": True,
        "primary_adjudication": "cadence_ratio_permutation_gate",
        "ratio_gate": {"passed": True},
        "permutation_gate": {"passed": False, "p_value": 0.11},
        "n_fast_cycles": 40,
        "n_slow_cycles": 40,
        "expected_pair_count": 40,
    }
    verdict = module.decision(endpoint, {"ratio_gate": {"passed": True}}, {
        "session_paired_ratio_sensitivity": {"passed": True}
    })
    assert verdict["headline_verdict"] == "NO-GO"
    assert verdict["calibrations_agree"] is False
    assert verdict["secondary_frozen_delta_method_passed"] is True
    assert "never used to decide" in verdict["secondary_readout_role"]


def test_the_drift_sensitivity_reruns_all_three_registered_shapes() -> None:
    pairs, endpoint = module.registered_cycle_residual_endpoint(
        _v4_rows(),
        expected_pair_count=40,
        expected_pairs_per_session=20,
        minimum_pair_count=30,
        pair_discard_granularity="cycle_pair",
        sessions_with_baseline={0, 1},
        primary_adjudication="cadence_ratio_permutation_gate",
    )
    hardware_report = {
        "session_shared_baseline": {
            "drift_qc": [
                {
                    "session_index": session,
                    "measured": True,
                    "fields": [
                        {"field": "h1", "phase_drift_z": 1.2},
                        {"field": "h2", "phase_drift_z": -0.8},
                    ],
                    "worst_absolute_z": 1.2,
                    "sigma_threshold": 3.0,
                    "static_within_shot_noise": True,
                }
                for session in (0, 1)
            ]
        }
    }
    result = module.shared_baseline_drift_sensitivity(
        hardware_report,
        pairs,
        {"endpoint_shot_floor_shared": 8.577935e-05},
        session_block_order={0: "fast", 1: "slow"},
        primary_passed=endpoint["permutation_gate"]["passed"],
    )
    assert result is not None
    assert result["available"] is True
    assert sorted(result["shapes"]) == result["registered_shapes"]
    assert len(result["registered_shapes"]) == 3
    assert result["endpoint_offset_at_upper_limit"] > 0.0
    assert result["all_shapes_agree_with_each_other"] is True
    assert result["all_shapes_agree_with_primary"] is True
    # Agreement must be measured against the verdict, not among the shapes: flipping the
    # claimed primary verdict has to flip the agreement flag, or the field says nothing.
    flipped = module.shared_baseline_drift_sensitivity(
        hardware_report,
        pairs,
        {"endpoint_shot_floor_shared": 8.577935e-05},
        session_block_order={0: "fast", 1: "slow"},
        primary_passed=not endpoint["permutation_gate"]["passed"],
    )
    assert flipped["all_shapes_agree_with_primary"] is False
    assert flipped["all_shapes_agree_with_each_other"] is True
    # The asymmetric shape has to load the two arms unequally, or the re-run is vacuous.
    ramp = result["shapes"]["linear_ramp"]
    assert ramp["removed_from_fast_arm"] != ramp["removed_from_slow_arm"]


def test_the_drift_sensitivity_is_absent_rather_than_assumed_without_a_readout() -> None:
    """No session produced both measurements: the drift is unbounded, not zero."""
    result = module.shared_baseline_drift_sensitivity(
        {"session_shared_baseline": {"drift_qc": [{"session_index": 0, "measured": False}]}},
        [{"session_index": 0, "fast_endpoint_residual_squared": 0.4, "slow_endpoint_residual_squared": 1.0}],
        {"endpoint_shot_floor_shared": 8.577935e-05},
    )
    assert result["available"] is False
    assert result["endpoint_offset_at_upper_limit"] is None
    assert "unbounded" in result["unavailable_reason"]
    assert module.shared_baseline_drift_sensitivity({}, [], None) is None
