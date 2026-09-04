from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts import drift_campaign_v4
from scripts import preflight_b4_cadence_collection_correction as preflight
from scripts import run_b4_cadence_pair_hardware as module
from scripts import run_cadence_pair_loop as cadence


ROOT = Path(__file__).resolve().parents[1]
LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_v1.json"
CORRECTED_LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v2.json"
AMENDED_LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v3.json"
FORTY_MINUTE_LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json"
BACKEND_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"
BACKEND_176_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"


@pytest.mark.parametrize(
    ("backend_path", "backend_id"),
    [
        (BACKEND_CONFIG, "tianyan-287"),
        (BACKEND_176_CONFIG, "tianyan176"),
    ],
)
def test_backend_pin_preserves_each_exact_platform_spelling(
    backend_path: Path,
    backend_id: str,
) -> None:
    loop_config = cadence.load_config(LOOP_CONFIG)
    loop_config["registered_backend_id"] = backend_id
    backend_config = drift_campaign_v4.load_config(backend_path)
    plan = module.build_plan_payload(
        loop_config,
        backend_config,
        operational_start_utc=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    )
    assert plan["backend_id"] == backend_id


def test_backend_pin_rejects_different_platform_spelling() -> None:
    loop_config = cadence.load_config(LOOP_CONFIG)
    loop_config["registered_backend_id"] = "tianyan-287"
    backend_config = drift_campaign_v4.load_config(BACKEND_176_CONFIG)
    with pytest.raises(ValueError, match="registered_backend_id"):
        module.build_plan_payload(
            loop_config,
            backend_config,
            operational_start_utc=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        )


def test_backend_pin_is_required() -> None:
    loop_config = cadence.load_config(LOOP_CONFIG)
    loop_config.pop("registered_backend_id")
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    with pytest.raises(ValueError, match="must declare"):
        module.build_plan_payload(
            loop_config,
            backend_config,
            operational_start_utc=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        )


def test_existing_plan_reuse_rejects_changed_backend_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_path = tmp_path / "backend.json"
    backend_path.write_text(BACKEND_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    stage1_manifest_path = tmp_path / "stage1.json"
    stage1_manifest_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "campaign"
    output.mkdir()
    plan = {
        "backend_id": "tianyan-287",
        "source_hashes": {
            "loop_config_sha256": module.digest_file(LOOP_CONFIG),
            "backend_config_sha256": module.digest_file(backend_path),
        },
    }
    (output / module.PLAN_NAME).write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(module, "verify_stage1_manifest", lambda _path: {})
    monkeypatch.setattr(module.drift_campaign_v4, "prepare_v4", lambda *_args: {})

    reused = module.prepare_plan(
        LOOP_CONFIG,
        backend_path,
        BACKEND_176_CONFIG,
        stage1_manifest_path,
        output,
        start_utc=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    )
    assert reused == plan

    backend_path.write_text(backend_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend-config hash changed"):
        module.prepare_plan(
            LOOP_CONFIG,
            backend_path,
            BACKEND_176_CONFIG,
            stage1_manifest_path,
            output,
            start_utc=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        )


def test_preflight_defaults_to_latest_frozen_v4_plan() -> None:
    assert preflight.DEFAULT_CONFIG == FORTY_MINUTE_LOOP_CONFIG
    assert preflight.DEFAULT_BACKEND_CONFIG == BACKEND_176_CONFIG
    assert preflight.DEFAULT_PEER_CONFIG == BACKEND_CONFIG


def test_runner_defaults_to_registered_t176_migration_and_quarantine() -> None:
    assert module.DEFAULT_LOOP_CONFIG == FORTY_MINUTE_LOOP_CONFIG
    assert module.DEFAULT_BACKEND_CONFIG == BACKEND_176_CONFIG
    assert module.DEFAULT_PEER_CONFIG == BACKEND_CONFIG
    assert module.DEFAULT_STAGE1_MANIFEST.name == "B4_T176_MIGRATION_CURATED_MANIFEST_20260823.json"
    assert str(module.DEFAULT_OUTPUT).replace("\\", "/").endswith(
        "/quarantine/tianyan176/B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823"
    )


def test_role_envelope_budget_recomputes_quota_and_parallel_wall_from_plan_counts() -> None:
    cycles = [
        {
            "session_index": 0,
            "sensing_shots_per_setting": 10,
            "mirror_seeds": [1],
            "mirror_shots_per_task": 20,
            "cadence_seconds": 90.0,
        },
        {
            "session_index": 0,
            "sensing_shots_per_setting": 10,
            "mirror_seeds": [],
            "mirror_shots_per_task": 20,
            "cadence_seconds": 360.0,
        },
    ]
    correction = {
        "sensing_settings_per_cycle": 2,
        "baseline_measurements_per_session": 2,
        "baseline_measurement_positions": ["session_start", "session_end"],
        "baseline_used_by_estimate": "session_start",
        "baseline_sharing_scope": "whole_session_both_cadence_blocks",
        "baseline_shots_per_setting": 30,
        "baseline_end_shots_per_setting": 30,
        "baseline_settings_per_measurement": 2,
        "timing_budget_model": "role_envelope_sum_task_runtime",
        "role_task_runtime_seconds": {"baseline": 5.0, "sense": 3.0, "mirror": 7.0},
        "role_settings_per_job": {"baseline": 2, "sense": 2, "mirror": 2},
        "role_jobs_per_session": {"baseline": 2, "sense": 2, "mirror": 1},
        "quota_seconds_per_session": 46.0,
        "execution_wall_seconds_per_session": 23.0,
    }
    backend_config = {
        "backend": {
            "tb6_measured_timing": {
                "effective_shots_per_second": 100.0,
                "fixed_overhead_seconds_per_setting": 1.0,
            }
        }
    }
    budget = module.measured_collection_budget(
        cycles,
        backend_config,
        correction,
        daily_window_seconds=30.0,
    )
    assert budget["timing_budget_model"] == "role_envelope_sum_task_runtime"
    assert budget["daily_budget_metric"] == "quota_seconds"
    assert budget["estimated_busy_seconds_total"] == 46.0
    assert budget["quota_seconds_total"] == 46.0
    assert budget["execution_wall_seconds_total"] == 23.0
    assert budget["session_rows"][0]["role_job_counts"] == {
        "baseline": 2,
        "sense": 2,
        "mirror": 1,
    }
    # Wall time fits 30 s, quota does not. Daily gate must use quota.
    assert budget["execution_wall_seconds_total"] < 30.0
    assert budget["daily_budget_passed"] is False


def test_role_envelope_budget_rejects_declared_job_counts_not_in_plan() -> None:
    cycles = [{
        "session_index": 0,
        "sensing_shots_per_setting": 10,
        "mirror_seeds": [],
        "mirror_shots_per_task": 20,
        "cadence_seconds": 90.0,
    }]
    correction = {
        "sensing_settings_per_cycle": 2,
        "timing_budget_model": "role_envelope_sum_task_runtime",
        "role_task_runtime_seconds": {"baseline": 5.0, "sense": 3.0, "mirror": 7.0},
        "role_settings_per_job": {"baseline": 2, "sense": 2, "mirror": 2},
        "role_jobs_per_session": {"baseline": 0, "sense": 2, "mirror": 0},
        "quota_seconds_per_session": 6.0,
        "execution_wall_seconds_per_session": 3.0,
    }
    backend_config = {
        "backend": {
            "tb6_measured_timing": {
                "effective_shots_per_second": 100.0,
                "fixed_overhead_seconds_per_setting": 1.0,
            }
        }
    }
    with pytest.raises(ValueError, match="role job counts disagree"):
        module.measured_collection_budget(cycles, backend_config, correction)


def test_plan_preserves_frozen_endpoint_under_operational_session_compression() -> None:
    loop_config = cadence.load_config(LOOP_CONFIG)
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    start = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    plan = module.build_plan_payload(loop_config, backend_config, operational_start_utc=start)
    assert plan["expected"] == {
        "sessions": 3,
        "blocks": 6,
        "cycles": 21,
        "loop_jobs": 21,
        "mirror_jobs": 21,
        "baseline_jobs": 0,
        "loop_tasks": 84,
        "mirror_tasks": 336,
        "baseline_tasks": 0,
        "total_tasks": 420,
        "complete_cadence_pairs": 24,
    }
    assert [row["block_order"] for row in plan["sessions"]] == [
        ["fast", "slow"],
        ["slow", "fast"],
        ["fast", "slow"],
    ]
    assert [row["gate_amplitude"] for row in plan["sessions"]] == [0.05, 0.1, 0.25]
    assert [row["virtual_start_seconds"] for row in plan["sessions"]] == [0.0, 86400.0, 172800.0]
    operational = [module.parse_utc(row["operational_start_utc"]) for row in plan["sessions"]]
    assert [(right - left).total_seconds() for left, right in zip(operational, operational[1:])] == [1200.0, 1200.0]
    assert all(len(cycle["mirror_seeds"]) == 8 for session in plan["sessions"] for cycle in session["cycles"])
    assert {cycle["sensing_shots_per_setting"] for session in plan["sessions"] for cycle in session["cycles"]} == {22050, 88200}


def test_program_batches_stay_below_platform_submission_limit() -> None:
    loop_config = cadence.load_config(LOOP_CONFIG)
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    plan = module.build_plan_payload(
        loop_config,
        backend_config,
        operational_start_utc=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    )
    cycle = plan["sessions"][0]["cycles"][0]
    physical = backend_config["backend"]["physical_qubits"]
    phase_time = loop_config["sensing"]["phase_time_seconds"]
    sensing = module.loop_programs(physical, cycle["sense_fields"], phase_time, cycle)
    mirrors = module.mirror_programs(
        physical,
        cycle["mirror_fields"],
        [0.0, 0.0],
        phase_time,
        loop_config["mirror"]["depth"],
        cycle,
    )
    assert len(sensing) == 4
    assert len(mirrors) == 16
    assert len(sensing) < 49
    assert len(mirrors) < 49
    assert {row["strategy"] for row in mirrors} == {"fixed", "adaptive"}
    assert len({row["pair_id"] for row in mirrors}) == 8


def test_corrected_plan_uses_three_complete_eight_pair_sessions() -> None:
    loop_config = cadence.load_config(CORRECTED_LOOP_CONFIG)
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    start = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
    plan = module.build_plan_payload(loop_config, backend_config, operational_start_utc=start)
    assert plan["expected"] == {
        "sessions": 3,
        "blocks": 6,
        "cycles": 48,
        "loop_jobs": 48,
        "mirror_jobs": 48,
        "baseline_jobs": 0,
        "loop_tasks": 192,
        "mirror_tasks": 96,
        "baseline_tasks": 0,
        "total_tasks": 288,
        "complete_cadence_pairs": 24,
        "cycles_per_cadence_per_session": 8,
        # The v2 plan carries its own zero-field condition in every cycle, so it has no
        # shared baseline and its pair block is still all-or-nothing.
        "sensing_settings_per_cycle": 4,
        "mirror_repetitions_per_cycle": 1,
        "mirror_qc_cycle_indices": None,
        "minimum_adjudicated_cycle_pairs": 24,
        "minimum_sessions_per_block_order": 0,
        "pair_discard_granularity": "session_pair_block",
        "pair_block_discard_condition": "either cadence block incomplete",
    }
    for session in plan["sessions"]:
        fast = [row for row in session["cycles"] if row["cadence"] == "fast"]
        slow = [row for row in session["cycles"] if row["cadence"] == "slow"]
        assert len(fast) == len(slow) == 8
        assert [row["registered_pair_id"] for row in fast] == [row["registered_pair_id"] for row in slow]
        assert all(row["sensing_shots_per_setting"] == 3072 for row in session["cycles"])
        assert all(len(row["mirror_seeds"]) == 1 for row in session["cycles"])
        assert [row["mirror_seeds"] for row in fast] == [row["mirror_seeds"] for row in slow]
        assert session["programmed_session_wallclock_seconds"] == 3600.0
        assert (
            module.parse_utc(session["operational_deadline_utc"])
            - module.parse_utc(session["operational_start_utc"])
        ).total_seconds() == 3600.0
        assert (
            module.parse_utc(session["operational_completion_deadline_utc"])
            - module.parse_utc(session["operational_start_utc"])
        ).total_seconds() == 4500.0
    operational = [module.parse_utc(row["operational_start_utc"]) for row in plan["sessions"]]
    assert [(right - left).total_seconds() for left, right in zip(operational, operational[1:])] == [86400.0, 86400.0]
    budget = plan["measured_collection_budget"]
    assert budget["total_cycles"] == 48
    assert budget["total_shots"] == 983040
    assert budget["total_settings"] == 288
    assert budget["programmed_hold_seconds_total"] == 10800.0
    assert budget["twenty_minute_daily_budget_passed"] is True
    assert all(row["estimated_busy_seconds"] < 1200.0 for row in budget["session_rows"])


def test_corrected_mirror_pair_id_matches_fast_and_slow_cycle_pair() -> None:
    loop_config = cadence.load_config(CORRECTED_LOOP_CONFIG)
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    plan = module.build_plan_payload(
        loop_config,
        backend_config,
        operational_start_utc=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
    )
    session = plan["sessions"][0]
    fast = next(row for row in session["cycles"] if row["cadence"] == "fast" and row["cycle_index"] == 0)
    slow = next(row for row in session["cycles"] if row["cadence"] == "slow" and row["cycle_index"] == 0)
    physical = backend_config["backend"]["physical_qubits"]
    phase_time = loop_config["sensing"]["phase_time_seconds"]
    fast_programs = module.mirror_programs(physical, fast["mirror_fields"], [0.0, 0.0], phase_time, 2, fast)
    slow_programs = module.mirror_programs(physical, slow["mirror_fields"], [0.0, 0.0], phase_time, 2, slow)
    assert {row["pair_id"] for row in fast_programs} == {row["pair_id"] for row in slow_programs}
    assert len(fast_programs) == len(slow_programs) == 2


def test_incomplete_corrected_session_discards_entire_pair_block() -> None:
    loop_config = cadence.load_config(CORRECTED_LOOP_CONFIG)
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    plan = module.build_plan_payload(
        loop_config,
        backend_config,
        operational_start_utc=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
    )
    completed = [
        row
        for row in plan["sessions"][0]["cycles"]
        if not (row["cadence"] == "slow" and row["cycle_index"] == 7)
    ]
    records = [{"event": "cadence_cycle_completed", **row} for row in completed]
    status = module.registered_pair_block_status(plan, records)
    assert status["complete_session_pair_blocks"] == []
    assert status["registered_cycle_pairs"] == 0
    assert status["discarded_session_pair_blocks"][0]["fast_completed"] == 8
    assert status["discarded_session_pair_blocks"][0]["slow_completed"] == 7


def test_explicitly_discarded_session_cannot_be_revived_by_late_cycles() -> None:
    loop_config = cadence.load_config(CORRECTED_LOOP_CONFIG)
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    plan = module.build_plan_payload(
        loop_config,
        backend_config,
        operational_start_utc=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
    )
    records = [
        {"event": "cadence_cycle_completed", **row}
        for row in plan["sessions"][0]["cycles"]
    ]
    records.append({
        "event": "session_pair_block_discarded",
        "session_index": 0,
        "reason": "same-session deadline expired",
    })
    status = module.registered_pair_block_status(plan, records)
    assert status["complete_session_pair_blocks"] == []
    assert status["registered_cycle_pairs"] == 0
    assert status["discarded_session_pair_blocks"][0]["reason"] == "same-session deadline expired"


def test_same_session_deadline_is_hard() -> None:
    deadline = "2026-08-16T13:00:00+00:00"
    module.require_session_open(
        deadline,
        stage="test",
        now_utc=datetime(2026, 8, 16, 13, 0, tzinfo=timezone.utc),
    )
    try:
        module.require_session_open(
            deadline,
            stage="test",
            now_utc=datetime(2026, 8, 16, 13, 0, 1, tzinfo=timezone.utc),
        )
    except module.SessionPairBlockExpired:
        pass
    else:
        raise AssertionError("expired session was accepted")


def test_amended_plan_adds_deadline_slack_without_adding_busy_time() -> None:
    loop_config = cadence.load_config(AMENDED_LOOP_CONFIG)
    backend_config = drift_campaign_v4.load_config(BACKEND_CONFIG)
    plan = module.build_plan_payload(
        loop_config,
        backend_config,
        operational_start_utc=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
    )
    assert plan["expected"]["sessions"] == 12
    assert plan["expected"]["cycles"] == 192
    assert plan["expected"]["complete_cadence_pairs"] == 96
    assert plan["expected"]["minimum_adjudicated_cycle_pairs"] == 80
    assert plan["expected"]["mirror_qc_cycle_indices"] == [0, 7]
    # 192 loop jobs of four settings; mirror QC brackets each cadence block, so
    # four mirror jobs per session rather than sixteen.
    assert plan["expected"]["loop_tasks"] == 768
    assert plan["expected"]["mirror_jobs"] == 48
    assert plan["expected"]["mirror_tasks"] == 96
    orders = [tuple(row["block_order"]) for row in plan["sessions"]]
    assert orders.count(("fast", "slow")) == orders.count(("slow", "fast")) == 6
    for session in plan["sessions"]:
        assert session["programmed_session_wallclock_seconds"] == 3600.0
        assert session["session_deadline_slack_seconds"] == 1800.0
        assert all(row["sensing_shots_per_setting"] == 14336 for row in session["cycles"])
        carried = [row for row in session["cycles"] if row["mirror_seeds"]]
        assert sorted({row["cycle_index"] for row in carried}) == [0, 7]
        assert len(carried) == 4
    budget = plan["measured_collection_budget"]
    assert budget["total_shots"] == 11403264
    assert budget["twenty_minute_daily_budget_passed"] is True
    assert all(row["estimated_busy_seconds"] < 1200.0 for row in budget["session_rows"])


def _v4_plan() -> dict:
    return module.build_plan_payload(
        cadence.load_config(FORTY_MINUTE_LOOP_CONFIG),
        drift_campaign_v4.load_config(BACKEND_176_CONFIG),
        operational_start_utc=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
    )


def test_forty_minute_plan_submits_two_setting_cycles_and_shared_baselines() -> None:
    """The whole point of v4: the zero-field condition leaves the cycle.

    80 cycles of two settings instead of four, plus two shared baseline jobs a session.
    The task count is what the machine time is actually spent on, so it is pinned here
    rather than inferred from the config's own declaration of itself.
    """
    plan = _v4_plan()
    expected = plan["expected"]
    assert expected["sessions"] == 2
    assert expected["cycles"] == 80
    assert expected["sensing_settings_per_cycle"] == 2
    assert expected["loop_tasks"] == 160
    assert expected["baseline_jobs"] == 4
    assert expected["baseline_tasks"] == 8
    assert expected["mirror_jobs"] == 4
    assert expected["mirror_tasks"] == 8
    assert expected["total_tasks"] == 176
    assert expected["complete_cadence_pairs"] == 40
    assert expected["minimum_adjudicated_cycle_pairs"] == 30
    assert expected["pair_discard_granularity"] == "cycle_pair"
    orders = [tuple(row["block_order"]) for row in plan["sessions"]]
    assert orders.count(("fast", "slow")) == orders.count(("slow", "fast")) == 1


def test_forty_minute_budget_uses_the_declared_t176_role_envelope() -> None:
    """Quota uses the per-task sum; parallel execution wall stays separate."""
    plan = _v4_plan()
    correction = plan["collection_correction"]
    budget = plan["measured_collection_budget"]
    assert plan["backend_id"] == "tianyan176"
    assert budget["timing_budget_model"] == "role_envelope_sum_task_runtime"
    assert budget["daily_budget_metric"] == "quota_seconds"
    assert budget["modelled_shots_per_second"] == correction[
        "shot_rate_per_second_used"
    ] == pytest.approx(1414.98122405956)
    assert budget["total_shots"] == correction["shots_total"] == 1243840
    assert budget["total_settings"] == 2 * correction["settings_per_session"] == 176
    assert budget["total_baseline_jobs"] == 4
    assert budget["quota_seconds_total"] == pytest.approx(1295.296)
    assert budget["execution_wall_seconds_total"] == pytest.approx(647.648)
    assert budget["estimated_busy_seconds_total"] == pytest.approx(1295.296)
    assert all(
        row["quota_seconds"] == pytest.approx(647.648)
        and row["execution_wall_seconds"] == pytest.approx(323.824)
        for row in budget["session_rows"]
    )
    assert budget["estimated_busy_seconds_total"] <= correction["machine_time_ceiling_seconds"]
    assert budget["daily_budget_passed"] is True


def test_forty_minute_sessions_hold_the_baselines_outside_the_cadence_grid() -> None:
    """A slow baseline queue must never push a cycle off its cadence tick.

    The lead and trail are wall clock only: they add no shots and no jobs beyond the two
    already counted, which is why the busy total is unchanged by them.
    """
    plan = _v4_plan()
    correction = plan["collection_correction"]
    lead = correction["baseline_lead_seconds"]
    assert lead == correction["baseline_trail_seconds"] == 1200.0
    for session in plan["sessions"]:
        measurements = session["baseline_measurements"]
        assert [row["position"] for row in measurements] == ["session_start", "session_end"]
        opening, closing = measurements
        assert opening["used_by_estimate"] is True
        assert closing["used_by_estimate"] is False
        assert all(row["settings"] == 2 for row in measurements)
        assert opening["shots_per_setting"] == correction["baseline_shots_per_setting"]
        assert closing["shots_per_setting"] == correction["baseline_end_shots_per_setting"]
        start = module.parse_utc(session["operational_start_utc"])
        assert module.parse_utc(opening["target_utc"]) == start
        first_sense = module.parse_utc(session["cycles"][0]["sense_target_utc"])
        assert (first_sense - start).total_seconds() == lead
        last_mirror = max(
            module.parse_utc(row["mirror_target_utc"]) for row in session["cycles"]
        )
        assert module.parse_utc(closing["target_utc"]) >= last_mirror
        completion = module.parse_utc(session["operational_completion_deadline_utc"])
        assert (completion - module.parse_utc(closing["target_utc"])).total_seconds() >= (
            correction["baseline_trail_seconds"]
        )


def test_a_lost_cycle_discards_its_own_pair_and_not_the_session() -> None:
    """The v3 rule threw away 20 pairs to lose one; v4 throws away one.

    It can, because the zero-field condition both members subtract is a separate job that
    survives a lost cycle, which is exactly the condition recorded in
    pair_block_discard_condition.
    """
    plan = _v4_plan()
    session = plan["sessions"][0]
    dropped = next(
        row for row in session["cycles"] if row["cadence"] == "slow" and row["cycle_index"] == 7
    )
    records: list[dict] = [
        {
            "event": "session_baseline_measured",
            "session_index": int(row["session_index"]),
            "position": str(row["position"]),
        }
        for other in plan["sessions"]
        for row in other["baseline_measurements"]
    ]
    records += [
        {"event": "cadence_cycle_completed", **row}
        for other in plan["sessions"]
        for row in other["cycles"]
        if row["cycle_id"] != dropped["cycle_id"]
    ]
    status = module.registered_pair_block_status(plan, records)
    assert status["discard_granularity"] == "cycle_pair"
    assert status["registered_cycle_pairs"] == 39
    assert status["complete_session_pair_blocks"] == [0, 1]
    assert status["discarded_session_pair_blocks"] == []
    assert [row["registered_pair_id"] for row in status["discarded_cycle_pairs"]] == [
        dropped["registered_pair_id"]
    ]
    assert "missing slow cycle" in status["discarded_cycle_pairs"][0]["reason"]
    assert status["registered_cycle_pairs"] >= plan["expected"][
        "minimum_adjudicated_cycle_pairs"
    ]


def test_a_missing_session_start_baseline_discards_the_whole_session() -> None:
    """Every cycle in both blocks subtracted it, so without it nothing in the session pairs.

    This is the one failure that still costs a whole block, and losing the *closing*
    measurement must not: it is a drift readout that no cycle ever subtracted.
    """
    plan = _v4_plan()
    completed = [
        {"event": "cadence_cycle_completed", **row}
        for session in plan["sessions"]
        for row in session["cycles"]
    ]
    baselines = [
        {
            "event": "session_baseline_measured",
            "session_index": int(row["session_index"]),
            "position": str(row["position"]),
        }
        for session in plan["sessions"]
        for row in session["baseline_measurements"]
    ]
    without_opening = [
        row
        for row in baselines
        if not (row["session_index"] == 0 and row["position"] == "session_start")
    ]
    status = module.registered_pair_block_status(plan, completed + without_opening)
    assert status["complete_session_pair_blocks"] == [1]
    assert status["registered_cycle_pairs"] == 20
    assert status["discarded_session_pair_blocks"][0]["session_index"] == 0
    assert "session_start baseline" in status["discarded_session_pair_blocks"][0]["reason"]
    assert len(status["discarded_cycle_pairs"]) == 20
    # And the registered consequence: one session is below the adjudicable minimum.
    assert status["registered_cycle_pairs"] < plan["expected"][
        "minimum_adjudicated_cycle_pairs"
    ]

    without_closing = [
        row
        for row in baselines
        if not (row["session_index"] == 0 and row["position"] == "session_end")
    ]
    kept = module.registered_pair_block_status(plan, completed + without_closing)
    assert kept["registered_cycle_pairs"] == 40
    assert kept["discarded_session_pair_blocks"] == []


def test_forty_minute_plan_carries_its_own_reachability_evidence() -> None:
    plan = _v4_plan()
    reachability = plan["registered_endpoint_reachability"]
    correction = plan["collection_correction"]
    assert reachability["rule"] == "cadence_ratio_permutation_gate"
    assert reachability["binding_pair_count"] == correction["minimum_adjudicated_cycle_pairs"]
    assert reachability["expected_ratio"] == pytest.approx(correction["expected_ratio"])
    assert reachability["binding_power"] >= correction["minimum_power"] - 0.016
    assert reachability["binding_boundary_size"] <= correction["maximum_boundary_size"] + 0.016
    assert reachability["reachable"] is True


def test_safe_metadata_never_writes_qcis() -> None:
    rows = [{"label": "x", "qcis": "M Q1", "kind": "test", "analysis_role": "test"}]
    safe = module.safe_task_metadata(rows)
    assert "qcis" not in safe[0]
    assert safe[0]["qcis_sha256"]
    json.dumps(safe)
