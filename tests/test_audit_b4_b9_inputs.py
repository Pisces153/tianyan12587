from __future__ import annotations

from scripts import audit_b4_b9_inputs as module


def test_primary_sf_audit_blocks_missing_platform_timestamps() -> None:
    rows = [{
        "event": "collected",
        "observations": [
            {
                "analysis_role": "primary_sf_only_when_non_event_and_same_regime",
                "primary_sf_eligible": True,
                "burst_flag": False,
                "platform_timestamp_raw": None,
            },
            {
                "analysis_role": "primary_sf_short_lag_reference",
                "primary_sf_eligible": True,
                "burst_flag": False,
                "platform_timestamp_raw": "2026-08-10T12:00:00Z",
            },
        ],
    }]
    report = module.audit_primary_sf_observations(rows)
    assert report["status"] == "blocked_missing_platform_timestamps"
    assert report["observations_missing_platform_timestamp"] == 1
    assert report["client_timestamp_substitution_used"] is False


def test_primary_sf_audit_uses_verified_platform_time_sidecar() -> None:
    rows = [{
        "event": "collected",
        "observations": [
            {
                "query_id": f"task-{index}",
                "analysis_role": "primary_sf_only_when_non_event_and_same_regime",
                "primary_sf_eligible": True,
                "burst_flag": False,
                "platform_timestamp_raw": None,
            }
            for index in range(3)
        ],
    }]
    index = {
        f"task-{value}": {"execution_start_time_utc": f"2026-08-10T13:0{value}:00.000Z"}
        for value in range(3)
    }
    report = module.audit_primary_sf_observations(rows, platform_time_index=index)
    assert report["status"] == "ready_for_t287_sf"
    assert report["observations_with_platform_timestamp"] == 3
    assert report["sidecar_ledger_observations"] == 3
    assert report["client_timestamp_substitution_used"] is False
    assert all(row["platform_timestamp_raw"] is None for row in rows[0]["observations"])


def test_cadence_audit_uses_cycle_pair_units_and_preserves_frozen_count() -> None:
    rows = [{
        "event": "cadence_cycle_completed",
        "cycle_id": "cycle-1",
        "cadence": "fast",
        "mirror_scores": [
            {"pair_id": "pair-1", "strategy": "adaptive"},
            {"pair_id": "pair-1", "strategy": "fixed"},
            {"pair_id": "pair-2", "strategy": "adaptive"},
            {"pair_id": "pair-2", "strategy": "fixed"},
        ],
    }]
    report = module.audit_cadence_completion(rows, frozen_pair_count=3)
    assert report["complete_registered_mirror_pairs"] == 2
    assert report["frozen_pair_count_matches_completed_cycles"] is False
    assert report["frozen_pair_count_matches_registered_mirror_pairs"] is False
    assert report["endpoint_status"] == "not_tested_frozen_pair_count_mismatch"


def test_cadence_audit_counts_reused_pair_id_once_when_every_cycle_is_complete() -> None:
    rows = [
        {
            "event": "cadence_cycle_completed",
            "cycle_id": "cycle-1",
            "cadence": "fast",
            "mirror_scores": [
                {"pair_id": "pair-1", "strategy": "adaptive"},
                {"pair_id": "pair-1", "strategy": "fixed"},
            ],
        },
        {
            "event": "cadence_cycle_completed",
            "cycle_id": "cycle-2",
            "cadence": "slow",
            "mirror_scores": [
                {"pair_id": "pair-1", "strategy": "adaptive"},
                {"pair_id": "pair-1", "strategy": "fixed"},
            ],
        },
    ]
    report = module.audit_cadence_completion(rows, frozen_pair_count=1)
    assert report["complete_registered_mirror_pairs"] == 1
    assert report["incomplete_registered_mirror_pairs"] == 0
    assert report["frozen_pair_count_matches_registered_mirror_pairs"] is True
