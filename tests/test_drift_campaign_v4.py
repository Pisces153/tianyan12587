from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import drift_campaign_v4 as module


ROOT = Path(__file__).resolve().parents[1]
CONFIG_287 = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"
CONFIG_176 = ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"


def _configs() -> tuple[dict, dict]:
    return module.load_config(CONFIG_287), module.load_config(CONFIG_176)


def test_non_t287_and_non_1024_probe_shots_pass_v4_validator() -> None:
    _, config = _configs()
    module.validate_v4_config(config)
    assert config["backend"]["backend_id"] == "tianyan176"
    assert config["measurement"]["probe_job"]["shots_per_setting"] == 16384


def test_config_pair_diff_is_only_backend_and_credential() -> None:
    left, right = _configs()
    differences = module.validate_config_pair(left, right)
    assert "credential_env" in differences
    assert "measurement.anchor_job.shots_per_setting" in differences
    assert all(
        path in {"credential_env", "measurement.anchor_job.shots_per_setting"}
        or path.startswith("backend.")
        for path in differences
    )
    assert left["cadence_origin_utc"] == right["cadence_origin_utc"]


def test_anchor_keeps_v3_positions_and_in_job_probes_while_burst_is_independent() -> None:
    config, _ = _configs()
    anchor = module.build_anchor_programs(config)
    probe = module.build_probe_programs(config)
    assert len(anchor) == 33
    assert [row["position_zero_indexed"] for row in anchor if row["kind"] == "interleaved_reference"] == [0, 11, 22, 32]
    in_job = [row for row in anchor if row.get("setting_tag") == "in_job_probe"]
    assert [row["position_zero_indexed"] for row in in_job] == [10, 21]
    assert all(row["analysis_role"] == "supportive_context_qc_only_not_primary_sf" for row in in_job)
    assert len(probe) == 2
    assert all(row["job_role"] == "probe_burst" for row in probe)
    assert config["measurement"]["anchor_job"]["shots_per_setting"] == 3072
    assert config["measurement"]["probe_job"]["shots_per_setting"] == 16384


def test_session_and_event_jobs_use_frozen_targets_and_burst_flag() -> None:
    config, _ = _configs()
    jobs = module.build_session_jobs(
        config,
        session_origin_platform_utc="2026-08-07T13:00:00+00:00",
        regime_id="regime-0002",
    )
    assert [row["planned_target_minutes"] for row in jobs] == [0, 1, 2, 4, 8, 16]
    assert all(row["job_role"] == "probe_burst" and not row["burst_flag"] for row in jobs)
    event = module.build_event_burst_jobs(
        config,
        event_platform_utc="2026-08-07T14:00:00+00:00",
        regime_id="regime-0003",
    )
    assert [row["planned_target_minutes"] for row in event] == [0, 1, 2, 4]
    assert all(row["job_role"] == "probe_burst" and row["burst_flag"] for row in event)
    assert not any(row["primary_sf_eligible"] for row in event)


def test_platform_timestamp_is_never_synthesized_and_elapsed_uses_platform_only() -> None:
    absent = module.observation_metadata(
        {},
        regime_id="regime-0000",
        burst_flag=False,
        job_role="probe_burst",
        analysis_role="primary_sf_only_when_non_event_and_same_regime",
        previous_platform_timestamp="2026-08-07T13:00:00+00:00",
    )
    assert absent["platform_timestamp_raw"] is None
    assert absent["elapsed_minutes_since_previous"] is None
    present = module.observation_metadata(
        {"executionTime": "2026-08-07T13:01:30+00:00"},
        regime_id="regime-0000",
        burst_flag=False,
        job_role="probe_burst",
        analysis_role="primary_sf_only_when_non_event_and_same_regime",
        previous_platform_timestamp="2026-08-07T13:00:00+00:00",
    )
    assert present["elapsed_minutes_since_previous"] == 1.5
    assert present["execution_timestamp_available"] is True


def test_sf_pairing_never_crosses_role_regime_or_event() -> None:
    base = {
        "backend_id": "tianyan-287",
        "regime_id": "regime-0000",
        "job_role": "probe_burst",
        "analysis_role": "primary_sf_only_when_non_event_and_same_regime",
        "burst_flag": False,
        "primary_sf_eligible": True,
    }
    assert module.sf_pair_eligible(base, dict(base))
    assert not module.sf_pair_eligible(base, {**base, "job_role": "anchor_33"})
    assert not module.sf_pair_eligible(base, {**base, "regime_id": "regime-0001"})
    assert not module.sf_pair_eligible(base, {**base, "burst_flag": True})


def test_regime_flip_schedules_new_supportive_regime() -> None:
    first = module.regime_transition(None, "2026-08-01T00:00:00Z")
    same = module.regime_transition(first, "2026-08-01T00:00:00Z")
    changed = module.regime_transition(same, "2026-08-04T00:00:00Z")
    assert same["flipped"] is False
    assert changed == {
        "regime_id": "regime-0001",
        "calibration_time_raw": "2026-08-04T00:00:00Z",
        "flipped": True,
    }


def test_prepare_manifest_mutually_records_peer_hash_and_role_mapping(tmp_path: Path) -> None:
    manifest = module.prepare_v4(CONFIG_287, CONFIG_176, tmp_path / "v4")
    assert manifest["peer_config_sha256"] == module.campaign.digest_file(CONFIG_176)
    assert manifest["role_shots_per_setting"] == {"anchor_33": 3072, "probe_burst": 16384}
    assert manifest["common_protocol_sha256"] == manifest["peer_common_protocol_sha256"]
    assert manifest["anchor_shots_per_setting_rule"] == module.ANCHOR_SHOTS_SCALING_RULE
    assert manifest["job_role_enum"] == ["anchor_33", "probe_burst", "loop", "mirror"]
    assert "never cross role" in manifest["pairing_rule"]


class _FakePlatform:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit_experiment(self, **kwargs):
        self.calls.append(kwargs)
        return [f"fake-{index}" for index in range(len(kwargs["circuit"]))]

    def query_experiment(self, query_ids, **_kwargs):
        return [
            {
                "experimentTaskId": query_id,
                "executionTime": f"2026-08-07T13:0{index}:00+00:00",
            }
            for index, query_id in enumerate(query_ids)
        ]


def test_fake_platform_probe_submission_is_two_settings_at_16384(tmp_path: Path) -> None:
    fake = _FakePlatform()
    record = module.submit_probe_job(
        CONFIG_176,
        CONFIG_287,
        tmp_path / "submit",
        planned_target_utc="2026-08-07T13:00:00+00:00",
        regime_id="regime-0000",
        burst_flag=False,
        confirm_hardware=True,
        platform_factory=lambda _config: fake,
    )
    assert record["event"] == "submitted"
    assert record["job_role"] == "probe_burst"
    assert record["execution_timestamp_available"] is False
    assert fake.calls[0]["machine_name"] == "tianyan176"
    assert fake.calls[0]["num_shots"] == 16384
    assert len(fake.calls[0]["circuit"]) == 2
    journal = [json.loads(line) for line in (tmp_path / "submit" / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert journal[-1]["job_role"] == "probe_burst"


def test_fake_platform_anchor_submission_is_33_settings_at_scaled_shots_and_at_most_once(tmp_path: Path) -> None:
    fake = _FakePlatform()
    arguments = {
        "config_path": CONFIG_287,
        "peer_config_path": CONFIG_176,
        "out": tmp_path / "anchor",
        "planned_target_utc": "2026-08-07T13:00:00+00:00",
        "regime_id": "regime-0000",
        "confirm_hardware": True,
        "platform_factory": lambda _config: fake,
    }
    record = module.submit_anchor_job(**arguments)
    assert record["job_role"] == "anchor_33"
    assert record["shots_per_setting"] == 3072
    assert record["elapsed_minutes_since_previous"] is None
    assert len(record["tasks"]) == 33
    assert len(fake.calls[0]["circuit"]) == 33
    assert fake.calls[0]["num_shots"] == 3072
    repeated = module.submit_anchor_job(**arguments)
    assert repeated["status"] == "already_recorded_no_resubmission"
    assert len(fake.calls) == 1


def test_t176_anchor_submission_uses_7840_shots(tmp_path: Path) -> None:
    fake = _FakePlatform()
    record = module.submit_anchor_job(
        CONFIG_176,
        CONFIG_287,
        tmp_path / "anchor176",
        planned_target_utc="2026-08-07T13:00:00+00:00",
        regime_id="regime-0000",
        confirm_hardware=True,
        platform_factory=lambda _config: fake,
    )
    assert record["shots_per_setting"] == 7840
    assert fake.calls[0]["num_shots"] == 7840


def test_fake_platform_collection_lands_role_regime_timestamp_and_elapsed(tmp_path: Path, monkeypatch) -> None:
    fake = _FakePlatform()
    out = tmp_path / "collect"
    module.submit_probe_job(
        CONFIG_287,
        CONFIG_176,
        out,
        planned_target_utc="2026-08-07T13:00:00+00:00",
        regime_id="regime-0000",
        burst_flag=False,
        confirm_hardware=True,
        platform_factory=lambda _config: fake,
    )
    monkeypatch.setattr(
        module.campaign,
        "result_counts",
        lambda _row, _physical, _shots: np.zeros(64, dtype=np.int32),
    )
    summary = module.collect_pending_jobs(
        CONFIG_287,
        CONFIG_176,
        out,
        confirm_hardware=True,
        platform_factory=lambda _config: fake,
    )
    assert summary["collected_jobs"] == 1
    journal = [json.loads(line) for line in (out / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    collected = journal[-1]
    assert collected["event"] == "collected"
    assert collected["job_role"] == "probe_burst"
    assert collected["regime_id"] == "regime-0000"
    assert collected["burst_flag"] is False
    assert collected["elapsed_minutes_since_previous"] == [None, 1.0]
    blob = np.load(collected["counts_path"], allow_pickle=False)
    assert set(blob["job_roles"]) == {"probe_burst"}
    assert list(blob["elapsed_minutes_since_previous"])[1] == 1.0


def test_trimming_order_reduces_probe_rounds_before_loop_then_mirror() -> None:
    config, _ = _configs()
    assert [module.trimming_state(config, step)["probe_rounds"] for step in range(3)] == [6, 5, 4]
    after_probe = module.trimming_state(config, 3)
    assert after_probe["probe_rounds"] == 4
    assert after_probe["closed_loop_updates"] == config["closed_loop"]["planned_updates"] - 1
    late = module.trimming_state(config, 8)
    assert late["closed_loop_updates"] == 0
    assert late["mirror_repetitions"] < config["mirror"]["planned_repetitions"]
