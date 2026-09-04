from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import b4_dry_run_common as common
from scripts import measure_b4_interface_floor
from scripts import scan_b4_mirror_depth
from scripts import verify_b4_t176_probe


ROOT = Path(__file__).resolve().parents[1]
CONFIG_287 = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"
CONFIG_176 = ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"


class FakePlatform:
    def __init__(self, physical_qubits: list[int], success_probability: float = 0.5) -> None:
        self.physical_qubits = physical_qubits
        self.success_probability = success_probability
        self.submissions: list[dict] = []
        self.last_shots = 0

    def download_config(self, machine: str):
        return {"machine": machine, "calibrationTime": "2026-08-04 09:01:23", "singleQubit": []}

    def submit_experiment(self, **kwargs):
        self.submissions.append(kwargs)
        self.last_shots = int(kwargs["num_shots"])
        return [f"fake-{len(self.submissions)}-{index}" for index in range(len(kwargs["circuit"]))]

    def query_experiment(self, query_ids, **_kwargs):
        successes = int(round(self.last_shots * self.success_probability))
        zeros = [0] * len(self.physical_qubits)
        ones = [1] * len(self.physical_qubits)
        samples = [zeros] * successes + [ones] * (self.last_shots - successes)
        return [
            {
                "experimentTaskId": query_id,
                "executionTime": "2026-08-07T13:00:00+00:00",
                "resultStatus": [self.physical_qubits, *samples],
            }
            for query_id in query_ids
        ]


class FakePlatformWithoutExecutionTime(FakePlatform):
    def query_experiment(self, query_ids, **kwargs):
        rows = super().query_experiment(query_ids, **kwargs)
        for row in rows:
            row.pop("executionTime", None)
        return rows


def test_common_job_and_differential_fit_are_fake_platform_executable() -> None:
    config = common.load_config(CONFIG_287)
    fake = FakePlatform(config["backend"]["physical_qubits"])
    clocks = iter([
        datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 4, 12, 0, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 4, 12, 0, 3, tzinfo=timezone.utc),
    ])
    timers = iter([10.0, 13.0])
    record, results = common.run_job(
        platform=fake,
        config=config,
        circuits=["M Q62"],
        shots_per_setting=10,
        name="fake",
        max_wait_seconds=1,
        poll_seconds=1,
        clock=lambda: next(clocks),
        timer=lambda: next(timers),
    )
    assert record["roundtrip_seconds"] == 3.0
    assert len(results) == 1
    estimate = common.estimate_rate_and_overhead([
        {"settings": 2, "shots_per_setting": 1024, "total_shots": 2048, "roundtrip_seconds": 10.0},
        {"settings": 2, "shots_per_setting": 1024, "total_shots": 2048, "roundtrip_seconds": 12.0},
        {"settings": 2, "shots_per_setting": 16384, "total_shots": 32768, "roundtrip_seconds": 40.0},
        {"settings": 2, "shots_per_setting": 16384, "total_shots": 32768, "roundtrip_seconds": 42.0},
    ])
    assert estimate["effective_shots_per_second"] > 0
    assert estimate["fixed_overhead_seconds_per_setting"] == 4.5


def test_interface_floor_and_t176_terminal_check_run_on_fake_platform(tmp_path: Path) -> None:
    config_287 = common.load_config(CONFIG_287)
    fake_287 = FakePlatform(config_287["backend"]["physical_qubits"])
    floor = measure_b4_interface_floor.execute(
        CONFIG_287,
        tmp_path / "floor.json",
        confirm_hardware=True,
        platform_factory=lambda _config: fake_287,
    )
    assert len(floor["jobs"]) == 10
    config_176 = common.load_config(CONFIG_176)
    fake_176 = FakePlatform(config_176["backend"]["physical_qubits"])
    terminal = verify_b4_t176_probe.execute(
        CONFIG_176,
        tmp_path / "t176.json",
        confirm_hardware=True,
        platform_factory=lambda _config: fake_176,
    )
    assert terminal["passed"] is True
    assert terminal["result4_downgrade_required"] is False
    assert json.loads((tmp_path / "t176.json").read_text(encoding="utf-8"))["first_job_only"] is True


def test_t176_terminal_check_allows_missing_optional_execution_time(tmp_path: Path) -> None:
    config = common.load_config(CONFIG_176)
    fake = FakePlatformWithoutExecutionTime(config["backend"]["physical_qubits"])
    terminal = verify_b4_t176_probe.execute(
        CONFIG_176,
        tmp_path / "t176-no-time.json",
        confirm_hardware=True,
        platform_factory=lambda _config: fake,
    )
    assert terminal["passed"] is True
    assert terminal["raw_counts_valid"] is True
    assert terminal["execution_timestamp_available"] is False
    assert terminal["execution_timestamp_required_for_terminal_pass"] is False


def test_mirror_depth_selection_is_frozen_and_deterministic() -> None:
    rows = [
        {"depth": 2, "median_success_probability": 0.68},
        {"depth": 4, "median_success_probability": 0.52},
        {"depth": 8, "median_success_probability": 0.48},
        {"depth": 16, "median_success_probability": 0.2},
    ]
    assert scan_b4_mirror_depth.select_depth(rows) == 4


def test_mirror_depth_score_checks_registered_shots(tmp_path: Path) -> None:
    config = common.load_config(CONFIG_287)
    fake = FakePlatform(config["backend"]["physical_qubits"], success_probability=0.5)
    report = scan_b4_mirror_depth.execute(
        CONFIG_287,
        tmp_path / "mirror-depth.json",
        confirm_hardware=True,
        depths=(2,),
        seeds_per_depth=1,
        shots=64,
        platform_factory=lambda _config: fake,
    )
    assert report["observations"][0]["shots"] == 64
