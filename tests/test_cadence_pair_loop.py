from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts import run_cadence_pair_loop as module


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "b4_cadence_pair_loop_v1.json"


def _config() -> dict:
    return module.load_config(CONFIG)


def test_frozen_simulation_config_keeps_ou_calibration_and_alternating_blocks() -> None:
    config = _config()
    module.validate_config(config)
    assert config["controlled_ou"]["hard_clip_absolute"] == 0.08
    assert config["controlled_ou"]["rng_seed"] == 2026080501
    assert config["block_order_by_day"] == [["fast", "slow"], ["slow", "fast"], ["fast", "slow"]]
    assert 2.0 <= module.fast_increment_to_sensing_sigma_ratio(config) <= 3.0
    assert config["cadence"]["hardware_lower_bound_rule"] == "T_fast >= 2 * measured interface latency P90"
    assert config["cadence"]["measured_interface_p90_seconds"] == pytest.approx(14.701512299997557)
    assert config["cadence"]["fast_seconds"] >= 2.0 * config["cadence"]["measured_interface_p90_seconds"]
    assert config["mirror"]["depth"] == 2
    assert config["mirror"]["depth_status"] == "frozen_after_tb6_hardware_ladder"


def test_five_gate_self_audit_and_three_level_amplitude_ladder() -> None:
    config = _config()
    audit = module.shield_self_audit(config)
    assert audit["observed"] == {
        "permit": None,
        "confidence": "confidence",
        "physical_range": "physical_range",
        "jz_reject": "jz_reject",
        "action_amplitude": "action_amplitude",
        "budget": "budget",
    }
    rows = module.gate_ladder(config)
    assert [row["observed_behavior"] for row in rows] == ["permit", "downscale", "abstain"]
    assert rows[1]["compensation"][0] == pytest.approx(-0.05)
    assert rows[2]["shield_gate"] == "action_amplitude"


def test_fake_backend_true_scheduler_path_runs_end_to_end_and_seals_log(tmp_path: Path) -> None:
    output = tmp_path / "tb5"
    report = module.run_simulation(CONFIG, output)
    assert report["acceptance"]["preflight_passed"] is True
    assert report["hardware_job_submitted"] is False
    assert report["tb1_cadence_endpoint_evidence"]["cell_count"] == 15
    assert report["tb1_cadence_endpoint_evidence"]["passed"] is True
    assert report["cadence_endpoint"]["pair_count"] == 24
    assert report["cadence_endpoint"]["direction_fast_loss_lower"] is True
    assert [row["block_order"] for row in report["days"]] == [["fast", "slow"], ["slow", "fast"], ["fast", "slow"]]
    assert [row["gate_test"]["amplitude"] for row in report["days"]] == [0.05, 0.1, 0.25]
    assert [row["gate_test"]["observed_behavior"] for row in report["days"]] == ["permit", "downscale", "abstain"]
    assert all(row["gate_test"]["execution_role"] == "single_gate_test_at_daily_block_end" for row in report["days"])

    observations = [json.loads(line) for line in (output / "cadence_observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert observations
    assert all(row["event_order"] == ["sense", "shield", "digital_inverse_compensation", "mirror_probe"] for row in observations)
    assert all(row["hardware_job_submitted"] is False for row in observations)
    assert all(task["primary_source"] == "raw_counts" for row in observations for task in row["mirror"]["tasks"])
    latencies = [row["mirror"]["simulated_interface_delay"]["waited_seconds"] for row in observations]
    assert min(latencies) > 0.0
    assert len({round(value, 8) for value in latencies}) > 1

    injection_path = output / "controlled_ou_injection.jsonl"
    injection = [json.loads(line) for line in injection_path.read_text(encoding="utf-8").splitlines()]
    assert injection
    assert max(abs(value) for row in injection for value in row["controlled_state_after"]) <= 0.08
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rng_seeds"]["controlled_ou"] == 2026080501
    assert manifest["sealed_injection_log"]["sha256"] == sha256(injection_path.read_bytes()).hexdigest()
    assert manifest["preflight_passed"] is True
    assert (output / "B4_TB5_SIMULATION_PREFLIGHT.md").exists()


def test_output_is_non_overwriting_and_hardware_mode_is_not_exposed(tmp_path: Path) -> None:
    output = tmp_path / "tb5"
    module.run_simulation(CONFIG, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.run_simulation(CONFIG, output)
    source = (ROOT / "scripts" / "run_cadence_pair_loop.py").read_text(encoding="utf-8")
    assert "submit_experiment" not in source
    assert "scenario ==" not in source
