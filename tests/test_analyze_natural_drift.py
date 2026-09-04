from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_natural_drift.py"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _self_hash(payload: dict) -> str:
    copied = dict(payload)
    copied.pop("self_sha256", None)
    canonical = json.dumps(copied, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(canonical).hexdigest().upper()


def _t3_artifact(*, gap: float) -> dict:
    payload = {
        "task": "T3_sigma_recalibration",
        "weights_unchanged": True,
        "hardware_used_for_fit": False,
        "calibration_unit": "disjoint before-target offsets from two_time_inverse_variance trajectories matching r4 inference",
        "pairing": {"source_trajectory_reused": False},
        "selected_simulation_coverage_1sigma": 0.68,
        "sim_to_real_coverage_gap": gap,
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    payload["self_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    return payload


def _t4_gate(*, passed: bool) -> dict:
    payload = {"task": "T4_zero_cz_twin", "fidelity": {"overall_gate_passed": passed, "validated_observables": ["X0", "Y0", "Z0", "X1", "Y1", "Z1", "XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]}, "n_parameters": 6, "self_hash_scope": "canonical JSON excluding self_sha256"}
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    payload["self_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    return payload


def _prior(t3: dict) -> dict:
    payload = {"task": "T3_sim_to_real_conservative_prior", "source_t3_sha256": t3["self_sha256"], "sigma_inflation_multiplier": 1.4, "separate_from_t3_calibration": True}
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    payload["self_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    return payload


def _t6(backends: tuple[str, ...] = ("tianyan-287",)) -> dict:
    payload = {"task": "T6_observable_environment_proxy", "feature_corpus_sha256": "A" * 64, "collected_snapshot_count": len(backends), "proxy_label_task_disjoint": True, "records_by_backend": {backend: 1 for backend in backends}}
    payload["self_sha256"] = _self_hash(payload)
    return payload


def _forecast(backend: str, *, passed: bool = True) -> dict:
    payload = {"analysis_task": "T7_forecast_head", "feature_corpus_sha256": "A" * 64, "corpus": {"backend_id": backend}, "cv": {"scheme": "rolling_origin_forward_chain", "shuffle_used": False}, "gate": {"forecasting_skill_claimed": passed}}
    payload["self_sha256"] = _self_hash(payload)
    return payload


def test_t3_domain_gap_blocks_even_when_other_artifacts_pass(tmp_path: Path) -> None:
    t6 = tmp_path / "t6.json"; forecast = tmp_path / "forecast.json"; t3 = tmp_path / "t3.json"
    t4 = tmp_path / "t4.json"
    _write(t6, _t6())
    _write(forecast, _forecast("tianyan-287"))
    _write(t3, _t3_artifact(gap=0.21))
    _write(t4, _t4_gate(passed=True))
    command = [sys.executable, str(SCRIPT), "bandit-preflight", "--t6-report", str(t6), "--forecast-report", str(forecast), "--t3-artifact", str(t3), "--t4-fidelity-gate", str(t4), "--out", str(tmp_path / "blocked")]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    blocked = json.loads((tmp_path / "blocked" / "prerequisite_gate.json").read_text(encoding="utf-8"))
    assert blocked["prerequisites"]["T3_sigma_calibrated"] is True
    assert blocked["prerequisites"]["T3_sim_to_real_conservative_prior"] is False
    assert blocked["policy_execution_permitted"] is False


def test_tampered_t3_artifact_cannot_unlock_preflight(tmp_path: Path) -> None:
    t6 = tmp_path / "t6.json"; forecast = tmp_path / "forecast.json"; t3 = tmp_path / "t3.json"; t4 = tmp_path / "t4.json"
    _write(t6, _t6())
    _write(forecast, _forecast("tianyan-287"))
    artifact = _t3_artifact(gap=0.10)
    artifact["selected_simulation_coverage_1sigma"] = 0.99
    _write(t3, artifact)
    _write(t4, _t4_gate(passed=True))
    command = [sys.executable, str(SCRIPT), "bandit-preflight", "--t6-report", str(t6), "--forecast-report", str(forecast), "--t3-artifact", str(t3), "--t4-fidelity-gate", str(t4), "--out", str(tmp_path / "blocked")]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    blocked = json.loads((tmp_path / "blocked" / "prerequisite_gate.json").read_text(encoding="utf-8"))
    assert blocked["t3_uncertainty_domain_status"]["self_hash_valid"] is False
    assert blocked["prerequisites"]["T3_sigma_calibrated"] is False


def test_valid_separate_conservative_prior_unlocks_only_t3_gap(tmp_path: Path) -> None:
    t6 = tmp_path / "t6.json"; forecast = tmp_path / "forecast.json"; t3 = tmp_path / "t3.json"; t4 = tmp_path / "t4.json"; prior = tmp_path / "prior.json"
    _write(t6, _t6())
    _write(forecast, _forecast("tianyan-287"))
    source = _t3_artifact(gap=0.21); _write(t3, source); _write(t4, _t4_gate(passed=False)); _write(prior, _prior(source))
    command = [sys.executable, str(SCRIPT), "bandit-preflight", "--t6-report", str(t6), "--forecast-report", str(forecast), "--t3-artifact", str(t3), "--t3-conservative-prior", str(prior), "--t4-fidelity-gate", str(t4), "--out", str(tmp_path / "blocked")]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "blocked" / "prerequisite_gate.json").read_text(encoding="utf-8"))
    assert result["prerequisites"]["T3_sim_to_real_conservative_prior"] is True
    assert result["policy_execution_permitted"] is False


def test_preflight_requires_a_passing_t7_report_for_every_collected_backend(tmp_path: Path) -> None:
    t6 = tmp_path / "t6.json"; t3 = tmp_path / "t3.json"; t4 = tmp_path / "t4.json"
    t287 = tmp_path / "t287.json"; t176 = tmp_path / "t176.json"
    _write(t6, _t6(("tianyan-287", "tianyan176")))
    _write(t287, _forecast("tianyan-287", passed=True))
    _write(t176, _forecast("tianyan176", passed=False))
    _write(t3, _t3_artifact(gap=0.10))
    _write(t4, _t4_gate(passed=True))
    command = [sys.executable, str(SCRIPT), "bandit-preflight", "--t6-report", str(t6), "--forecast-report", str(t287), "--forecast-report", str(t176), "--t3-artifact", str(t3), "--t4-fidelity-gate", str(t4), "--out", str(tmp_path / "blocked")]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "blocked" / "prerequisite_gate.json").read_text(encoding="utf-8"))
    assert result["prerequisites"]["T7_forecast"] is False
    assert result["t7_forecast_status"]["forecasting_skill_claimed_by_backend"] == {"tianyan-287": True, "tianyan176": False}


def test_tampered_t7_report_cannot_unlock_preflight(tmp_path: Path) -> None:
    t6 = tmp_path / "t6.json"; forecast = tmp_path / "forecast.json"; t3 = tmp_path / "t3.json"; t4 = tmp_path / "t4.json"
    _write(t6, _t6())
    tampered = _forecast("tianyan-287")
    tampered["gate"]["forecasting_skill_claimed"] = False
    _write(forecast, tampered)
    _write(t3, _t3_artifact(gap=0.10))
    _write(t4, _t4_gate(passed=True))
    command = [sys.executable, str(SCRIPT), "bandit-preflight", "--t6-report", str(t6), "--forecast-report", str(forecast), "--t3-artifact", str(t3), "--t4-fidelity-gate", str(t4), "--out", str(tmp_path / "blocked")]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "blocked" / "prerequisite_gate.json").read_text(encoding="utf-8"))
    assert result["prerequisites"]["T7_forecast"] is False
    assert result["t7_forecast_status"]["report_validation_by_backend"]["tianyan-287"]["self_hash_valid"] is False
