from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_b4_t287_sensing_map",
    ROOT / "scripts" / "analyze_b4_t287_sensing_map.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_channel_arrays_keep_shared_anchors_and_channel_probe() -> None:
    observations = []
    for index in range(3):
        observations.append({
            "query_id": f"a{index}",
            "analysis_channels": "e0_readout_all_zero;e1_readout_all_one",
            "effective_observation_time_utc": f"2026-08-10T13:30:0{index}Z",
            "value": str(0.2 + index / 100.0),
            "shots": "3072",
            "regime_id": "r0",
            "burst_flag": "False",
            "instrument_id": "anchor_33",
        })
    for index in range(3):
        observations.append({
            "query_id": f"p{index}",
            "analysis_channels": "e1_readout_all_one",
            "effective_observation_time_utc": f"2026-08-10T13:31:0{index}Z",
            "value": str(0.1 + index / 100.0),
            "shots": "16384",
            "regime_id": "r0",
            "burst_flag": "False",
            "instrument_id": "probe_burst",
        })
    selected, arrays = module.channel_arrays(observations, "e1_readout_all_one")
    assert len(selected) == 6
    assert list(arrays["instruments"]) == ["anchor_33"] * 3 + ["probe_burst"] * 3
    assert np.all(np.diff(arrays["times"]) >= 0.0)


def test_detection_failure_blocks_map_even_with_nonparametric_optimum() -> None:
    result = module.evaluate_floor(
        mean_probability=0.1,
        effective_rate=100.0,
        maximum_interval=1000.0,
        floor={"label": "protocol_reachable", "seconds": 60.0, "role": "primary", "primary": True},
        variance_gate={
            "passed": False,
            "process_variance_ci_lower": -0.001,
        },
        fit={"ok": False},
        process_bounds=None,
        tau_bounds=None,
        structure_function=[
            {"lag_mid_seconds": 10.0, "sf_debiased": 0.001, "n_effective_points": 4},
            {"lag_mid_seconds": 100.0, "sf_debiased": 0.002, "n_effective_points": 4},
        ],
    )
    assert result["nonparametric_sensitivity"] is not None
    assert result["economic_gate_evaluated"] is False
    assert result["worth_sensing"] is False
    assert result["verdict_reason"] == "detection_gate_failed"


def test_confidence_corner_can_fail_when_point_curve_passes() -> None:
    result = module.evaluate_floor(
        mean_probability=0.1,
        effective_rate=100.0,
        maximum_interval=1000.0,
        floor={"label": "protocol_reachable", "seconds": 60.0, "role": "primary", "primary": True},
        variance_gate={
            "passed": True,
            "process_variance_ci_lower": 0.001,
        },
        fit={
            "ok": True,
            "process_variance": 0.002,
            "tau_seconds": 300.0,
        },
        process_bounds=(1e-12, 0.2),
        tau_bounds=(1.0, 10000.0),
        structure_function=[
            {"lag_mid_seconds": 10.0, "sf_debiased": 0.001, "n_effective_points": 4},
            {"lag_mid_seconds": 100.0, "sf_debiased": 0.002, "n_effective_points": 4},
        ],
    )
    assert result["point_estimate_economic_separation"] is True
    assert result["economic_gate_evaluated"] is True
    assert result["confidence_economic_separation"] is False
    assert result["worth_sensing"] is False
    assert result["frozen_economic_gate_verdict"] == "NO-GO"
    assert result["verdict_reason"] == "economic_ci_not_separated"


def test_fit_boundary_diagnostic_rejects_bound_hitting_interval() -> None:
    payload = {
        "ou_fit": {
            "ok": True,
            "process_variance_ci_lower": np.finfo(float).tiny,
            "process_variance_ci_upper": 1.0,
            "tau_ci_lower_seconds": 0.01,
            "tau_ci_upper_seconds": 100000.0,
        },
        "structure_function": [
            {"lag_mid_seconds": 10.0},
            {"lag_mid_seconds": 100.0},
        ],
        "effective_time_span_seconds": 1000.0,
    }
    bootstrap = {
        "available": True,
        "tau_seconds_interval": [10.0, 100000.0],
        "t_star_seconds_interval": [20.0, 1000.0],
    }
    diagnostic = module.fit_boundary_diagnostics(payload, bootstrap)
    assert diagnostic["identified"] is False
    assert diagnostic["asymptotic_ci_hits_bound"]["process_variance_lower"] is True
    assert diagnostic["bootstrap_interval_hits_bound"]["t_star_upper"] is True


def test_unidentified_fit_maps_frozen_no_go_to_inconclusive() -> None:
    classification, reason = module.decision_classification(
        detection_gate_passed=True,
        fit_identified=False,
        frozen_worth_sensing=False,
    )
    assert classification == "INCONCLUSIVE"
    assert reason == "ou_parameters_not_identified"


def test_identified_fit_preserves_frozen_go_no_go() -> None:
    assert module.decision_classification(
        detection_gate_passed=True,
        fit_identified=True,
        frozen_worth_sensing=True,
    )[0] == "GO"
    assert module.decision_classification(
        detection_gate_passed=True,
        fit_identified=True,
        frozen_worth_sensing=False,
    )[0] == "NO-GO"


def test_fallacy_scan_is_complete() -> None:
    scan = module.fallacy_scan()
    assert len(scan) == 11
    assert all(row["status"] == "checked" for row in scan)


def test_write_csv_accepts_channel_specific_field_union() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rows.csv"
        module.write_csv(path, [{"channel": "e0", "a": 1}, {"channel": "e1", "b": 2}])
        text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "channel,a,b"
    assert "e0,1," in text
    assert "e1,,2" in text
