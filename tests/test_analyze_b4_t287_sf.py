from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_b4_t287_sf",
    ROOT / "scripts" / "analyze_b4_t287_sf.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_reference_error_probability_uses_logical_z0z1_parity() -> None:
    counts = np.zeros(64, dtype=np.int64)
    counts[0] = 70
    counts[16] = 30
    assert np.isclose(module.reference_error_probability(counts), 0.3)


def test_readout_error_probability_uses_expected_bitstring() -> None:
    zero = np.zeros(64, dtype=np.int64)
    zero[0] = 90
    zero[1] = 10
    one = np.zeros(64, dtype=np.int64)
    one[63] = 80
    one[0] = 20
    assert np.isclose(module.readout_error_probability("readout_all_zero", zero, 100), 0.1)
    assert np.isclose(module.readout_error_probability("readout_all_one", one, 100), 0.2)


def test_effective_observation_time_applies_frozen_anchor_positions() -> None:
    start = datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc)
    end = datetime(2026, 8, 10, 13, 31, tzinfo=timezone.utc)
    anchor, source, offset = module.effective_observation_time(
        label="interleaved_reference_4",
        execution_start_utc=start,
        execution_end_utc=end,
        anchor_setting_seconds=9.0,
    )
    assert anchor == start.replace(minute=34, second=48)
    assert offset == 288.0
    assert "frozen within-job" in source
    probe, source, offset = module.effective_observation_time(
        label="readout_all_zero",
        execution_start_utc=start,
        execution_end_utc=end,
        anchor_setting_seconds=9.0,
    )
    assert probe == start.replace(second=30)
    assert offset is None
    assert "midpoint" in source


def test_channel_analysis_keeps_shared_anchor_and_one_readout_label() -> None:
    start = datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc)
    observations = []
    for index in range(12):
        observations.append({
            "query_id": f"a{index}",
            "analysis_channels": sorted(module.CHANNEL_LABEL),
            "instrument_id": "anchor_33",
            "regime_id": "regime-0000",
            "burst_flag": False,
            "shots": 3072,
            "value": 0.5 + index / 1000.0,
            "effective_observation_time_utc": module.iso_utc(start + module.timedelta(seconds=100 * index)),
        })
    for channel, label in module.CHANNEL_LABEL.items():
        for index in range(18):
            observations.append({
                "query_id": f"{label}-{index}",
                "analysis_channels": [channel],
                "instrument_id": "probe_burst",
                "regime_id": "regime-0000",
                "burst_flag": False,
                "shots": 16384,
                "value": 0.1 + index / 1000.0,
                "effective_observation_time_utc": module.iso_utc(start + module.timedelta(seconds=70 * index + 10)),
            })
    for channel in module.CHANNEL_LABEL:
        result = module.analyze_channel(observations, channel)
        assert result["observation_count"] == 30
        assert result["observation_counts_by_instrument"] == {"anchor_33": 12, "probe_burst": 18}
        assert result["eligible_pair_counts_by_instrument"] == {"anchor_33": 66, "probe_burst": 153}


def test_channel_analysis_does_not_fit_ou_when_variance_gate_fails() -> None:
    start = datetime(2026, 8, 10, 13, 30, tzinfo=timezone.utc)
    observations = []
    for index in range(12):
        observations.append({
            "query_id": f"a{index}",
            "analysis_channels": ["e0_readout_all_zero"],
            "instrument_id": "anchor_33",
            "regime_id": "regime-0000",
            "burst_flag": False,
            "shots": 3072,
            "value": 0.5,
            "effective_observation_time_utc": module.iso_utc(start + module.timedelta(seconds=100 * index)),
        })
    for index in range(18):
        observations.append({
            "query_id": f"p{index}",
            "analysis_channels": ["e0_readout_all_zero"],
            "instrument_id": "probe_burst",
            "regime_id": "regime-0000",
            "burst_flag": False,
            "shots": 16384,
            "value": 0.1,
            "effective_observation_time_utc": module.iso_utc(start + module.timedelta(seconds=70 * index + 10)),
        })
    result = module.analyze_channel(observations, "e0_readout_all_zero")
    assert not result["variance_gate"]["passed"]
    assert not result["ou_fit"]["ok"]
