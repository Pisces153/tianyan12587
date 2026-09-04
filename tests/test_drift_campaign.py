from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("drift_campaign", ROOT / "scripts" / "drift_campaign.py")
assert SPEC and SPEC.loader
dc = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(dc)


CONFIG = ROOT / "config" / "tianyan_h1h2_dual_backend_drift_campaign_v1.json"
QUBITS = {
    "tianyan-287": [62, 55, 61, 68, 76, 69],
    "tianyan176": [42, 36, 31, 37, 44, 49],
}


class FakePlatform:
    next_identifier = 0
    submitted_calls = 0

    def __init__(self, backend_id: str, *, partial: bool = False):
        self.backend_id = backend_id
        self.partial = partial

    def get_machine_config(self, params):
        return {"overview": {"backend": self.backend_id, "calibration": "fake"}, "params": params}

    def download_config(self, machine: str):
        return {"machine": machine, "qubit": {"T1": [1.0], "T2": [2.0]}}

    def submit_experiment(self, *, circuit, name, num_shots, machine_name):
        assert machine_name == self.backend_id
        assert len(circuit) == 33
        assert num_shots == 1024
        FakePlatform.submitted_calls += 1
        identifiers = [f"{self.backend_id}-{FakePlatform.next_identifier + index}" for index in range(len(circuit))]
        FakePlatform.next_identifier += len(circuit)
        return identifiers

    def query_experiment(self, identifiers, max_wait_time, sleep_time):
        rows = list(identifiers[:1] if self.partial else identifiers)
        return [{
            "experimentTaskId": identifier,
            "resultStatus": [QUBITS[self.backend_id], *([[0, 0, 0, 0, 0, 0]] * 1024)],
        } for identifier in rows]


def frozen_config(tmp_path: Path, **overrides) -> Path:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_protocol_has_33_settings_and_fixed_interleaving() -> None:
    config = dc.load_json(CONFIG)
    dc.validate_config(config)
    for backend in dc.backend_rows(config):
        programs = dc.build_programs(config, backend)
        assert len(programs) == 33
        assert sum(row["kind"] == "anchor" for row in programs) == 27
        references = [row for row in programs if row["kind"] == "interleaved_reference"]
        assert [row["position_zero_indexed"] for row in references] == [0, 11, 22, 32]
        assert len({row["qcis"] for row in references}) == 1
        assert [(row["label"], row["position_zero_indexed"]) for row in programs if row["kind"] == "readout_probe"] == [
            ("readout_all_zero", 10), ("readout_all_one", 21)
        ]


def test_hash_chain_detects_tampering(tmp_path: Path) -> None:
    store = dc.CampaignStore(tmp_path)
    store.append("test", {"snapshot_id": "first", "execution_time_available": False})
    store.append("test", {"snapshot_id": "second", "execution_time_available": False})
    assert dc.CampaignStore(tmp_path).last_hash == store.last_hash
    journal = tmp_path / "snapshots.jsonl"
    journal.write_text(journal.read_text(encoding="utf-8").replace("second", "tampered"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        dc.CampaignStore(tmp_path)


def test_resume_collects_persisted_query_ids_without_resubmission(tmp_path: Path) -> None:
    FakePlatform.next_identifier = 0
    FakePlatform.submitted_calls = 0
    config_path = frozen_config(tmp_path)
    out = tmp_path / "campaign"
    factory = lambda backend: FakePlatform(backend)
    first = dc.run_once(
        config_path, out, confirm_hardware=True, scheduled_utc="2026-08-01T00:00:00+00:00",
        collect_wait_seconds=1, poll_seconds=0, platform_factory=factory,
    )
    assert first["submission_outcomes"] == {"tianyan-287": "submitted", "tianyan176": "submitted"}
    assert FakePlatform.submitted_calls == 2
    second = dc.run_once(
        config_path, out, confirm_hardware=True, scheduled_utc="2026-08-01T00:00:00+00:00",
        collect_wait_seconds=1, poll_seconds=0, platform_factory=factory,
    )
    assert FakePlatform.submitted_calls == 2
    assert set(second["collection_outcomes"].values()) == {"collected"}
    report = dc.verify(config_path, out)
    assert report["submitted_snapshots"] == 2
    assert report["committed_shots"] == 2 * 33 * 1024
    for path in (out / "raw").glob("*_counts.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert data["counts"].shape == (33, 64)
            assert np.all(data["counts"].sum(axis=1) == 1024)


def test_partial_result_is_not_failed_and_retains_missing_ids(tmp_path: Path) -> None:
    FakePlatform.next_identifier = 0
    config_path = frozen_config(tmp_path)
    out = tmp_path / "campaign"
    factory = lambda backend: FakePlatform(backend, partial=True)
    dc.run_once(config_path, out, confirm_hardware=True, scheduled_utc="2026-08-01T00:00:00+00:00", collect_wait_seconds=1, poll_seconds=0, platform_factory=factory)
    dc.run_once(config_path, out, confirm_hardware=True, scheduled_utc="2026-08-01T00:00:00+00:00", collect_wait_seconds=1, poll_seconds=0, platform_factory=factory)
    records = dc.CampaignStore(out).records
    partial = [row for row in records if row["event"] == "partial"]
    assert len(partial) == 2
    assert all(len(row["missing_query_ids"]) == 32 for row in partial)
    assert not any(row["event"] == "collection_failed" for row in records)


def test_budget_guard_stops_before_platform_submission(tmp_path: Path) -> None:
    FakePlatform.submitted_calls = 0
    config_path = frozen_config(tmp_path, shot_budget_hard_cap=1)
    with pytest.raises(RuntimeError, match="shot_budget_hard_cap"):
        dc.run_once(
            config_path, tmp_path / "campaign", confirm_hardware=True,
            scheduled_utc="2026-08-01T00:00:00+00:00", collect_wait_seconds=1, poll_seconds=0,
            platform_factory=lambda backend: FakePlatform(backend),
        )
    assert FakePlatform.submitted_calls == 0


def test_power_plan_covers_registered_effect_range(tmp_path: Path) -> None:
    config_path = frozen_config(tmp_path)
    manifest = dc.prepare(config_path, tmp_path / "campaign")
    report = dc.power_report(dc.load_json(config_path), manifest)
    assert {row["target_effect"] for row in report["rows"]} == {0.01, 0.02, 0.05}
    pauli_rows = [row for row in report["rows"] if row["family"] == "pauli15"]
    assert next(row for row in pauli_rows if row["target_effect"] == 0.01)["required_independent_pair_equivalents"] > 100
