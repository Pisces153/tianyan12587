from __future__ import annotations

import json
from pathlib import Path

from scripts import b4_dry_run_common as common
from scripts import run_cadence_pair_hardware_smoke as module


ROOT = Path(__file__).resolve().parents[1]
LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_v1.json"
BACKEND_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"


class FakePlatform:
    def __init__(self, physical_qubits: list[int]) -> None:
        self.physical_qubits = physical_qubits
        self.submission_index = 0
        self.shots = 0

    def submit_experiment(self, **kwargs):
        self.submission_index += 1
        self.shots = int(kwargs["num_shots"])
        return [f"fake-{self.submission_index}-{index}" for index in range(len(kwargs["circuit"]))]

    def query_experiment(self, query_ids, **_kwargs):
        zeros = [0] * len(self.physical_qubits)
        if self.submission_index == 1:
            rows = []
            for index, query_id in enumerate(query_ids):
                samples = [zeros] * self.shots
                rows.append({"experimentTaskId": query_id, "resultStatus": [self.physical_qubits, *samples]})
            return rows
        return [
            {"experimentTaskId": query_id, "resultStatus": [self.physical_qubits, *([zeros] * self.shots)]}
            for query_id in query_ids
        ]


def test_hardware_smoke_uses_estimates_for_shield_and_seals_raw_results(tmp_path: Path) -> None:
    depth = tmp_path / "depth.json"
    depth.write_text(json.dumps({"selection_passed": True, "selected_depth": 2}), encoding="utf-8")
    config = common.load_config(BACKEND_CONFIG)
    fake = FakePlatform(config["backend"]["physical_qubits"])
    output = tmp_path / "hardware-smoke"
    report = module.execute(
        LOOP_CONFIG,
        BACKEND_CONFIG,
        depth,
        output,
        confirm_hardware=True,
        sensing_shots=4096,
        mirror_shots=128,
        platform_factory=lambda _config: fake,
    )
    assert report["smoke_passed"] is True
    assert report["shield"]["permitted"] is True
    assert report["event_order"] == [
        "controlled_injection",
        "sense",
        "five_gate_shield",
        "digital_inverse_compensation",
        "matched_mirror_pair",
    ]
    assert "injected_effective_fields" not in report["observable_shield_state"]
    assert [row["strategy"] for row in report["mirror_scores"]] == ["fixed", "adaptive"]
    assert all(row["shots"] == 128 for row in report["mirror_scores"])
    assert (output / "raw_query_results.json").exists()
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["smoke_passed"] is True
