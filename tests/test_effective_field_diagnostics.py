from __future__ import annotations

import json
from pathlib import Path

from src.adaptive.effective_field_diagnostics import analyze_effective_field_identifiability, write_effective_field_diagnostics


def _record(index: int, *, h2: float = 2.88) -> dict:
    per_time = []
    for time, h1_estimate, h2_estimate in ((0.16, -0.5, 4.12), (0.31, 0.37, 2.68), (0.47, 0.81, 1.78)):
        per_time.append({
            "time": time,
            "estimate": {"h1": h1_estimate, "h2": h2_estimate},
            "covariance_h1_h2": [[0.04, 0.0], [0.0, 0.09]],
        })
    return {
        "backend_id": "tianyan-287",
        "snapshot_index": index,
        "snapshot_id": f"snapshot-{index}",
        "effective_field_state": {
            "h1": {"value": 0.75, "shot_sigma": 0.15},
            "h2": {"value": h2, "shot_sigma": 0.38},
            "per_time": per_time,
        },
    }


def test_diagnostic_selects_route_b_and_expands_sigma() -> None:
    identifiability, sigma = analyze_effective_field_identifiability([_record(0)])
    assert identifiability["selected_route"] == "B_readout_only"
    h2 = next(row for row in identifiability["fields"] if row["field"] == "h2")["snapshots"][0]
    assert h2["minimum_branch_margin_from_odd_pi_rad"] < 0.5
    assert h2["conservative_sigma"] > h2["shot_sigma"]
    assert len(h2["pairwise_consistency"]) == 3
    assert len(sigma["rows"]) == 2


def test_diagnostic_writes_signed_non_overwriting_artifacts(tmp_path: Path) -> None:
    corpus = tmp_path / "features.jsonl"
    corpus.write_text(json.dumps(_record(0)) + "\n", encoding="utf-8")
    output = tmp_path / "diagnostic"
    result = write_effective_field_diagnostics(corpus, output)
    persisted = json.loads((output / "effective_field_identifiability.json").read_text(encoding="utf-8"))
    assert result["selected_route"] == persisted["selected_route"]
    assert len(persisted["self_sha256"]) == 64
    try:
        write_effective_field_diagnostics(corpus, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("diagnostic artifact overwrite was accepted")
