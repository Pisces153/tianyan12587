from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.adaptive import task_metric_mirror as module


def _row(pair_id: str, strategy: str, successes: int, *, window: str = "w0") -> dict[str, object]:
    shots = 100
    return {
        "pair_id": pair_id,
        "strategy": strategy,
        "backend_id": "tianyan-287",
        "time_window_id": window,
        "shots": shots,
        "task_family": "mirror_v1",
        "depth": 8,
        "task_duration_seconds": 1.0,
        "total_strategy_shots_in_window": 200,
        "ideal_bitstring": "000000",
        "raw_counts": {"000000": successes, "111111": shots - successes},
        "readout_mitigated_success_probability": successes / shots + 0.01,
    }


def test_raw_counts_define_primary_success_probability() -> None:
    result = module.success_probability_from_raw_counts({"000000": 73, "111111": 27}, "000000", shots=100)
    assert result["success_count"] == 73
    assert result["success_probability"] == pytest.approx(0.73)


def test_readout_mitigation_is_secondary_only() -> None:
    score = module.score_observation(_row("p0", "adaptive", 70))
    assert score.success_probability == pytest.approx(0.70)
    assert score.readout_mitigated_success_probability_secondary == pytest.approx(0.71)


def test_pair_mismatch_is_rejected() -> None:
    rows = [_row("p0", "adaptive", 70), _row("p0", "fixed", 60, window="w1")]
    with pytest.raises(ValueError, match="matching fields"):
        module.compare_strategies(rows, resamples=100, seed=1)


def test_minimum_actual_improvement_must_be_explicit_probability_scale() -> None:
    with pytest.raises(ValueError, match="minimum_actual_improvement"):
        module.compare_strategies([], resamples=100, seed=1, minimum_actual_improvement=-0.01)


def test_incomplete_pair_is_reported_without_imputation() -> None:
    rows = [_row("p0", "adaptive", 70), _row("p1", "adaptive", 70), _row("p1", "fixed", 60)]
    report = module.compare_strategies(rows, resamples=200, seed=2)
    assert report["pairing"]["complete_pairs"] == 1
    assert report["pairing"]["incomplete_pairs"] == 1
    assert "no imputation" in report["pairing"]["missing_policy"]
    assert report["endpoint"]["available"] is False


def test_random_clifford_mirror_has_exact_structural_inverse() -> None:
    circuit = module.build_random_clifford_mirror([62, 55, 61, 68, 76, 69], depth=7, seed=20260804)
    expected = tuple(
        tuple(module._invert_operation(operation) for operation in reversed(layer))
        for layer in reversed(circuit.forward_layers)
    )
    assert circuit.inverse_layers == expected
    assert circuit.ideal_bitstring == "000000"
    assert circuit.qcis.splitlines()[-6:] == [f"M Q{qubit}" for qubit in circuit.physical_qubits]


def test_depth_ladder_manifest_has_frozen_random_seeds() -> None:
    manifest = module.build_depth_ladder_manifest(
        [62, 55, 61, 68, 76, 69],
        [2, 8],
        seeds_per_depth=3,
        seed=20260804,
    )
    assert len(manifest["tasks"]) == 6
    assert len({row["seed"] for row in manifest["tasks"]}) == 6
    assert all(row["qcis_sha256"] for row in manifest["tasks"])


def test_analysis_module_has_no_oracle_branch_terms() -> None:
    source = (__import__("pathlib").Path(__file__).parents[1] / "src" / "adaptive" / "task_metric_mirror.py").read_text(encoding="utf-8")
    assert "scenario ==" not in source
    assert "dgp_label" not in source
    assert "true_parameter" not in source
    assert "latent" not in source


def test_score_cli_smoke_uses_raw_counts_and_writes_report(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for index in range(4):
        rows.extend([
            _row(f"p{index}", "adaptive", 70 + index),
            _row(f"p{index}", "fixed", 60 + index),
        ])
    input_path = tmp_path / "mirror_rows.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "score"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "run_mirror_metric.py"),
            "score",
            "--input",
            str(input_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    endpoint = json.loads(completed.stdout)
    report = json.loads((output / "mirror_metric_report.json").read_text(encoding="utf-8"))
    assert endpoint["source"] == "raw_counts"
    assert report["endpoint"]["mean_difference"] == pytest.approx(0.10)
    assert report["config"]["minimum_actual_improvement_status"] == "frozen_after_tb6_before_stage1"
    assert (output / "paired_scores.csv").exists()
