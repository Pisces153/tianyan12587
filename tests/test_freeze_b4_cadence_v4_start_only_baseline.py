from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import freeze_b4_cadence_v4_start_only_baseline as freeze_module


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json"


def _backend_config(*, mirror_runtime: float = 72.057) -> dict:
    registered = 2.373
    measured = 2.3639
    return {
        "backend": {
            "backend_id": "tianyan176",
            "cadence_supplement_timing": {
                "source_ledger": "E:/evidence/t176_platform_task_time_ledger.json",
                "source_ledger_sha256": "a" * 64,
                "queue_free_slope_shots_per_second": 1414.98122405956,
                "unconstrained_intercept_seconds": -0.815789458981282,
                "nonnegative_overhead_seconds_used": 0.0,
                "budget_model": "role_envelope_sum_task_runtime",
                "role_task_runtime_seconds": {
                    "baseline": 18.735,
                    "sense": 3.556,
                    "mirror": mirror_runtime,
                },
                "role_settings_per_job": {"baseline": 2, "sense": 2, "mirror": 2},
                "floor_constant_verification": {
                    "registered_constant": registered,
                    "measured_constant": measured,
                    "ratio_to_registered": measured / registered,
                    "source_artifact": (
                        "E:/quarantine/tianyan176/amplified_closed_loop_report.json"
                    ),
                    "source_artifact_sha256": "b" * 64,
                    "pooling_permitted": False,
                    "registered_endpoint_contribution": "none",
                },
            },
        }
    }


def _write_inputs(tmp_path: Path, *, loop: dict | None = None, backend: dict | None = None):
    loop_path = tmp_path / "loop.json"
    backend_path = tmp_path / "backend.json"
    if loop is None:
        loop = json.loads(SOURCE_LOOP_CONFIG.read_text(encoding="utf-8"))
    if backend is None:
        backend = _backend_config()
    loop_path.write_text(json.dumps(loop, indent=2) + "\n", encoding="utf-8")
    backend_path.write_text(json.dumps(backend, indent=2) + "\n", encoding="utf-8")
    return loop_path, backend_path


def test_role_envelope_retains_all_40_pairs_under_both_budget_gates(tmp_path: Path) -> None:
    loop_path, backend_path = _write_inputs(tmp_path)
    loop = json.loads(loop_path.read_text(encoding="utf-8"))
    timing = freeze_module.read_cadence_supplement_timing(
        json.loads(backend_path.read_text(encoding="utf-8"))
    )

    design = freeze_module.design(loop, timing)

    assert design["role_jobs_per_session"] == {"baseline": 2, "sense": 40, "mirror": 2}
    assert design["role_settings_per_job"] == {"baseline": 2, "sense": 2, "mirror": 2}
    assert design["execution_wall_seconds_per_session"] == pytest.approx(323.824)
    assert design["quota_seconds_per_session"] == pytest.approx(647.648)
    assert design["quota_seconds_total"] == pytest.approx(1295.296)
    assert design["quota_seconds_per_session"] < loop["daily_window_seconds"]
    assert design["quota_seconds_total"] < loop["collection_correction"][
        "machine_time_ceiling_seconds"
    ]
    assert loop["collection_correction"]["registered_cycle_pairs_total"] == 40


def test_default_freeze_changes_timing_only_and_preserves_original_hash(tmp_path: Path) -> None:
    loop_path, backend_path = _write_inputs(tmp_path)
    before = json.loads(loop_path.read_text(encoding="utf-8"))
    before_bytes = loop_path.read_bytes()
    artifact_dir = tmp_path / "artifacts"

    result = freeze_module.freeze(
        loop_config_path=loop_path,
        backend_config_path=backend_path,
        artifact_dir=artifact_dir,
    )

    after = json.loads(loop_path.read_text(encoding="utf-8"))
    old_correction = before["collection_correction"]
    new_correction = after["collection_correction"]
    for field in freeze_module.REGISTERED_STATISTICAL_FIELDS:
        assert new_correction[field] == old_correction[field], field
    assert new_correction["reachability_evidence"] == old_correction["reachability_evidence"]
    assert new_correction["reachability_evidence_sha256"] == old_correction[
        "reachability_evidence_sha256"
    ]

    permitted_changes = set(freeze_module.TIMING_UPDATE_FIELDS) | {
        "timing_migration_status",
        "timing_backend_id",
        "timing_budget_model",
        "role_task_runtime_seconds",
        "role_settings_per_job",
        "role_jobs_per_session",
        "execution_wall_seconds_per_session",
        "execution_wall_seconds_total",
        "quota_seconds_per_session",
        "quota_seconds_total",
        "queue_free_slope_shots_per_second",
        "unconstrained_intercept_seconds",
        "nonnegative_overhead_seconds_used",
        "timing_source_ledger",
        "timing_source_ledger_sha256",
        "floor_constant_backend_verification",
        "timing_reachability_evidence",
        "timing_reachability_evidence_sha256",
        "timing_reachability_evidence_note",
    }
    for field, value in old_correction.items():
        if field not in permitted_changes:
            assert new_correction[field] == value, field

    assert new_correction["timing_budget_model"] == "role_envelope_sum_task_runtime"
    assert new_correction["modelled_busy_seconds_per_session"] == pytest.approx(647.648)
    assert new_correction["modelled_busy_seconds_total"] == pytest.approx(1295.296)
    assert new_correction["shot_rate_per_second_used"] == pytest.approx(1414.98122405956)
    assert new_correction["seconds_per_job_used"] == 0.0
    assert new_correction["seconds_per_setting_used"] == 0.0
    assert new_correction["timing_reachability_evidence_sha256"] == result[
        "artifact_sha256"
    ]

    payload = json.loads(result["artifact"].read_text(encoding="utf-8"))
    assert hashlib.sha256(result["artifact"].read_bytes()).hexdigest() == result[
        "artifact_sha256"
    ]
    assert payload["schema"] == "b4_cadence_v4_backend_migration_timing_v1"
    assert payload["not_a_new_preregistration"] is True
    assert payload["original_registered_statistical_evidence"]["sha256"] == (
        old_correction["reachability_evidence_sha256"]
    )
    assert payload["source_hashes"]["loop_config_sha256_before_timing_update"] == (
        hashlib.sha256(before_bytes).hexdigest()
    )
    assert payload["integrity"]["pooling_permitted"] is False
    assert payload["integrity"]["registered_endpoint_contribution"] == "none"
    assert payload["integrity"]["monte_carlo_recomputed_by_this_artifact"] is False


def test_timing_only_fails_before_writing_if_statistic_would_change(tmp_path: Path) -> None:
    loop = json.loads(SOURCE_LOOP_CONFIG.read_text(encoding="utf-8"))
    loop["collection_correction"]["expected_ratio"] += 1e-12
    loop_path, backend_path = _write_inputs(tmp_path, loop=loop)
    before = loop_path.read_bytes()
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="expected_ratio"):
        freeze_module.freeze(
            loop_config_path=loop_path,
            backend_config_path=backend_path,
            artifact_dir=artifact_dir,
        )

    assert loop_path.read_bytes() == before
    assert not artifact_dir.exists()


def test_explicit_rewrite_statistics_is_required_to_repair_a_mismatch(tmp_path: Path) -> None:
    loop = json.loads(SOURCE_LOOP_CONFIG.read_text(encoding="utf-8"))
    registered_ratio = loop["collection_correction"]["expected_ratio"]
    loop["collection_correction"]["expected_ratio"] = 0.75
    loop_path, backend_path = _write_inputs(tmp_path, loop=loop)

    result = freeze_module.freeze(
        loop_config_path=loop_path,
        backend_config_path=backend_path,
        artifact_dir=tmp_path / "artifacts",
        rewrite_statistics=True,
    )

    after = json.loads(loop_path.read_text(encoding="utf-8"))
    assert after["collection_correction"]["expected_ratio"] == registered_ratio
    payload = json.loads(result["artifact"].read_text(encoding="utf-8"))
    assert payload["statistics_guard"]["rewrite_statistics_requested"] is True
    assert payload["statistics_guard"]["mismatches"][0]["field"] == "expected_ratio"
    assert payload["integrity"]["statistical_fields_written"] is True


def test_floor_probe_is_provenance_only_and_never_enters_statistics(tmp_path: Path) -> None:
    loop_path, backend_path = _write_inputs(tmp_path)
    before = json.loads(loop_path.read_text(encoding="utf-8"))["collection_correction"]

    result = freeze_module.freeze(
        loop_config_path=loop_path,
        backend_config_path=backend_path,
        artifact_dir=tmp_path / "artifacts",
    )

    after = json.loads(loop_path.read_text(encoding="utf-8"))["collection_correction"]
    assert after["endpoint_shot_floor_total"] == before["endpoint_shot_floor_total"]
    verification = after["floor_constant_backend_verification"]
    assert verification["registered_constant"] == 2.373
    assert verification["measured_constant"] == 2.3639
    payload = json.loads(result["artifact"].read_text(encoding="utf-8"))
    assert payload["floor_constant_backend_verification"]["use_in_this_artifact"] == (
        "provenance_only"
    )
    assert payload["floor_constant_backend_verification"]["statistical_constant_changed"] is False


def test_daily_gate_uses_role_quota_envelope_not_rate_only_model(tmp_path: Path) -> None:
    backend = _backend_config(mirror_runtime=400.0)
    loop_path, backend_path = _write_inputs(tmp_path, backend=backend)

    with pytest.raises(ValueError, match="daily quota envelope"):
        freeze_module.freeze(
            loop_config_path=loop_path,
            backend_config_path=backend_path,
            artifact_dir=tmp_path / "artifacts",
        )

    assert not (tmp_path / "artifacts").exists()


def test_cli_accepts_directed_loop_backend_and_artifact_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loop_path, backend_path = _write_inputs(tmp_path)
    artifact_dir = tmp_path / "directed"

    assert freeze_module.main(
        [
            "--loop-config",
            str(loop_path),
            "--backend-config",
            str(backend_path),
            "--artifact-dir",
            str(artifact_dir),
            "--artifact-name",
            "timing.json",
        ]
    ) == 0

    assert (artifact_dir / "timing.json").exists()
    output = capsys.readouterr().out
    assert "647.648 s/session" in output
    assert "40 retained" in output
