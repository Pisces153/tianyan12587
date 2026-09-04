from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from scripts import drift_campaign_v4
from scripts import preflight_b4_cadence_collection_correction as module
from scripts import run_b4_cadence_pair_hardware as runner
from scripts import run_cadence_pair_loop as cadence


ROOT = Path(__file__).resolve().parents[1]
LOOP_CONFIG = ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json"
BACKEND_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan176.json"
PEER_CONFIG = ROOT / "config" / "b4_drift_campaign_v4_tianyan287.json"


def timing_only_plan() -> dict:
    return runner.build_plan_payload(
        cadence.load_config(LOOP_CONFIG),
        drift_campaign_v4.load_config(BACKEND_CONFIG),
        operational_start_utc=datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc),
        validate_registered_reachability=False,
    )


def test_t176_timing_summary_reports_frozen_role_envelope_and_forty_pairs() -> None:
    plan = timing_only_plan()
    summary = module.timing_summary(plan)
    assert plan["registered_endpoint_reachability"]["validation_performed"] is False
    assert summary["timing_budget_model"] == "role_envelope_sum_task_runtime"
    assert summary["daily_budget_metric"] == "quota_seconds"
    assert summary["registered_cycle_pairs"] == 40
    assert summary["quota_seconds_per_session"] == pytest.approx(647.648)
    assert summary["quota_seconds_total"] == pytest.approx(1295.296)
    assert summary["execution_wall_seconds_per_session"] == pytest.approx(323.824)
    assert summary["execution_wall_seconds_total"] == pytest.approx(647.648)
    assert summary["daily_budget_passed"] is True


def test_statistics_are_copied_from_correction_without_recomputation() -> None:
    correction = timing_only_plan()["collection_correction"]
    snapshot = module.frozen_statistical_snapshot(correction)
    assert snapshot["source"] == "plan.collection_correction"
    assert snapshot["recomputed_by_preflight"] is False
    for field in module.FROZEN_STATISTICAL_FIELDS:
        assert snapshot[field] == correction[field]


def test_readiness_status_reflects_manifest_and_isolation_state() -> None:
    assert module.readiness_status(
        daily_budget_passed=True,
        quarantine_read=False,
        curated_manifest_verified=False,
    ) == "PENDING_MIGRATION_CURATED_MANIFEST"
    assert module.readiness_status(
        daily_budget_passed=True,
        quarantine_read=False,
        curated_manifest_verified=True,
    ) == "READY_FOR_REGISTERED_COLLECTION"
    assert module.readiness_status(
        daily_budget_passed=True,
        quarantine_read=True,
        curated_manifest_verified=True,
    ) == "BLOCKED_INPUT_ISOLATION_VIOLATION"


def test_recommended_command_uses_registered_quarantine_path_and_manifest_placeholder() -> None:
    command = module.recommended_hardware_command(
        config_path=LOOP_CONFIG,
        backend_config_path=BACKEND_CONFIG,
        peer_config_path=PEER_CONFIG,
    )
    assert "--loop-config config/b4_cadence_pair_loop_cycle_paired_v4.json" in command
    assert "--backend-config config/b4_drift_campaign_v4_tianyan176.json" in command
    assert "--peer-config config/b4_drift_campaign_v4_tianyan287.json" in command
    assert "--stage1-manifest docs/B4_T176_MIGRATION_CURATED_MANIFEST_20260823.json" in command
    assert (
        "--output E:/TianYan/XA-202609/quarantine/tianyan176/"
        "B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823"
    ) in command
    assert command.endswith("--wait-for-running --confirm-hardware")


def test_reachability_integrity_accepts_byte_exact_or_crlf_storage_only(tmp_path: Path) -> None:
    canonical = b'{\n  "registered": true\n}\n'
    expected = sha256(canonical).hexdigest()
    exact_path = tmp_path / "exact.json"
    exact_path.write_bytes(canonical)
    exact = module.verify_reachability_evidence({
        "reachability_evidence": str(exact_path),
        "reachability_evidence_sha256": expected,
    })
    assert exact["match_mode"] == "byte_exact"
    assert exact["statistical_content_drift_detected"] is False

    crlf_path = tmp_path / "crlf.json"
    crlf_path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    crlf = module.verify_reachability_evidence({
        "reachability_evidence": str(crlf_path),
        "reachability_evidence_sha256": expected,
    })
    assert crlf["match_mode"] == "crlf_storage_canonical_lf_match"
    assert crlf["canonical_lf_sha256"] == expected.upper()
    assert crlf["artifact_modified_by_preflight"] is False


def test_reachability_integrity_rejects_content_change(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_bytes(b'{\r\n  "registered": false\r\n}\r\n')
    expected = sha256(b'{\n  "registered": true\n}\n').hexdigest()
    with pytest.raises(RuntimeError, match="changed beyond accepted newline storage"):
        module.verify_reachability_evidence({
            "reachability_evidence": str(path),
            "reachability_evidence_sha256": expected,
        })


def test_migration_manifest_is_verified_by_runner_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    calls: list[Path] = []

    def fake_verify(path: Path) -> dict:
        calls.append(path)
        return {"passed": True}

    monkeypatch.setattr(module.runner, "verify_stage1_manifest", fake_verify)
    result = module.verify_migration_curated_manifest(manifest)
    assert calls == [manifest]
    assert result["verified"] is True
    assert result["runner_audit"] == {"passed": True}


def test_missing_migration_manifest_keeps_preflight_pending(tmp_path: Path) -> None:
    result = module.verify_migration_curated_manifest(tmp_path / "missing.json")
    assert result["exists"] is False
    assert result["verified"] is False
