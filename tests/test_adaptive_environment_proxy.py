from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.adaptive.environment_proxy import FORBIDDEN_TERMS, extract_campaign, extract_snapshot
from src.features.pauli import BASIS_ORDER


def _write_counts(path: Path, *, all_zero_success: int, all_one_success: int) -> None:
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for time_index in range(3):
        for basis in BASIS_ORDER:
            row = np.zeros(64, dtype=np.int64)
            row[0] = 1024
            labels.append(f"anchor_t{time_index}_{basis}")
            rows.append(row)
    for index in range(4):
        row = np.zeros(64, dtype=np.int64)
        row[0] = 1024
        labels.append(f"interleaved_reference_{index + 1}")
        rows.append(row)
    for label, expected, success in (("readout_all_zero", 0, all_zero_success), ("readout_all_one", 63, all_one_success)):
        row = np.zeros(64, dtype=np.int64)
        row[expected] = success
        row[1 if expected == 0 else 62] = 1024 - success
        labels.append(label)
        rows.append(row)
    np.savez_compressed(path, labels=np.asarray(labels), counts=np.asarray(rows), shots=np.asarray(1024), snapshot_id=np.asarray("snap"), probe_manifest_sha256=np.asarray("probe"))


def test_proxy_extraction_separates_tasks_and_blocks_first_snapshot_drift(tmp_path: Path) -> None:
    counts = tmp_path / "counts.npz"
    _write_counts(counts, all_zero_success=1000, all_one_success=980)
    record = extract_snapshot(counts, campaign_id="campaign", backend_id="tianyan-287", scheduled_utc="2026-08-01T00:00:00+00:00", snapshot_index=0, previous_record=None)
    assert not set(record["proxy_task_labels"]).intersection(record["label_task_labels"])
    assert all(not value["drift"]["claim_permitted"] for value in record["observable_environment_proxy"].values())
    covariance = np.asarray(record["effective_field_state"]["covariance_h1_h2"])
    assert covariance.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    serialized = str(record).lower()
    assert not any(word in serialized for word in FORBIDDEN_TERMS)


def test_proxy_extraction_claim_rule_uses_combined_floor(tmp_path: Path) -> None:
    first_path, second_path = tmp_path / "first.npz", tmp_path / "second.npz"
    _write_counts(first_path, all_zero_success=1024, all_one_success=1024)
    first = extract_snapshot(first_path, campaign_id="campaign", backend_id="tianyan-287", scheduled_utc="2026-08-01T00:00:00+00:00", snapshot_index=0, previous_record=None)
    _write_counts(second_path, all_zero_success=900, all_one_success=900)
    second = extract_snapshot(second_path, campaign_id="campaign", backend_id="tianyan-287", scheduled_utc="2026-08-01T00:25:00+00:00", snapshot_index=1, previous_record=first)
    assert second["observable_environment_proxy"]["readout_mean_error"]["drift"]["claim_permitted"]


def test_effective_fields_and_platform_summary_use_only_snapshot_inputs(tmp_path: Path) -> None:
    counts = tmp_path / "counts.npz"
    _write_counts(counts, all_zero_success=1000, all_one_success=980)
    telemetry = {
        "raw": {
            "download_config": {
                "calibrationTime": "2026-08-01 00:00:00",
                "qubit": {
                    "relatime": {
                        "T1": {"qubit_used": ["Q1", "Q2"], "param_list": [10.0, 14.0], "unit": "us"},
                        "T2": {"qubit_used": ["Q1", "Q2"], "param_list": [8.0, 12.0], "unit": "us"},
                    },
                    "frequency": {"f01": {"qubit_used": ["Q1", "Q2"], "param_list": [4.9, 5.1], "unit": "GHz"}},
                },
                "singleQubit": {"gate error": {"qubit_used": ["Q1", "Q2"], "param_list": [0.01, 0.03], "unit": "percent"}},
                "overview": {"cz_error": 0.2, "readout_error": 0.4},
            }
        }
    }
    record = extract_snapshot(
        counts, campaign_id="campaign", backend_id="tianyan-287", scheduled_utc="2026-08-01T00:00:00+00:00",
        snapshot_index=0, previous_record=None, physical_qubits=(1, 2), telemetry=telemetry,
    )
    fields = record["platform_calibration_metadata"]["fields"]
    assert record["platform_calibration_metadata"]["calibration_timestamp_raw"] == "2026-08-01 00:00:00"
    assert fields["mean_T1"]["value"] == 12.0
    assert fields["mean_T2"]["value"] == 10.0
    assert fields["mean_f01"]["value"] == 5.0
    assert fields["mean_1q_gate_error"]["value"] == 0.02


def test_campaign_report_binds_signed_t6_evidence_to_written_corpus(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    counts = campaign_root / "counts.npz"
    _write_counts(counts, all_zero_success=1000, all_one_success=980)
    (campaign_root / "campaign_manifest.json").write_text(json.dumps({"campaign_id": "campaign"}), encoding="utf-8")
    journal = [
        {"event": "submitted", "snapshot_id": "snap", "backend_id": "tianyan-287", "scheduled_utc": "2026-08-01T00:00:00+00:00"},
        {"event": "collected", "snapshot_id": "snap", "counts_path": str(counts)},
    ]
    (campaign_root / "snapshots.jsonl").write_text("\n".join(json.dumps(row) for row in journal) + "\n", encoding="utf-8")
    result = extract_campaign(campaign_root, tmp_path / "out")
    report = result["report"]
    corpus = Path(result["corpus_path"])
    expected_corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest().upper()
    canonical = dict(report)
    supplied_hash = canonical.pop("self_sha256")
    expected_self_hash = hashlib.sha256(json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest().upper()
    assert report["feature_corpus_sha256"] == expected_corpus_sha
    assert supplied_hash == expected_self_hash
