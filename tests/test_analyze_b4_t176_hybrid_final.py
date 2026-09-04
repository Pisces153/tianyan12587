from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import analyze_b4_t176_hybrid_final as hybrid
from src.adaptive import shared_baseline_sensing as shared_baseline


def _append(records: list[dict], event: str, payload: dict) -> dict:
    row = {
        "sequence": len(records),
        "recorded_at_utc": f"2026-08-29T00:00:{len(records):02d}+00:00",
        "event": event,
        "previous_record_sha256": None if not records else records[-1]["record_sha256"],
        **payload,
    }
    row["record_sha256"] = hybrid.digest_payload(row)
    records.append(row)
    return row


def _source_plan() -> dict:
    sessions = []
    for session_index, order in ((0, ["fast", "slow"]), (1, ["slow", "fast"])):
        cycles = []
        for block_index, cadence in enumerate(order):
            for cycle_index in range(20):
                cycles.append({
                    "cycle_id": f"session{session_index:02d}-block{block_index:02d}-cycle{cycle_index:02d}",
                    "session_index": session_index,
                    "block_index": block_index,
                    "cadence": cadence,
                    "cadence_seconds": 90.0 if cadence == "fast" else 360.0,
                    "cycle_index": cycle_index,
                    "registered_pair_id": f"session{session_index:02d}-cyclepair{cycle_index:02d}",
                    "mirror_fields": [0.2 + cycle_index * 1e-4, 0.0],
                })
        sessions.append({"session_index": session_index, "block_order": order, "cycles": cycles})
    return {
        "schema": hybrid.SOURCE_PLAN_SCHEMA,
        "backend_id": hybrid.BACKEND_ID,
        "expected": {
            "sessions": 2,
            "cycles": 80,
            "complete_cadence_pairs": 40,
            "cycles_per_cadence_per_session": 20,
            "minimum_adjudicated_cycle_pairs": 30,
            "minimum_sessions_per_block_order": 1,
            "pair_discard_granularity": "cycle_pair",
        },
        "sessions": sessions,
    }


def _loop_config() -> dict:
    return {
        "sensing": {"phase_time_seconds": 0.47},
        "collection_correction": {
            "primary_adjudication": "cadence_ratio_permutation_gate",
            "simulation_pooling_permitted": False,
            "endpoint_shot_floor_shared": 8.577935222672066e-05,
        },
    }


def _baseline(session_index: int, position: str, y: float) -> dict:
    return shared_baseline.baseline_record(
        [
            {"observed_y": y, "observed_z": 0.99},
            {"observed_y": -y, "observed_z": 0.99},
        ],
        27_664,
        session_index=session_index,
        position=position,
    )


def _hardware_records(source_plan: dict) -> list[dict]:
    records: list[dict] = []
    _append(records, "session_baseline_measured", {
        "measurement_id": "session00-baseline-session_start",
        "session_index": 0,
        "position": "session_start",
        "baseline": _baseline(0, "session_start", 0.0),
    })
    for cycle in source_plan["sessions"][0]["cycles"]:
        compensation = [-0.15, 0.0, 0.0] if cycle["cadence"] == "fast" else [-0.10, 0.0, 0.0]
        _append(records, "cadence_cycle_completed", {
            **cycle,
            "shield": {"compensation": compensation, "permitted": True},
        })
    _append(records, "session_baseline_measured", {
        "measurement_id": "session00-baseline-session_end",
        "session_index": 0,
        "position": "session_end",
        "baseline": _baseline(0, "session_end", 0.001),
    })
    return records


def test_record_chain_rejects_tampering() -> None:
    records: list[dict] = []
    _append(records, "safe", {"value": 1})
    _append(records, "safe", {"value": 2})
    assert hybrid.verify_record_chain(records)["verified"] is True
    records[1]["value"] = 3
    with pytest.raises(ValueError, match="record SHA-256 mismatch"):
        hybrid.verify_record_chain(records)


def test_record_chain_accepts_uppercase_sha_links_without_rehashing_content() -> None:
    records: list[dict] = []
    first = {
        "sequence": 0,
        "recorded_at_utc": "2026-08-29T00:00:00+00:00",
        "event": "safe",
        "previous_record_sha256": None,
    }
    first["record_sha256"] = hybrid.digest_payload(first).upper()
    second = {
        "sequence": 1,
        "recorded_at_utc": "2026-08-29T00:00:01+00:00",
        "event": "safe",
        "previous_record_sha256": first["record_sha256"],
    }
    second["record_sha256"] = hybrid.digest_payload(second).upper()
    records.extend([first, second])
    result = hybrid.verify_record_chain(records)
    assert result["verified"] is True
    assert result["last_record_sha256"] == second["record_sha256"].lower()


def test_source_schedule_keeps_v4_order_and_pair_mapping() -> None:
    result = hybrid.validate_source_schedule(_source_plan(), _loop_config())
    assert result["verified"] is True
    assert [row["block_order"] for row in result["sessions"]] == [
        ["fast", "slow"],
        ["slow", "fast"],
    ]
    broken = _source_plan()
    broken["sessions"][1]["block_order"] = ["fast", "slow"]
    with pytest.raises(ValueError, match="block order changed"):
        hybrid.validate_source_schedule(broken, _loop_config())


def test_extract_hardware_session0_uses_derived_endpoint_only() -> None:
    source = _source_plan()
    records = _hardware_records(source)
    pairs, cycles, baseline, integrity = hybrid.extract_hardware_session0(
        source,
        _loop_config(),
        records,
    )
    assert len(pairs) == 20
    assert len(cycles) == 40
    assert integrity["raw_counts_read"] is False
    assert integrity["npz_read"] is False
    assert integrity["shield_permitted_cycles"] == 40
    assert baseline["qc"]["session_index"] == 0
    first = pairs[0]
    assert first["fast_endpoint_squared_residual"] == pytest.approx(0.05**2)
    assert first["slow_endpoint_squared_residual"] == pytest.approx(0.10**2)


def _write_simulation_fixture(root: Path) -> tuple[Path, Path, dict]:
    journal_path = root / "session1_simulated_snapshots.jsonl"
    pair_path = root / "session1_simulated_pair_rows.csv"
    records: list[dict] = []
    pair_rows = []
    for cycle_index in range(20):
        pair_id = f"session01-cyclepair{cycle_index:02d}"
        values = {"fast": 0.002 + cycle_index * 1e-6, "slow": 0.004 + cycle_index * 1e-6}
        cycle_ids = {}
        for cadence, block_index in (("slow", 0), ("fast", 1)):
            cycle_id = f"session01-block{block_index:02d}-cycle{cycle_index:02d}"
            cycle_ids[cadence] = cycle_id
            _append(records, "simulated_cadence_cycle_completed", {
                "record_origin": "simulation",
                "simulation_only": True,
                "hardware_job_submitted": False,
                "registered_hardware_endpoint_contribution": "none",
                "pooling_permitted": False,
                "source_registered_pair_id": pair_id,
                "session_index": 1,
                "block_index": block_index,
                "cycle_index": cycle_index,
                "cycle_id": cycle_id,
                "cadence": cadence,
                "endpoint_squared_residual": values[cadence],
            })
        pair_rows.append({
            "record_origin": "simulation",
            "simulation_only": True,
            "hardware_job_submitted": False,
            "registered_hardware_endpoint_contribution": "none",
            "pooling_permitted": False,
            "source_registered_pair_id": pair_id,
            "session_index": 1,
            "fast_cycle_id": cycle_ids["fast"],
            "slow_cycle_id": cycle_ids["slow"],
            "fast_endpoint_squared_residual": values["fast"],
            "slow_endpoint_squared_residual": values["slow"],
        })
    journal_path.write_text("".join(hybrid.canonical_json(row) + "\n" for row in records), encoding="utf-8")
    with pair_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    report = {
        "hardware_jobs_submitted": 0,
        "hardware_results_read": False,
        "registered_hardware_endpoint_contribution": "none",
        "simulation_only_counterfactual_endpoint": {"ratio": 0.5, "p_value": 0.01},
    }
    return pair_path, journal_path, report


def test_extract_simulation_session1_verifies_table_against_chain(tmp_path: Path) -> None:
    pair_path, journal_path, report = _write_simulation_fixture(tmp_path)
    pairs, integrity = hybrid.extract_simulation_session1(pair_path, journal_path, report)
    assert len(pairs) == 20
    assert integrity["record_chain"]["rows"] == 40
    assert pairs[0]["evidence_origin"] == "simulation_session1"
    assert pairs[0]["block_order"] == "slow_then_fast"


def _gate(ratio: float, passed: bool) -> dict:
    return {"ratio": ratio, "passed": passed}


def test_hybrid_decision_can_preserve_only_simulation_assisted_status() -> None:
    decision = hybrid.classify_decision(
        hardware_gate=_gate(0.70, False),
        simulation_gate=_gate(0.40, True),
        hybrid_gate=_gate(0.55, True),
        prediction_interval=[0.326, 0.788],
        integrity_passed=True,
        pair_counts_exact=True,
    )
    assert decision["simulation_assisted_status"] == "B4_PRESERVED_SIMULATION_ASSISTED"
    assert decision["registered_hardware_endpoint_status"] == "INCONCLUSIVE_MISSING_HARDWARE_SESSION1"
    assert decision["registered_hardware_endpoint_passed"] is False
    assert decision["hardware_session0_evidence_grade"] == "HARDWARE_SESSION0_DIRECTIONAL_SUPPORT_ONLY"


def test_origin_reversal_blocks_hybrid_preservation() -> None:
    decision = hybrid.classify_decision(
        hardware_gate=_gate(1.10, False),
        simulation_gate=_gate(0.30, True),
        hybrid_gate=_gate(0.60, True),
        prediction_interval=[0.326, 0.788],
        integrity_passed=True,
        pair_counts_exact=True,
    )
    assert decision["simulation_assisted_passed"] is False
    assert decision["criteria"]["no_origin_direction_reversal"] is False


def test_fallacy_scan_always_covers_all_eleven() -> None:
    decision = hybrid.classify_decision(
        hardware_gate=_gate(0.70, False),
        simulation_gate=_gate(0.40, True),
        hybrid_gate=_gate(0.55, True),
        prediction_interval=[0.326, 0.788],
        integrity_passed=True,
        pair_counts_exact=True,
    )
    scan = hybrid.build_fallacy_scan(decision=decision, hardware_pairs=20, simulation_pairs=20)
    assert scan["coverage"] == "11/11 checked"
    assert [row["index"] for row in scan["items"]] == list(range(1, 12))
    assert {row["fallacy"] for row in scan["items"]} == {
        "Simpson's Paradox",
        "Ecological Fallacy",
        "Berkson's Paradox",
        "Collider Bias",
        "Base Rate Neglect",
        "Regression to the Mean",
        "Survivorship Bias",
        "Look-Elsewhere Effect",
        "Garden of Forking Paths",
        "Correlation != Causation",
        "Reverse Causality",
    }


def test_hybrid_drift_sensitivity_respects_opposite_block_orders() -> None:
    pairs = []
    for session, order in ((0, "fast_then_slow"), (1, "slow_then_fast")):
        for index in range(3):
            pairs.append({
                "session_index": session,
                "block_order": order,
                "fast_endpoint_squared_residual": 0.2 + index * 0.01,
                "slow_endpoint_squared_residual": 0.5 + index * 0.01,
            })
    offsets = {
        session: {
            "shapes": {
                name: {"block_one_offset": first * 0.01, "block_two_offset": second * 0.01}
                for name, (first, second) in shared_baseline.DRIFT_SHAPES.items()
            }
        }
        for session in (0, 1)
    }
    result = hybrid.hybrid_drift_sensitivity(
        pairs,
        offsets,
        alpha=0.05,
        permutations=200,
        seed=7,
        primary_passed=True,
    )
    linear = result["shapes"]["linear_ramp"]
    assert linear["mean_removed_from_fast"] == pytest.approx(linear["mean_removed_from_slow"])
    assert set(result["shape_verdicts"]) == set(shared_baseline.DRIFT_SHAPES)
