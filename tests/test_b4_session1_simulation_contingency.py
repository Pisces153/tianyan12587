from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from scripts import run_b4_session1_simulation_contingency as simulation


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def _cycle(
    *,
    cadence: str,
    block_index: int,
    cycle_index: int,
    shots: int,
    mirror_shots: int,
) -> dict:
    cadence_seconds = 90.0 if cadence == "fast" else 360.0
    sense = 86400.0 + block_index * 5000.0 + cycle_index * cadence_seconds
    pair_id = f"session01-pair{cycle_index:02d}"
    return {
        "cycle_id": f"session01-block{block_index:02d}-cycle{cycle_index:02d}",
        "session_index": 1,
        "block_index": block_index,
        "cadence": cadence,
        "cadence_seconds": cadence_seconds,
        "cycle_index": cycle_index,
        "registered_pair_id": pair_id,
        "virtual_sense_seconds": sense,
        "virtual_mirror_seconds": sense + cadence_seconds,
        "sense_target_utc": "2026-08-25T10:00:00+00:00",
        "mirror_target_utc": "2026-08-25T10:01:30+00:00",
        "sense_fields": [0.01 + cycle_index * 1e-4, -0.008],
        "mirror_fields": [0.012 + cycle_index * 1e-4, -0.006],
        "sense_ou_advance": {},
        "mirror_ou_advance": {},
        "sensing_shots_per_setting": shots,
        "mirror_shots_per_task": mirror_shots,
        "mirror_seeds": [1700 + block_index] if cycle_index == 0 else [],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    original_config = Path(simulation.ROOT / "config" / "b4_cadence_pair_loop_cycle_paired_v4.json")
    loop_config = tmp_path / "loop.json"
    loop_config.write_bytes(original_config.read_bytes())
    config = json.loads(loop_config.read_text(encoding="utf-8"))
    correction = config["collection_correction"]
    cycles = [
        *[
            _cycle(
                cadence="slow",
                block_index=0,
                cycle_index=index,
                shots=int(correction["sensing_shots_per_setting"]),
                mirror_shots=int(config["mirror"]["shots_per_task"]),
            )
            for index in range(20)
        ],
        *[
            _cycle(
                cadence="fast",
                block_index=1,
                cycle_index=index,
                shots=int(correction["sensing_shots_per_setting"]),
                mirror_shots=int(config["mirror"]["shots_per_task"]),
            )
            for index in range(20)
        ],
    ]
    baselines = [
        {
            "measurement_id": "session01-baseline-session_start",
            "session_index": 1,
            "position": "session_start",
            "shots_per_setting": int(correction["baseline_shots_per_setting"]),
            "used_by_estimate": True,
        },
        {
            "measurement_id": "session01-baseline-session_end",
            "session_index": 1,
            "position": "session_end",
            "shots_per_setting": int(correction["baseline_end_shots_per_setting"]),
            "used_by_estimate": False,
        },
    ]
    base = {
        "schema": simulation.SOURCE_PLAN_SCHEMA,
        "backend_id": "tianyan176",
        "source_hashes": {
            "loop_config_sha256": simulation.digest_file(loop_config),
            "runner_sha256": "hardware-runner-frozen-hash",
        },
        "collection_correction": correction,
        "expected": {
            "sessions": 2,
            "complete_cadence_pairs": 40,
            "cycles_per_cadence_per_session": 20,
            "minimum_adjudicated_cycle_pairs": 30,
            "minimum_sessions_per_block_order": 1,
        },
        "sessions": [
            {
                "session_index": 0,
                "block_order": ["fast", "slow"],
                "virtual_start_seconds": 0.0,
                "baseline_measurements": [],
                "cycles": [],
            },
            {
                "session_index": 1,
                "block_order": ["slow", "fast"],
                "virtual_start_seconds": 86400.0,
                "baseline_measurements": baselines,
                "cycles": cycles,
            },
        ],
    }
    base_plan = tmp_path / "supplement_plan.json"
    base_plan.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    plan = tmp_path / "session1-simulation.plan.json"
    output = tmp_path / "simulation-output"
    return base_plan, loop_config, plan, output


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_freeze_and_run_session1_without_touching_sources(tmp_path: Path) -> None:
    base, config, plan_path, output = _fixture(tmp_path)
    base_before = base.read_bytes()
    config_before = config.read_bytes()
    frozen = simulation.freeze_plan(
        base_plan_path=base,
        loop_config_path=config,
        plan_path=plan_path,
        output_dir=output,
        monte_carlo_replicates=200,
        permutation_count=200,
        created_at_utc=NOW,
    )
    assert frozen["status"] == "frozen_before_hardware_outcome_unsealing"
    assert frozen["session1_contract"]["block_order"] == ["slow", "fast"]
    assert frozen["separation_contract"]["hardware_raw_counts_read"] is False
    assert base.read_bytes() == base_before
    assert config.read_bytes() == config_before

    result = simulation.run_frozen_plan(plan_path)
    assert result["report"]["project_status"] == "COMPLETED_WITH_SIMULATION_CONTINGENCY"
    assert result["report"]["registered_hardware_endpoint_status"] == "PENDING_HARDWARE"
    assert result["report"]["session1"]["cycles"] == 40
    assert result["report"]["session1"]["counterfactual_pair_count"] == 20
    assert result["report"]["hardware_jobs_submitted"] == 0
    assert result["manifest"]["pooling_permitted"] is False
    assert base.read_bytes() == base_before
    assert config.read_bytes() == config_before

    rows = [json.loads(line) for line in (output / "session1_simulated_snapshots.jsonl").read_text().splitlines()]
    assert len(rows) == 43
    assert [row["sequence"] for row in rows] == list(range(43))
    assert all(row["record_origin"] == "simulation" for row in rows)
    assert all(row["pooling_permitted"] is False for row in rows)
    assert all(row["hardware_job_submitted"] is False for row in rows)
    assert all(not _contains_key(row, "query_id") for row in rows)
    assert all(not _contains_key(row, "raw_counts") for row in rows)
    assert rows[-1]["event"] == "simulated_session1_counterfactual_gate"


def test_simulation_is_deterministic_for_the_same_frozen_sources(tmp_path: Path) -> None:
    base, config, plan_path, output = _fixture(tmp_path)
    frozen = simulation.freeze_plan(
        base_plan_path=base,
        loop_config_path=config,
        plan_path=plan_path,
        output_dir=output,
        monte_carlo_replicates=100,
        permutation_count=100,
        created_at_utc=NOW,
    )
    source = json.loads(base.read_text())
    loop = json.loads(config.read_text())
    session = simulation._session(source, 1)
    first = simulation.simulate_session1(frozen, source, loop, session)
    second = simulation.simulate_session1(frozen, source, loop, session)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]
    assert first[0]["simulation_only_counterfactual_endpoint"] == second[0][
        "simulation_only_counterfactual_endpoint"
    ]
    assert first[0]["session1_model_envelope"] == second[0]["session1_model_envelope"]


def test_freeze_rejects_pooling_and_overwrite(tmp_path: Path) -> None:
    base, config, plan_path, output = _fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["collection_correction"]["simulation_pooling_permitted"] = True
    config.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    source = json.loads(base.read_text())
    source["collection_correction"] = payload["collection_correction"]
    source["source_hashes"]["loop_config_sha256"] = simulation.digest_file(config)
    base.write_text(json.dumps(source) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pooling must be explicitly forbidden"):
        simulation.freeze_plan(
            base_plan_path=base,
            loop_config_path=config,
            plan_path=plan_path,
            output_dir=output,
            monte_carlo_replicates=100,
            permutation_count=100,
            created_at_utc=NOW,
        )

    base, config, plan_path, output = _fixture(tmp_path / "second")
    simulation.freeze_plan(
        base_plan_path=base,
        loop_config_path=config,
        plan_path=plan_path,
        output_dir=output,
        monte_carlo_replicates=100,
        permutation_count=100,
        created_at_utc=NOW,
    )
    with pytest.raises(FileExistsError, match="overwrite frozen simulation plan"):
        simulation.freeze_plan(
            base_plan_path=base,
            loop_config_path=config,
            plan_path=plan_path,
            output_dir=output,
            monte_carlo_replicates=100,
            permutation_count=100,
            created_at_utc=NOW,
        )


def test_run_fails_closed_when_a_frozen_source_changes(tmp_path: Path) -> None:
    base, config, plan_path, output = _fixture(tmp_path)
    simulation.freeze_plan(
        base_plan_path=base,
        loop_config_path=config,
        plan_path=plan_path,
        output_dir=output,
        monte_carlo_replicates=100,
        permutation_count=100,
        created_at_utc=NOW,
    )
    config.write_text(config.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source hash changed"):
        simulation.run_frozen_plan(plan_path)


def test_runner_contains_no_hardware_client_or_submission_path() -> None:
    source = Path(simulation.__file__).read_text(encoding="utf-8")
    assert "import cqlib" not in source
    assert "submit_job(" not in source
    assert "query_quantum" not in source
    assert "platform.login" not in source
