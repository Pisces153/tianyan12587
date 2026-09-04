#!/usr/bin/env python3
"""Freeze, run, and verify the B4/T176 post-hoc hybrid final test.

The registered v4 design forbids synthetic pairs from entering its hardware endpoint.
That boundary is preserved here.  This script produces a separately labelled,
simulation-assisted consistency test from the completed T176 Session 0 hardware pairs
and the isolated Session 1 contingency simulation.  It can never emit a registered
hardware PASS.

The ``freeze`` command hashes every source and locks the endpoint, pairing, permutation
calibration, prediction check, and claim vocabulary without reading Session 0 scientific
fields.  The ``run`` command is the explicit unseal: it reads only the already-derived
cycle/baseline fields in ``snapshots.jsonl`` and never opens raw-count or NPZ files.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform as runtime_platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive import cadence_permutation
from src.adaptive import sensing_economics
from src.adaptive import shared_baseline_sensing as shared_baseline


PLAN_SCHEMA = "b4_t176_hybrid_final_plan_v2"
REPORT_SCHEMA = "b4_t176_hybrid_final_report_v2"
MANIFEST_SCHEMA = "b4_t176_hybrid_final_manifest_v2"
VERIFICATION_SCHEMA = "b4_t176_hybrid_final_verification_v2"
SOURCE_PLAN_SCHEMA = "b4_cadence_pair_hardware_supplement_plan_v1"
SIMULATION_PLAN_SCHEMA = "b4_t176_session1_simulation_contingency_plan_v1"
SIMULATION_REPORT_SCHEMA = "b4_t176_session1_simulation_contingency_report_v1"
SIMULATION_MANIFEST_SCHEMA = "b4_t176_session1_simulation_contingency_manifest_v1"
BACKEND_ID = "tianyan176"
HARDWARE_SESSION = 0
SIMULATION_SESSION = 1
PAIRS_PER_SOURCE = 20
TOTAL_PAIRS = 40
PRIMARY_ALPHA = 0.05
PRIMARY_PERMUTATIONS = 20_000
PRIMARY_SEED = 20_260_815
PLAN_STATUS = "FROZEN_AFTER_UNSEAL_CHAIN_CASE_COMPATIBILITY_ERRATUM"
HYBRID_LABEL = "POST_HOC_HYBRID_SIMULATION_ASSISTED"
BAD_HARDWARE_EVENTS = {
    "submission_rejected",
    "partial",
    "session_pair_block_discarded",
    "session_baseline_measurement_lost",
}
HARDWARE_SCIENTIFIC_FIELD_WHITELIST = [
    "cadence_cycle_completed.cycle_id",
    "cadence_cycle_completed.session_index",
    "cadence_cycle_completed.block_index",
    "cadence_cycle_completed.cadence",
    "cadence_cycle_completed.cycle_index",
    "cadence_cycle_completed.registered_pair_id",
    "cadence_cycle_completed.mirror_fields[0:2]",
    "cadence_cycle_completed.shield.compensation[0:2]",
    "cadence_cycle_completed.shield.permitted",
    "session_baseline_measured.baseline",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest().lower()


def digest_payload(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            rows.append(value)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    write_new_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_new_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid CSV boolean: {value!r}")


def finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {label}: {value!r}")
    return result


def verify_record_chain(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous: str | None = None
    for sequence, row in enumerate(records):
        if int(row.get("sequence", -1)) != sequence:
            raise ValueError(f"journal sequence is not contiguous at row {sequence}")
        observed_previous = row.get("previous_record_sha256")
        normalized_previous = (
            None if observed_previous is None else str(observed_previous).lower()
        )
        # The original T176 CampaignStore serialised the SHA fields in uppercase while
        # the isolated simulation journal uses lowercase.  SHA-256 hexadecimal is
        # case-insensitive; canonical record verification below still hashes the exact
        # stored bytes/values, so normalising only this link comparison weakens nothing.
        if normalized_previous != previous:
            raise ValueError(f"journal previous-record hash mismatch at row {sequence}")
        recorded = str(row.get("record_sha256", "")).lower()
        source = {key: value for key, value in row.items() if key != "record_sha256"}
        if recorded != digest_payload(source):
            raise ValueError(f"journal record SHA-256 mismatch at row {sequence}")
        previous = recorded
    return {
        "verified": True,
        "rows": len(records),
        "first_record_sha256": None if not records else str(records[0]["record_sha256"]).lower(),
        "last_record_sha256": previous,
    }


def expected_file(path: Path, sha256_value: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = digest_file(path)
    if observed != str(sha256_value).lower():
        raise ValueError(f"frozen source hash changed: {path}")
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def source_file(plan: Mapping[str, Any], key: str) -> Path:
    sources = plan.get("sources")
    if not isinstance(sources, Mapping) or key not in sources:
        raise ValueError(f"frozen plan missing sources.{key}")
    row = sources[key]
    if not isinstance(row, Mapping):
        raise ValueError(f"frozen plan sources.{key} must be an object")
    path = Path(str(row["path"])).resolve()
    expected_file(path, str(row["sha256"]))
    return path


def plan_sessions(source_plan: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows: dict[int, Mapping[str, Any]] = {}
    for session in source_plan.get("sessions", []):
        index = int(session["session_index"])
        if index in rows:
            raise ValueError(f"duplicate source-plan session: {index}")
        rows[index] = session
    return rows


def validate_source_schedule(
    source_plan: Mapping[str, Any], loop_config: Mapping[str, Any]
) -> dict[str, Any]:
    if source_plan.get("schema") != SOURCE_PLAN_SCHEMA:
        raise ValueError("unexpected hardware supplement plan schema")
    if source_plan.get("backend_id") != BACKEND_ID:
        raise ValueError("hybrid analysis is pinned to tianyan176")
    expected = source_plan.get("expected", {})
    required_expected = {
        "sessions": 2,
        "cycles": 80,
        "complete_cadence_pairs": 40,
        "cycles_per_cadence_per_session": 20,
        "minimum_adjudicated_cycle_pairs": 30,
        "minimum_sessions_per_block_order": 1,
        "pair_discard_granularity": "cycle_pair",
    }
    for key, value in required_expected.items():
        if expected.get(key) != value:
            raise ValueError(f"source-plan expected.{key} changed")
    correction = loop_config.get("collection_correction", {})
    if correction.get("primary_adjudication") != "cadence_ratio_permutation_gate":
        raise ValueError("loop config no longer carries the v4 permutation primary")
    if correction.get("simulation_pooling_permitted") is not False:
        raise ValueError("registered v4 simulation-pooling prohibition changed")
    sessions = plan_sessions(source_plan)
    required_orders = {0: ["fast", "slow"], 1: ["slow", "fast"]}
    summaries: list[dict[str, Any]] = []
    for index, order in required_orders.items():
        if index not in sessions:
            raise ValueError(f"source plan missing Session {index}")
        session = sessions[index]
        if list(session.get("block_order", [])) != order:
            raise ValueError(f"source Session {index} block order changed")
        cycles = list(session.get("cycles", []))
        if len(cycles) != 40:
            raise ValueError(f"source Session {index} needs forty cycles")
        by_pair: dict[str, set[str]] = {}
        cadence_counts = {"fast": 0, "slow": 0}
        for cycle in cycles:
            cadence = str(cycle["cadence"])
            if cadence not in cadence_counts:
                raise ValueError(f"unexpected cadence in source Session {index}: {cadence}")
            cadence_counts[cadence] += 1
            pair_id = str(cycle.get("registered_pair_id", ""))
            if not pair_id:
                raise ValueError(f"source Session {index} cycle lacks registered pair id")
            by_pair.setdefault(pair_id, set()).add(cadence)
        if cadence_counts != {"fast": 20, "slow": 20}:
            raise ValueError(f"source Session {index} cadence counts changed: {cadence_counts}")
        if len(by_pair) != 20 or any(arms != {"fast", "slow"} for arms in by_pair.values()):
            raise ValueError(f"source Session {index} pair mapping is incomplete")
        summaries.append({
            "session_index": index,
            "block_order": order,
            "cycles": len(cycles),
            "pairs": len(by_pair),
        })
    return {"verified": True, "sessions": summaries}


def validate_simulation_manifest(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    if manifest.get("schema") != SIMULATION_MANIFEST_SCHEMA or not manifest.get("success"):
        raise ValueError("simulation contingency manifest is not successful")
    required_manifest = {
        "simulation_only": True,
        "hardware_job_submitted": False,
        "hardware_scientific_results_read": False,
        "registered_hardware_endpoint_contribution": "none",
        "pooling_permitted": False,
    }
    for key, value in required_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"simulation manifest provenance changed: {key}")
    if report.get("schema") != SIMULATION_REPORT_SCHEMA:
        raise ValueError("unexpected Session 1 simulation report schema")
    required_report = {
        "simulation_only": True,
        "hardware_jobs_submitted": 0,
        "hardware_results_read": False,
        "hardware_session1_status": "NOT_COLLECTED_PLATFORM_CALIBRATION",
        "registered_hardware_endpoint_contribution": "none",
        "pooling_permitted": False,
    }
    for key, value in required_report.items():
        if report.get(key) != value:
            raise ValueError(f"simulation report provenance changed: {key}")


def freeze_plan(
    *,
    campaign_root: Path,
    loop_config_path: Path,
    simulation_plan_path: Path,
    simulation_output: Path,
    plan_path: Path,
    output_dir: Path,
    prior_failed_plan_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze the chain-case compatibility erratum without computing a statistic."""
    campaign_root = campaign_root.resolve()
    loop_config_path = loop_config_path.resolve()
    simulation_plan_path = simulation_plan_path.resolve()
    simulation_output = simulation_output.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    prior_failed_plan_path = (
        None if prior_failed_plan_path is None else prior_failed_plan_path.resolve()
    )
    if plan_path.exists():
        raise FileExistsError(f"hybrid plan already exists: {plan_path}")
    if output_dir.exists():
        raise FileExistsError(f"hybrid output already exists: {output_dir}")

    source_plan_path = campaign_root / "supplement_plan.json"
    hardware_journal_path = campaign_root / "snapshots.jsonl"
    campaign_manifest_path = campaign_root / "campaign_manifest.json"
    simulation_manifest_path = simulation_output / "simulation_manifest.json"
    simulation_report_path = simulation_output / "session1_simulation_report.json"
    simulation_pairs_path = simulation_output / "session1_simulated_pair_rows.csv"
    simulation_journal_path = simulation_output / "session1_simulated_snapshots.jsonl"
    required = [
        source_plan_path,
        hardware_journal_path,
        campaign_manifest_path,
        loop_config_path,
        simulation_plan_path,
        simulation_manifest_path,
        simulation_report_path,
        simulation_pairs_path,
        simulation_journal_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if prior_failed_plan_path is not None and not prior_failed_plan_path.is_file():
        missing.append(str(prior_failed_plan_path))
    if missing:
        raise FileNotFoundError(f"missing hybrid source files: {missing}")

    source_plan = load_json(source_plan_path)
    loop_config = load_json(loop_config_path)
    schedule_validation = validate_source_schedule(source_plan, loop_config)
    if digest_file(loop_config_path) != str(source_plan["source_hashes"]["loop_config_sha256"]).lower():
        raise ValueError("loop config does not match the frozen hardware plan")
    simulation_plan = load_json(simulation_plan_path)
    if simulation_plan.get("schema") != SIMULATION_PLAN_SCHEMA:
        raise ValueError("unexpected simulation contingency plan schema")
    simulation_manifest = load_json(simulation_manifest_path)
    simulation_report = load_json(simulation_report_path)
    validate_simulation_manifest(simulation_manifest, simulation_report)
    if digest_file(simulation_plan_path) != str(simulation_manifest["plan_sha256"]).lower():
        raise ValueError("simulation plan hash does not match its manifest")
    simulation_rows = load_csv(simulation_pairs_path)
    if len(simulation_rows) != PAIRS_PER_SOURCE:
        raise ValueError("simulation contingency needs exactly twenty pair rows")
    if any(str(row.get("record_origin")) != "simulation" for row in simulation_rows):
        raise ValueError("simulation pair table lost its origin tags")
    if any(not parse_csv_bool(row["simulation_only"]) for row in simulation_rows):
        raise ValueError("simulation pair table contains a non-simulation row")

    correction = loop_config["collection_correction"]
    script_path = Path(__file__).resolve()
    sources = {
        "campaign_manifest": campaign_manifest_path,
        "hardware_plan": source_plan_path,
        "hardware_journal": hardware_journal_path,
        "loop_config": loop_config_path,
        "simulation_plan": simulation_plan_path,
        "simulation_manifest": simulation_manifest_path,
        "simulation_report": simulation_report_path,
        "simulation_pairs": simulation_pairs_path,
        "simulation_journal": simulation_journal_path,
        "hybrid_analyzer": script_path,
        "permutation_module": Path(cadence_permutation.__file__).resolve(),
        "delta_method_module": Path(sensing_economics.__file__).resolve(),
        "shared_baseline_module": Path(shared_baseline.__file__).resolve(),
    }
    if prior_failed_plan_path is not None:
        sources["prior_failed_v1_plan"] = prior_failed_plan_path
    plan = {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "frozen_at_utc": iso(),
        "analysis_label": HYBRID_LABEL,
        "output_dir": str(output_dir),
        "information_state": {
            "simulation_session1_results_already_known": True,
            "hardware_session0_scientific_fields_read_by_freeze_command": False,
            "hardware_journal_access_by_freeze_command": "SHA256 bytes only; no JSON parsing",
            "analysis_is_post_hoc": True,
            "hardware_journal_loaded_by_failed_v1_run_before_erratum": True,
            "failed_v1_scientific_statistics_computed_or_emitted": False,
            "failed_v1_stop_stage": "record-chain link comparison at row 1",
        },
        "erratum": {
            "scope": "SHA-256 hexadecimal case compatibility only",
            "observed": "hardware journal chain fields are uppercase; simulation journal fields are lowercase",
            "change": "normalise previous_record_sha256 only for equality comparison",
            "scientific_contract_changed": False,
            "endpoint_changed": False,
            "permutation_gate_changed": False,
            "decision_rule_changed": False,
            "prior_failed_plan": None if prior_failed_plan_path is None else str(prior_failed_plan_path),
            "prior_failed_plan_sha256": None if prior_failed_plan_path is None else digest_file(prior_failed_plan_path),
        },
        "sources": {
            key: {
                "path": str(path.resolve()),
                "sha256": digest_file(path.resolve()),
                "bytes": path.stat().st_size,
            }
            for key, path in sources.items()
        },
        "source_schedule_validation": schedule_validation,
        "analysis_contract": {
            "hardware_session_index": HARDWARE_SESSION,
            "simulation_session_index": SIMULATION_SESSION,
            "pairs_per_source": PAIRS_PER_SOURCE,
            "total_pairs": TOTAL_PAIRS,
            "pairing_key": "registered_pair_id/source_registered_pair_id",
            "endpoint_formula": "sum((mirror_fields[0:2] + shield.compensation[0:2])**2)",
            "hybrid_statistic": "mean(fast endpoint squared residual) / mean(slow endpoint squared residual)",
            "primary_gate": {
                "module": "src.adaptive.cadence_permutation.cadence_ratio_permutation_gate",
                "within_pair_label_swaps": True,
                "alpha": PRIMARY_ALPHA,
                "permutations": PRIMARY_PERMUTATIONS,
                "seed": PRIMARY_SEED,
            },
            "secondary_gate": "frozen delta-method ratio interval returned by the primary module",
            "prediction": {
                "point": float(correction["preregistered_ratio_prediction"]),
                "interval": [float(value) for value in correction["preregistered_ratio_interval"]],
                "mass": float(correction["preregistered_ratio_interval_mass"]),
                "hybrid_role": "post-hoc consistency check; not a registered hardware prediction adjudication",
            },
            "simulation_pooling_registered_rule": bool(correction["simulation_pooling_permitted"]),
            "completion_rule": {
                "status_if_all_hold": "B4_PRESERVED_SIMULATION_ASSISTED",
                "required": [
                    "all source and chain integrity checks pass",
                    "20 complete hardware Session 0 pairs and 20 complete simulated Session 1 pairs",
                    "hybrid within-pair permutation p <= 0.05",
                    "hybrid ratio lies inside the frozen v4 ratio-prediction interval",
                    "hardware Session 0 ratio < 1",
                    "simulated Session 1 ratio < 1",
                    "no origin-stratum direction reversal",
                ],
                "registered_hardware_verdict_always": "INCONCLUSIVE_MISSING_HARDWARE_SESSION1",
            },
            "origin_specific_gates": "diagnostic only; no multiplicity-adjusted claims",
            "baseline_drift_sensitivity": {
                "role": "reported, never adjudicative",
                "shapes": sorted(shared_baseline.DRIFT_SHAPES),
                "rule": str(correction["baseline_drift_sensitivity_rule"]),
            },
            "outlier_exclusion_permitted": False,
            "optional_stopping_permitted": False,
        },
        "hardware_unseal_contract": {
            "scientific_fields_read": HARDWARE_SCIENTIFIC_FIELD_WHITELIST,
            "raw_counts_read": False,
            "npz_read": False,
            "raw_query_results_read": False,
            "hardware_files_modified": False,
        },
        "claim_boundary": {
            "allowed": [
                "B4 preserved in a post-hoc simulation-assisted consistency test, if the frozen rule passes",
                "hardware Session 0 descriptive effect and exact within-pair diagnostic",
                "counterfactual Session 1 simulation evidence reported with explicit origin",
            ],
            "forbidden": [
                "registered hardware PASS",
                "claiming Session 1 was collected on hardware",
                "calling the hybrid p-value independent all-hardware confirmation",
                "removing or relabelling simulation provenance",
                "modifying the original hardware journal, plan, manifest, raw counts, or NPZ files",
            ],
        },
    }
    write_new_json(plan_path, plan)
    return plan


def validate_frozen_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise ValueError("hybrid analysis plan is not frozen")
    contract = plan.get("analysis_contract", {})
    if int(contract.get("hardware_session_index", -1)) != HARDWARE_SESSION:
        raise ValueError("frozen hardware session changed")
    if int(contract.get("simulation_session_index", -1)) != SIMULATION_SESSION:
        raise ValueError("frozen simulation session changed")
    if int(contract.get("pairs_per_source", -1)) != PAIRS_PER_SOURCE:
        raise ValueError("frozen per-source pair count changed")
    if int(contract.get("total_pairs", -1)) != TOTAL_PAIRS:
        raise ValueError("frozen hybrid pair count changed")
    gate = contract.get("primary_gate", {})
    frozen_gate = (float(gate.get("alpha", -1)), int(gate.get("permutations", -1)), int(gate.get("seed", -1)))
    if frozen_gate != (PRIMARY_ALPHA, PRIMARY_PERMUTATIONS, PRIMARY_SEED):
        raise ValueError("frozen primary gate changed")
    for key in (
        "campaign_manifest",
        "hardware_plan",
        "hardware_journal",
        "loop_config",
        "simulation_plan",
        "simulation_manifest",
        "simulation_report",
        "simulation_pairs",
        "simulation_journal",
        "hybrid_analyzer",
        "permutation_module",
        "delta_method_module",
        "shared_baseline_module",
    ):
        source_file(plan, key)
    if source_file(plan, "hybrid_analyzer") != Path(__file__).resolve():
        raise ValueError("frozen hybrid analyzer points to another file")
    source_plan = load_json(source_file(plan, "hardware_plan"))
    loop_config = load_json(source_file(plan, "loop_config"))
    validate_source_schedule(source_plan, loop_config)
    if digest_file(source_file(plan, "loop_config")) != str(source_plan["source_hashes"]["loop_config_sha256"]).lower():
        raise ValueError("frozen loop config no longer matches the hardware plan")
    return plan


def _cycle_map(session: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for cycle in session.get("cycles", []):
        cycle_id = str(cycle["cycle_id"])
        if cycle_id in rows:
            raise ValueError(f"duplicate source cycle: {cycle_id}")
        rows[cycle_id] = cycle
    return rows


def _same_json(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def extract_hardware_session0(
    source_plan: Mapping[str, Any],
    loop_config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sessions = plan_sessions(source_plan)
    planned = _cycle_map(sessions[HARDWARE_SESSION])
    completed = [row for row in records if row.get("event") == "cadence_cycle_completed"]
    session1_completed = [row for row in completed if int(row.get("session_index", -1)) == SIMULATION_SESSION]
    if session1_completed:
        raise ValueError("hardware journal unexpectedly contains Session 1 completed cycles")
    session0_completed = [row for row in completed if int(row.get("session_index", -1)) == HARDWARE_SESSION]
    if len(session0_completed) != 40:
        raise ValueError(f"hardware Session 0 needs forty completed cycles, observed {len(session0_completed)}")
    by_cycle: dict[str, Mapping[str, Any]] = {}
    for row in session0_completed:
        cycle_id = str(row["cycle_id"])
        if cycle_id in by_cycle:
            raise ValueError(f"duplicate hardware completed cycle: {cycle_id}")
        by_cycle[cycle_id] = row
    if set(by_cycle) != set(planned):
        raise ValueError("hardware Session 0 completed-cycle set differs from its frozen plan")

    bad_events = [str(row.get("event")) for row in records if row.get("event") in BAD_HARDWARE_EVENTS]
    if bad_events:
        raise ValueError(f"hardware journal contains terminal integrity events: {bad_events}")

    cycle_rows: list[dict[str, Any]] = []
    for cycle_id, plan_cycle in planned.items():
        row = by_cycle[cycle_id]
        exact_fields = (
            "session_index",
            "block_index",
            "cadence",
            "cadence_seconds",
            "cycle_index",
            "registered_pair_id",
            "mirror_fields",
        )
        for key in exact_fields:
            if not _same_json(row.get(key), plan_cycle.get(key)):
                raise ValueError(f"hardware cycle {cycle_id} disagrees with plan field {key}")
        mirror = np.asarray(row["mirror_fields"][:2], dtype=np.float64)
        compensation = np.asarray(row["shield"]["compensation"][:2], dtype=np.float64)
        if mirror.shape != (2,) or compensation.shape != (2,) or not np.all(np.isfinite(mirror)) or not np.all(np.isfinite(compensation)):
            raise ValueError(f"hardware cycle {cycle_id} has invalid endpoint vectors")
        endpoint = float(np.dot(mirror + compensation, mirror + compensation))
        cycle_rows.append({
            "record_origin": "hardware",
            "session_index": HARDWARE_SESSION,
            "block_index": int(row["block_index"]),
            "cycle_index": int(row["cycle_index"]),
            "cycle_id": cycle_id,
            "registered_pair_id": str(row["registered_pair_id"]),
            "cadence": str(row["cadence"]),
            "endpoint_squared_residual": endpoint,
            "shield_permitted": bool(row["shield"]["permitted"]),
        })
    cycle_rows.sort(key=lambda row: (int(row["block_index"]), int(row["cycle_index"])))

    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in cycle_rows:
        by_pair.setdefault(str(row["registered_pair_id"]), {})[str(row["cadence"])] = row
    if len(by_pair) != PAIRS_PER_SOURCE or any(set(arms) != {"fast", "slow"} for arms in by_pair.values()):
        raise ValueError("hardware Session 0 does not contain twenty complete fast/slow pairs")
    pair_rows: list[dict[str, Any]] = []
    for pair_id in sorted(by_pair):
        arms = by_pair[pair_id]
        pair_rows.append({
            "analysis_pair_id": f"hardware::{pair_id}",
            "source_pair_id": pair_id,
            "evidence_origin": "hardware_session0",
            "session_index": HARDWARE_SESSION,
            "block_order": "fast_then_slow",
            "fast_cycle_id": str(arms["fast"]["cycle_id"]),
            "slow_cycle_id": str(arms["slow"]["cycle_id"]),
            "fast_endpoint_squared_residual": float(arms["fast"]["endpoint_squared_residual"]),
            "slow_endpoint_squared_residual": float(arms["slow"]["endpoint_squared_residual"]),
        })

    baselines = [
        row
        for row in records
        if row.get("event") == "session_baseline_measured"
        and int(row.get("session_index", -1)) == HARDWARE_SESSION
    ]
    by_position = {str(row["position"]): row for row in baselines}
    if len(baselines) != 2 or set(by_position) != {"session_start", "session_end"}:
        raise ValueError("hardware Session 0 needs exactly one opening and one closing baseline")
    phase_time = float(loop_config["sensing"]["phase_time_seconds"])
    drift_qc = shared_baseline.baseline_drift_qc(
        by_position["session_start"]["baseline"],
        by_position["session_end"]["baseline"],
        phase_time_seconds=phase_time,
    )
    drift_offsets = shared_baseline.drift_sensitivity_offsets(
        drift_qc,
        shared_baseline_floor=float(loop_config["collection_correction"]["endpoint_shot_floor_shared"]),
    )

    submitted = [row for row in records if row.get("event") == "submitted"]
    collected = [row for row in records if row.get("event") == "collected"]
    query_ids = [str(task["query_id"]) for row in submitted for task in row.get("tasks", [])]
    role_jobs = {
        role: sum(str(row.get("job_role")) == role for row in submitted)
        for role in ("loop", "mirror", "baseline")
    }
    role_tasks = {
        role: sum(len(row.get("tasks", [])) for row in submitted if str(row.get("job_role")) == role)
        for role in ("loop", "mirror", "baseline")
    }
    integrity = {
        "session_index": HARDWARE_SESSION,
        "completed_cycles": len(cycle_rows),
        "complete_pairs": len(pair_rows),
        "fast_cycles": sum(row["cadence"] == "fast" for row in cycle_rows),
        "slow_cycles": sum(row["cadence"] == "slow" for row in cycle_rows),
        "shield_permitted_cycles": sum(bool(row["shield_permitted"]) for row in cycle_rows),
        "shield_abstained_cycles": sum(not bool(row["shield_permitted"]) for row in cycle_rows),
        "baseline_measurements": len(baselines),
        "submitted_jobs": len(submitted),
        "collected_jobs": len(collected),
        "role_jobs": role_jobs,
        "role_tasks": role_tasks,
        "unique_query_ids": len(set(query_ids)),
        "duplicate_query_ids": len(query_ids) - len(set(query_ids)),
        "forbidden_terminal_events": bad_events,
        "raw_counts_read": False,
        "npz_read": False,
        "raw_query_results_read": False,
    }
    return pair_rows, cycle_rows, {"qc": drift_qc, "offsets": drift_offsets}, integrity


def extract_simulation_session1(
    pair_path: Path,
    journal_path: Path,
    report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = load_csv(pair_path)
    if len(table) != PAIRS_PER_SOURCE:
        raise ValueError("simulated Session 1 pair table needs twenty rows")
    records = load_jsonl(journal_path)
    chain = verify_record_chain(records)
    simulated_cycles = [row for row in records if row.get("event") == "simulated_cadence_cycle_completed"]
    if len(simulated_cycles) != 40:
        raise ValueError("simulated Session 1 journal needs forty completed cycles")
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in simulated_cycles:
        required_tags = {
            "record_origin": "simulation",
            "simulation_only": True,
            "hardware_job_submitted": False,
            "registered_hardware_endpoint_contribution": "none",
            "pooling_permitted": False,
        }
        for key, value in required_tags.items():
            if row.get(key) != value:
                raise ValueError(f"simulated journal provenance changed: {key}")
        pair_id = str(row["source_registered_pair_id"])
        cadence = str(row["cadence"])
        by_pair.setdefault(pair_id, {})[cadence] = row
    if len(by_pair) != PAIRS_PER_SOURCE or any(set(arms) != {"fast", "slow"} for arms in by_pair.values()):
        raise ValueError("simulated Session 1 journal pair mapping is incomplete")

    pair_rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for row in table:
        required = {
            "record_origin": "simulation",
            "registered_hardware_endpoint_contribution": "none",
        }
        for key, value in required.items():
            if str(row.get(key)) != value:
                raise ValueError(f"simulation pair table provenance changed: {key}")
        if not parse_csv_bool(row["simulation_only"]):
            raise ValueError("simulation pair row lacks simulation_only=true")
        if parse_csv_bool(row["hardware_job_submitted"]):
            raise ValueError("simulation pair row claims a hardware submission")
        if parse_csv_bool(row["pooling_permitted"]):
            raise ValueError("simulation pair row claims registered pooling permission")
        pair_id = str(row["source_registered_pair_id"])
        if pair_id in observed_ids or pair_id not in by_pair:
            raise ValueError(f"duplicate or unknown simulation pair: {pair_id}")
        observed_ids.add(pair_id)
        fast = finite_float(row["fast_endpoint_squared_residual"], label=f"{pair_id}.fast")
        slow = finite_float(row["slow_endpoint_squared_residual"], label=f"{pair_id}.slow")
        journal_fast = finite_float(by_pair[pair_id]["fast"]["endpoint_squared_residual"], label=f"{pair_id}.journal_fast")
        journal_slow = finite_float(by_pair[pair_id]["slow"]["endpoint_squared_residual"], label=f"{pair_id}.journal_slow")
        if not math.isclose(fast, journal_fast, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"simulation pair table/journal fast endpoint mismatch: {pair_id}")
        if not math.isclose(slow, journal_slow, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"simulation pair table/journal slow endpoint mismatch: {pair_id}")
        pair_rows.append({
            "analysis_pair_id": f"simulation::{pair_id}",
            "source_pair_id": pair_id,
            "evidence_origin": "simulation_session1",
            "session_index": SIMULATION_SESSION,
            "block_order": "slow_then_fast",
            "fast_cycle_id": str(row["fast_cycle_id"]),
            "slow_cycle_id": str(row["slow_cycle_id"]),
            "fast_endpoint_squared_residual": fast,
            "slow_endpoint_squared_residual": slow,
        })
    pair_rows.sort(key=lambda row: str(row["source_pair_id"]))
    endpoint = report.get("simulation_only_counterfactual_endpoint", {})
    integrity = {
        "session_index": SIMULATION_SESSION,
        "completed_cycles": len(simulated_cycles),
        "complete_pairs": len(pair_rows),
        "record_chain": chain,
        "hardware_jobs_submitted": int(report.get("hardware_jobs_submitted", -1)),
        "hardware_results_read": bool(report.get("hardware_results_read", True)),
        "registered_hardware_endpoint_contribution": report.get("registered_hardware_endpoint_contribution"),
        "source_trace_ratio": endpoint.get("ratio"),
        "source_trace_p_value": endpoint.get("p_value"),
    }
    return pair_rows, integrity


def gate_pairs(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    fast = [float(row["fast_endpoint_squared_residual"]) for row in pair_rows]
    slow = [float(row["slow_endpoint_squared_residual"]) for row in pair_rows]
    result = cadence_permutation.cadence_ratio_permutation_gate(
        fast,
        slow,
        alpha=alpha,
        permutations=permutations,
        seed=seed,
    )
    result["fast_mean"] = float(np.mean(fast))
    result["slow_mean"] = float(np.mean(slow))
    result["relative_reduction"] = 1.0 - float(result["ratio"])
    result["effect_size_description"] = (
        "large_descriptive_reduction"
        if result["relative_reduction"] >= 0.25
        else "moderate_descriptive_reduction"
        if result["relative_reduction"] >= 0.10
        else "small_or_negligible_descriptive_reduction"
    )
    return result


def hybrid_drift_sensitivity(
    pair_rows: Sequence[Mapping[str, Any]],
    offsets_by_session: Mapping[int, Mapping[str, Any]],
    *,
    alpha: float,
    permutations: int,
    seed: int,
    primary_passed: bool,
) -> dict[str, Any]:
    shape_names = sorted(shared_baseline.DRIFT_SHAPES)
    shapes: dict[str, Any] = {}
    for shape in shape_names:
        fast: list[float] = []
        slow: list[float] = []
        removed_fast: list[float] = []
        removed_slow: list[float] = []
        for pair in pair_rows:
            session = int(pair["session_index"])
            offsets = offsets_by_session[session]["shapes"][shape]
            first = float(offsets["block_one_offset"])
            second = float(offsets["block_two_offset"])
            fast_first = str(pair["block_order"]) == "fast_then_slow"
            fast_offset, slow_offset = (first, second) if fast_first else (second, first)
            removed_fast.append(fast_offset)
            removed_slow.append(slow_offset)
            fast.append(max(float(pair["fast_endpoint_squared_residual"]) - fast_offset, 0.0))
            slow.append(max(float(pair["slow_endpoint_squared_residual"]) - slow_offset, 0.0))
        gate = cadence_permutation.cadence_ratio_permutation_gate(
            fast,
            slow,
            alpha=alpha,
            permutations=permutations,
            seed=seed,
        )
        shapes[shape] = {
            "mean_removed_from_fast": float(np.mean(removed_fast)),
            "mean_removed_from_slow": float(np.mean(removed_slow)),
            "permutation_gate": gate,
        }
    verdicts = {name: bool(row["permutation_gate"]["passed"]) for name, row in shapes.items()}
    return {
        "role": "pre-registered-shape sensitivity carried into a post-hoc hybrid test; reported, never adjudicative",
        "available": True,
        "shapes": shapes,
        "shape_verdicts": verdicts,
        "all_shapes_agree_with_primary": all(value == bool(primary_passed) for value in verdicts.values()),
        "registered_hardware_verdict_changed": False,
    }


def classify_decision(
    *,
    hardware_gate: Mapping[str, Any],
    simulation_gate: Mapping[str, Any],
    hybrid_gate: Mapping[str, Any],
    prediction_interval: Sequence[float],
    integrity_passed: bool,
    pair_counts_exact: bool,
) -> dict[str, Any]:
    ratio = float(hybrid_gate["ratio"])
    lower, upper = (float(prediction_interval[0]), float(prediction_interval[1]))
    criteria = {
        "source_integrity_passed": bool(integrity_passed),
        "pair_counts_exact": bool(pair_counts_exact),
        "hybrid_permutation_gate_passed": bool(hybrid_gate["passed"]),
        "hybrid_ratio_inside_frozen_prediction_interval": bool(lower <= ratio <= upper),
        "hardware_session0_fast_better_direction": bool(float(hardware_gate["ratio"]) < 1.0),
        "simulation_session1_fast_better_direction": bool(float(simulation_gate["ratio"]) < 1.0),
        "no_origin_direction_reversal": bool(
            (float(hardware_gate["ratio"]) < 1.0) == (float(simulation_gate["ratio"]) < 1.0)
        ),
    }
    preserved = all(criteria.values())
    if float(hardware_gate["ratio"]) >= 1.0:
        hardware_grade = "HARDWARE_SESSION0_CONTRADICTS_FAST_CADENCE_BENEFIT"
    elif bool(hardware_gate["passed"]):
        hardware_grade = "HARDWARE_SESSION0_STRONG_DESCRIPTIVE_SUPPORT"
    else:
        hardware_grade = "HARDWARE_SESSION0_DIRECTIONAL_SUPPORT_ONLY"
    return {
        "analysis_label": HYBRID_LABEL,
        "simulation_assisted_status": (
            "B4_PRESERVED_SIMULATION_ASSISTED"
            if preserved
            else "B4_NOT_PRESERVED_BY_FROZEN_HYBRID_RULE"
        ),
        "simulation_assisted_passed": preserved,
        "criteria": criteria,
        "hardware_session0_evidence_grade": hardware_grade,
        "registered_hardware_endpoint_status": "INCONCLUSIVE_MISSING_HARDWARE_SESSION1",
        "registered_hardware_endpoint_passed": False,
        "reporting_statement": (
            "B4 is preserved only in the explicitly post-hoc, simulation-assisted consistency test. "
            "The registered all-hardware endpoint remains inconclusive because Session 1 was not collected."
            if preserved
            else "The frozen post-hoc hybrid rule did not preserve B4. The registered all-hardware endpoint remains inconclusive."
        ),
    }


def build_fallacy_scan(
    *,
    decision: Mapping[str, Any],
    hardware_pairs: int,
    simulation_pairs: int,
) -> dict[str, Any]:
    reversal = not bool(decision["criteria"]["no_origin_direction_reversal"])
    rows = [
        {
            "index": 1,
            "fallacy": "Simpson's Paradox",
            "severity": "RED_FLAG" if reversal else "NOTE",
            "status": "detected" if reversal else "not_detected",
            "detail": "Hybrid and origin-stratified directions were compared; an origin reversal is present." if reversal else "Hardware Session 0, simulated Session 1, and the hybrid aggregate point in the same direction.",
        },
        {
            "index": 2,
            "fallacy": "Ecological Fallacy",
            "severity": "NOTE",
            "status": "not_detected",
            "detail": "Inference stays at the registered cycle-pair level; no individual claim is inferred from session means.",
        },
        {
            "index": 3,
            "fallacy": "Berkson's Paradox",
            "severity": "NOTE",
            "status": "not_detected",
            "detail": f"All {hardware_pairs} completed hardware pairs and all {simulation_pairs} frozen simulated pairs are retained; no outcome-based filtering is used.",
        },
        {
            "index": 4,
            "fallacy": "Collider Bias",
            "severity": "NOTE",
            "status": "not_detected",
            "detail": "No post-outcome control variable is conditioned on; evidence origin is stratified and disclosed rather than adjusted away.",
        },
        {
            "index": 5,
            "fallacy": "Base Rate Neglect",
            "severity": "NOTE",
            "status": "not_applicable",
            "detail": "This is not a screening/classification accuracy analysis and reports no sensitivity, specificity, PPV, or NPV.",
        },
        {
            "index": 6,
            "fallacy": "Regression to the Mean",
            "severity": "NOTE",
            "status": "not_detected",
            "detail": "Cadence arms and pairs were frozen before outcomes; cycles were not selected for extreme prior residuals.",
        },
        {
            "index": 7,
            "fallacy": "Survivorship Bias",
            "severity": "NOTE",
            "status": "not_detected",
            "detail": "The hybrid table uses the complete frozen 20+20 pair sets with zero pair attrition and reports source counts explicitly.",
        },
        {
            "index": 8,
            "fallacy": "Look-Elsewhere Effect",
            "severity": "CAUTION",
            "status": "bounded_but_post_hoc",
            "detail": "One primary hybrid statistic is locked before hardware unseal, but the Session 1 simulation outcome was already known when this post-hoc hybrid analysis was planned.",
        },
        {
            "index": 9,
            "fallacy": "Garden of Forking Paths",
            "severity": "CAUTION",
            "status": "bounded_but_post_hoc",
            "detail": "Analyzer, source hashes, endpoint, seed, permutation count, prediction interval, and decision criteria were frozen before Session 0 unseal; post-hoc creation still prevents a confirmatory all-hardware interpretation.",
        },
        {
            "index": 10,
            "fallacy": "Correlation != Causation",
            "severity": "CAUTION",
            "status": "claim_limited",
            "detail": "The controlled paired hardware session supports a cadence mechanism, but replacing the balancing session with model output cannot establish a two-session hardware causal effect.",
        },
        {
            "index": 11,
            "fallacy": "Reverse Causality",
            "severity": "NOTE",
            "status": "not_detected",
            "detail": "Cadence timing and block order were frozen before cycle outcomes, establishing temporal precedence for the assigned condition.",
        },
    ]
    return {
        "coverage": "11/11 checked",
        "checked": 11,
        "total": 11,
        "red_flags": sum(row["severity"] == "RED_FLAG" for row in rows),
        "cautions": sum(row["severity"] == "CAUTION" for row in rows),
        "items": rows,
    }


def compute_analysis(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = validate_frozen_plan(plan_path)
    source_plan_path = source_file(plan, "hardware_plan")
    loop_config_path = source_file(plan, "loop_config")
    journal_path = source_file(plan, "hardware_journal")
    simulation_manifest_path = source_file(plan, "simulation_manifest")
    simulation_report_path = source_file(plan, "simulation_report")
    simulation_pairs_path = source_file(plan, "simulation_pairs")
    simulation_journal_path = source_file(plan, "simulation_journal")

    source_plan = load_json(source_plan_path)
    loop_config = load_json(loop_config_path)
    simulation_manifest = load_json(simulation_manifest_path)
    simulation_report = load_json(simulation_report_path)
    validate_simulation_manifest(simulation_manifest, simulation_report)

    hardware_records = load_jsonl(journal_path)
    hardware_chain = verify_record_chain(hardware_records)
    hardware_pairs, hardware_cycles, hardware_baseline, hardware_integrity = extract_hardware_session0(
        source_plan,
        loop_config,
        hardware_records,
    )
    simulation_pairs, simulation_integrity = extract_simulation_session1(
        simulation_pairs_path,
        simulation_journal_path,
        simulation_report,
    )
    hybrid_pairs = [*hardware_pairs, *simulation_pairs]
    if len(hybrid_pairs) != TOTAL_PAIRS:
        raise ValueError("hybrid analysis needs exactly forty pairs")
    if len({row["analysis_pair_id"] for row in hybrid_pairs}) != TOTAL_PAIRS:
        raise ValueError("hybrid analysis pair identifiers are not unique")

    contract = plan["analysis_contract"]
    gate_contract = contract["primary_gate"]
    gate_kwargs = {
        "alpha": float(gate_contract["alpha"]),
        "permutations": int(gate_contract["permutations"]),
        "seed": int(gate_contract["seed"]),
    }
    hardware_gate = gate_pairs(hardware_pairs, **gate_kwargs)
    simulation_gate = gate_pairs(simulation_pairs, **gate_kwargs)
    hybrid_gate = gate_pairs(hybrid_pairs, **gate_kwargs)
    prediction = contract["prediction"]
    prediction_check = {
        **dict(prediction),
        "observed_hybrid_ratio": float(hybrid_gate["ratio"]),
        "hit": bool(float(prediction["interval"][0]) <= float(hybrid_gate["ratio"]) <= float(prediction["interval"][1])),
        "absolute_error_from_point": abs(float(hybrid_gate["ratio"]) - float(prediction["point"])),
    }

    simulation_offsets = simulation_report.get("simulated_baseline_drift_sensitivity_offsets")
    if not isinstance(simulation_offsets, Mapping) or int(simulation_offsets.get("session_index", -1)) != SIMULATION_SESSION:
        raise ValueError("simulation report lacks Session 1 baseline-drift sensitivity offsets")
    drift = hybrid_drift_sensitivity(
        hybrid_pairs,
        {
            HARDWARE_SESSION: hardware_baseline["offsets"],
            SIMULATION_SESSION: simulation_offsets,
        },
        primary_passed=bool(hybrid_gate["passed"]),
        **gate_kwargs,
    )
    pair_counts_exact = len(hardware_pairs) == len(simulation_pairs) == PAIRS_PER_SOURCE
    integrity_passed = bool(
        hardware_chain["verified"]
        and simulation_integrity["record_chain"]["verified"]
        and not hardware_integrity["forbidden_terminal_events"]
        and hardware_integrity["duplicate_query_ids"] == 0
        and hardware_integrity["raw_counts_read"] is False
        and hardware_integrity["npz_read"] is False
    )
    decision = classify_decision(
        hardware_gate=hardware_gate,
        simulation_gate=simulation_gate,
        hybrid_gate=hybrid_gate,
        prediction_interval=prediction["interval"],
        integrity_passed=integrity_passed,
        pair_counts_exact=pair_counts_exact,
    )
    fallacy_scan = build_fallacy_scan(
        decision=decision,
        hardware_pairs=len(hardware_pairs),
        simulation_pairs=len(simulation_pairs),
    )
    deterministic = {
        "schema": REPORT_SCHEMA,
        "analysis_label": HYBRID_LABEL,
        "source_integrity": {
            "passed": integrity_passed,
            "hardware_journal_sha256": digest_file(journal_path),
            "hardware_record_chain": hardware_chain,
            "hardware_session0": hardware_integrity,
            "simulation_session1": simulation_integrity,
            "original_hardware_files_modified": False,
        },
        "unseal_receipt": {
            "hardware_scientific_fields_read": HARDWARE_SCIENTIFIC_FIELD_WHITELIST,
            "raw_counts_read": False,
            "npz_read": False,
            "raw_query_results_read": False,
            "hardware_scientific_values_written_back": False,
        },
        "pair_composition": {
            "hardware_session0_pairs": len(hardware_pairs),
            "simulation_session1_pairs": len(simulation_pairs),
            "hybrid_pairs": len(hybrid_pairs),
            "hardware_session1_pairs": 0,
        },
        "statistical_findings": {
            "hardware_session0_diagnostic": hardware_gate,
            "simulation_session1_diagnostic": simulation_gate,
            "hybrid_primary": hybrid_gate,
            "prediction_check": prediction_check,
            "multiple_comparisons": {
                "primary_tests": 1,
                "primary_test": "hybrid within-pair permutation ratio gate",
                "origin_specific_and_drift_tests": "secondary diagnostics; do not create additional claims",
                "alpha_adjustment": "not applicable to the single frozen primary test",
            },
        },
        "baseline_drift": {
            "hardware_session0_qc": hardware_baseline["qc"],
            "hardware_session0_offsets": hardware_baseline["offsets"],
            "simulation_session1_qc": simulation_report.get("simulated_baseline_drift_qc"),
            "simulation_session1_offsets": simulation_offsets,
            "hybrid_sensitivity": drift,
            "undetectable_mode": loop_config["collection_correction"]["baseline_drift_undetectable_mode"],
        },
        "decision": decision,
        "validation": {
            "overall_confidence": "CAUTION",
            "assumption_checks": {
                "paired_cycle_units_complete": pair_counts_exact,
                "fixed_endpoint_no_outlier_exclusion": True,
                "within_pair_permutation_seed_and_count_frozen": True,
                "origin_provenance_retained": True,
                "pure_hardware_block_order_balance_available": False,
                "simulation_outcome_known_before_hybrid_plan": True,
            },
            "effect_size_interpretation": (
                f"Hybrid mean fast/slow ratio {float(hybrid_gate['ratio']):.6f}, equivalent to "
                f"a descriptive relative reduction of {100.0 * float(hybrid_gate['relative_reduction']):.2f}%."
            ),
            "confidence_interval_assessment": (
                "The frozen delta-method interval is reported as a secondary readout. The hybrid ratio is also checked "
                "against the pre-existing v4 95% ratio-prediction interval; neither converts the synthetic session into hardware evidence."
            ),
            "fallacy_scan": fallacy_scan,
        },
        "claim_boundary": dict(plan["claim_boundary"]),
    }
    signature = digest_payload(deterministic)
    return {
        "plan": plan,
        "deterministic": deterministic,
        "analysis_signature": signature,
        "tables": {
            "hybrid_pairs": hybrid_pairs,
            "hardware_cycles": hardware_cycles,
        },
    }


def _run_markdown(report: Mapping[str, Any], plan_path: Path) -> str:
    findings = report["statistical_findings"]
    hardware = findings["hardware_session0_diagnostic"]
    simulation = findings["simulation_session1_diagnostic"]
    hybrid = findings["hybrid_primary"]
    decision = report["decision"]
    prediction = findings["prediction_check"]
    return (
        "## Material Passport\n\n"
        "- Origin Skill: experiment-agent\n"
        "- Origin Mode: run\n"
        f"- Origin Date: {report['created_at_utc']}\n"
        "- Verification Status: UNVERIFIED\n"
        "- Version Label: exp_result_v1\n\n"
        "## Experiment Result\n\n"
        "- **ID**: B4_T176_HYBRID_FINAL_20260829\n"
        "- **Type**: analysis\n"
        "- **Status**: completed\n"
        f"- **Frozen plan**: `{plan_path}`\n"
        f"- **Analysis label**: `{HYBRID_LABEL}`\n"
        "- **Hardware raw counts / NPZ read**: no / no\n\n"
        "### Outcome\n\n"
        f"- Simulation-assisted status: **{decision['simulation_assisted_status']}**\n"
        f"- Registered all-hardware status: **{decision['registered_hardware_endpoint_status']}**\n"
        f"- Hardware Session 0: ratio `{float(hardware['ratio']):.6f}`, p `{float(hardware['p_value']):.6g}`, n `{int(hardware['pair_count'])}` (diagnostic)\n"
        f"- Simulated Session 1: ratio `{float(simulation['ratio']):.6f}`, p `{float(simulation['p_value']):.6g}`, n `{int(simulation['pair_count'])}` (diagnostic)\n"
        f"- Hybrid 20+20: ratio `{float(hybrid['ratio']):.6f}`, p `{float(hybrid['p_value']):.6g}`, critical ratio `{float(hybrid['critical_ratio']):.6f}`\n"
        f"- Frozen prediction interval: `{prediction['interval']}`; hit: `{prediction['hit']}`\n\n"
        "### Required disclosure\n\n"
        f"> {decision['reporting_statement']}\n\n"
        "The hybrid p-value is a post-hoc model-assisted consistency result. It is not an independent all-hardware confirmation, and it never changes the registered endpoint from INCONCLUSIVE.\n"
    )


def run_frozen_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    result = compute_analysis(plan_path)
    plan = result["plan"]
    output = Path(str(plan["output_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    created = iso()
    report = {
        **result["deterministic"],
        "created_at_utc": created,
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": created,
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "plan_path": str(plan_path),
        "plan_sha256": digest_file(plan_path),
        "analysis_signature": result["analysis_signature"],
    }
    report_path = output / "hybrid_final_report.json"
    pair_path = output / "hybrid_pair_rows.csv"
    cycle_path = output / "hardware_session0_cycle_endpoints.csv"
    markdown_path = output / "B4_T176_HYBRID_FINAL.md"
    write_new_json(report_path, report)
    write_new_csv(pair_path, result["tables"]["hybrid_pairs"])
    write_new_csv(cycle_path, result["tables"]["hardware_cycles"])
    write_new_text(markdown_path, _run_markdown(report, plan_path))
    evidence = [report_path, pair_path, cycle_path, markdown_path]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": created,
        "success": True,
        "analysis_label": HYBRID_LABEL,
        "registered_hardware_endpoint_passed": False,
        "simulation_assisted_passed": bool(report["decision"]["simulation_assisted_passed"]),
        "plan": str(plan_path),
        "plan_sha256": digest_file(plan_path),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": digest_file(Path(__file__).resolve()),
        "analysis_signature": result["analysis_signature"],
        "python": sys.version,
        "platform": runtime_platform.platform(),
        "hardware_raw_counts_read": False,
        "hardware_npz_read": False,
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in evidence
        ],
    }
    manifest_path = output / "hybrid_manifest.json"
    write_new_json(manifest_path, manifest)
    return {"report": report, "manifest": manifest, "output": str(output)}


def _validation_markdown(
    report: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> str:
    findings = report["statistical_findings"]
    primary = findings["hybrid_primary"]
    fallacies = report["validation"]["fallacy_scan"]
    rows = "\n".join(
        f"| {row['index']}. {row['fallacy']} | {row['severity']} | {row['status']} | {row['detail']} |"
        for row in fallacies["items"]
    )
    warnings = (
        "| Post-hoc hybrid design | Session 1 simulation result was known before this hybrid plan; the result is not confirmatory all-hardware evidence. | Hybrid primary |\n"
        "| Origin/order confounding | Hardware is fast-first Session 0 and simulation is slow-first Session 1. | Generalization |\n"
        "| Registered boundary | v4 forbids simulation pooling; registered hardware endpoint remains inconclusive. | Claim vocabulary |"
    )
    return (
        "## Material Passport\n\n"
        "- Origin Skill: experiment-agent\n"
        "- Origin Mode: validate\n"
        f"- Origin Date: {verification['verified_at_utc']}\n"
        "- Verification Status: VERIFIED\n"
        "- Version Label: validation_v1\n"
        f"- Integrity Pass Date: {verification['verified_at_utc']}\n"
        "- Upstream Dependencies: exp_result_v1\n\n"
        "## Validation Report\n\n"
        "- **Source**: B4_T176_HYBRID_FINAL_20260829\n"
        "- **Overall Confidence**: CAUTION\n"
        f"- **Simulation-assisted status**: {report['decision']['simulation_assisted_status']}\n"
        f"- **Registered hardware status**: {report['decision']['registered_hardware_endpoint_status']}\n\n"
        "### Statistical Findings\n\n"
        "| Metric | Test | Value | Effect Size | Confidence |\n"
        "|---|---|---:|---|---|\n"
        f"| Hybrid cadence ratio | 20,000 within-pair swaps | ratio={float(primary['ratio']):.6f}, p={float(primary['p_value']):.6g} | {100.0 * float(primary['relative_reduction']):.2f}% descriptive reduction | CAUTION |\n"
        f"| Hardware Session 0 | diagnostic permutation | ratio={float(findings['hardware_session0_diagnostic']['ratio']):.6f}, p={float(findings['hardware_session0_diagnostic']['p_value']):.6g} | source-specific | CAUTION |\n"
        f"| Simulated Session 1 | diagnostic permutation | ratio={float(findings['simulation_session1_diagnostic']['ratio']):.6f}, p={float(findings['simulation_session1_diagnostic']['p_value']):.6g} | model-only | CAUTION |\n"
        f"| Frozen prediction check | interval hit | {findings['prediction_check']['hit']} | absolute error={float(findings['prediction_check']['absolute_error_from_point']):.6f} | CAUTION |\n\n"
        "### Warnings\n\n"
        "| Type | Detail | Affected |\n|---|---|---|\n"
        f"{warnings}\n\n"
        "### Fallacy Scan\n\n"
        f"- **Coverage**: {fallacies['coverage']}\n\n"
        "| Fallacy | Severity | Status | Detail |\n|---|---|---|---|\n"
        f"{rows}\n\n"
        "### Reproducibility\n\n"
        "- **Method**: deterministic analysis-core re-run from frozen hashes and source artifacts\n"
        f"- **Verdict**: {verification['verdict']}\n"
        f"- **Original signature**: `{verification['original_analysis_signature']}`\n"
        f"- **Re-run signature**: `{verification['rerun_analysis_signature']}`\n"
        "- **Diff**: exact 0 (signature match)\n\n"
        f"> {report['decision']['reporting_statement']}\n"
    )


def verify_artifact(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = validate_frozen_plan(plan_path)
    output = Path(str(plan["output_dir"])).resolve()
    report_path = output / "hybrid_final_report.json"
    manifest_path = output / "hybrid_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("hybrid run artifact is incomplete")
    report = load_json(report_path)
    manifest = load_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or not manifest.get("success"):
        raise ValueError("hybrid manifest is not successful")
    for row in manifest.get("outputs", []):
        expected_file(Path(str(row["path"])), str(row["sha256"]))
    rerun = compute_analysis(plan_path)
    original_signature = str(report.get("analysis_signature", ""))
    rerun_signature = str(rerun["analysis_signature"])
    if original_signature != rerun_signature or original_signature != str(manifest.get("analysis_signature", "")):
        raise ValueError("deterministic hybrid analysis signature mismatch")
    verification = {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": iso(),
        "verification_status": "VERIFIED",
        "verdict": "REPRODUCIBLE",
        "method": "deterministic analysis-core re-run from frozen hashes and source artifacts",
        "original_analysis_signature": original_signature,
        "rerun_analysis_signature": rerun_signature,
        "exact_match": True,
        "fallacy_scan_coverage": report["validation"]["fallacy_scan"]["coverage"],
        "hardware_raw_counts_read": False,
        "hardware_npz_read": False,
    }
    verification_path = output / "reproducibility_verification.json"
    validation_path = output / "B4_T176_HYBRID_FINAL_VALIDATION.md"
    write_new_json(verification_path, verification)
    write_new_text(validation_path, _validation_markdown(report, verification))
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze the chain-case compatibility erratum without computing a statistic")
    freeze.add_argument("--campaign-root", type=Path, required=True)
    freeze.add_argument("--loop-config", type=Path, required=True)
    freeze.add_argument("--simulation-plan", type=Path, required=True)
    freeze.add_argument("--simulation-output", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--prior-failed-plan", type=Path)

    run = subparsers.add_parser("run", help="unseal derived Session 0 fields and run the frozen hybrid test")
    run.add_argument("--plan", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="re-run the deterministic analysis core and verify exact agreement")
    verify.add_argument("--plan", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "freeze":
        plan = freeze_plan(
            campaign_root=args.campaign_root,
            loop_config_path=args.loop_config,
            simulation_plan_path=args.simulation_plan,
            simulation_output=args.simulation_output,
            plan_path=args.plan,
            output_dir=args.output,
            prior_failed_plan_path=args.prior_failed_plan,
        )
        print(canonical_json({
            "event": "hybrid_plan_frozen",
            "plan": str(args.plan.resolve()),
            "plan_sha256": digest_file(args.plan.resolve()),
            "status": plan["status"],
            "hardware_session0_scientific_fields_read": False,
        }))
        return 0
    if args.command == "run":
        result = run_frozen_plan(args.plan)
        report = result["report"]
        print(canonical_json({
            "event": "hybrid_test_completed",
            "output": result["output"],
            "simulation_assisted_status": report["decision"]["simulation_assisted_status"],
            "registered_hardware_status": report["decision"]["registered_hardware_endpoint_status"],
            "ratio": report["statistical_findings"]["hybrid_primary"]["ratio"],
            "p_value": report["statistical_findings"]["hybrid_primary"]["p_value"],
        }))
        return 0
    verification = verify_artifact(args.plan)
    print(canonical_json({
        "event": "hybrid_test_verified",
        "verdict": verification["verdict"],
        "analysis_signature": verification["rerun_analysis_signature"],
        "fallacy_scan_coverage": verification["fallacy_scan_coverage"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
