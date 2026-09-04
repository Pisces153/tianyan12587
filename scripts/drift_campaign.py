#!/usr/bin/env python3
"""Run an append-only, dual-backend natural-drift anchor campaign.

The campaign observes a fixed digital effective-field task.  It never claims
temperature or electromagnetic sensing, pulse calibration, or sub-queue
latency control.  Hardware submission is impossible without
``--confirm-hardware`` and a process-local ``TIANYAN_LOGIN_KEY``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backends.tianyan_native import assert_native_qcis, gate_counts
from src.backends.tianyan_v8_entangling import program
from src.features.pauli import BASIS_ORDER


PROFILE = "xa202609_tianyan_h1h2_dual_backend_natural_drift_v1"
SOURCE_FILES = (
    "scripts/drift_campaign.py",
    "docs/NATURAL_DRIFT_CAMPAIGN_PREREGISTRATION_v1.md",
    "src/backends/tianyan_v8_entangling.py",
    "src/backends/tianyan_native.py",
    "src/features/pauli.py",
)
TERMINAL_EVENTS = {"collected", "collection_failed", "submission_unknown"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp must include an offset")
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def digest_payload(payload: Any) -> str:
    return digest_bytes(canonical_json(payload).encode("utf-8"))


def json_ready(value: Any) -> Any:
    """Convert SDK and NumPy values without changing their observable content."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return iso(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def x_gate(qubit: int) -> list[str]:
    return [f"X2P Q{qubit}", f"X2P Q{qubit}"]


def readout_program(prepared_bits: str, physical_qubits: Sequence[int]) -> str:
    if len(prepared_bits) != len(physical_qubits) or set(prepared_bits).difference({"0", "1"}):
        raise ValueError("Readout probe must specify one bit for every physical qubit")
    lines: list[str] = []
    for bit, qubit in zip(prepared_bits, physical_qubits, strict=True):
        if bit == "1":
            lines.extend(x_gate(int(qubit)))
    lines.extend(f"M Q{qubit}" for qubit in physical_qubits)
    result = "\n".join(lines)
    assert_native_qcis(result)
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("status") != "preregistered_not_submitted":
        raise ValueError("Unexpected campaign status")
    if not isinstance(config.get("campaign_id"), str):
        raise ValueError("Campaign ID is required")
    cadence = config.get("cadence_minutes")
    if not isinstance(cadence, int) or cadence < 20 or cadence > 30:
        raise ValueError("Cadence must be an integer from 20 through 30 minutes")
    if not isinstance(config.get("shot_budget_hard_cap"), int) or config["shot_budget_hard_cap"] <= 0:
        raise ValueError("A positive campaign shot budget is required")
    measurement = config.get("measurement", {})
    if tuple(measurement.get("bases", ())) != BASIS_ORDER:
        raise ValueError("Campaign must use frozen V8 Pauli basis order")
    if tuple(measurement.get("times", ())) != (0.16, 0.31, 0.47):
        raise ValueError("Campaign must use frozen V8 anchor times")
    if measurement.get("nominal_parameters") != {"h1": 0.25, "h2": -0.35}:
        raise ValueError("Campaign must use frozen V8 h1/h2 nominal point")
    if measurement.get("shots_per_setting") != 1024:
        raise ValueError("Campaign currently requires 1024 shots per setting")
    reference = measurement.get("reference", {})
    if reference.get("time") != 0.31 or reference.get("basis") != "ZZ":
        raise ValueError("Reference probe must be nominal V8 t=0.31 ZZ")
    if reference.get("fixed_positions_zero_indexed") != [0, 11, 22, 32]:
        raise ValueError("Reference positions must be [0, 11, 22, 32]")
    readout = measurement.get("readout_probes", [])
    if len(readout) != 2 or {row.get("prepared_bits") for row in readout} != {"000000", "111111"}:
        raise ValueError("Campaign requires all-zero and all-one readout probes")
    if {row.get("fixed_position_zero_indexed") for row in readout} != {10, 21}:
        raise ValueError("Readout probe positions must be 10 and 21")
    backends = config.get("backends", [])
    expected = {
        "tianyan-287": [62, 55, 61, 68, 76, 69],
        "tianyan176": [42, 36, 31, 37, 44, 49],
    }
    if {item.get("backend_id"): item.get("physical_qubits") for item in backends} != expected:
        raise ValueError("Campaign backend layouts do not match qualifying h1/h2 routes")
    retry = config.get("retry_policy", {})
    if retry.get("max_attempts") != 5 or retry.get("base_backoff_seconds") != 30:
        raise ValueError("Retry policy must be five attempts with 30-second base backoff")
    if config.get("observable_environment_proxy", {}).get("execution_time_available") is not False:
        raise ValueError("Cloud execution timestamp must be explicitly unavailable")
    parse_utc(str(config["campaign_end_utc"]))


def backend_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in config["backends"]]


def build_programs(config: Mapping[str, Any], backend: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build exactly 27 anchors, four interleaved copies, and two readout probes."""
    measurement = config["measurement"]
    physical = list(backend["physical_qubits"])
    nominal = dict(measurement["nominal_parameters"])
    anchors: list[dict[str, Any]] = []
    for time_index, time_value in enumerate(measurement["times"]):
        for basis_index, basis in enumerate(BASIS_ORDER):
            qcis = program(nominal, float(time_value), basis, physical)
            anchors.append({
                "label": f"anchor_t{time_index}_{basis}",
                "kind": "anchor",
                "time": float(time_value),
                "basis": basis,
                "basis_index": basis_index,
                "qcis": qcis,
                "native_gate_counts": gate_counts(qcis),
            })
    reference_qcis = program(nominal, float(measurement["reference"]["time"]), str(measurement["reference"]["basis"]), physical)
    references = [{
        "label": f"interleaved_reference_{copy_index + 1}",
        "kind": "interleaved_reference",
        "copy_index": copy_index + 1,
        "time": float(measurement["reference"]["time"]),
        "basis": str(measurement["reference"]["basis"]),
        "qcis": reference_qcis,
        "native_gate_counts": gate_counts(reference_qcis),
    } for copy_index in range(4)]
    readouts = [{
        "label": str(item["label"]),
        "kind": "readout_probe",
        "prepared_bits": str(item["prepared_bits"]),
        "qcis": readout_program(str(item["prepared_bits"]), physical),
        "native_gate_counts": gate_counts(readout_program(str(item["prepared_bits"]), physical)),
    } for item in measurement["readout_probes"]]
    fixed: dict[int, dict[str, Any]] = {}
    for position, item in zip(measurement["reference"]["fixed_positions_zero_indexed"], references, strict=True):
        fixed[int(position)] = item
    for item in readouts:
        position = next(probe["fixed_position_zero_indexed"] for probe in measurement["readout_probes"] if probe["label"] == item["label"])
        fixed[int(position)] = item
    program_rows: list[dict[str, Any]] = []
    anchor_iter = iter(anchors)
    for position in range(33):
        item = dict(fixed[position]) if position in fixed else dict(next(anchor_iter))
        item["position_zero_indexed"] = position
        item["qcis_sha256"] = digest_bytes(item["qcis"].encode("utf-8"))
        program_rows.append(item)
    if list(anchor_iter):
        raise AssertionError("Anchor placement left unassigned probes")
    if len(program_rows) != 33 or sum(item["kind"] == "anchor" for item in program_rows) != 27:
        raise AssertionError("Campaign setting count is not 27+4+2")
    if [item["position_zero_indexed"] for item in program_rows if item["kind"] == "interleaved_reference"] != [0, 11, 22, 32]:
        raise AssertionError("Reference settings are not interleaved at frozen positions")
    return program_rows


def program_manifest(config: Mapping[str, Any], backend: Mapping[str, Any]) -> dict[str, Any]:
    programs = build_programs(config, backend)
    safe_rows = [{key: value for key, value in row.items() if key != "qcis"} for row in programs]
    payload = {
        "campaign_id": config["campaign_id"],
        "backend_id": backend["backend_id"],
        "physical_qubits": backend["physical_qubits"],
        "shots_per_setting": config["measurement"]["shots_per_setting"],
        "programs": safe_rows,
    }
    payload["probe_manifest_sha256"] = digest_payload(payload)
    return payload


def snapshot_id(campaign_id: str, backend_id: str, scheduled_utc: str) -> str:
    return digest_bytes(f"{campaign_id}|{backend_id}|{scheduled_utc}".encode("utf-8"))[:16].lower()


def cadence_slot(config: Mapping[str, Any], current: datetime | None = None) -> str:
    current = (current or utc_now()).astimezone(timezone.utc).replace(second=0, microsecond=0)
    cadence = int(config["cadence_minutes"])
    minute = current.minute - (current.minute % cadence)
    return iso(current.replace(minute=minute))


def frozen_sources(config_path: Path) -> list[dict[str, str]]:
    paths = [config_path.resolve(), *(ROOT / value for value in SOURCE_FILES)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("A campaign source file is missing")
    return [{"path": str(path.resolve()), "sha256": digest_file(path.resolve())} for path in paths]


class CampaignStore:
    """Append-only lifecycle journal with a verifiable SHA-256 chain."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.records_path = self.root / "snapshots.jsonl"
        self.records = self._read_records()
        self.last_hash = self._verify_chain()

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Blank line in append-only journal at {line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Journal record {line_number} is not an object")
                rows.append(row)
        return rows

    def _verify_chain(self) -> str | None:
        previous: str | None = None
        for sequence, row in enumerate(self.records):
            if row.get("sequence") != sequence:
                raise ValueError("Journal sequence is not contiguous")
            if row.get("previous_record_sha256") != previous:
                raise ValueError("Journal previous-record hash mismatch")
            recorded_hash = row.get("record_sha256")
            source = {key: value for key, value in row.items() if key != "record_sha256"}
            if recorded_hash != digest_payload(source):
                raise ValueError("Journal record SHA-256 mismatch")
            previous = str(recorded_hash)
        return previous

    def append(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            "sequence": len(self.records),
            "recorded_at_utc": iso(),
            "event": event,
            "previous_record_sha256": self.last_hash,
            **json_ready(payload),
        }
        row["record_sha256"] = digest_payload(row)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(row)
        self.last_hash = str(row["record_sha256"])
        return row

    def by_snapshot(self, identifier: str) -> list[dict[str, Any]]:
        return [row for row in self.records if row.get("snapshot_id") == identifier]

    def submitted(self) -> dict[str, dict[str, Any]]:
        return {str(row["snapshot_id"]): row for row in self.records if row.get("event") == "submitted"}

    def terminal_event(self, identifier: str) -> str | None:
        events = [str(row.get("event")) for row in self.by_snapshot(identifier)]
        return next((event for event in reversed(events) if event in TERMINAL_EVENTS), None)

    def latest(self, identifier: str, event: str) -> dict[str, Any] | None:
        return next((row for row in reversed(self.by_snapshot(identifier)) if row.get("event") == event), None)


def prepare(config_path: Path, out: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config)
    out = out.resolve()
    manifest_path = out / "campaign_manifest.json"
    source_rows = frozen_sources(config_path)
    manifest = {
        "profile": PROFILE,
        "campaign_id": config["campaign_id"],
        "prepared_at_utc": iso(),
        "config_path": str(config_path),
        "config_sha256": digest_file(config_path),
        "frozen_sources": source_rows,
        "backend_probe_manifests": [program_manifest(config, backend) for backend in backend_rows(config)],
        "per_backend_settings": 33,
        "per_backend_shots": 33 * int(config["measurement"]["shots_per_setting"]),
        "per_round_shots": 2 * 33 * int(config["measurement"]["shots_per_setting"]),
        "hardware_submission": False,
    }
    if manifest_path.exists():
        existing = load_json(manifest_path)
        immutable = ("profile", "campaign_id", "config_sha256", "frozen_sources", "backend_probe_manifests")
        if any(existing.get(key) != manifest.get(key) for key in immutable):
            raise ValueError("Existing campaign manifest conflicts with current frozen sources")
        return existing
    write_json_new(manifest_path, manifest)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    CampaignStore(out)
    return manifest


def assert_frozen(manifest: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    config = load_json(config_path.resolve())
    validate_config(config)
    if manifest.get("config_sha256") != digest_file(config_path.resolve()):
        raise ValueError("Campaign config differs from frozen manifest")
    if manifest.get("frozen_sources") != frozen_sources(config_path.resolve()):
        raise ValueError("Campaign source differs from frozen manifest")
    return config


def platform_for_backend(backend_id: str) -> Any:
    key = os.environ.get("TIANYAN_LOGIN_KEY")
    if not key:
        raise RuntimeError("Set TIANYAN_LOGIN_KEY only in this process; never write it to disk")
    from cqlib.quantum_platform import TianYanPlatform

    return TianYanPlatform(login_key=key, auto_login=True, machine_name=backend_id)


def capture_calibration_metadata(platform: Any, backend_id: str) -> dict[str, Any]:
    """Save raw available calibration responses; absent fields remain null, never invented."""
    calls: list[tuple[str, Callable[[], Any]]] = [
        ("get_machine_config", lambda: platform.get_machine_config({"computerCode": backend_id})),
        ("download_config", lambda: platform.download_config(machine=backend_id)),
    ]
    raw: dict[str, Any] = {}
    errors: dict[str, str | None] = {}
    for label, method in calls:
        try:
            raw[label] = json_ready(method())
            errors[label] = None
        except Exception as error:  # Raw platform fields vary across API revisions.
            raw[label] = None
            errors[label] = f"{type(error).__name__}: {error}"
    return {
        "backend_id": backend_id,
        "captured_at_utc": iso(),
        "raw": raw,
        "errors": errors,
        "available": any(value is not None for value in raw.values()),
        "execution_time_available": False,
    }


def _safe_result_id(row: Mapping[str, Any]) -> str | None:
    identifier = row.get("experimentTaskId")
    return None if identifier is None else str(identifier)


def result_counts(row: Mapping[str, Any], physical_qubits: Sequence[int], shots: int) -> np.ndarray:
    samples = row.get("resultStatus")
    if not isinstance(samples, list) or len(samples) != shots + 1 or samples[0] != list(physical_qubits):
        raise ValueError("Unexpected TianYan bit-order header or shot count")
    counts = np.zeros(64, dtype=np.int32)
    for sample in samples[1:]:
        if not isinstance(sample, list) or len(sample) != 6 or any(bit not in (0, 1) for bit in sample):
            raise ValueError("Malformed TianYan sample")
        counts[int("".join(str(bit) for bit in sample), 2)] += 1
    return counts


def raw_result_rows(store: CampaignStore, identifier: str) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in store.by_snapshot(identifier):
        raw_path = row.get("raw_results_path")
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Journal references missing raw result artifact: {path}")
        payload = load_json(path)
        for result in payload.get("results", []):
            if isinstance(result, Mapping) and _safe_result_id(result) is not None:
                merged[_safe_result_id(result) or ""] = dict(result)
    return merged


def next_retry_allowed(store: CampaignStore, identifier: str, current: datetime) -> bool:
    latest = store.latest(identifier, "collection_retry")
    if latest is None:
        return True
    return current >= parse_utc(str(latest["retry_not_before_utc"]))


def collection_attempts(store: CampaignStore, identifier: str) -> int:
    return sum(row.get("event") == "collection_retry" for row in store.by_snapshot(identifier))


def materialize_counts(
    store: CampaignStore,
    submitted: Mapping[str, Any],
    result_map: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> Path:
    identifier = str(submitted["snapshot_id"])
    tasks = submitted["tasks"]
    physical = list(backend["physical_qubits"])
    shots = int(config["measurement"]["shots_per_setting"])
    counts = np.stack([result_counts(result_map[str(task["query_id"])], physical, shots) for task in tasks])
    path = store.root / "raw" / f"{identifier}_counts.npz"
    if path.exists():
        return path
    np.savez_compressed(
        path,
        labels=np.asarray([task["label"] for task in tasks]),
        kinds=np.asarray([task["kind"] for task in tasks]),
        positions=np.asarray([task["position_zero_indexed"] for task in tasks], dtype=np.int16),
        counts=counts,
        shots=np.asarray(shots),
        snapshot_id=np.asarray(identifier),
        probe_manifest_sha256=np.asarray(submitted["probe_manifest_sha256"]),
    )
    return path


def collect_snapshot(
    store: CampaignStore,
    submitted: Mapping[str, Any],
    config: Mapping[str, Any],
    backend: Mapping[str, Any],
    platform: Any,
    *,
    wait_seconds: int,
    poll_seconds: int,
) -> str:
    identifier = str(submitted["snapshot_id"])
    current = utc_now()
    terminal = store.terminal_event(identifier)
    if terminal in TERMINAL_EVENTS:
        return terminal
    if not next_retry_allowed(store, identifier, current):
        return "backoff"
    existing = raw_result_rows(store, identifier)
    missing = [str(task["query_id"]) for task in submitted["tasks"] if str(task["query_id"]) not in existing]
    if not missing:
        counts_path = materialize_counts(store, submitted, existing, config, backend)
        store.append("collected", {
            "snapshot_id": identifier,
            "backend_id": backend["backend_id"],
            "result_count": len(existing),
            "missing_query_ids": [],
            "counts_path": str(counts_path),
            "counts_sha256": digest_file(counts_path),
            "execution_time_available": False,
        })
        return "collected"
    try:
        queried = platform.query_experiment(missing, max_wait_time=wait_seconds, sleep_time=poll_seconds)
        if not isinstance(queried, list):
            raise ValueError("TianYan query did not return a result list")
        raw_payload = {
            "profile": PROFILE,
            "snapshot_id": identifier,
            "backend_id": backend["backend_id"],
            "wallclock_retrieve_utc": iso(),
            "requested_query_ids": missing,
            "results": json_ready(queried),
            "execution_time_available": False,
        }
        raw_path = store.root / "raw" / f"{identifier}_query_{collection_attempts(store, identifier) + 1:02d}.json"
        write_json_new(raw_path, raw_payload)
        observed = {_safe_result_id(row) for row in queried if isinstance(row, Mapping)}
        merged = {**existing, **{_safe_result_id(row): dict(row) for row in queried if isinstance(row, Mapping) and _safe_result_id(row) is not None}}
        still_missing = [query_id for query_id in missing if query_id not in observed]
        event_payload = {
            "snapshot_id": identifier,
            "backend_id": backend["backend_id"],
            "wallclock_retrieve_utc": raw_payload["wallclock_retrieve_utc"],
            "raw_results_path": str(raw_path),
            "raw_results_sha256": digest_file(raw_path),
            "result_count": len(merged),
            "missing_query_ids": still_missing,
            "execution_time_available": False,
        }
        if still_missing:
            store.append("partial", event_payload)
            return "partial"
        counts_path = materialize_counts(store, submitted, merged, config, backend)
        event_payload.update({"counts_path": str(counts_path), "counts_sha256": digest_file(counts_path)})
        store.append("collected", event_payload)
        return "collected"
    except Exception as error:
        attempts = collection_attempts(store, identifier) + 1
        if attempts >= int(config["retry_policy"]["max_attempts"]):
            store.append("collection_failed", {
                "snapshot_id": identifier,
                "backend_id": backend["backend_id"],
                "attempts": attempts,
                "error": f"{type(error).__name__}: {error}",
                "execution_time_available": False,
            })
            return "collection_failed"
        delay = int(config["retry_policy"]["base_backoff_seconds"]) * (2 ** (attempts - 1))
        store.append("collection_retry", {
            "snapshot_id": identifier,
            "backend_id": backend["backend_id"],
            "attempt": attempts,
            "retry_not_before_utc": iso(current + timedelta(seconds=delay)),
            "error": f"{type(error).__name__}: {error}",
            "execution_time_available": False,
        })
        return "retry"


def committed_shots(store: CampaignStore) -> int:
    return sum(int(row.get("shots_committed", 0)) for row in store.records if row.get("event") == "submitted")


def emit_failure_alert_if_needed(store: CampaignStore, identifier: str) -> None:
    failed = [row for row in store.records if row.get("event") in {"collection_failed", "submission_unknown"}]
    if len(failed) < 3:
        return
    tail = failed[-3:]
    if all(row.get("snapshot_id") for row in tail):
        store.append("campaign_alert", {
            "alert_type": "three_consecutive_snapshot_failures",
            "snapshot_ids": [row["snapshot_id"] for row in tail],
            "latest_snapshot_id": identifier,
            "execution_time_available": False,
        })


def submit_snapshot(
    store: CampaignStore,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    backend: Mapping[str, Any],
    scheduled_utc: str,
    platform: Any,
) -> str:
    identifier = snapshot_id(str(config["campaign_id"]), str(backend["backend_id"]), scheduled_utc)
    existing_events = store.by_snapshot(identifier)
    if any(row.get("event") == "submitted" for row in existing_events):
        return "already_submitted"
    if existing_events:
        # Submission response may have reached hardware before a client disconnect.
        # At-most-once behavior is safer than guessing and duplicating raw counts.
        return "submission_requires_manual_reconciliation"
    program_rows = build_programs(config, backend)
    probe = next(item for item in manifest["backend_probe_manifests"] if item["backend_id"] == backend["backend_id"])
    telemetry = capture_calibration_metadata(platform, str(backend["backend_id"]))
    store.append("submission_attempt", {
        "snapshot_id": identifier,
        "campaign_id": config["campaign_id"],
        "backend_id": backend["backend_id"],
        "scheduled_utc": scheduled_utc,
        "probe_manifest_sha256": probe["probe_manifest_sha256"],
        "program_count": len(program_rows),
        "telemetry": telemetry,
        "execution_time_available": False,
    })
    try:
        identifiers = platform.submit_experiment(
            circuit=[row["qcis"] for row in program_rows],
            name=f"XA202609_NATURAL_DRIFT_{backend['backend_id']}_{identifier}",
            num_shots=int(config["measurement"]["shots_per_setting"]),
            machine_name=str(backend["backend_id"]),
        )
        if not isinstance(identifiers, list) or len(identifiers) != len(program_rows):
            raise RuntimeError("TianYan returned incomplete task-ID list")
        tasks = [{
            "query_id": str(query_id),
            "label": row["label"],
            "kind": row["kind"],
            "position_zero_indexed": row["position_zero_indexed"],
            "qcis_sha256": row["qcis_sha256"],
        } for query_id, row in zip(identifiers, program_rows, strict=True)]
        store.append("submitted", {
            "snapshot_id": identifier,
            "campaign_id": config["campaign_id"],
            "backend_id": backend["backend_id"],
            "physical_qubits": backend["physical_qubits"],
            "scheduled_utc": scheduled_utc,
            "wallclock_submit_utc": iso(),
            "probe_manifest_sha256": probe["probe_manifest_sha256"],
            "tasks": tasks,
            "shots_committed": len(tasks) * int(config["measurement"]["shots_per_setting"]),
            "execution_time_available": False,
        })
        return "submitted"
    except Exception as error:
        store.append("submission_unknown", {
            "snapshot_id": identifier,
            "campaign_id": config["campaign_id"],
            "backend_id": backend["backend_id"],
            "scheduled_utc": scheduled_utc,
            "error": f"{type(error).__name__}: {error}",
            "resolution": "Manual query-ID reconciliation required before any re-submission; automatic retry would risk duplicate hardware work.",
            "execution_time_available": False,
        })
        emit_failure_alert_if_needed(store, identifier)
        return "submission_unknown"


def pending_submissions(store: CampaignStore) -> Iterable[dict[str, Any]]:
    for identifier, submitted in store.submitted().items():
        if store.terminal_event(identifier) not in TERMINAL_EVENTS:
            yield submitted


def run_once(
    config_path: Path,
    out: Path,
    *,
    confirm_hardware: bool,
    scheduled_utc: str | None,
    collect_wait_seconds: int,
    poll_seconds: int,
    backend_ids: Sequence[str] | None = None,
    platform_factory: Callable[[str], Any] = platform_for_backend,
) -> dict[str, Any]:
    manifest = prepare(config_path, out)
    config = assert_frozen(manifest, config_path)
    store = CampaignStore(out)
    scheduled = parse_utc(scheduled_utc).isoformat() if scheduled_utc else cadence_slot(config)
    if parse_utc(scheduled) > parse_utc(str(config["campaign_end_utc"])):
        return {"status": "campaign_closed", "scheduled_utc": scheduled}
    selected_backends = [backend for backend in backend_rows(config) if backend_ids is None or backend["backend_id"] in backend_ids]
    if not selected_backends or (backend_ids is not None and set(backend_ids).difference({backend["backend_id"] for backend in selected_backends})):
        raise ValueError("Requested backend is not in the frozen dual-backend campaign")
    if not confirm_hardware:
        return dry_run_summary(config, manifest, scheduled, selected_backends)
    platforms = {backend["backend_id"]: platform_factory(str(backend["backend_id"])) for backend in selected_backends}
    collected: dict[str, str] = {}
    for submitted in list(pending_submissions(store)):
        if submitted["backend_id"] not in platforms:
            continue
        backend = next(item for item in backend_rows(config) if item["backend_id"] == submitted["backend_id"])
        collected[str(submitted["snapshot_id"])] = collect_snapshot(
            store, submitted, config, backend, platforms[backend["backend_id"]], wait_seconds=collect_wait_seconds, poll_seconds=poll_seconds
        )
    outcomes: dict[str, str] = {}
    missing_backends = [
        backend for backend in selected_backends
        if not store.by_snapshot(snapshot_id(str(config["campaign_id"]), str(backend["backend_id"]), scheduled))
    ]
    required = len(missing_backends) * int(manifest["per_backend_shots"])
    if committed_shots(store) + required > int(config["shot_budget_hard_cap"]):
        raise RuntimeError("shot_budget_hard_cap would be exceeded; campaign stopped before submission")
    for backend in selected_backends:
        outcomes[str(backend["backend_id"])] = submit_snapshot(store, manifest, config, backend, scheduled, platforms[backend["backend_id"]])
    return {
        "status": "submitted_or_resumed",
        "scheduled_utc": scheduled,
        "submission_outcomes": outcomes,
        "collection_outcomes": collected,
        "committed_shots": committed_shots(store),
        "shot_budget_hard_cap": config["shot_budget_hard_cap"],
        "selected_backends": [backend["backend_id"] for backend in selected_backends],
    }


def dry_run_summary(
    config: Mapping[str, Any], manifest: Mapping[str, Any], scheduled_utc: str, selected_backends: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_backends = list(selected_backends or backend_rows(config))
    return {
        "will_submit_hardware": False,
        "campaign_id": config["campaign_id"],
        "scheduled_utc": scheduled_utc,
        "backends": [{
            "backend_id": backend["backend_id"],
            "settings": len(build_programs(config, backend)),
            "shots": len(build_programs(config, backend)) * int(config["measurement"]["shots_per_setting"]),
            "probe_manifest_sha256": next(item["probe_manifest_sha256"] for item in manifest["backend_probe_manifests"] if item["backend_id"] == backend["backend_id"]),
        } for backend in selected_backends],
        "per_round_shots": len(selected_backends) * manifest["per_backend_shots"],
        "execution_time_available": False,
    }


def power_report(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    plan = config["power_plan"]
    shots = int(config["measurement"]["shots_per_setting"])
    rho = float(plan["ar1_correlation_assumption"])
    if not -1.0 < rho < 1.0:
        raise ValueError("AR(1) correlation assumption must be in (-1, 1)")
    design_effect = (1.0 + rho) / (1.0 - rho)
    normal = NormalDist()
    rows: list[dict[str, Any]] = []
    for family, count in plan["multiplicity_families"].items():
        maximum_single_snapshot_variance = 1.0 / shots if family == "pauli15" else 0.25 / shots
        alpha_per_test = float(plan["familywise_alpha"]) / int(count)
        z_alpha = normal.inv_cdf(1.0 - alpha_per_test / 2.0)
        z_power = normal.inv_cdf(float(plan["power"]))
        for effect in plan["target_effect_sizes"]:
            pairs = math.ceil(
                2.0 * maximum_single_snapshot_variance * design_effect * (z_alpha + z_power) ** 2 / float(effect) ** 2
            )
            mdd_at_100_pairs = (z_alpha + z_power) * math.sqrt(2.0 * maximum_single_snapshot_variance * design_effect / 100.0)
            rows.append({
                "family": family,
                "multiplicity": int(count),
                "target_effect": float(effect),
                "required_independent_pair_equivalents": pairs,
                "mdd_at_100_pair_equivalents": mdd_at_100_pairs,
                "single_snapshot_variance_upper_bound": maximum_single_snapshot_variance,
                "ar1_design_effect": design_effect,
                "alpha_per_test_two_sided": alpha_per_test,
            })
    return {
        "profile": PROFILE,
        "created_at_utc": iso(),
        "method": "Normal-approximation paired change test with binomial shot-noise upper bound, Bonferroni family control, and AR(1) effective-sample penalty.",
        "not_a_hardware_result": True,
        "shots_per_setting": shots,
        "rows": rows,
        "decision": "Cadence and 1024 shots were selected before collection. Observed per-feature shot variance and serial correlation must replace these conservative design inputs in final analysis.",
        "config_sha256": manifest["config_sha256"],
    }


def verify(config_path: Path, out: Path, *, write_report: bool = True) -> dict[str, Any]:
    manifest = prepare(config_path, out)
    config = assert_frozen(manifest, config_path)
    store = CampaignStore(out)
    submitted = store.submitted()
    expected_programs = {backend["backend_id"]: build_programs(config, backend) for backend in backend_rows(config)}
    prohibited_keys: list[str] = []
    for row in store.records:
        stack: list[Any] = [row]
        while stack:
            current = stack.pop()
            if isinstance(current, Mapping):
                for key, value in current.items():
                    if "execution_time" in str(key) and key != "execution_time_available":
                        prohibited_keys.append(str(key))
                    stack.append(value)
            elif isinstance(current, list):
                stack.extend(current)
    if prohibited_keys:
        raise ValueError(f"Prohibited timestamp field names in journal: {sorted(set(prohibited_keys))}")
    for identifier, row in submitted.items():
        backend_id = str(row["backend_id"])
        expected_identifier = snapshot_id(str(config["campaign_id"]), backend_id, str(row["scheduled_utc"]))
        if identifier != expected_identifier:
            raise ValueError("Snapshot identifier does not bind campaign, backend, and schedule")
        expected = expected_programs[backend_id]
        tasks = row.get("tasks", [])
        if len(tasks) != 33 or [task.get("label") for task in tasks] != [item["label"] for item in expected]:
            raise ValueError("Submitted settings differ from frozen 27+4+2 protocol")
        if any(not task.get("query_id") for task in tasks):
            raise ValueError("Submitted snapshot lacks persisted query IDs")
    total = committed_shots(store)
    if total > int(config["shot_budget_hard_cap"]):
        raise ValueError("Campaign budget gate was violated")
    report = {
        "profile": PROFILE,
        "verified_at_utc": iso(),
        "journal_records": len(store.records),
        "submitted_snapshots": len(submitted),
        "committed_shots": total,
        "shot_budget_hard_cap": config["shot_budget_hard_cap"],
        "journal_tip_sha256": store.last_hash,
        "hash_chain_valid": True,
        "execution_time_available": False,
        "result": "pass",
    }
    if write_report:
        write_json_atomic(out / "verification_report.json", report)
    return report


def run_continuously(arguments: argparse.Namespace) -> None:
    if not arguments.confirm_hardware:
        raise RuntimeError("Continuous hardware submission requires --confirm-hardware")
    while True:
        result = run_once(
            arguments.config, arguments.out, confirm_hardware=True, scheduled_utc=None,
            collect_wait_seconds=arguments.collect_wait_seconds, poll_seconds=arguments.poll_seconds, backend_ids=arguments.backend,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("status") == "campaign_closed":
            return
        time.sleep(max(1, int(arguments.sleep_seconds)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "dry-run", "verify", "power"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--out", required=True, type=Path)
    sub = commands.add_parser("run")
    sub.add_argument("--config", required=True, type=Path)
    sub.add_argument("--out", required=True, type=Path)
    sub.add_argument("--confirm-hardware", action="store_true")
    sub.add_argument("--once", action="store_true")
    sub.add_argument("--scheduled-utc")
    sub.add_argument("--collect-wait-seconds", type=int, default=1)
    sub.add_argument("--poll-seconds", type=int, default=0)
    sub.add_argument("--sleep-seconds", type=int, default=60)
    sub.add_argument("--backend", action="append", choices=["tianyan-287", "tianyan176"])
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        print(json.dumps(prepare(arguments.config, arguments.out), ensure_ascii=False))
    elif arguments.command == "dry-run":
        manifest = prepare(arguments.config, arguments.out)
        config = assert_frozen(manifest, arguments.config)
        print(json.dumps(dry_run_summary(config, manifest, cadence_slot(config)), ensure_ascii=False))
    elif arguments.command == "power":
        manifest = prepare(arguments.config, arguments.out)
        config = assert_frozen(manifest, arguments.config)
        report = power_report(config, manifest)
        write_json_atomic(arguments.out.resolve() / "power_report.json", report)
        print(json.dumps(report, ensure_ascii=False))
    elif arguments.command == "verify":
        print(json.dumps(verify(arguments.config, arguments.out), ensure_ascii=False))
    elif arguments.command == "run":
        if arguments.once:
            result = run_once(
                arguments.config, arguments.out, confirm_hardware=arguments.confirm_hardware,
                scheduled_utc=arguments.scheduled_utc, collect_wait_seconds=arguments.collect_wait_seconds,
                poll_seconds=arguments.poll_seconds, backend_ids=arguments.backend,
            )
            print(json.dumps(result, ensure_ascii=False))
        else:
            run_continuously(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
