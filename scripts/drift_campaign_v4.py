#!/usr/bin/env python3
"""B-4 v4 single-backend collector protocol and guarded probe submission.

One process owns one backend and one credential environment variable.  The
module builds the frozen 33-setting anchor job plus independent two-setting
readout-probe jobs, records only platform-returned execution timestamps, and
marks event-triggered bursts as supportive-only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import drift_campaign as campaign


SCHEMA = "xa202609_b4_drift_campaign_v4"
PROFILE = "xa202609_b4_single_backend_drift_campaign_v4"
ANCHOR_REFERENCE_SHOTS_PER_SETTING = 3072
ANCHOR_REFERENCE_RATE_SHOTS_PER_SECOND = 1486.0918995653424
ANCHOR_SHOTS_SCALING_RULE = (
    "round(3072 * backend.tb6_measured_timing.effective_shots_per_second / "
    "1486.0918995653424)"
)
PROBE_SHOTS_PER_SETTING = 16384
BURST_TARGET_MINUTES = (0, 1, 2, 4, 8, 16)
EVENT_BURST_TARGET_MINUTES = (0, 1, 2, 4)
REFERENCE_POSITIONS = (0, 11, 22, 32)
LEGACY_READOUT_POSITIONS = (10, 21)
PROBE_LABELS = ("readout_all_zero", "readout_all_one")
PLATFORM_TIMESTAMP_FIELDS = (
    "executionTime",
    "execution_time",
    "executeTime",
    "executedAt",
    "finishTime",
    "finishedAt",
    "platformTimestamp",
)
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_embedded_credentials(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()
            if lowered in {"api_key", "apikey", "login_key", "token", "secret", "password"}:
                raise ValueError(f"credential material is forbidden in config: {child_path}")
            _reject_embedded_credentials(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_credentials(child, f"{path}[{index}]")


def expected_anchor_shots_per_setting(config: Mapping[str, Any]) -> int:
    """Return backend-scaled anchor shots selected after the map root-cause A/B."""
    try:
        rate = float(config["backend"]["tb6_measured_timing"]["effective_shots_per_second"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("backend requires a valid T-B6 effective shot rate") from error
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("T-B6 effective shot rate must be positive and finite")
    return int(round(ANCHOR_REFERENCE_SHOTS_PER_SETTING * rate / ANCHOR_REFERENCE_RATE_SHOTS_PER_SECOND))


def validate_v4_config(config: Mapping[str, Any]) -> None:
    """Validate one backend-specific instance of the shared v4 protocol."""
    if config.get("schema") != SCHEMA or config.get("status") != "preregistered_not_submitted":
        raise ValueError("unexpected v4 schema or status")
    if config.get("campaign_id") != "xa202609_b4_drift_campaign_v4":
        raise ValueError("v4 campaign_id changed")
    campaign.parse_utc(str(config.get("cadence_origin_utc")))
    campaign.parse_utc(str(config.get("campaign_end_utc")))
    backend = config.get("backend", {})
    qubits = backend.get("physical_qubits", [])
    if not isinstance(backend.get("backend_id"), str) or not backend["backend_id"]:
        raise ValueError("one backend_id is required")
    if len(qubits) != 6 or len(set(qubits)) != 6 or not all(isinstance(value, int) for value in qubits):
        raise ValueError("backend requires six distinct physical qubits")
    credential_env = config.get("credential_env")
    if not isinstance(credential_env, str) or not _ENVIRONMENT_NAME.fullmatch(credential_env):
        raise ValueError("credential_env must name one process environment variable")
    _reject_embedded_credentials(config)

    measurement = config.get("measurement", {})
    anchor = measurement.get("anchor_job", {})
    expected_anchor_shots = expected_anchor_shots_per_setting(config)
    if anchor.get("shots_per_setting") != expected_anchor_shots:
        raise ValueError(
            f"anchor_33 requires {expected_anchor_shots} shots per setting for this backend"
        )
    if anchor.get("shots_per_setting_rule") != ANCHOR_SHOTS_SCALING_RULE:
        raise ValueError("anchor_33 backend scaling rule changed")
    if tuple(anchor.get("times", ())) != (0.16, 0.31, 0.47):
        raise ValueError("anchor times changed")
    if tuple(anchor.get("bases", ())) != campaign.BASIS_ORDER:
        raise ValueError("anchor basis order changed")
    if anchor.get("nominal_parameters") != {"h1": 0.25, "h2": -0.35}:
        raise ValueError("anchor nominal parameters changed")
    if tuple(anchor.get("reference", {}).get("fixed_positions_zero_indexed", ())) != REFERENCE_POSITIONS:
        raise ValueError("reference positions must remain [0, 11, 22, 32]")
    if anchor.get("include_legacy_readout_copies") is not True:
        raise ValueError(
            "33-setting semantics unresolved: retaining positions [0,11,22,32] requires legacy readout copies at [10,21]"
        )
    if tuple(anchor.get("legacy_readout_positions_zero_indexed", ())) != LEGACY_READOUT_POSITIONS:
        raise ValueError("legacy readout positions changed")
    probes = measurement.get("probe_job", {}).get("settings", [])
    if measurement.get("probe_job", {}).get("shots_per_setting") != PROBE_SHOTS_PER_SETTING:
        raise ValueError("probe_burst requires 16384 shots per setting")
    if len(probes) != 2 or tuple(row.get("label") for row in probes) != PROBE_LABELS:
        raise ValueError("independent probe job requires ordered all-zero/all-one settings")
    if {row.get("prepared_bits") for row in probes} != {"000000", "111111"}:
        raise ValueError("independent probe prepared states changed")
    if tuple(config.get("scheduling", {}).get("probe_target_minutes", ())) != BURST_TARGET_MINUTES:
        raise ValueError("probe burst targets must be [0,1,2,4,8,16] minutes")
    if tuple(config.get("event_policy", {}).get("supportive_burst_target_minutes", ())) != EVENT_BURST_TARGET_MINUTES:
        raise ValueError("event burst target minutes changed")
    if config.get("event_policy", {}).get("primary_sf_eligible") is not False:
        raise ValueError("event bursts must be excluded from primary SF")
    timestamp = config.get("timestamp_policy", {})
    if timestamp.get("platform_timestamp_only") is not True or timestamp.get("synthesize_execution_timestamp") is not False:
        raise ValueError("execution timestamp must be platform-returned or null")
    trimming = config.get("trimming_order", {})
    if trimming.get("probe_round_counts") != [6, 5, 4]:
        raise ValueError("probe rounds must trim 6 to 5 to 4 first")
    if trimming.get("then") != ["closed_loop_updates", "mirror_repetitions"]:
        raise ValueError("trimming order changed")
    retry = config.get("retry_policy", {})
    if retry != {"max_attempts": 5, "base_backoff_seconds": 30}:
        raise ValueError("retry policy changed")


def _base_program_config(config: Mapping[str, Any]) -> dict[str, Any]:
    anchor = config["measurement"]["anchor_job"]
    probes = config["measurement"]["probe_job"]["settings"]
    return {
        "campaign_id": config["campaign_id"],
        "measurement": {
            "nominal_parameters": anchor["nominal_parameters"],
            "times": anchor["times"],
            "bases": anchor["bases"],
            "shots_per_setting": anchor["shots_per_setting"],
            "reference": anchor["reference"],
            "readout_probes": [
                {**row, "fixed_position_zero_indexed": position}
                for row, position in zip(probes, LEGACY_READOUT_POSITIONS, strict=True)
            ],
        },
    }


def build_anchor_programs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_v4_config(config)
    rows = campaign.build_programs(_base_program_config(config), config["backend"])
    for row in rows:
        row["job_role"] = "anchor_33"
        if row["kind"] == "interleaved_reference":
            row["analysis_role"] = "primary_sf_short_lag_reference"
        elif row["kind"] == "readout_probe":
            row["kind"] = "legacy_readout_diagnostic"
            row["setting_tag"] = "in_job_probe"
            row["analysis_role"] = "supportive_context_qc_only_not_primary_sf"
        else:
            row["analysis_role"] = "anchor_diagnostic"
    assert [row["position_zero_indexed"] for row in rows if row["kind"] == "interleaved_reference"] == list(REFERENCE_POSITIONS)
    return rows


def build_probe_programs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_v4_config(config)
    physical = list(config["backend"]["physical_qubits"])
    rows: list[dict[str, Any]] = []
    for position, setting in enumerate(config["measurement"]["probe_job"]["settings"]):
        qcis = campaign.readout_program(str(setting["prepared_bits"]), physical)
        rows.append({
            "label": str(setting["label"]),
            "kind": "independent_readout_probe",
            "job_role": "probe_burst",
            "analysis_role": "primary_sf_only_when_non_event_and_same_regime",
            "prepared_bits": str(setting["prepared_bits"]),
            "position_zero_indexed": position,
            "qcis": qcis,
            "qcis_sha256": campaign.digest_bytes(qcis.encode("utf-8")),
        })
    return rows


def build_session_jobs(
    config: Mapping[str, Any],
    *,
    session_origin_platform_utc: str,
    regime_id: str,
    active_probe_rounds: int = 6,
    burst_flag: bool = False,
) -> list[dict[str, Any]]:
    validate_v4_config(config)
    origin = campaign.parse_utc(session_origin_platform_utc)
    allowed_rounds = config["trimming_order"]["probe_round_counts"]
    if active_probe_rounds not in allowed_rounds:
        raise ValueError("active_probe_rounds must follow frozen trimming ladder")
    targets = list(config["scheduling"]["probe_target_minutes"])[:active_probe_rounds]
    jobs: list[dict[str, Any]] = []
    for round_index, minutes in enumerate(targets):
        planned = origin + timedelta(minutes=float(minutes))
        jobs.append({
            "job_id": campaign.digest_payload({
                "campaign_id": config["campaign_id"],
                "backend_id": config["backend"]["backend_id"],
                "session_origin_platform_utc": campaign.iso(origin),
                "round_index": round_index,
                "burst_flag": bool(burst_flag),
            })[:20].lower(),
            "job_role": "probe_burst",
            "round_index": round_index,
            "planned_target_minutes": float(minutes),
            "planned_target_utc": campaign.iso(planned),
            "regime_id": str(regime_id),
            "burst_flag": bool(burst_flag),
            "primary_sf_eligible": not burst_flag,
            "settings": build_probe_programs(config),
        })
    return jobs


def build_event_burst_jobs(config: Mapping[str, Any], *, event_platform_utc: str, regime_id: str) -> list[dict[str, Any]]:
    copied = deepcopy(config)
    copied["scheduling"]["probe_target_minutes"] = list(EVENT_BURST_TARGET_MINUTES) + [8, 16]
    return build_session_jobs(
        copied,
        session_origin_platform_utc=event_platform_utc,
        regime_id=regime_id,
        active_probe_rounds=4,
        burst_flag=True,
    )


def trimming_state(config: Mapping[str, Any], step: int) -> dict[str, int]:
    """Deterministic order: probe rounds, closed-loop updates, mirror repeats."""
    validate_v4_config(config)
    rounds = list(config["trimming_order"]["probe_round_counts"])
    updates = int(config["closed_loop"]["planned_updates"])
    mirrors = int(config["mirror"]["planned_repetitions"])
    if step < 0:
        raise ValueError("trimming step must be non-negative")
    if step < len(rounds):
        return {"probe_rounds": rounds[step], "closed_loop_updates": updates, "mirror_repetitions": mirrors}
    step -= len(rounds) - 1
    reduced_updates = max(0, updates - step)
    if reduced_updates > 0:
        return {"probe_rounds": rounds[-1], "closed_loop_updates": reduced_updates, "mirror_repetitions": mirrors}
    mirror_step = max(0, step - updates)
    return {"probe_rounds": rounds[-1], "closed_loop_updates": 0, "mirror_repetitions": max(0, mirrors - mirror_step)}


def calibration_time_raw(payload: Any) -> Any | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() == "calibrationtime":
                return value
        for value in payload.values():
            found = calibration_time_raw(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = calibration_time_raw(value)
            if found is not None:
                return found
    return None


def regime_transition(previous: Mapping[str, Any] | None, calibration_time: Any | None) -> dict[str, Any]:
    raw = None if calibration_time is None else str(calibration_time)
    if previous is None:
        return {"regime_id": "regime-0000", "calibration_time_raw": raw, "flipped": False}
    previous_raw = previous.get("calibration_time_raw")
    previous_id = str(previous.get("regime_id", "regime-0000"))
    if raw is None or previous_raw is None or raw == previous_raw:
        return {"regime_id": previous_id, "calibration_time_raw": raw or previous_raw, "flipped": False}
    index = int(previous_id.rsplit("-", 1)[-1]) + 1
    return {"regime_id": f"regime-{index:04d}", "calibration_time_raw": raw, "flipped": True}


def platform_timestamp(record: Mapping[str, Any]) -> dict[str, Any]:
    for field in PLATFORM_TIMESTAMP_FIELDS:
        value = record.get(field)
        if value is not None:
            return {"platform_timestamp_raw": value, "platform_timestamp_field": field, "execution_timestamp_available": True}
    return {"platform_timestamp_raw": None, "platform_timestamp_field": None, "execution_timestamp_available": False}


def observation_metadata(
    result_record: Mapping[str, Any],
    *,
    regime_id: str,
    burst_flag: bool,
    job_role: str,
    analysis_role: str,
    previous_platform_timestamp: str | None = None,
) -> dict[str, Any]:
    timestamp = platform_timestamp(result_record)
    elapsed: float | None = None
    current = timestamp["platform_timestamp_raw"]
    if current is not None and previous_platform_timestamp is not None:
        try:
            elapsed = (campaign.parse_utc(str(current)) - campaign.parse_utc(previous_platform_timestamp)).total_seconds() / 60.0
        except (TypeError, ValueError):
            elapsed = None
    return {
        **timestamp,
        "regime_id": str(regime_id),
        "burst_flag": bool(burst_flag),
        "job_role": str(job_role),
        "elapsed_minutes_since_previous": elapsed,
        "analysis_role": str(analysis_role),
        "primary_sf_eligible": bool(
            not burst_flag
            and (
                (job_role == "probe_burst" and analysis_role == "primary_sf_only_when_non_event_and_same_regime")
                or (job_role == "anchor_33" and analysis_role == "primary_sf_short_lag_reference")
            )
        ),
        "client_timestamp_substitution_forbidden": True,
    }


def sf_pair_eligible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Freeze no-cross-role, no-cross-regime, no-event primary SF pairing."""
    if left.get("backend_id") != right.get("backend_id"):
        return False
    if left.get("regime_id") != right.get("regime_id"):
        return False
    if left.get("job_role") != right.get("job_role"):
        return False
    if bool(left.get("burst_flag")) or bool(right.get("burst_flag")):
        return False
    if not bool(left.get("primary_sf_eligible")) or not bool(right.get("primary_sf_eligible")):
        return False
    return left.get("analysis_role") == right.get("analysis_role")


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths: list[str] = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            paths.extend(_diff_paths(a, b, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def validate_config_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    validate_v4_config(left)
    validate_v4_config(right)
    differences = _diff_paths(left, right)
    allowed_common_differences = {
        "credential_env",
        "measurement.anchor_job.shots_per_setting",
    }
    forbidden = [
        path
        for path in differences
        if path not in allowed_common_differences and not path.startswith("backend.")
    ]
    if forbidden:
        raise ValueError(f"v4 config pair differs outside backend/credential fields: {forbidden}")
    if left["backend"]["backend_id"] == right["backend"]["backend_id"]:
        raise ValueError("v4 config pair must target two distinct backends")
    return differences


def _protocol_sha256(config: Mapping[str, Any]) -> str:
    common = deepcopy(config)
    common.pop("backend", None)
    common.pop("credential_env", None)
    common["measurement"]["anchor_job"]["shots_per_setting"] = {
        "normalized_by_rule": ANCHOR_SHOTS_SCALING_RULE,
    }
    return campaign.digest_payload(common)


def prepare_v4(config_path: Path, peer_config_path: Path, out: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    peer_config_path = peer_config_path.resolve()
    config = load_config(config_path)
    peer = load_config(peer_config_path)
    differences = validate_config_pair(config, peer)
    out = out.resolve()
    manifest_path = out / "campaign_manifest.json"
    anchor = build_anchor_programs(config)
    probes = build_probe_programs(config)
    anchor_public = [{key: value for key, value in row.items() if key != "qcis"} for row in anchor]
    probe_public = [{key: value for key, value in row.items() if key != "qcis"} for row in probes]
    manifest = {
        "schema": "xa202609_b4_drift_campaign_v4_manifest",
        "profile": PROFILE,
        "campaign_id": config["campaign_id"],
        "backend_id": config["backend"]["backend_id"],
        "credential_source": {"environment_variable": config["credential_env"], "credential_stored": False},
        "config_path": str(config_path),
        "config_sha256": campaign.digest_file(config_path),
        "peer_config_path": str(peer_config_path),
        "peer_config_sha256": campaign.digest_file(peer_config_path),
        "collector_source_sha256": campaign.digest_file(Path(__file__).resolve()),
        "config_pair_difference_paths": differences,
        "common_protocol_sha256": _protocol_sha256(config),
        "peer_common_protocol_sha256": _protocol_sha256(peer),
        "cadence_origin_utc": config["cadence_origin_utc"],
        "job_role_enum": ["anchor_33", "probe_burst", "loop", "mirror"],
        "role_shots_per_setting": {
            "anchor_33": int(config["measurement"]["anchor_job"]["shots_per_setting"]),
            "probe_burst": PROBE_SHOTS_PER_SETTING,
        },
        "anchor_shots_per_setting_rule": ANCHOR_SHOTS_SCALING_RULE,
        "anchor_settings": len(anchor),
        "probe_settings_per_job": len(probes),
        "anchor_program_manifest_sha256": campaign.digest_payload(anchor_public),
        "probe_program_manifest_sha256": campaign.digest_payload(probe_public),
        "reference_positions_zero_indexed": list(REFERENCE_POSITIONS),
        "probe_target_minutes": list(BURST_TARGET_MINUTES),
        "lag_axis_mapping": config["lag_axis_mapping"],
        "pairing_rule": "same backend + same regime_id + same job_role + same analysis_role; never cross role",
        "anchor_pacing_status": "T-B6 dry-run must calibrate actual per-setting pacing; v3 lags 23/46/67 seconds are not constants",
        "event_data_role": "supportive_only_excluded_from_primary_sf",
        "hardware_submission": False,
    }
    if manifest["common_protocol_sha256"] != manifest["peer_common_protocol_sha256"]:
        raise AssertionError("paired configs do not share one protocol payload")
    if manifest_path.exists():
        existing = campaign.load_json(manifest_path)
        immutable = (
            "config_sha256",
            "peer_config_sha256",
            "collector_source_sha256",
            "common_protocol_sha256",
            "backend_id",
            "anchor_program_manifest_sha256",
            "probe_program_manifest_sha256",
        )
        if any(existing.get(key) != manifest.get(key) for key in immutable):
            raise ValueError("existing v4 manifest conflicts with config pair")
        return existing
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    campaign.write_json_new(manifest_path, manifest)
    campaign.CampaignStore(out)
    return manifest


def platform_from_config(config: Mapping[str, Any]) -> Any:
    credential_env = str(config["credential_env"])
    key = os.environ.get(credential_env)
    if not key:
        raise RuntimeError(f"{credential_env} is not set in this process")
    from cqlib.quantum_platform import TianYanPlatform

    return TianYanPlatform(login_key=key, auto_login=True, machine_name=str(config["backend"]["backend_id"]))


def submit_probe_job(
    config_path: Path,
    peer_config_path: Path,
    out: Path,
    *,
    planned_target_utc: str,
    regime_id: str,
    burst_flag: bool,
    confirm_hardware: bool,
    platform_factory: Callable[[Mapping[str, Any]], Any] = platform_from_config,
) -> dict[str, Any]:
    """Submit exactly one two-setting probe job; hardware requires explicit flag."""
    config = load_config(config_path.resolve())
    return _submit_role_job(
        config_path,
        peer_config_path,
        out,
        programs=build_probe_programs(config),
        shots_per_setting=PROBE_SHOTS_PER_SETTING,
        job_role="probe_burst",
        planned_target_utc=planned_target_utc,
        regime_id=regime_id,
        burst_flag=burst_flag,
        confirm_hardware=confirm_hardware,
        platform_factory=platform_factory,
    )


def submit_anchor_job(
    config_path: Path,
    peer_config_path: Path,
    out: Path,
    *,
    planned_target_utc: str,
    regime_id: str,
    confirm_hardware: bool,
    platform_factory: Callable[[Mapping[str, Any]], Any] = platform_from_config,
) -> dict[str, Any]:
    """Submit one 33-setting anchor job at the frozen backend-scaled shots."""
    config = load_config(config_path.resolve())
    return _submit_role_job(
        config_path,
        peer_config_path,
        out,
        programs=build_anchor_programs(config),
        shots_per_setting=int(config["measurement"]["anchor_job"]["shots_per_setting"]),
        job_role="anchor_33",
        planned_target_utc=planned_target_utc,
        regime_id=regime_id,
        burst_flag=False,
        confirm_hardware=confirm_hardware,
        platform_factory=platform_factory,
    )


def _submit_role_job(
    config_path: Path,
    peer_config_path: Path,
    out: Path,
    *,
    programs: Sequence[Mapping[str, Any]],
    shots_per_setting: int,
    job_role: str,
    planned_target_utc: str,
    regime_id: str,
    burst_flag: bool,
    confirm_hardware: bool,
    platform_factory: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    if job_role not in {"anchor_33", "probe_burst", "loop", "mirror"}:
        raise ValueError("unknown v4 job_role")
    manifest = prepare_v4(config_path, peer_config_path, out)
    config = load_config(config_path.resolve())
    planned = campaign.iso(campaign.parse_utc(planned_target_utc))
    identifier = campaign.digest_payload({
        "campaign_id": config["campaign_id"],
        "backend_id": config["backend"]["backend_id"],
        "job_role": job_role,
        "planned_target_utc": planned,
        "burst_flag": bool(burst_flag),
    })[:20].lower()
    store = campaign.CampaignStore(out.resolve())
    existing = store.by_snapshot(identifier)
    if existing:
        return {
            "status": "already_recorded_no_resubmission",
            "snapshot_id": identifier,
            "existing_events": [row["event"] for row in existing],
            "will_submit_hardware": False,
        }
    summary = {
        "will_submit_hardware": bool(confirm_hardware),
        "snapshot_id": identifier,
        "backend_id": config["backend"]["backend_id"],
        "job_role": job_role,
        "planned_target_utc": planned,
        "regime_id": str(regime_id),
        "burst_flag": bool(burst_flag),
        "elapsed_minutes_since_previous": None,
        "primary_sf_eligible": not burst_flag and job_role in {"anchor_33", "probe_burst"},
        "settings": len(programs),
        "shots_per_setting": int(shots_per_setting),
        "total_shots": len(programs) * int(shots_per_setting),
        "config_sha256": manifest["config_sha256"],
        "peer_config_sha256": manifest["peer_config_sha256"],
    }
    if not confirm_hardware:
        return summary
    platform = platform_factory(config)
    telemetry = campaign.capture_calibration_metadata(platform, str(config["backend"]["backend_id"]))
    previous_regime = next((row for row in reversed(store.records) if row.get("event") == "calibration_regime"), None)
    transition = regime_transition(previous_regime, calibration_time_raw(telemetry))
    if previous_regime is None:
        transition["regime_id"] = str(regime_id)
    store.append("calibration_regime", {
        "snapshot_id": identifier,
        "backend_id": config["backend"]["backend_id"],
        **transition,
        "telemetry": telemetry,
        "job_role": job_role,
        "burst_flag": bool(burst_flag or transition["flipped"]),
        "elapsed_minutes_since_previous": None,
    })
    effective_burst = bool(burst_flag or transition["flipped"])
    summary.update({
        "regime_id": transition["regime_id"],
        "burst_flag": effective_burst,
        "primary_sf_eligible": not effective_burst and job_role in {"anchor_33", "probe_burst"},
        "calibration_time_raw": transition["calibration_time_raw"],
        "calibration_time_flipped": transition["flipped"],
    })
    identifiers = platform.submit_experiment(
        circuit=[row["qcis"] for row in programs],
        name=f"XA202609_B4_{job_role.upper()}_{config['backend']['backend_id']}_{identifier[:12]}",
        num_shots=int(shots_per_setting),
        machine_name=str(config["backend"]["backend_id"]),
    )
    if not isinstance(identifiers, list) or len(identifiers) != len(programs):
        raise RuntimeError("platform returned incomplete task-ID list")
    submitted = {
        **summary,
        "event": "submitted",
        "wallclock_submit_utc": campaign.iso(),
        "platform_execution_timestamp_raw": None,
        "execution_timestamp_available": False,
        "event_burst_schedule": (
            build_event_burst_jobs(
                config,
                event_platform_utc=campaign.iso(),
                regime_id=str(transition["regime_id"]),
            )
            if transition["flipped"]
            else []
        ),
        "tasks": [
            {"query_id": str(identifier), **{key: value for key, value in row.items() if key != "qcis"}}
            for identifier, row in zip(identifiers, programs, strict=True)
        ],
    }
    store.append("submitted", submitted)
    return submitted


def collect_pending_jobs(
    config_path: Path,
    peer_config_path: Path,
    out: Path,
    *,
    confirm_hardware: bool,
    max_wait_seconds: int = 300,
    poll_seconds: int = 5,
    platform_factory: Callable[[Mapping[str, Any]], Any] = platform_from_config,
) -> dict[str, Any]:
    """Collect submitted v4 jobs; preserve platform timestamps verbatim or null."""
    prepare_v4(config_path, peer_config_path, out)
    config = load_config(config_path.resolve())
    store = campaign.CampaignStore(out.resolve())
    submitted = [row for row in store.records if row.get("event") == "submitted"]
    terminal_ids = {
        str(row["snapshot_id"])
        for row in store.records
        if row.get("event") in {"collected", "collection_failed"} and row.get("snapshot_id") is not None
    }
    pending = [row for row in submitted if str(row["snapshot_id"]) not in terminal_ids]
    summary = {
        "will_query_hardware": bool(confirm_hardware),
        "backend_id": config["backend"]["backend_id"],
        "pending_jobs": len(pending),
        "collected_jobs": 0,
        "partial_jobs": 0,
        "job_roles": [str(row["job_role"]) for row in pending],
    }
    if not confirm_hardware or not pending:
        return summary
    platform = platform_factory(config)
    previous_platform_timestamp = next(
        (
            row.get("last_platform_timestamp_raw")
            for row in reversed(store.records)
            if row.get("event") == "collected" and row.get("last_platform_timestamp_raw") is not None
        ),
        None,
    )
    physical = list(config["backend"]["physical_qubits"])
    for submitted_row in pending:
        identifier = str(submitted_row["snapshot_id"])
        tasks = list(submitted_row["tasks"])
        query_ids = [str(task["query_id"]) for task in tasks]
        try:
            results = platform.query_experiment(query_ids, max_wait_time=max_wait_seconds, sleep_time=poll_seconds)
            if not isinstance(results, list):
                raise ValueError("platform query did not return a result list")
            by_id = {
                str(result["experimentTaskId"]): result
                for result in results
                if isinstance(result, Mapping) and result.get("experimentTaskId") is not None
            }
            raw_payload = {
                "schema": "xa202609_b4_v4_raw_query",
                "snapshot_id": identifier,
                "backend_id": config["backend"]["backend_id"],
                "job_role": submitted_row["job_role"],
                "regime_id": submitted_row["regime_id"],
                "burst_flag": bool(submitted_row["burst_flag"]),
                "wallclock_retrieve_utc": campaign.iso(),
                "requested_query_ids": query_ids,
                "results": campaign.json_ready(results),
            }
            raw_path = out.resolve() / "raw" / f"{identifier}_query.json"
            if not raw_path.exists():
                campaign.write_json_new(raw_path, raw_payload)
            missing = [query_id for query_id in query_ids if query_id not in by_id]
            if missing:
                store.append("partial", {
                    "snapshot_id": identifier,
                    "backend_id": config["backend"]["backend_id"],
                    "job_role": submitted_row["job_role"],
                    "regime_id": submitted_row["regime_id"],
                    "burst_flag": bool(submitted_row["burst_flag"]),
                    "elapsed_minutes_since_previous": None,
                    "missing_query_ids": missing,
                    "raw_results_path": str(raw_path),
                    "raw_results_sha256": campaign.digest_file(raw_path),
                })
                summary["partial_jobs"] += 1
                continue
            observations: list[dict[str, Any]] = []
            counts: list[np.ndarray] = []
            shots = int(submitted_row["shots_per_setting"])
            for task in tasks:
                result = by_id[str(task["query_id"])]
                metadata = observation_metadata(
                    result,
                    regime_id=str(submitted_row["regime_id"]),
                    burst_flag=bool(submitted_row["burst_flag"]),
                    job_role=str(submitted_row["job_role"]),
                    analysis_role=str(task["analysis_role"]),
                    previous_platform_timestamp=None if previous_platform_timestamp is None else str(previous_platform_timestamp),
                )
                metadata.update({
                    "query_id": str(task["query_id"]),
                    "label": str(task["label"]),
                    "setting_tag": task.get("setting_tag"),
                })
                observations.append(metadata)
                if metadata["platform_timestamp_raw"] is not None:
                    previous_platform_timestamp = metadata["platform_timestamp_raw"]
                counts.append(campaign.result_counts(result, physical, shots))
            counts_path = out.resolve() / "raw" / f"{identifier}_counts.npz"
            if not counts_path.exists():
                np.savez_compressed(
                    counts_path,
                    labels=np.asarray([task["label"] for task in tasks]),
                    kinds=np.asarray([task["kind"] for task in tasks]),
                    job_roles=np.asarray([submitted_row["job_role"]] * len(tasks)),
                    analysis_roles=np.asarray([task["analysis_role"] for task in tasks]),
                    regime_ids=np.asarray([submitted_row["regime_id"]] * len(tasks)),
                    burst_flags=np.asarray([bool(submitted_row["burst_flag"])] * len(tasks), dtype=bool),
                    counts=np.stack(counts),
                    shots=np.asarray(shots),
                    platform_timestamps_raw=np.asarray([
                        "" if row["platform_timestamp_raw"] is None else str(row["platform_timestamp_raw"])
                        for row in observations
                    ]),
                    elapsed_minutes_since_previous=np.asarray([
                        np.nan if row["elapsed_minutes_since_previous"] is None else float(row["elapsed_minutes_since_previous"])
                        for row in observations
                    ]),
                )
            elapsed_values = [row["elapsed_minutes_since_previous"] for row in observations]
            store.append("collected", {
                "snapshot_id": identifier,
                "backend_id": config["backend"]["backend_id"],
                "job_role": submitted_row["job_role"],
                "regime_id": submitted_row["regime_id"],
                "burst_flag": bool(submitted_row["burst_flag"]),
                "elapsed_minutes_since_previous": elapsed_values,
                "last_platform_timestamp_raw": previous_platform_timestamp,
                "observations": observations,
                "raw_results_path": str(raw_path),
                "raw_results_sha256": campaign.digest_file(raw_path),
                "counts_path": str(counts_path),
                "counts_sha256": campaign.digest_file(counts_path),
            })
            summary["collected_jobs"] += 1
        except Exception as error:
            store.append("collection_failed", {
                "snapshot_id": identifier,
                "backend_id": config["backend"]["backend_id"],
                "job_role": submitted_row["job_role"],
                "regime_id": submitted_row["regime_id"],
                "burst_flag": bool(submitted_row["burst_flag"]),
                "elapsed_minutes_since_previous": None,
                "error": f"{type(error).__name__}: {error}",
            })
            raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--peer-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    dry = subparsers.add_parser("dry-run-probe")
    dry.add_argument("--planned-target-utc", required=True)
    dry.add_argument("--regime-id", default="regime-0000")
    dry.add_argument("--burst-flag", action="store_true")
    dry_anchor = subparsers.add_parser("dry-run-anchor")
    dry_anchor.add_argument("--planned-target-utc", required=True)
    dry_anchor.add_argument("--regime-id", default="regime-0000")
    submit = subparsers.add_parser("submit-probe")
    submit.add_argument("--planned-target-utc", required=True)
    submit.add_argument("--regime-id", required=True)
    submit.add_argument("--burst-flag", action="store_true")
    submit.add_argument("--confirm-hardware", action="store_true", required=True)
    submit_anchor = subparsers.add_parser("submit-anchor")
    submit_anchor.add_argument("--planned-target-utc", required=True)
    submit_anchor.add_argument("--regime-id", required=True)
    submit_anchor.add_argument("--confirm-hardware", action="store_true", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--max-wait-seconds", type=int, default=300)
    collect.add_argument("--poll-seconds", type=int, default=5)
    collect.add_argument("--confirm-hardware", action="store_true", required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        payload = prepare_v4(arguments.config, arguments.peer_config, arguments.out)
    elif arguments.command in {"dry-run-probe", "submit-probe"}:
        payload = submit_probe_job(
            arguments.config,
            arguments.peer_config,
            arguments.out,
            planned_target_utc=arguments.planned_target_utc,
            regime_id=arguments.regime_id,
            burst_flag=arguments.burst_flag,
            confirm_hardware=arguments.command == "submit-probe" and bool(arguments.confirm_hardware),
        )
    elif arguments.command in {"dry-run-anchor", "submit-anchor"}:
        payload = submit_anchor_job(
            arguments.config,
            arguments.peer_config,
            arguments.out,
            planned_target_utc=arguments.planned_target_utc,
            regime_id=arguments.regime_id,
            confirm_hardware=arguments.command == "submit-anchor" and bool(arguments.confirm_hardware),
        )
    else:
        payload = collect_pending_jobs(
            arguments.config,
            arguments.peer_config,
            arguments.out,
            confirm_hardware=bool(arguments.confirm_hardware),
            max_wait_seconds=arguments.max_wait_seconds,
            poll_seconds=arguments.poll_seconds,
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
