"""Extract auditable observable-environment proxies from campaign raw counts.

This module deliberately reports measurement-derived proxies only.  It does not
infer unavailable physical sensor values, and it keeps proxy tasks disjoint from
the anchor-derived task label used by the forecasting layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from src.features.pauli import BASIS_ORDER, counts_array_to_pauli15, select_xobs6


FEATURE_VERSION = "t6_observable_environment_proxy_v2"
FORBIDDEN_TERMS = ("temperature", "thermal", "emi", "electromagnetic", "sensor_reading")
NORMAL = NormalDist()
ANCHOR_TIMES = (0.16, 0.31, 0.47)


def _json_hash(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    copied = dict(value)
    copied.pop("self_sha256", None)
    canonical = json.dumps(copied, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(canonical).hexdigest().upper()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_json_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _scalar(value: np.ndarray) -> str:
    return str(value.item())


def _binomial_floor(probability: float, shots: int) -> float:
    probability = float(np.clip(probability, 0.0, 1.0))
    return float(np.sqrt(max(probability * (1.0 - probability), 0.0) / shots))


def _mdd_80(floor: float, *, family_size: int) -> float:
    """Two-sided Bonferroni MDD for one paired snapshot comparison."""
    alpha_per_test = 0.05 / max(family_size, 1)
    z_alpha = NORMAL.inv_cdf(1.0 - alpha_per_test / 2.0)
    z_power = NORMAL.inv_cdf(0.8)
    return float((z_alpha + z_power) * np.sqrt(2.0) * floor)


def _logical_z0z1(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (64,) or values.sum() <= 0:
        raise ValueError("logical correlation requires one 64-outcome count vector")
    indices = np.arange(64, dtype=np.uint8)
    q0 = 1.0 - 2.0 * ((indices >> 5) & 1)
    q1 = 1.0 - 2.0 * ((indices >> 4) & 1)
    return float(np.dot(values, q0 * q1) / values.sum())


def _outcome_signs() -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(64, dtype=np.uint8)
    return 1.0 - 2.0 * ((indices >> 5) & 1), 1.0 - 2.0 * ((indices >> 4) & 1)


def _jeffreys_signed_mean_variance(mean: float, shots: int) -> float:
    """Posterior variance for a +/-1 expectation under Beta(1/2, 1/2)."""
    successes = float(np.clip((mean + 1.0) * shots / 2.0, 0.0, shots))
    alpha, beta = successes + 0.5, shots - successes + 0.5
    return float(4.0 * alpha * beta / ((shots + 1.0) ** 2 * (shots + 2.0)))


def _anchor_counts(labels: np.ndarray, counts: np.ndarray, time_index: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for basis in BASIS_ORDER:
        label = f"anchor_t{time_index}_{basis}"
        matches = np.flatnonzero(labels == label)
        if matches.size != 1:
            raise ValueError(f"Missing or duplicated anchor label: {label}")
        rows.append(counts[int(matches[0])])
    return np.stack(rows)


def _effective_fields_at_time(anchor_counts: np.ndarray, *, time: float, shots: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Infer two local effective fields and their within-time delta covariance.

    This is a measurement-defined effective-field summary, not a claim that a
    complete hardware Hamiltonian or noise channel has been reconstructed.
    """
    values = np.asarray(anchor_counts, dtype=np.float64)
    if values.shape != (len(BASIS_ORDER), 64):
        raise ValueError("One anchor time requires nine 64-outcome count vectors")
    if np.any(values.sum(axis=1) != shots):
        raise ValueError("Anchor counts must have the frozen shot count")
    sign0, sign1 = _outcome_signs()
    signs = (sign0, sign1)
    means: dict[tuple[int, str], float] = {}
    index_by_variable: dict[tuple[int, str], tuple[int, ...]] = {}
    for qubit in (0, 1):
        for axis in "YZ":
            indexes = tuple(index for index, basis in enumerate(BASIS_ORDER) if basis[qubit] == axis)
            if len(indexes) != 3:
                raise AssertionError("Frozen Pauli basis does not provide three local estimates")
            local_means = [float(np.dot(values[index], signs[qubit]) / shots) for index in indexes]
            means[(qubit, axis)] = float(np.mean(local_means))
            index_by_variable[(qubit, axis)] = indexes

    variable_order = ((0, "Y"), (0, "Z"), (1, "Y"), (1, "Z"))
    observation_covariance = np.zeros((4, 4), dtype=np.float64)
    for left_index, left in enumerate(variable_order):
        for right_index, right in enumerate(variable_order):
            if right_index < left_index:
                continue
            shared = set(index_by_variable[left]).intersection(index_by_variable[right])
            if left == right:
                covariance = sum(
                    _jeffreys_signed_mean_variance(float(np.dot(values[index], signs[left[0]]) / shots), shots)
                    for index in shared
                ) / (len(index_by_variable[left]) ** 2)
            elif left[0] == right[0]:
                # Different single-qubit axes are measured in disjoint circuits.
                covariance = 0.0
            else:
                covariance = sum(
                    (
                        float(np.dot(values[index], signs[0] * signs[1]) / shots)
                        - float(np.dot(values[index], signs[left[0]]) / shots)
                        * float(np.dot(values[index], signs[right[0]]) / shots)
                    )
                    / shots
                    for index in shared
                ) / (len(index_by_variable[left]) * len(index_by_variable[right]))
            observation_covariance[left_index, right_index] = covariance
            observation_covariance[right_index, left_index] = covariance

    estimate = np.empty(2, dtype=np.float64)
    jacobian = np.zeros((2, 4), dtype=np.float64)
    for qubit, start in ((0, 0), (1, 2)):
        y, z = means[(qubit, "Y")], means[(qubit, "Z")]
        radius_squared = max(y * y + z * z, 1e-12)
        estimate[qubit] = np.arctan2(-y, z) / (2.0 * time)
        jacobian[qubit, start] = -z / (2.0 * time * radius_squared)
        jacobian[qubit, start + 1] = y / (2.0 * time * radius_squared)
    covariance = jacobian @ observation_covariance @ jacobian.T
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0.0):
        raise ValueError("Effective-field delta covariance is invalid")
    limit = float(np.sqrt(covariance[0, 0] * covariance[1, 1]))
    if abs(float(covariance[0, 1])) >= limit:
        # Sampling arithmetic can produce an endpoint outside the PSD cone by
        # roundoff. Keep the measured sign and move only to its open boundary.
        covariance[0, 1] = covariance[1, 0] = np.copysign(limit * (1.0 - 1e-12), covariance[0, 1])
    return estimate, covariance, {
        "time": float(time),
        "local_pauli_means": {"Y0": means[(0, "Y")], "Z0": means[(0, "Z")], "Y1": means[(1, "Y")], "Z1": means[(1, "Z")]},
        "estimate": {"h1": float(estimate[0]), "h2": float(estimate[1])},
        "covariance_h1_h2": covariance.tolist(),
    }


def _effective_field_state(labels: np.ndarray, counts: np.ndarray, shots: int) -> dict[str, Any]:
    estimates: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for time_index, time in enumerate(ANCHOR_TIMES):
        estimate, covariance, detail = _effective_fields_at_time(
            _anchor_counts(labels, counts, time_index), time=time, shots=shots
        )
        estimates.append(estimate)
        covariances.append(covariance)
        details.append(detail)
    precision_h1 = np.asarray([1.0 / covariance[0, 0] for covariance in covariances])
    precision_h2 = np.asarray([1.0 / covariance[1, 1] for covariance in covariances])
    denominator_h1, denominator_h2 = float(precision_h1.sum()), float(precision_h2.sum())
    combined = np.asarray([
        sum(weight * estimate[0] for weight, estimate in zip(precision_h1, estimates, strict=True)) / denominator_h1,
        sum(weight * estimate[1] for weight, estimate in zip(precision_h2, estimates, strict=True)) / denominator_h2,
    ])
    covariance = np.asarray([
        [1.0 / denominator_h1, sum(weight0 * weight1 * block[0, 1] for weight0, weight1, block in zip(precision_h1, precision_h2, covariances, strict=True)) / (denominator_h1 * denominator_h2)],
        [0.0, 1.0 / denominator_h2],
    ], dtype=np.float64)
    covariance[1, 0] = covariance[0, 1]
    if not np.all(np.linalg.eigvalsh(covariance) > 0.0):
        raise ValueError("Combined effective-field covariance is not positive definite")
    return {
        "definition": "three-time local Y/Z phase inverse; measurement-defined effective field",
        "anchor_times": list(ANCHOR_TIMES),
        "h1": {"value": float(combined[0]), "shot_sigma": float(np.sqrt(covariance[0, 0]))},
        "h2": {"value": float(combined[1]), "shot_sigma": float(np.sqrt(covariance[1, 1]))},
        "covariance_h1_h2": covariance.tolist(),
        "per_time": details,
    }


def _per_qubit_mean(raw: Mapping[str, Any], group: str, field: str, physical_qubits: Sequence[int]) -> tuple[float | None, str | None]:
    try:
        payload = raw[group][field]
        values = payload["param_list"]
        labels = payload["qubit_used"]
        index = {str(label): position for position, label in enumerate(labels)}
        selected = [float(values[index[f"Q{int(qubit)}"]]) for qubit in physical_qubits]
        if not all(np.isfinite(selected)):
            return None, None
        return float(np.mean(selected)), payload.get("unit")
    except (KeyError, TypeError, ValueError, IndexError):
        return None, None


def _platform_metadata(telemetry: Mapping[str, Any] | None, physical_qubits: Sequence[int]) -> dict[str, Any]:
    raw = None
    if isinstance(telemetry, Mapping):
        candidate = telemetry.get("raw")
        if isinstance(candidate, Mapping):
            candidate = candidate.get("download_config")
        if isinstance(candidate, Mapping):
            raw = candidate
    if raw is None:
        return {"available": False, "calibration_timestamp_raw": None, "fields": {}, "reason": "download_config unavailable"}
    extracted: dict[str, Any] = {}
    for name, group, field in (
        ("mean_T1", "qubit", "relatime.T1"),
        ("mean_T2", "qubit", "relatime.T2"),
        ("mean_f01", "qubit", "frequency.f01"),
        ("mean_1q_gate_error", "singleQubit", "gate error"),
    ):
        if "." in field:
            parent, child = field.split(".", 1)
            nested = raw.get(group)
            value, unit = _per_qubit_mean(nested, parent, child, physical_qubits) if isinstance(nested, Mapping) else (None, None)
        else:
            value, unit = _per_qubit_mean(raw, group, field, physical_qubits)
        extracted[name] = {"value": value, "unit": unit}
    overview = raw.get("overview") if isinstance(raw.get("overview"), Mapping) else {}
    for name, field in (("platform_cz_error", "cz_error"), ("platform_readout_error", "readout_error")):
        value = overview.get(field)
        extracted[name] = {"value": float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None, "unit": None}
    return {
        "available": True,
        "calibration_timestamp_raw": raw.get("calibrationTime"),
        "fields": extracted,
        "physical_qubits": [int(qubit) for qubit in physical_qubits],
        "source": "submission_attempt.telemetry.raw.download_config",
    }


def _anchor_task_metric(labels: np.ndarray, counts: np.ndarray, shots: int) -> tuple[float, list[str]]:
    values: list[float] = []
    task_labels: list[str] = []
    for time_index in range(3):
        rows: list[np.ndarray] = []
        for basis in BASIS_ORDER:
            label = f"anchor_t{time_index}_{basis}"
            matches = np.flatnonzero(labels == label)
            if matches.size != 1:
                raise ValueError(f"Missing or duplicated anchor label: {label}")
            rows.append(counts[int(matches[0])])
            task_labels.append(label)
        values.append(float(np.linalg.norm(select_xobs6(counts_array_to_pauli15(np.stack(rows), shots=shots)), ord=2)))
    return float(np.mean(values)), task_labels


def _measurement_features(labels: np.ndarray, counts: np.ndarray, shots: int) -> tuple[dict[str, dict[str, Any]], list[str]]:
    references: list[float] = []
    proxy_labels: list[str] = []
    for copy_index in range(1, 5):
        label = f"interleaved_reference_{copy_index}"
        matches = np.flatnonzero(labels == label)
        if matches.size != 1:
            raise ValueError(f"Missing or duplicated reference label: {label}")
        references.append(_logical_z0z1(counts[int(matches[0])]))
        proxy_labels.append(label)
    readout: dict[str, float] = {}
    for label, expected_index in (("readout_all_zero", 0), ("readout_all_one", 63)):
        matches = np.flatnonzero(labels == label)
        if matches.size != 1:
            raise ValueError(f"Missing or duplicated readout label: {label}")
        success = float(counts[int(matches[0]), expected_index] / shots)
        readout[label] = 1.0 - success
        proxy_labels.append(label)

    reference_mean = float(np.mean(references))
    reference_floor = float(np.sqrt(max(1.0 - reference_mean**2, 0.0) / (4 * shots)))
    reference_spread = float(np.std(references, ddof=1))
    zero_error = readout["readout_all_zero"]
    one_error = readout["readout_all_one"]
    feature_rows = {
        "reference_z0z1": (reference_mean, reference_floor, "derived_from_quantum_probe"),
        "reference_within_batch_spread": (reference_spread, reference_floor, "derived_from_quantum_probe"),
        "readout_all_zero_error": (zero_error, _binomial_floor(zero_error, shots), "derived_from_quantum_probe"),
        "readout_all_one_error": (one_error, _binomial_floor(one_error, shots), "derived_from_quantum_probe"),
        "readout_mean_error": ((zero_error + one_error) / 2.0, np.hypot(_binomial_floor(zero_error, shots), _binomial_floor(one_error, shots)) / 2.0, "derived_from_quantum_probe"),
        "readout_error_asymmetry": (zero_error - one_error, np.hypot(_binomial_floor(zero_error, shots), _binomial_floor(one_error, shots)), "derived_from_quantum_probe"),
    }
    return {
        name: {
            "value": float(value),
            "shot_noise_floor": float(floor),
            "mdd_80": _mdd_80(float(floor), family_size=6),
            "provenance": provenance,
        }
        for name, (value, floor, provenance) in feature_rows.items()
    }, proxy_labels


def feature_spec() -> dict[str, Any]:
    return {
        "version": FEATURE_VERSION,
        "proxy_estimators": [
            "logical_z0z1_from_interleaved_reference",
            "within_batch_reference_spread",
            "all_zero_and_all_one_error_rates",
        ],
        "task_label_estimator": "mean_l2_xobs6_over_three_anchor_times",
        "effective_field_state_estimator": "three_time_local_yz_phase_inverse_with_full_2x2_delta_covariance",
        "platform_metadata_estimator": "selected_qubit_summary_from_submission_download_config",
        "proxy_label_shot_separation": True,
        "normalization": "T287-only constants, frozen only after at least two snapshots",
        "claim_rule": "absolute_delta_over_combined_floor >= 2",
    }


def extract_snapshot(
    counts_path: Path,
    *,
    campaign_id: str,
    backend_id: str,
    scheduled_utc: str,
    snapshot_index: int,
    previous_record: Mapping[str, Any] | None,
    physical_qubits: Sequence[int] = (),
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with np.load(counts_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"], dtype=str)
        counts = np.asarray(archive["counts"], dtype=np.int64)
        shots = int(archive["shots"].item())
        snapshot_id = _scalar(archive["snapshot_id"])
        probe_manifest_sha256 = _scalar(archive["probe_manifest_sha256"])
    if counts.shape != (33, 64) or labels.shape != (33,):
        raise ValueError("Natural-drift raw counts must contain exactly 33 settings by 64 outcomes")
    if np.any(counts.sum(axis=1) != shots):
        raise ValueError("Every setting must have the frozen shot count")

    proxies, proxy_task_labels = _measurement_features(labels, counts, shots)
    label_task_metric, label_task_labels = _anchor_task_metric(labels, counts, shots)
    effective_fields = _effective_field_state(labels, counts, shots)
    if set(proxy_task_labels).intersection(label_task_labels):
        raise AssertionError("Proxy and label task IDs overlap; forecast would be noise-leaky")
    spec = feature_spec()
    spec_hash = _json_hash(spec)
    for name, payload in proxies.items():
        prior = None if previous_record is None else previous_record["observable_environment_proxy"].get(name)
        if prior is None:
            assessment = {"baseline_value": None, "drift_magnitude": None, "ratio_to_floor": None, "claim_permitted": False}
        else:
            delta = float(payload["value"] - float(prior["value"]))
            combined_floor = float(np.hypot(float(payload["shot_noise_floor"]), float(prior["shot_noise_floor"])))
            ratio = abs(delta) / combined_floor if combined_floor > 0 else 0.0
            assessment = {
                "baseline_value": float(prior["value"]),
                "drift_magnitude": delta,
                "ratio_to_floor": float(ratio),
                "claim_permitted": bool(ratio >= 2.0),
            }
        payload["drift"] = assessment

    record = {
        "campaign_id": campaign_id,
        "backend_id": backend_id,
        "snapshot_id": snapshot_id,
        "scheduled_utc": scheduled_utc,
        "snapshot_index": snapshot_index,
        "counts_path": str(counts_path),
        "shots_per_setting": shots,
        "feature_spec_sha256": spec_hash,
        "probe_manifest_sha256": probe_manifest_sha256,
        "observable_environment_proxy": proxies,
        "effective_field_state": effective_fields,
        "platform_calibration_metadata": _platform_metadata(telemetry, physical_qubits),
        "task_label": {
            "name": "anchor_xobs6_l2_mean",
            "value": label_task_metric,
            "task_labels": label_task_labels,
            "feature_lag_audit": "label is contemporaneous and is only used at a future index by forecast evaluation",
        },
        "proxy_task_labels": proxy_task_labels,
        "label_task_labels": label_task_labels,
        "execution_time_available": False,
    }
    assert_clean_vocabulary(record)
    return record


def assert_clean_vocabulary(value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False).lower()
    found = [word for word in FORBIDDEN_TERMS if word in serialized]
    if found:
        raise ValueError(f"Observable proxy output contains forbidden terms: {found}")


def _normalization(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    t287 = [record for record in records if record["backend_id"] == "tianyan-287"]
    names = sorted(t287[0]["observable_environment_proxy"]) if t287 else []
    enough = len(t287) >= 2
    return {
        "source_backend": "tianyan-287",
        "source_snapshot_count": len(t287),
        "frozen": enough,
        "reason": None if enough else "Requires at least two T287 snapshots; T176 must not define its own transfer normalization.",
        "features": {
            name: {
                "mean": float(np.mean([record["observable_environment_proxy"][name]["value"] for record in t287])) if enough else None,
                "std": float(max(np.std([record["observable_environment_proxy"][name]["value"] for record in t287], ddof=1), 1e-12)) if enough else None,
            }
            for name in names
        },
    }


def extract_campaign(
    campaign_root: Path,
    output_root: Path,
    *,
    included_snapshot_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Extract a T6 corpus, optionally from a complete-pair snapshot whitelist.

    The whitelist is analysis-side only.  It never modifies the append-only
    journal and prevents an early single-backend collection from entering a
    dual-backend refresh.
    """
    journal_path = campaign_root / "snapshots.jsonl"
    rows = _read_json_lines(journal_path)
    submitted = {str(row["snapshot_id"]): row for row in rows if row.get("event") == "submitted"}
    telemetry_by_snapshot = {
        str(row["snapshot_id"]): row.get("telemetry")
        for row in rows if row.get("event") == "submission_attempt"
    }
    collected = [row for row in rows if row.get("event") == "collected"]
    source_manifest = json.loads((campaign_root / "campaign_manifest.json").read_text(encoding="utf-8"))
    candidates: list[tuple[datetime, Mapping[str, Any], Mapping[str, Any]]] = []
    for row in collected:
        if included_snapshot_ids is not None and str(row["snapshot_id"]) not in included_snapshot_ids:
            continue
        snapshot = submitted.get(str(row["snapshot_id"]))
        counts_path = Path(str(row["counts_path"]))
        if snapshot is None or not counts_path.is_file():
            continue
        candidates.append((datetime.fromisoformat(str(snapshot["scheduled_utc"])), snapshot, row))
    candidates.sort(key=lambda item: (str(item[1]["backend_id"]), item[0]))

    histories: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for _, snapshot, collected_row in candidates:
        backend_id = str(snapshot["backend_id"])
        history = histories.setdefault(backend_id, [])
        records.append(extract_snapshot(
            Path(str(collected_row["counts_path"])), campaign_id=str(source_manifest["campaign_id"]),
            backend_id=backend_id, scheduled_utc=str(snapshot["scheduled_utc"]), snapshot_index=len(history),
            previous_record=history[-1] if history else None,
            physical_qubits=tuple(int(qubit) for qubit in snapshot.get("physical_qubits", ())),
            telemetry=telemetry_by_snapshot.get(str(snapshot["snapshot_id"])),
        ))
        history.append(records[-1])
    records.sort(key=lambda row: (row["backend_id"], row["snapshot_index"]))

    output_root.mkdir(parents=True, exist_ok=True)
    corpus_path = output_root / f"features_{source_manifest['campaign_id']}.jsonl"
    normalization = _normalization(records)
    _write_json_lines(corpus_path, records)
    feature_corpus_sha256 = sha256(corpus_path.read_bytes()).hexdigest().upper()
    report = {
        "task": "T6_observable_environment_proxy",
        "campaign_id": source_manifest["campaign_id"],
        "feature_spec_sha256": _json_hash(feature_spec()),
        "feature_corpus_sha256": feature_corpus_sha256,
        "collected_snapshot_count": len(records),
        "included_snapshot_id_count": len(included_snapshot_ids) if included_snapshot_ids is not None else len(records),
        "records_by_backend": {backend: len(history) for backend, history in histories.items()},
        "proxy_label_task_disjoint": all(not set(row["proxy_task_labels"]).intersection(row["label_task_labels"]) for row in records),
        "effective_field_state_available": all("effective_field_state" in row for row in records),
        "platform_metadata_available_by_backend": {
            backend: sum(bool(row["platform_calibration_metadata"]["available"]) for row in records if row["backend_id"] == backend)
            for backend in histories
        },
        "normalization_frozen": normalization["frozen"],
        "drift_claim_count": sum(
            bool(feature["drift"]["claim_permitted"])
            for row in records for feature in row["observable_environment_proxy"].values()
        ),
        "execution_time_available": False,
        "self_hash_scope": "canonical JSON excluding self_sha256",
    }
    report["self_sha256"] = _self_hash(report)
    _write_json(output_root / "normalization_artifact.json", normalization)
    _write_json(output_root / "feature_extraction_report.json", report)
    _write_json(output_root / "manifest.json", {
        "campaign_manifest_sha256": sha256((campaign_root / "campaign_manifest.json").read_bytes()).hexdigest(),
        "journal_sha256": sha256(journal_path.read_bytes()).hexdigest(),
        "feature_corpus_sha256": feature_corpus_sha256,
        "feature_spec_sha256": _json_hash(feature_spec()),
    })
    return {"corpus_path": str(corpus_path), "report": report, "normalization_path": str(output_root / "normalization_artifact.json")}
