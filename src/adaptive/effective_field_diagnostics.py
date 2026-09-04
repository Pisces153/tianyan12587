"""Protocol-only effective-field identifiability and uncertainty diagnostics.

This module reads an existing feature corpus but never rewrites it. It is not
a forecast and deliberately contains no Brier, BSS, or model-selection code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


BACKEND_ID = "tianyan-287"
FIELD_NAMES = ("h1", "h2")
ROUTE_A_RULE = {
    "minimum_branch_margin_from_odd_pi_rad": 0.5,
    "maximum_pairwise_standardized_difference": 3.0,
    "maximum_median_inconsistency_to_shot_sigma_ratio": 1.0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _self_hash(value: Mapping[str, Any]) -> str:
    copied = dict(value)
    copied.pop("self_sha256", None)
    payload = json.dumps(copied, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(payload).hexdigest().upper()


def _sign(value: dict[str, Any], *, corpus_sha256: str) -> None:
    value["source_feature_corpus_sha256"] = corpus_sha256
    value["self_hash_scope"] = "canonical JSON excluding self_sha256"
    value["self_sha256"] = _self_hash(value)


def _branch_margin_from_odd_pi(phase: float) -> float:
    """Return shortest angular distance to any odd multiple of pi."""
    return float(abs((phase % (2.0 * np.pi)) - np.pi))


def _field_snapshot(record: Mapping[str, Any], field: str) -> tuple[float, float, list[dict[str, float]]]:
    try:
        state = record["effective_field_state"]
        combined = float(state[field]["value"])
        shot_sigma = float(state[field]["shot_sigma"])
        per_time = state["per_time"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing effective-field diagnostic inputs for {field}") from error
    if not np.isfinite(combined) or not np.isfinite(shot_sigma) or shot_sigma < 0.0:
        raise ValueError(f"Non-finite effective-field diagnostic input for {field}")
    if not isinstance(per_time, Sequence) or isinstance(per_time, (str, bytes)) or len(per_time) != 3:
        raise ValueError("Effective-field diagnostic requires exactly three anchor times")
    rows: list[dict[str, float]] = []
    field_index = FIELD_NAMES.index(field)
    for item in per_time:
        try:
            time = float(item["time"])
            estimate = float(item["estimate"][field])
            covariance = np.asarray(item["covariance_h1_h2"], dtype=np.float64)
            variance = float(covariance[field_index, field_index])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"Malformed per-time effective-field diagnostic input for {field}") from error
        if not np.isfinite(time) or not np.isfinite(estimate) or not np.isfinite(variance) or variance <= 0.0:
            raise ValueError(f"Non-finite per-time effective-field diagnostic input for {field}")
        phase = 2.0 * combined * time
        rows.append({
            "time": time,
            "per_time_estimate": estimate,
            "per_time_shot_sigma": float(np.sqrt(variance)),
            "phase_2_h_t_rad": phase,
            "signal_sensitivity_abs_d_sin_2ht_dh": float(abs(2.0 * time * np.cos(phase))),
            "branch_margin_from_odd_pi_rad": _branch_margin_from_odd_pi(phase),
        })
    return combined, shot_sigma, rows


def _pairwise_consistency(rows: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
    pairs: list[dict[str, float]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            estimate_difference = float(rows[left]["per_time_estimate"] - rows[right]["per_time_estimate"])
            sigma = float(np.hypot(rows[left]["per_time_shot_sigma"], rows[right]["per_time_shot_sigma"]))
            pairs.append({
                "left_time": float(rows[left]["time"]),
                "right_time": float(rows[right]["time"]),
                "estimate_difference": estimate_difference,
                "combined_shot_sigma": sigma,
                "absolute_standardized_difference": abs(estimate_difference) / sigma,
            })
    return pairs


def analyze_effective_field_identifiability(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return signed-ready identifiability and corrected-sigma artifacts for T287."""
    selected = sorted(
        (record for record in records if record.get("backend_id") == BACKEND_ID),
        key=lambda record: int(record["snapshot_index"]),
    )
    if not selected:
        raise ValueError("Effective-field diagnostic requires at least one tianyan-287 snapshot")

    identifiability_rows: list[dict[str, Any]] = []
    sigma_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for field in FIELD_NAMES:
        field_rows: list[dict[str, Any]] = []
        for record in selected:
            combined, shot_sigma, per_time = _field_snapshot(record, field)
            pairwise = _pairwise_consistency(per_time)
            estimates = np.asarray([row["per_time_estimate"] for row in per_time], dtype=np.float64)
            inconsistency_sigma = float(np.std(estimates, ddof=1))
            conservative_sigma = float(np.hypot(shot_sigma, inconsistency_sigma))
            ratio = inconsistency_sigma / shot_sigma if shot_sigma > 0.0 else float("inf")
            max_z = max((float(row["absolute_standardized_difference"]) for row in pairwise), default=0.0)
            min_margin = min(float(row["branch_margin_from_odd_pi_rad"]) for row in per_time)
            field_row = {
                "snapshot_index": int(record["snapshot_index"]),
                "snapshot_id": record.get("snapshot_id"),
                "combined_h_estimate": combined,
                "shot_sigma": shot_sigma,
                "inconsistency_sigma": inconsistency_sigma,
                "conservative_sigma": conservative_sigma,
                "inconsistency_to_shot_sigma_ratio": ratio,
                "per_time": per_time,
                "pairwise_consistency": pairwise,
                "minimum_branch_margin_from_odd_pi_rad": min_margin,
                "maximum_pairwise_standardized_difference": max_z,
            }
            field_rows.append(field_row)
            sigma_rows.append({
                "backend_id": BACKEND_ID,
                "field": field,
                "snapshot_index": field_row["snapshot_index"],
                "snapshot_id": field_row["snapshot_id"],
                "shot_sigma": shot_sigma,
                "inconsistency_sigma": inconsistency_sigma,
                "conservative_sigma": conservative_sigma,
                "combination": "hypot(shot_sigma, inconsistency_sigma)",
            })
        minimum_margin = min(float(row["minimum_branch_margin_from_odd_pi_rad"]) for row in field_rows)
        maximum_z = max(float(row["maximum_pairwise_standardized_difference"]) for row in field_rows)
        median_ratio = float(np.median([row["inconsistency_to_shot_sigma_ratio"] for row in field_rows]))
        eligible = bool(
            minimum_margin >= ROUTE_A_RULE["minimum_branch_margin_from_odd_pi_rad"]
            and maximum_z <= ROUTE_A_RULE["maximum_pairwise_standardized_difference"]
            and median_ratio <= ROUTE_A_RULE["maximum_median_inconsistency_to_shot_sigma_ratio"]
        )
        summaries[field] = {
            "n_snapshots": len(field_rows),
            "minimum_branch_margin_from_odd_pi_rad": minimum_margin,
            "maximum_pairwise_standardized_difference": maximum_z,
            "median_inconsistency_to_shot_sigma_ratio": median_ratio,
            "route_a_eligible": eligible,
        }
        identifiability_rows.append({"field": field, "snapshots": field_rows})

    route_a_eligible = all(summary["route_a_eligible"] for summary in summaries.values())
    identifiability = {
        "analysis_task": "effective_field_identifiability_v3_protocol_diagnostic",
        "generated_at_utc": _utc_now(),
        "backend_id": BACKEND_ID,
        "source_corpus_role": "sealed_v2_protocol_diagnostic_only_not_v3_forecast_corpus",
        "criterion": ROUTE_A_RULE,
        "summary": summaries,
        "selected_route": "A_reselect_anchors" if route_a_eligible else "B_readout_only",
        "route_a_eligible": route_a_eligible,
        "route_b_statement": "h1_effective and h2_effective are not T7 forecast targets under this protocol." if not route_a_eligible else None,
        "fields": identifiability_rows,
    }
    sigma = {
        "analysis_task": "effective_field_sigma_v3_protocol_diagnostic",
        "generated_at_utc": _utc_now(),
        "backend_id": BACKEND_ID,
        "source_corpus_role": "sealed_v2_protocol_diagnostic_only_not_v3_forecast_corpus",
        "definition": {
            "shot_component": "effective_field_state[field].shot_sigma",
            "inconsistency_component": "sample standard deviation of three per_time estimates, ddof=1",
            "conservative_combination": "hypot(shot_sigma, inconsistency_sigma)",
        },
        "rows": sigma_rows,
    }
    return identifiability, sigma


def write_effective_field_diagnostics(corpus_path: Path, output_root: Path) -> dict[str, Any]:
    """Write non-overwriting diagnostic artifacts and return identifiability output."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic artifacts: {output_root}")
    corpus_bytes = corpus_path.read_bytes()
    records = [json.loads(line) for line in corpus_bytes.decode("utf-8").splitlines() if line.strip()]
    identifiability, sigma = analyze_effective_field_identifiability(records)
    corpus_sha256 = sha256(corpus_bytes).hexdigest().upper()
    _sign(identifiability, corpus_sha256=corpus_sha256)
    _sign(sigma, corpus_sha256=corpus_sha256)
    output_root.mkdir(parents=True)
    (output_root / "effective_field_identifiability.json").write_text(
        json.dumps(identifiability, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "effective_field_sigma_artifact.json").write_text(
        json.dumps(sigma, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identifiability
