#!/usr/bin/env python3
"""Run B9 stage 1: frozen T287 structure-function estimation.

This loader joins the append-only T287 campaign journal to the verified
platform task-time ledger.  Cross-job time comes only from platform
``runStartTime``/``finishTime``.  The four anchor references share one
platform job interval, so their within-job relative axis follows the frozen
positions [0, 11, 22, 32] and the frozen T-B6 setting-duration model.

The script does not evaluate the sensing-economics map, read T176 quarantine,
or submit hardware work.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_sensing_economics
from scripts import build_b4_platform_time_ledger as platform_time_ledger
from scripts import drift_campaign
from scripts import simulate_b4_design_power as design_power
from src.adaptive import sensing_economics


SCHEMA = "b4_b9_t287_sf_v1"
PRIMARY_SF_ROLES = {
    "primary_sf_short_lag_reference",
    "primary_sf_only_when_non_event_and_same_regime",
}
REFERENCE_POSITIONS = {
    "interleaved_reference_1": 0,
    "interleaved_reference_2": 11,
    "interleaved_reference_3": 22,
    "interleaved_reference_4": 32,
}
READOUT_SUCCESS_INDEX = {
    "readout_all_zero": 0,
    "readout_all_one": 63,
}
CHANNEL_LABEL = {
    "e0_readout_all_zero": "readout_all_zero",
    "e1_readout_all_one": "readout_all_one",
}
FROZEN_PATHS = (
    "config/b4_drift_campaign_v4_tianyan287.json",
    "scripts/analyze_sensing_economics.py",
    "scripts/simulate_b4_design_power.py",
    "src/adaptive/sensing_economics.py",
)


def digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest().upper()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def reference_error_probability(counts: Sequence[int | float]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (64,) or float(values.sum()) <= 0.0:
        raise ValueError("anchor reference requires one 64-outcome count vector")
    indices = np.arange(64, dtype=np.uint8)
    q0 = 1.0 - 2.0 * ((indices >> 5) & 1)
    q1 = 1.0 - 2.0 * ((indices >> 4) & 1)
    correlation = float(np.dot(values, q0 * q1) / values.sum())
    return float((1.0 - correlation) / 2.0)


def readout_error_probability(label: str, counts: Sequence[int | float], shots: int) -> float:
    if label not in READOUT_SUCCESS_INDEX:
        raise ValueError(f"unknown readout label: {label}")
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (64,) or shots <= 1 or not np.isclose(values.sum(), shots):
        raise ValueError("readout probe requires one complete 64-outcome count vector")
    return float(1.0 - values[READOUT_SUCCESS_INDEX[label]] / shots)


def frozen_anchor_setting_seconds(config: Mapping[str, Any]) -> float:
    shots = float(config["measurement"]["anchor_job"]["shots_per_setting"])
    timing = config["backend"]["tb6_measured_timing"]
    rate = float(timing["effective_shots_per_second"])
    overhead = float(timing["fixed_overhead_seconds_per_setting"])
    if shots <= 0.0 or rate <= 0.0 or overhead < 0.0:
        raise ValueError("invalid frozen anchor timing parameters")
    return shots / rate + overhead


def effective_observation_time(
    *,
    label: str,
    execution_start_utc: datetime,
    execution_end_utc: datetime,
    anchor_setting_seconds: float,
) -> tuple[datetime, str, float | None]:
    if execution_end_utc < execution_start_utc:
        raise ValueError("platform execution end precedes start")
    if label in REFERENCE_POSITIONS:
        offset = float(REFERENCE_POSITIONS[label]) * float(anchor_setting_seconds)
        return (
            execution_start_utc + timedelta(seconds=offset),
            "platform runStartTime plus frozen within-job reference-position model",
            offset,
        )
    if label in READOUT_SUCCESS_INDEX:
        midpoint = execution_start_utc + (execution_end_utc - execution_start_utc) / 2
        return midpoint, "midpoint of platform runStartTime and finishTime", None
    raise ValueError(f"unknown primary-SF label: {label}")


def verify_stage1_freeze(manifest_path: Path, config_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["path"]): str(row["sha256"]).upper() for row in manifest["files"]}
    actual_paths = {
        "config/b4_drift_campaign_v4_tianyan287.json": config_path,
        "scripts/analyze_sensing_economics.py": ROOT / "scripts" / "analyze_sensing_economics.py",
        "scripts/simulate_b4_design_power.py": ROOT / "scripts" / "simulate_b4_design_power.py",
        "src/adaptive/sensing_economics.py": ROOT / "src" / "adaptive" / "sensing_economics.py",
    }
    rows: list[dict[str, Any]] = []
    for relative in FROZEN_PATHS:
        if relative not in expected:
            raise ValueError(f"Stage-1 manifest lacks frozen path: {relative}")
        path = actual_paths[relative]
        actual = digest_file(path)
        row = {
            "path": relative,
            "resolved_path": str(path.resolve()),
            "expected_sha256": expected[relative],
            "actual_sha256": actual,
            "matched": actual == expected[relative],
        }
        rows.append(row)
    mismatches = [row for row in rows if not row["matched"]]
    if mismatches:
        raise ValueError(f"Stage-1 frozen-file hash mismatch: {mismatches}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": digest_file(manifest_path),
        "files": rows,
        "passed": True,
    }


def _verified_ledger_index(ledger_path: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    verification = platform_time_ledger.verify_ledger_artifact(ledger_path)
    if not verification["valid"]:
        raise ValueError(f"platform task-time ledger failed verification: {verification['issues']}")
    ledger = verification["ledger"]
    entries = list(ledger.get("entries", []))
    index = {str(entry["query_id"]): entry for entry in entries}
    if len(index) != len(entries):
        raise ValueError("platform task-time ledger contains duplicate query IDs")
    return index, verification


def collect_primary_observations(
    *,
    campaign_root: Path,
    config: Mapping[str, Any],
    ledger_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = drift_campaign.CampaignStore(campaign_root).records
    collected = [row for row in records if row.get("event") == "collected"]
    anchor_setting_seconds = frozen_anchor_setting_seconds(config)
    observations: list[dict[str, Any]] = []
    counts_files: list[dict[str, str]] = []
    jobs_used: set[str] = set()
    for collected_row in collected:
        metadata_rows = list(collected_row.get("observations", []))
        eligible_indices = [
            index
            for index, row in enumerate(metadata_rows)
            if (
                str(row.get("analysis_role", "")) in PRIMARY_SF_ROLES
                and bool(row.get("primary_sf_eligible"))
                and not bool(row.get("burst_flag"))
            )
        ]
        if not eligible_indices:
            continue
        counts_path = Path(str(collected_row["counts_path"]))
        actual_counts_sha256 = digest_file(counts_path)
        expected_counts_sha256 = str(collected_row["counts_sha256"]).upper()
        if actual_counts_sha256 != expected_counts_sha256:
            raise ValueError(f"counts SHA-256 mismatch: {counts_path}")
        counts_files.append({"path": str(counts_path.resolve()), "sha256": actual_counts_sha256})
        with np.load(counts_path, allow_pickle=False) as archive:
            labels = [str(value) for value in archive["labels"]]
            counts = np.asarray(archive["counts"], dtype=np.int64)
            shots = int(np.asarray(archive["shots"]).item())
        if counts.shape != (len(labels), 64) or len(metadata_rows) != len(labels):
            raise ValueError(f"campaign metadata/count shape mismatch: {counts_path}")
        jobs_used.add(str(collected_row["snapshot_id"]))
        for index in eligible_indices:
            metadata = metadata_rows[index]
            label = str(metadata["label"])
            if labels[index] != label:
                raise ValueError(f"campaign label order mismatch: {counts_path}")
            query_id = str(metadata["query_id"])
            ledger_entry = ledger_index.get(query_id)
            if ledger_entry is None:
                raise ValueError(f"missing platform time for query ID: {query_id}")
            execution_start = parse_utc(str(ledger_entry["execution_start_time_utc"]))
            execution_end = parse_utc(str(ledger_entry["execution_end_time_utc"]))
            effective_time, timing_source, modeled_offset = effective_observation_time(
                label=label,
                execution_start_utc=execution_start,
                execution_end_utc=execution_end,
                anchor_setting_seconds=anchor_setting_seconds,
            )
            if label in REFERENCE_POSITIONS:
                value = reference_error_probability(counts[index])
                channels = sorted(CHANNEL_LABEL)
                instrument = "anchor_33"
            elif label in READOUT_SUCCESS_INDEX:
                value = readout_error_probability(label, counts[index], shots)
                channels = [channel for channel, expected_label in CHANNEL_LABEL.items() if expected_label == label]
                instrument = "probe_burst"
            else:
                raise ValueError(f"unexpected primary-SF label: {label}")
            observations.append({
                "snapshot_id": str(collected_row["snapshot_id"]),
                "query_id": query_id,
                "label": label,
                "analysis_role": str(metadata["analysis_role"]),
                "analysis_channels": channels,
                "instrument_id": instrument,
                "regime_id": str(metadata["regime_id"]),
                "burst_flag": False,
                "shots": shots,
                "value": value,
                "platform_execution_start_utc": iso_utc(execution_start),
                "platform_execution_end_utc": iso_utc(execution_end),
                "platform_runtime_seconds": float(ledger_entry["runtime_seconds"]),
                "effective_observation_time_utc": iso_utc(effective_time),
                "effective_time_source": timing_source,
                "within_job_modeled_offset_seconds": modeled_offset,
            })
    observations.sort(key=lambda row: (row["effective_observation_time_utc"], row["query_id"]))
    if len({row["query_id"] for row in observations}) != len(observations):
        raise ValueError("duplicate query IDs in primary-SF observation set")
    return observations, {
        "campaign_journal_records": len(records),
        "collected_jobs": len(collected),
        "primary_sf_jobs": len(jobs_used),
        "counts_files": counts_files,
        "anchor_setting_seconds": anchor_setting_seconds,
        "anchor_reference_offsets_seconds": {
            label: float(position) * anchor_setting_seconds
            for label, position in REFERENCE_POSITIONS.items()
        },
    }


def minimum_eligible_lag_seconds(rows: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, float]]:
    minimum_by_instrument: dict[str, float] = {}
    for instrument in sorted({str(row["instrument_id"]) for row in rows}):
        block = [row for row in rows if str(row["instrument_id"]) == instrument]
        times = np.asarray([parse_utc(str(row["effective_observation_time_utc"])).timestamp() for row in block])
        regimes = np.asarray([str(row["regime_id"]) for row in block])
        left, right = np.triu_indices(len(block), k=1)
        eligible = regimes[left] == regimes[right]
        lag = times[right[eligible]] - times[left[eligible]]
        positive = lag[lag > 0.0]
        if positive.size:
            minimum_by_instrument[instrument] = float(np.min(positive))
    if not minimum_by_instrument:
        raise ValueError("no positive same-instrument, same-regime lag")
    return min(minimum_by_instrument.values()), minimum_by_instrument


def analyze_channel(observations: Sequence[Mapping[str, Any]], channel: str) -> dict[str, Any]:
    selected = [row for row in observations if channel in row["analysis_channels"]]
    selected.sort(key=lambda row: (row["effective_observation_time_utc"], row["query_id"]))
    if len(selected) < 3:
        raise ValueError(f"insufficient observations for channel: {channel}")
    absolute = np.asarray([
        parse_utc(str(row["effective_observation_time_utc"])).timestamp()
        for row in selected
    ])
    times = absolute - absolute[0]
    values = np.asarray([float(row["value"]) for row in selected])
    shots = np.asarray([int(row["shots"]) for row in selected])
    regimes = np.asarray([str(row["regime_id"]) for row in selected])
    bursts = np.asarray([bool(row["burst_flag"]) for row in selected])
    instruments = np.asarray([str(row["instrument_id"]) for row in selected])
    minimum_lag, minimum_by_instrument = minimum_eligible_lag_seconds(selected)
    edges = design_power.lag_edges(times, minimum_lag)
    variance_gate = sensing_economics.variance_component_gate(
        values,
        shots,
        regimes,
        bursts,
        instruments,
    )
    sf_rows = sensing_economics.structure_function(
        values,
        times,
        shots,
        regimes,
        bursts,
        edges,
        instrument_ids=instruments,
    )
    ou_fit = (
        sensing_economics.fit_ou_structure(sf_rows)
        if variance_gate.passed
        else sensing_economics.OUFit(False, None, None, None, None, None, None, len(sf_rows))
    )
    pair_counts: Counter[str] = Counter()
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            if (
                instruments[left] == instruments[right]
                and regimes[left] == regimes[right]
                and not bursts[left]
                and not bursts[right]
            ):
                pair_counts[str(instruments[left])] += 1
    return {
        "channel": channel,
        "readout_label": CHANNEL_LABEL[channel],
        "observation_count": len(selected),
        "observation_counts_by_instrument": dict(sorted(Counter(instruments).items())),
        "eligible_pair_counts_by_instrument": dict(sorted(pair_counts.items())),
        "minimum_eligible_lag_seconds": minimum_lag,
        "minimum_eligible_lag_seconds_by_instrument": minimum_by_instrument,
        "lag_edges_seconds": [float(value) for value in edges],
        "effective_time_start_utc": str(selected[0]["effective_observation_time_utc"]),
        "effective_time_end_utc": str(selected[-1]["effective_observation_time_utc"]),
        "effective_time_span_seconds": float(times[-1]),
        "variance_gate": variance_gate.__dict__,
        "structure_function": sf_rows,
        "retained_sf_bins": len(sf_rows),
        "ou_fit": ou_fit.__dict__,
        "power_law_fit": analyze_sensing_economics.fit_power_law(sf_rows),
        "query_ids": [str(row["query_id"]) for row in selected],
        "method_boundary": "same instrument, same regime, non-burst pairs only; no cross-role pairs",
    }


def build_report(
    *,
    campaign_root: Path,
    config_path: Path,
    ledger_path: Path,
    stage1_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config["backend"]["backend_id"]) != "tianyan-287":
        raise ValueError("B9 stage 1 accepts T287 config only")
    freeze_audit = verify_stage1_freeze(stage1_manifest_path, config_path)
    ledger_index, ledger_verification = _verified_ledger_index(ledger_path)
    observations, campaign_audit = collect_primary_observations(
        campaign_root=campaign_root,
        config=config,
        ledger_index=ledger_index,
    )
    target_query_ids = set(str(value) for value in ledger_verification["ledger"]["target_query_ids"])
    observed_query_ids = {str(row["query_id"]) for row in observations}
    missing = sorted(target_query_ids - observed_query_ids)
    unexpected = sorted(observed_query_ids - target_query_ids)
    if missing or unexpected or len(observations) != 48:
        raise ValueError({
            "reason": "primary-SF target set mismatch",
            "expected": len(target_query_ids),
            "observed": len(observations),
            "missing": missing,
            "unexpected": unexpected,
        })
    platform_starts = [parse_utc(str(row["platform_execution_start_utc"])) for row in observations]
    platform_ends = [parse_utc(str(row["platform_execution_end_utc"])) for row in observations]
    unique_job_intervals = {
        (str(row["snapshot_id"]), str(row["platform_execution_start_utc"]), str(row["platform_execution_end_utc"]))
        for row in observations
    }
    channels = {channel: analyze_channel(observations, channel) for channel in sorted(CHANNEL_LABEL)}
    report = {
        "schema": SCHEMA,
        "status": "completed_t287_sf",
        "b9_stage": 1,
        "b9_stage_name": "T287 SF",
        "hardware_submission_performed": False,
        "t176_quarantine_read": False,
        "scope": "T287 primary structure function only; sensing-economics map not evaluated",
        "input_integrity": {
            "stage1_freeze": freeze_audit,
            "campaign_root": str(campaign_root.resolve()),
            "campaign_journal_sha256": digest_file(campaign_root / "snapshots.jsonl"),
            "config_path": str(config_path.resolve()),
            "config_sha256": digest_file(config_path),
            "platform_time_ledger_path": str(ledger_path.resolve()),
            "platform_time_ledger_sha256": ledger_verification["ledger_sha256"],
            "platform_time_freeze_sha256": ledger_verification["freeze_sha256"],
            "target_query_ids_sha256": canonical_sha256(sorted(target_query_ids)),
            "counts_files": campaign_audit["counts_files"],
        },
        "recovery": {
            "expected_query_ids": len(target_query_ids),
            "recovered_query_ids": len(observed_query_ids),
            "missing_query_ids": missing,
            "unexpected_query_ids": unexpected,
            "primary_sf_jobs": campaign_audit["primary_sf_jobs"],
            "unique_platform_job_intervals": len(unique_job_intervals),
            "platform_time_records": len(observations),
        },
        "platform_time_window": {
            "timezone": "UTC",
            "execution_start_min_utc": iso_utc(min(platform_starts)),
            "execution_start_max_utc": iso_utc(max(platform_starts)),
            "execution_end_max_utc": iso_utc(max(platform_ends)),
            "asia_shanghai_execution_start_min": min(platform_starts).astimezone(
                timezone(timedelta(hours=8))
            ).isoformat(timespec="milliseconds"),
            "asia_shanghai_execution_end_max": max(platform_ends).astimezone(
                timezone(timedelta(hours=8))
            ).isoformat(timespec="milliseconds"),
        },
        "timeline_method": {
            "cross_job_axis": "verified platform runStartTime/finishTime only",
            "probe_observation_time": "platform job midpoint",
            "anchor_within_job_axis": "platform runStartTime plus frozen positions [0,11,22,32] times frozen T-B6 setting duration",
            "anchor_setting_seconds": campaign_audit["anchor_setting_seconds"],
            "anchor_reference_offsets_seconds": campaign_audit["anchor_reference_offsets_seconds"],
            "scheduled_target_minutes_used_as_analysis_lag": False,
            "client_wallclock_used": False,
        },
        "observation_inventory": {
            "unique_task_records": len(observations),
            "labels": dict(sorted(Counter(str(row["label"]) for row in observations).items())),
            "analysis_roles": dict(sorted(Counter(str(row["analysis_role"]) for row in observations).items())),
            "instruments": dict(sorted(Counter(str(row["instrument_id"]) for row in observations).items())),
        },
        "channels": channels,
        "next_permitted_action": "Evaluate the frozen T287 sensing-economics map from this SF artifact.",
    }
    return report, observations


def write_outputs(output: Path, report: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite T287 SF artifact: {output}")
    output.mkdir(parents=True)
    report_path = output / "t287_sf_report.json"
    observations_path = output / "t287_sf_observations.csv"
    summary_path = output / "T287_SF_SUMMARY.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_rows: list[dict[str, Any]] = []
    for row in observations:
        csv_rows.append({
            **{key: value for key, value in row.items() if key != "analysis_channels"},
            "analysis_channels": ";".join(str(value) for value in row["analysis_channels"]),
        })
    with observations_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = [
        "# B9 Stage 1 — T287 SF",
        "",
        f"- 平台任务时刻：{report['recovery']['recovered_query_ids']}/{report['recovery']['expected_query_ids']}；缺失 0。",
        f"- 独立平台 job 时间窗：{report['recovery']['unique_platform_job_intervals']}。",
        f"- 平台原始范围（北京时间）：{report['platform_time_window']['asia_shanghai_execution_start_min']} 至 {report['platform_time_window']['asia_shanghai_execution_end_max']}。",
        "- 跨 job 只用平台时刻；anchor job 内参考点用冻结位置模型；客户端时钟与计划目标分钟未进入 lag。",
        "- T176 未读取；判据地图未运行。",
        "",
    ]
    for channel, result in report["channels"].items():
        variance = result["variance_gate"]
        ou = result["ou_fit"]
        power = result["power_law_fit"]
        summary.extend([
            f"## {channel}",
            "",
            f"- observations: {result['observation_count']}；SF bins: {result['retained_sf_bins']}。",
            f"- variance gate: {variance['passed']}；p={float(variance['p_value']):.6g}。",
            f"- OU fit: {ou['ok']}；tau={ou['tau_seconds']} s；process variance={ou['process_variance']}。",
            f"- power-law alpha: {power.get('alpha')} ± {power.get('se_alpha')}。",
            "",
        ])
    summary.append("下一步：冻结 T287 sensing-economics map。\n")
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    artifact_files = [report_path, observations_path, summary_path]
    manifest = {
        "schema": "b4_b9_t287_sf_artifact_manifest_v1",
        "files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in artifact_files
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t287-campaign-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--platform-time-ledger", type=Path, required=True)
    parser.add_argument(
        "--stage1-manifest",
        type=Path,
        default=ROOT / "docs" / "B4_STAGE1_CURATED_MANIFEST_20260806.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report, observations = build_report(
        campaign_root=arguments.t287_campaign_root,
        config_path=arguments.config,
        ledger_path=arguments.platform_time_ledger,
        stage1_manifest_path=arguments.stage1_manifest,
    )
    write_outputs(arguments.output, report, observations)
    print(json.dumps({
        "status": report["status"],
        "recovered": report["recovery"]["recovered_query_ids"],
        "output": str(arguments.output.resolve()),
        "next": report["next_permitted_action"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
