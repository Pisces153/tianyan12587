"""Analysis-side cadence accounting; never infers platform execution times."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


def _self_hash(payload: Mapping[str, Any]) -> str:
    copied = dict(payload)
    copied.pop("self_sha256", None)
    canonical = json.dumps(copied, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(canonical).hexdigest().upper()


def build_cadence_ledger(journal_path: Path, manifest_path: Path, config_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nominal = int(config["cadence_minutes"])
    required_backends = sorted(str(row["backend_id"]) for row in manifest["backend_probe_manifests"])
    scheduled_by_backend: dict[str, list[datetime]] = defaultdict(list)
    for row in rows:
        if row.get("event") == "submitted" and str(row.get("backend_id")) in required_backends:
            scheduled_by_backend[str(row["backend_id"])].append(datetime.fromisoformat(str(row["scheduled_utc"])))

    intervals: dict[str, list[float]] = {}
    deviations: list[dict[str, Any]] = []
    for backend, stamps in scheduled_by_backend.items():
        ordered = sorted(stamps)
        gaps = [round((later - earlier).total_seconds() / 60.0, 9) for earlier, later in zip(ordered, ordered[1:])]
        intervals[backend] = gaps
        deviations.extend({"backend_id": backend, "previous_scheduled_utc": earlier.isoformat(), "scheduled_utc": later.isoformat(), "observed_schedule_gap_minutes": gap, "nominal_cadence_minutes": nominal} for earlier, later, gap in zip(ordered, ordered[1:], gaps) if gap != nominal)
    observed = [gap for gaps in intervals.values() for gap in gaps]
    result = {
        "task": "natural_drift_cadence_deviation_ledger",
        "campaign_id": manifest["campaign_id"],
        "nominal_cadence_minutes": nominal,
        "required_backends": required_backends,
        "scheduled_gap_frequency_minutes": {str(key): value for key, value in sorted(Counter(observed).items())},
        "scheduled_intervals_by_backend_minutes": intervals,
        "strict_nominal_cadence_observed": bool(observed) and all(gap == nominal for gap in observed),
        "deviations": deviations,
        "prediction_time_axis": "snapshot_index",
        "platform_execution_time_available": False,
        "client_event_times": "provenance only; never used as platform execution timestamps or equal-spacing evidence",
        "wording_guard": "T7 horizon is measured in snapshots. Reports must not call these samples equally spaced 25-minute execution intervals.",
        "frozen_campaign_sources_unmodified": True,
        "self_hash_scope": "canonical JSON excluding self_sha256",
    }
    result["self_sha256"] = _self_hash(result)
    return result
