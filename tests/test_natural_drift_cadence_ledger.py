from __future__ import annotations

import json
from pathlib import Path

from src.adaptive.cadence_ledger import build_cadence_ledger


def test_ledger_preserves_ten_minute_schedule_deviation(tmp_path: Path) -> None:
    journal = tmp_path / "snapshots.jsonl"
    rows = [
        {"event": "submitted", "backend_id": backend, "scheduled_utc": stamp}
        for backend in ("tianyan-287", "tianyan176")
        for stamp in ("2026-08-01T00:00:00+00:00", "2026-08-01T00:25:00+00:00", "2026-08-01T00:50:00+00:00", "2026-08-01T01:00:00+00:00")
    ]
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = tmp_path / "campaign_manifest.json"
    manifest.write_text(json.dumps({"campaign_id": "test", "backend_probe_manifests": [{"backend_id": "tianyan-287"}, {"backend_id": "tianyan176"}]}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cadence_minutes": 25}), encoding="utf-8")
    ledger = build_cadence_ledger(journal, manifest, config)
    assert ledger["strict_nominal_cadence_observed"] is False
    assert ledger["scheduled_gap_frequency_minutes"]["10.0"] == 2
    assert ledger["prediction_time_axis"] == "snapshot_index"
    assert ledger["platform_execution_time_available"] is False
