from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("refresh_natural_drift_analysis", ROOT / "scripts" / "refresh_natural_drift_analysis.py")
assert SPEC and SPEC.loader
refresh = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(refresh)


def test_complete_pairs_exclude_half_collected_slots(tmp_path: Path) -> None:
    (tmp_path / "campaign_manifest.json").write_text(json.dumps({"backend_probe_manifests": [{"backend_id": "tianyan-287"}, {"backend_id": "tianyan176"}]}), encoding="utf-8")
    rows = []
    for stamp, collected_backends in (("2026-08-01T00:00:00+00:00", ("tianyan-287", "tianyan176")), ("2026-08-01T00:25:00+00:00", ("tianyan-287",))):
        for backend in ("tianyan-287", "tianyan176"):
            snapshot_id = f"{stamp}:{backend}"
            rows.append({"event": "submitted", "snapshot_id": snapshot_id, "backend_id": backend, "scheduled_utc": stamp})
            if backend in collected_backends:
                rows.append({"event": "collected", "snapshot_id": snapshot_id})
    (tmp_path / "snapshots.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    backends, snapshot_ids = refresh.complete_pair_snapshot_ids(tmp_path)
    assert backends == ["tianyan-287", "tianyan176"]
    assert len(snapshot_ids) == 2
    assert all("00:00:00" in snapshot_id for snapshot_id in snapshot_ids)
