from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("poll_platform_config", ROOT / "scripts" / "poll_platform_config.py")
assert SPEC and SPEC.loader
poller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = poller
SPEC.loader.exec_module(poller)


class FakePlatform:
    def __init__(self) -> None:
        self.download_calls: list[str] = []

    def download_config(self, machine: str):
        self.download_calls.append(machine)
        return {
            "calibrationTime": "2026-08-04 10:20:30",
            "singleQubit": [{"readoutError": 0.012, "gateError": 0.003}],
            "twoQubit": {"czError": {"value": 0.04, "unit": None}},
        }

    def submit(self, *args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("poller submitted a hardware job")


def test_poll_once_records_raw_metadata_without_hardware_job(tmp_path: Path) -> None:
    platform = FakePlatform()
    stamps = iter([
        datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 4, 12, 0, 2, tzinfo=timezone.utc),
    ])
    ticks = iter([10.0, 12.0])
    row, success = poller.poll_once(
        backend_id="tianyan-287",
        login_key="not-written",
        account_label="t287_account",
        output_root=tmp_path,
        previous_success_utc=datetime(2026, 8, 4, 10, 40, tzinfo=timezone.utc),
        interval_seconds=3600.0,
        jitter_seconds=300.0,
        platform_factory=lambda key, backend: platform,
        clock=lambda: next(stamps),
        timer=lambda: next(ticks),
    )
    assert success == datetime(2026, 8, 4, 12, 0, 2, tzinfo=timezone.utc)
    assert platform.download_calls == ["tianyan-287"]
    assert row["hardware_job_submitted"] is False
    assert row["account_label"] == "t287_account"
    assert row["calibrationTime_raw"] == "2026-08-04 10:20:30"
    assert set(row["platform_error_fields_raw"]) == {
        "singleQubit[0].readoutError",
        "singleQubit[0].gateError",
        "twoQubit.czError",
    }
    assert row["polling_gap_detected"] is True
    path = next((tmp_path / "tianyan-287").glob("*.jsonl"))
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["status"] == "ok"
    assert "not-written" not in path.read_text(encoding="utf-8")


def test_latest_success_and_jitter_bounds(tmp_path: Path) -> None:
    directory = tmp_path / "tianyan176"
    directory.mkdir(parents=True)
    path = directory / "platform_config_20260804.jsonl"
    path.write_text(
        json.dumps({"status": "error", "query_finished_utc": "2026-08-04T10:00:00+00:00"}) + "\n"
        + json.dumps({"status": "ok", "query_finished_utc": "2026-08-04T11:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    assert poller.latest_success_utc(tmp_path, "tianyan176") == datetime(2026, 8, 4, 11, 0, tzinfo=timezone.utc)
    rng = random.Random(7)
    delays = [poller.next_delay(3600.0, 300.0, rng) for _ in range(100)]
    assert min(delays) >= 3300.0
    assert max(delays) <= 3900.0


def test_poller_has_no_job_submission_path_and_supervisor_restarts_fast() -> None:
    source = (ROOT / "scripts" / "poll_platform_config.py").read_text(encoding="utf-8")
    executable = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert ".submit(" not in executable
    assert ".run(" not in executable
    assert "download_config(machine=backend_id)" in executable
    supervisor = (ROOT / "scripts" / "run_platform_config_poll_supervisor.bat").read_text(encoding="utf-8")
    assert "timeout /t 60" in supervisor
    assert "goto restart" in supervisor


def test_b0_acceptance_requires_full_24h_and_zero_flagged_gaps() -> None:
    from scripts import audit_b4_poll_acceptance as audit

    rows = [
        {
            "status": "ok",
            "query_finished_utc": "2026-08-04T12:00:00+00:00",
            "polling_gap_detected": False,
            "hardware_job_submitted": False,
        },
        {
            "status": "ok",
            "query_finished_utc": "2026-08-05T12:00:01+00:00",
            "polling_gap_detected": False,
            "hardware_job_submitted": False,
        },
    ]
    summary = audit.summarize_backend(rows, "tianyan-287")
    assert summary["coverage_24h_passed"] is True
    rows[-1]["polling_gap_detected"] = True
    assert audit.summarize_backend(rows, "tianyan-287")["coverage_24h_passed"] is False


def test_b0_acceptance_restarts_continuous_clock_after_last_gap() -> None:
    from scripts import audit_b4_poll_acceptance as audit

    rows = [
        {
            "status": "ok",
            "query_finished_utc": "2026-08-04T00:00:00+00:00",
            "polling_gap_detected": False,
            "hardware_job_submitted": False,
        },
        {
            "status": "ok",
            "query_finished_utc": "2026-08-05T00:00:00+00:00",
            "polling_gap_detected": True,
            "hardware_job_submitted": False,
        },
        {
            "status": "ok",
            "query_finished_utc": "2026-08-06T00:00:01+00:00",
            "polling_gap_detected": False,
            "hardware_job_submitted": False,
        },
    ]
    summary = audit.summarize_backend(rows, "tianyan-287")
    assert summary["gap_count"] == 1
    assert summary["current_segment_gap_count"] == 0
    assert summary["current_continuous_segment_start_utc"] == "2026-08-05T00:00:00+00:00"
    assert summary["continuous_span_seconds"] == 86401.0
    assert summary["coverage_24h_passed"] is True
