#!/usr/bin/env python3
"""Generate T6 proxy and T7 forecast artifacts from an append-only campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive.environment_proxy import extract_campaign
from src.adaptive.forecast import run_rolling_origin


def _self_hash(payload: dict) -> str:
    copied = dict(payload)
    copied.pop("self_sha256", None)
    canonical = json.dumps(copied, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(canonical).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--campaign-root", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--corpus", type=Path, required=True)
    forecast.add_argument("--out", type=Path, required=True)
    forecast.add_argument("--backend", choices=("tianyan-287", "tianyan176"), required=True)
    forecast.add_argument("--horizon", type=int, default=1)
    forecast.add_argument("--window", type=int, default=3)
    forecast.add_argument("--tolerance", type=float, default=0.02)
    preflight = commands.add_parser("bandit-preflight")
    preflight.add_argument("--t6-report", type=Path, required=True)
    preflight.add_argument("--forecast-report", type=Path, action="append", required=True)
    preflight.add_argument("--out", type=Path, required=True)
    preflight.add_argument("--t3-artifact", type=Path, required=True)
    preflight.add_argument("--t3-conservative-prior", type=Path)
    preflight.add_argument("--t4-fidelity-gate", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract_campaign(args.campaign_root, args.out)
    elif args.command == "forecast":
        result = run_rolling_origin(args.corpus, args.out, backend_id=args.backend, horizon=args.horizon, window=args.window, tolerance=args.tolerance)
    else:
        t6 = json.loads(args.t6_report.read_text(encoding="utf-8"))
        forecast_reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.forecast_report]
        t3_artifact = json.loads(args.t3_artifact.read_text(encoding="utf-8"))
        t4_gate = json.loads(args.t4_fidelity_gate.read_text(encoding="utf-8"))
        t3_status = {
            "artifact_provided": True,
            "valid_recalibration_artifact": False,
            "self_hash_valid": False,
            "sim_to_real_coverage_gap": None,
            "conservative_prior_required": False,
            "conservative_prior_applied": False,
        }
        simulation_coverage = float(t3_artifact.get("selected_simulation_coverage_1sigma", float("nan")))
        gap = abs(float(t3_artifact.get("sim_to_real_coverage_gap", float("inf"))))
        t3_status["self_hash_valid"] = t3_artifact.get("self_sha256") == _self_hash(t3_artifact)
        t3_status["valid_recalibration_artifact"] = bool(
            t3_status["self_hash_valid"]
            and t3_artifact.get("task") == "T3_sigma_recalibration"
            and t3_artifact.get("weights_unchanged") is True
            and t3_artifact.get("hardware_used_for_fit") is False
            and t3_artifact.get("calibration_unit") == "disjoint before-target offsets from two_time_inverse_variance trajectories matching r4 inference"
            and t3_artifact.get("pairing", {}).get("source_trajectory_reused") is False
            and 0.63 <= simulation_coverage <= 0.73
        )
        t3_status["sim_to_real_coverage_gap"] = gap
        t3_status["conservative_prior_required"] = gap > 0.20
        if args.t3_conservative_prior is not None:
            prior = json.loads(args.t3_conservative_prior.read_text(encoding="utf-8"))
            t3_status["conservative_prior_applied"] = bool(
                prior.get("self_sha256") == _self_hash(prior)
                and prior.get("task") == "T3_sim_to_real_conservative_prior"
                and prior.get("source_t3_sha256") == t3_artifact.get("self_sha256")
                and prior.get("separate_from_t3_calibration") is True
                and float(prior.get("sigma_inflation_multiplier", 0.0)) >= 1.0
            )
        t4_status = {
            "valid_fidelity_gate": bool(
                t4_gate.get("self_sha256") == _self_hash(t4_gate)
                and t4_gate.get("task") == "T4_zero_cz_twin"
                and t4_gate.get("fidelity", {}).get("overall_gate_passed") is True
                and sorted(t4_gate.get("fidelity", {}).get("validated_observables", [])) == sorted(["X0", "Y0", "Z0", "X1", "Y1", "Z1", "XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"])
                and int(t4_gate.get("n_parameters", 9)) <= 8
            ),
            "self_hash_valid": t4_gate.get("self_sha256") == _self_hash(t4_gate),
        }
        t6_self_hash_valid = t6.get("self_sha256") == _self_hash(t6)
        t6_feature_corpus_sha256 = t6.get("feature_corpus_sha256")
        backend_counts = t6.get("records_by_backend", {})
        expected_backends = sorted(
            str(backend)
            for backend, count in backend_counts.items()
            if isinstance(count, int) and not isinstance(count, bool) and count > 0
        ) if isinstance(backend_counts, dict) else []
        t6_status = {
            "self_hash_valid": t6_self_hash_valid,
            "feature_corpus_sha256": t6_feature_corpus_sha256,
            "valid_feature_report": bool(
                t6_self_hash_valid
                and t6.get("task") == "T6_observable_environment_proxy"
                and isinstance(t6_feature_corpus_sha256, str)
                and len(t6_feature_corpus_sha256) == 64
                and t6.get("collected_snapshot_count", 0) > 0
                and t6.get("proxy_label_task_disjoint") is True
                and expected_backends
            ),
        }
        reports_by_backend: dict[str, dict] = {}
        duplicate_backends: list[str] = []
        invalid_report_count = 0
        for forecast_report in forecast_reports:
            corpus = forecast_report.get("corpus", {})
            backend = corpus.get("backend_id") if isinstance(corpus, dict) else None
            if not isinstance(backend, str) or not backend:
                invalid_report_count += 1
                continue
            if backend in reports_by_backend:
                duplicate_backends.append(backend)
                continue
            reports_by_backend[backend] = forecast_report
        report_backends = sorted(reports_by_backend)
        missing_backends = sorted(set(expected_backends).difference(report_backends))
        unexpected_backends = sorted(set(report_backends).difference(expected_backends))
        t7_report_validation: dict[str, dict[str, bool]] = {}
        for backend in sorted(set(expected_backends).intersection(reports_by_backend)):
            forecast_report = reports_by_backend[backend]
            cv = forecast_report.get("cv", {})
            gate = forecast_report.get("gate", {})
            self_hash_valid = forecast_report.get("self_sha256") == _self_hash(forecast_report)
            schema_valid = bool(
                forecast_report.get("analysis_task") == "T7_forecast_head"
                and isinstance(cv, dict)
                and cv.get("scheme") == "rolling_origin_forward_chain"
                and cv.get("shuffle_used") is False
                and isinstance(gate, dict)
            )
            corpus_match = forecast_report.get("feature_corpus_sha256") == t6_feature_corpus_sha256
            claimed = bool(isinstance(gate, dict) and gate.get("forecasting_skill_claimed") is True)
            t7_report_validation[backend] = {
                "self_hash_valid": self_hash_valid,
                "schema_valid": schema_valid,
                "feature_corpus_matches_t6": corpus_match,
                "forecasting_skill_claimed": claimed,
                "passed": bool(self_hash_valid and schema_valid and corpus_match and claimed),
            }
        t7_by_backend = {backend: status["passed"] for backend, status in t7_report_validation.items()}
        t7_status = {
            "required_backends": expected_backends,
            "reported_backends": report_backends,
            "missing_backends": missing_backends,
            "unexpected_backends": unexpected_backends,
            "duplicate_backends": sorted(set(duplicate_backends)),
            "invalid_report_count": invalid_report_count,
            "forecasting_skill_claimed_by_backend": t7_by_backend,
            "report_validation_by_backend": t7_report_validation,
            "all_required_backends_passed": bool(
                expected_backends
                and not missing_backends
                and not unexpected_backends
                and not duplicate_backends
                and invalid_report_count == 0
                and all(t7_by_backend.get(backend) is True for backend in expected_backends)
            ),
        }
        prerequisites = {
            "T3_sigma_calibrated": t3_status["valid_recalibration_artifact"],
            "T3_sim_to_real_conservative_prior": (not t3_status["conservative_prior_required"]) or t3_status["conservative_prior_applied"],
            "T4_twin_gate_passed": t4_status["valid_fidelity_gate"],
            "T6_features": t6_status["valid_feature_report"],
            "T7_forecast": t7_status["all_required_backends_passed"],
        }
        result = {
            "task": "T8_bandit_preflight",
            "prerequisites": prerequisites,
            "policy_execution_permitted": all(prerequisites.values()),
            "t3_uncertainty_domain_status": t3_status,
            "t4_twin_status": t4_status,
            "t6_feature_status": t6_status,
            "t7_forecast_status": t7_status,
            "reason": "Policy execution is blocked until every artifact-backed prerequisite is true; T3 sim-to-real coverage uses the sealed conservative prior when required." if t3_status["conservative_prior_applied"] else "Policy execution is blocked until every artifact-backed prerequisite is true; a T3 sim-to-real coverage gap above 0.20 blocks policy execution pending a separately specified conservative-prior artifact.",
        }
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite T8 artifact: {args.out}")
        args.out.mkdir(parents=True)
        (args.out / "prerequisite_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
