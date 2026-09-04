#!/usr/bin/env python3
"""Per-replicate diagnosis of B-4 map-power timing nonmonotonicity."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import simulate_b4_design_power as simulation
from src.adaptive import sensing_economics


DEFAULT_OUTPUT = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_timing_diagnostic_20260804_v2"
)
DEFAULT_REUSE_DIRECTORY = Path(
    r"E:\TianYan\XA-202609\artifacts\analysis\B4_map_timing_diagnostic_20260804"
)
DEFAULT_PROFILES = ((850, 0.0), (1000, 0.0), (1000, 0.5), (1000, 1.1))


def _failure_gate(analysis: Mapping[str, Any]) -> str:
    if not bool(analysis["variance_gate"]["passed"]):
        return "detection_gate"
    if not bool(analysis["ou_fit"]["ok"]):
        return "ou_fit"
    if not bool(analysis.get("economic_separation")):
        return "economic_separation"
    return "passed"


def _run_profile(arguments: tuple[int, float, int, int]) -> dict[str, Any]:
    shot_rate, overhead, replicates, seed = arguments
    schedule = simulation.analysis_schedule(shot_rate, overhead)
    times = schedule["times_seconds"]
    shots = schedule["shots"]
    instruments = schedule["instrument_ids"]
    floor = simulation.design_resolvable_floor_seconds(shot_rate, overhead)
    edges = simulation.lag_edges(times, floor)
    effective_rate = simulation.effective_probe_shots_per_second(shot_rate, overhead)
    rng = np.random.default_rng(simulation._seed_for({
        "endpoint": "map",
        "shot_rate": shot_rate,
        "overhead": overhead,
    }, seed))
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        # Match run_map_endpoint RNG order exactly: e0 observation first, then e1.
        simulation.simulate_observations(np.full(len(times), simulation.E0_PROBABILITY), shots, rng)
        tau_minutes = 15.0 if replicate % 2 == 0 else 30.0
        probability = simulation.PRIMARY_PROBABILITY + simulation._ou_component(
            times,
            6.1e-4,
            tau_minutes * 60.0,
            rng,
        )
        latent = simulation.LatentSeries(
            np.clip(probability, 1e-5, 1.0 - 1e-5),
            np.zeros(len(times), dtype=int),
            (),
            None,
            {},
        )
        analysis = sensing_economics.analyze_ou_sensing(
            values=simulation.simulate_observations(latent.probability, shots, rng),
            times_seconds=times,
            shots=shots,
            regime_ids=latent.regime_ids,
            burst_flags=np.zeros(len(times), dtype=bool),
            instrument_ids=instruments,
            lag_edges_seconds=edges,
            effective_shots_per_second=effective_rate,
            maximum_interval_seconds=float(times[-1] - times[0]),
            interface_floor_seconds=floor,
        )
        fit = analysis["ou_fit"]
        tau_lower = fit.get("tau_ci_lower_seconds")
        tau_upper = fit.get("tau_ci_upper_seconds")
        t_star = analysis.get("t_star_seconds")
        rows.append({
            "replicate": replicate,
            "shot_rate_per_second": shot_rate,
            "fixed_overhead_seconds_per_setting": overhead,
            "true_tau_minutes": tau_minutes,
            "failure_gate": _failure_gate(analysis),
            "detection_pass": bool(analysis["variance_gate"]["passed"]),
            "ou_fit_ok": bool(fit["ok"]),
            "economic_separation_pass": bool(analysis.get("economic_separation")),
            "worth_sensing": bool(analysis["worth_sensing"]),
            "tau_hat_seconds": fit.get("tau_seconds"),
            "tau_ci_lower_seconds": tau_lower,
            "tau_ci_upper_seconds": tau_upper,
            "tau_ci_width_seconds": None if tau_lower is None or tau_upper is None else float(tau_upper) - float(tau_lower),
            "tau_ci_width_octaves": None if tau_lower is None or tau_upper is None else float(np.log2(float(tau_upper) / float(tau_lower))),
            "t_star_seconds": t_star,
            "design_floor_seconds": floor,
            "floor_minus_t_star_seconds": None if t_star is None else floor - float(t_star),
            "floor_right_of_t_star": None if t_star is None else bool(floor > float(t_star)),
            "economic_separation_margin": analysis.get("economic_separation_margin"),
            "worst_corner_residual_variance": analysis.get("worst_corner_residual_variance"),
            "process_variance_ci_lower": analysis.get("process_variance_ci_lower_for_economic_gate"),
        })
    return {
        "profile": {
            **simulation.timing_fields(shot_rate, overhead),
            "design_resolvable_floor_seconds": floor,
        },
        "lag_edges_seconds": [float(value) for value in edges],
        "rows": rows,
    }


def _nullable_median(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return None if not values else float(np.median(values))


def summarize(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(result["rows"])
    counts = {
        gate: sum(row["failure_gate"] == gate for row in rows)
        for gate in ("passed", "detection_gate", "ou_fit", "economic_separation")
    }
    trials = len(rows)
    return {
        **result["profile"],
        "n_replicates": trials,
        "power": counts["passed"] / trials,
        "detection_failure_rate": counts["detection_gate"] / trials,
        "ou_fit_failure_rate": counts["ou_fit"] / trials,
        "economic_separation_failure_rate": counts["economic_separation"] / trials,
        "tau_ci_width_seconds_median": _nullable_median(rows, "tau_ci_width_seconds"),
        "tau_ci_width_octaves_median": _nullable_median(rows, "tau_ci_width_octaves"),
        "t_star_seconds_median": _nullable_median(rows, "t_star_seconds"),
        "floor_right_of_t_star_rate": float(np.mean([
            float(row["floor_right_of_t_star"])
            for row in rows
            if row["floor_right_of_t_star"] is not None
        ])),
        "economic_separation_margin_median": _nullable_median(rows, "economic_separation_margin"),
        "lag_edges_seconds": result["lag_edges_seconds"],
    }


def mechanism_verdict(summaries: Sequence[Mapping[str, Any]]) -> str:
    by_profile = {
        (int(row["shot_rate_per_second"]), float(row["fixed_overhead_seconds_per_setting"])): row
        for row in summaries
    }
    low = by_profile[(850, 0.0)]
    high = by_profile[(1000, 0.0)]
    middle = by_profile[(1000, 0.5)]
    high_overhead = by_profile[(1000, 1.1)]
    shorter_lag = float(high["design_resolvable_floor_seconds"]) < float(low["design_resolvable_floor_seconds"])
    economic_drop = float(high["economic_separation_failure_rate"]) > float(low["economic_separation_failure_rate"])
    short_lag_widens_tau_ci = float(high["tau_ci_width_seconds_median"]) > float(low["tau_ci_width_seconds_median"])
    middle_restores_power = float(middle["power"]) > float(high["power"])
    middle_narrows_tau_ci = float(middle["tau_ci_width_seconds_median"]) < float(high["tau_ci_width_seconds_median"])
    high_overhead_pushes_floor_right = float(high_overhead["floor_right_of_t_star_rate"]) > float(middle["floor_right_of_t_star_rate"])
    high_overhead_loses_power = float(high_overhead["power"]) < float(middle["power"])
    if (
        shorter_lag
        and economic_drop
        and short_lag_widens_tau_ci
        and middle_restores_power
        and middle_narrows_tau_ci
        and high_overhead_pushes_floor_right
        and high_overhead_loses_power
    ):
        return "physical_timing_tradeoff_supported"
    return "mechanism_not_closed_check_binning_or_clamping"


def run(
    output: Path,
    profiles: Sequence[tuple[int, float]],
    replicates: int,
    seed: int,
    workers: int,
    reuse_directory: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    frozen_profiles = tuple((int(rate), float(overhead)) for rate, overhead in profiles)
    if frozen_profiles != DEFAULT_PROFILES:
        raise ValueError("diagnostic requires the frozen 850/1000/counterfactual profile set")
    reused_summaries: list[dict[str, Any]] = []
    replicate_sources: list[str] = []
    reused_profiles: set[tuple[int, float]] = set()
    if reuse_directory is not None and (reuse_directory / "diagnostic_summary.json").exists():
        prior = json.loads((reuse_directory / "diagnostic_summary.json").read_text(encoding="utf-8"))
        for summary in prior["summaries"]:
            profile = (
                int(summary["shot_rate_per_second"]),
                float(summary["fixed_overhead_seconds_per_setting"]),
            )
            if profile in frozen_profiles:
                reused_summaries.append(summary)
                reused_profiles.add(profile)
                replicate_sources.append(str((reuse_directory / f"replicates_R{profile[0]}_c{profile[1]:.1f}.csv").resolve()))
    arguments = [
        (rate, overhead, replicates, seed)
        for rate, overhead in frozen_profiles
        if (rate, overhead) not in reused_profiles
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
            results = list(executor.map(_run_profile, arguments))
    else:
        results = [_run_profile(item) for item in arguments]
    summaries = reused_summaries + [summarize(result) for result in results]
    summaries.sort(key=lambda row: (
        int(row["shot_rate_per_second"]),
        float(row["fixed_overhead_seconds_per_setting"]),
    ))
    verdict = mechanism_verdict(summaries)
    report = {
        "schema": "b4_map_timing_diagnostic_v2",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "replicates_per_profile": replicates,
        "seed": seed,
        "verdict": verdict,
        "summaries": summaries,
        "replicate_sources": replicate_sources,
        "gate_definition": {
            "detection_gate": "variance component gate failed",
            "ou_fit": "detection passed but OU fit unavailable",
            "economic_separation": "worst CI-corner residual did not beat process-variance CI lower bound",
            "passed": "detection, OU fit, and economic separation all passed",
        },
    }
    output.mkdir(parents=True)
    for result in results:
        profile = result["profile"]
        name = f"replicates_R{profile['shot_rate_per_second']}_c{profile['fixed_overhead_seconds_per_setting']:.1f}.csv"
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
        report["replicate_sources"].append(str((output / name).resolve()))
    (output / "diagnostic_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B-4 850 vs 1000 非单调机制诊断",
        "",
        f"判定：`{verdict}`。每档 {replicates} 个 e1 replicate；逐 replicate 明细另存 CSV。",
        "",
        "| R | c/setting | floor s | power | detection fail | OU fit fail | economic fail | tau CI width s (median) | T* s (median) | margin (median) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['shot_rate_per_second']} | {row['fixed_overhead_seconds_per_setting']:.1f} | "
            f"{row['design_resolvable_floor_seconds']:.4f} | {row['power']:.4f} | "
            f"{row['detection_failure_rate']:.4f} | {row['ou_fit_failure_rate']:.4f} | "
            f"{row['economic_separation_failure_rate']:.4f} | {row['tau_ci_width_seconds_median']:.2f} | "
            f"{row['t_star_seconds_median']:.2f} | {row['economic_separation_margin_median']:.6g} |"
        )
    lines.extend([
        "",
        "解释规则：c=0 过短 lag 扩大 τ CI；c=0.5 恢复识别与 power；c=1.1 又把 floor 推近/推过 T*。三段同时成立即物理时序权衡，不是分箱或钳位 bug。",
        "",
    ])
    (output / "B4_MAP_TIMING_MECHANISM.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse-directory", type=Path, default=DEFAULT_REUSE_DIRECTORY)
    arguments = parser.parse_args()
    report = run(
        arguments.output,
        DEFAULT_PROFILES,
        arguments.replicates,
        arguments.seed,
        max(1, arguments.workers),
        arguments.reuse_directory,
    )
    print(json.dumps({"verdict": report["verdict"], "summaries": report["summaries"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
