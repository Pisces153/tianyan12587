"""Frozen map from stage-1 hardware timing to the stage-2 collection design.

This module is registered *before* any stage-2 hardware record exists.  It is a pure
function of stage-1 measurements and the machine-time budget; there is no human choice
between measuring the timing and firing the collection.  What it fixes:

* ``pairs`` and ``shots_per_setting`` for the collection,
* the endpoint means those imply,
* a 95% predictive interval for the ratio the collection will report.

Every one of those depends only on quantities that are orthogonal to the endpoint
values themselves -- queue timing and estimator shot noise -- so sizing the design from
stage 1 is not data-dependent tuning of the endpoint.

Stage 1 measures the *fixed cost of one sensing cycle* directly, at the exact job shape
stage 2 will use.  That dissolves the per-job / per-setting ambiguity in the T-B6
throughput fit, which held ``settings_per_job`` fixed at 2 and therefore had no power
to separate the two readings: ``c = 6.990118 s/setting`` there is literally
``13.980237 / 2``, a definition rather than a measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive.cadence_permutation import permutation_claim_rate

SCHEMA = "b4_cadence_stage2_design_from_timing_v1"

# Frozen selection rule.  Candidate pair counts are enumerated in ascending order and
# the design with the highest predicted permutation power wins; ties break to the
# smaller pair count, which is the one with more shots per setting and therefore the
# larger effect size.  No other candidate is considered.
CANDIDATE_PAIR_COUNTS = tuple(range(12, 161, 2))
MINIMUM_SHOTS_PER_SETTING = 2048
POWER_REPLICATES = 4000
POWER_PERMUTATIONS = 600
POWER_SEED = 20260815
PREDICTION_REPLICATES = 200000
PREDICTION_SEED = 20260816
PREDICTION_MASS = 0.95


def drift_endpoint_term(cadence_seconds: float, process_variance: float, tau_seconds: float) -> float:
    return 4.0 * float(process_variance) * (1.0 - math.exp(-float(cadence_seconds) / float(tau_seconds)))


def busy_seconds(
    *,
    pairs: int,
    shots_per_setting: int,
    timing: Mapping[str, Any],
) -> float:
    """Modelled machine-time cost of the whole stage-2 collection."""
    cycles = 2 * int(pairs)
    sensing_shots = cycles * int(timing["settings_per_cycle"]) * int(shots_per_setting)
    baseline_shots = int(timing["baseline_jobs"]) * int(timing["baseline_shots_per_job"])
    fixed = (cycles + int(timing["baseline_jobs"])) * float(timing["seconds_per_cycle_fixed"])
    return (sensing_shots + baseline_shots) / float(timing["shot_rate_per_second"]) + fixed


def endpoint_means(
    *,
    shots_per_setting: int,
    timing: Mapping[str, Any],
    ou: Mapping[str, Any],
    cadence: Mapping[str, Any],
) -> tuple[float, float]:
    floor = float(timing["shot_floor_constant"]) / float(shots_per_setting)
    variance = float(ou["stationary_process_variance"])
    tau = float(ou["tau_seconds"])
    fast = floor + drift_endpoint_term(float(cadence["fast_seconds"]), variance, tau)
    slow = floor + drift_endpoint_term(float(cadence["slow_seconds"]), variance, tau)
    return fast, slow


def largest_shots_within_budget(
    *,
    pairs: int,
    budget_seconds: float,
    timing: Mapping[str, Any],
) -> int:
    """Invert the busy-time model for the largest affordable shots per setting."""
    cycles = 2 * int(pairs)
    baseline_shots = int(timing["baseline_jobs"]) * int(timing["baseline_shots_per_job"])
    rate = float(timing["shot_rate_per_second"])
    fixed = (cycles + int(timing["baseline_jobs"])) * float(timing["seconds_per_cycle_fixed"])
    transfer_budget = float(budget_seconds) - fixed
    if transfer_budget <= 0.0:
        return 0
    affordable = transfer_budget * rate - baseline_shots
    per_setting = cycles * int(timing["settings_per_cycle"])
    if affordable <= 0.0 or per_setting <= 0:
        return 0
    return int(affordable // per_setting)


def ratio_prediction_interval(
    *,
    pairs: int,
    fast_mean: float,
    slow_mean: float,
    replicates: int = PREDICTION_REPLICATES,
    seed: int = PREDICTION_SEED,
    mass: float = PREDICTION_MASS,
) -> dict[str, float]:
    """Predictive interval for the ratio the collection will report.

    This is the pre-registered numeric prediction.  It is a statement about the
    hardware run made before the hardware run, and it is checked by hit/miss rather
    than by a p-value, so it remains informative at pair counts where the endpoint
    test itself is underpowered.
    """
    generator = np.random.default_rng(seed)
    fast = generator.exponential(fast_mean, (replicates, pairs))
    slow = generator.exponential(slow_mean, (replicates, pairs))
    ratios = fast.mean(axis=1) / slow.mean(axis=1)
    tail = (1.0 - mass) / 2.0
    return {
        "predicted_ratio_median": float(np.quantile(ratios, 0.5)),
        "predicted_ratio_mean": float(np.mean(ratios)),
        "prediction_interval_lower": float(np.quantile(ratios, tail)),
        "prediction_interval_upper": float(np.quantile(ratios, 1.0 - tail)),
        "prediction_mass": float(mass),
        "prediction_replicates": int(replicates),
        "prediction_seed": int(seed),
    }


def design(
    *,
    timing: Mapping[str, Any],
    ou: Mapping[str, Any],
    cadence: Mapping[str, Any],
    budget_seconds: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for pairs in CANDIDATE_PAIR_COUNTS:
        shots = largest_shots_within_budget(
            pairs=pairs, budget_seconds=budget_seconds, timing=timing
        )
        if shots < MINIMUM_SHOTS_PER_SETTING:
            continue
        fast_mean, slow_mean = endpoint_means(
            shots_per_setting=shots, timing=timing, ou=ou, cadence=cadence
        )
        power = permutation_claim_rate(
            pair_count=pairs,
            fast_mean=fast_mean,
            slow_mean=slow_mean,
            replicates=POWER_REPLICATES,
            permutations=POWER_PERMUTATIONS,
            seed=POWER_SEED,
            alpha=alpha,
        )
        candidates.append({
            "pairs": int(pairs),
            "shots_per_setting": int(shots),
            "fast_endpoint_mean": fast_mean,
            "slow_endpoint_mean": slow_mean,
            "expected_ratio": fast_mean / slow_mean,
            "permutation_power": power,
            "busy_seconds": busy_seconds(
                pairs=pairs, shots_per_setting=shots, timing=timing
            ),
        })
    if not candidates:
        raise ValueError("no design fits the machine-time budget")
    # Frozen tie-break: highest power, then fewest pairs.
    chosen = max(candidates, key=lambda row: (row["permutation_power"], -row["pairs"]))
    prediction = ratio_prediction_interval(
        pairs=chosen["pairs"],
        fast_mean=chosen["fast_endpoint_mean"],
        slow_mean=chosen["slow_endpoint_mean"],
    )
    return {
        "schema": SCHEMA,
        "budget_seconds": float(budget_seconds),
        "alpha": float(alpha),
        "timing_inputs": dict(timing),
        "selection_rule": (
            "enumerate CANDIDATE_PAIR_COUNTS, take the largest shots per setting that "
            "fits the busy-time budget, keep the candidate with the highest permutation "
            "power, break ties toward fewer pairs"
        ),
        "chosen": chosen,
        "prediction": prediction,
        "candidates": candidates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-artifact", type=Path, required=True)
    parser.add_argument("--loop-config", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    timing = json.loads(arguments.timing_artifact.read_text(encoding="utf-8"))
    config = json.loads(arguments.loop_config.read_text(encoding="utf-8"))
    report = design(
        timing=timing["stage2_timing_inputs"],
        ou=config["controlled_ou"],
        cadence=config["cadence"],
        budget_seconds=arguments.budget_seconds,
    )
    report["timing_artifact"] = str(arguments.timing_artifact)
    report["loop_config"] = str(arguments.loop_config)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(f"refusing to overwrite evidence artifact: {arguments.output}")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
