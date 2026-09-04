"""Power and boundary size of the frozen cadence gate on the true hardware endpoint.

The T-B1 design simulation (``scripts/simulate_b4_design_power.py``) remains frozen and
is not touched here.  Its cadence cell does not transfer to the corrected hardware
collection for three independent reasons:

1. It draws a *scalar* squared Gaussian, so its per-cycle observable is chi-square with
   one degree of freedom.  The hardware endpoint is the squared norm of a residual over
   *two* controlled field components, which is chi-square with two degrees of freedom,
   i.e. Exponential.  The coefficient of variation is sqrt(2) there and exactly 1 here.
2. It uses the time-average residual variance ``pq/(R T) + 2V[1 - (tau/T)(1 - e^{-T/tau})]``.
   The hardware endpoint is the end-of-interval residual ``sum(sigma_shot^2) + 4V(1 - e^{-T/tau})``.
3. Its shot term scales as ``1/T``, so under its null the fast arm carries four times the
   slow arm's variance and the null sits at ratio 4.0.  The corrected collection fixes the
   shots per setting across both cadences, so its null sits on the boundary at ratio 1.0.
   A size measured at ratio 4.0 says nothing about type-I error at the boundary.

This module measures both quantities directly through ``cadence_ratio_gate`` itself.  The
gate is imported, never reimplemented and never modified.

The estimator shot floor is the empirically calibrated ``F(S) = shot_floor_constant / S``,
where the constant was fitted on the r2 hardware arms: S=22050 gave F*S=4.7552 and
S=88200 gave F*S=4.7364, agreeing to 0.4% across a fourfold difference in shots.
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

from src.adaptive.sensing_economics import cadence_ratio_gate

SCHEMA = "b4_cadence_endpoint_power_v1"
SHOT_FLOOR_CONSTANT = 4.746
CALIBRATION_ARMS = (
    {"shots_per_setting": 22050, "observed_shot_floor": 2.156562e-4},
    {"shots_per_setting": 88200, "observed_shot_floor": 5.370067e-5},
)


def estimator_shot_floor(shots_per_setting: int, *, constant: float = SHOT_FLOOR_CONSTANT) -> float:
    """Summed two-field estimator variance at a fixed number of shots per setting."""
    if shots_per_setting <= 0:
        raise ValueError("shots per setting must be positive")
    return float(constant) / float(shots_per_setting)


def drift_endpoint_term(cadence_seconds: float, process_variance: float, tau_seconds: float) -> float:
    """Two-field end-of-interval OU drift variance over one cadence interval."""
    return 4.0 * float(process_variance) * (1.0 - math.exp(-float(cadence_seconds) / float(tau_seconds)))


def endpoint_mean(
    *,
    cadence_seconds: float,
    shots_per_setting: int,
    process_variance: float,
    tau_seconds: float,
    extra_floor: float = 0.0,
    constant: float = SHOT_FLOOR_CONSTANT,
) -> float:
    return (
        estimator_shot_floor(shots_per_setting, constant=constant)
        + float(extra_floor)
        + drift_endpoint_term(cadence_seconds, process_variance, tau_seconds)
    )


def gate_claim_rate(
    *,
    pair_count: int,
    fast_mean: float,
    slow_mean: float,
    replicates: int,
    seed: int,
) -> float:
    """Fraction of replicates in which the frozen gate claims a fast-cadence benefit."""
    generator = np.random.default_rng(seed)
    fast = generator.exponential(fast_mean, (replicates, pair_count))
    slow = generator.exponential(slow_mean, (replicates, pair_count))
    claims = 0
    for index in range(replicates):
        if bool(cadence_ratio_gate(fast[index], slow[index])["passed"]):
            claims += 1
    return claims / float(replicates)


def evaluate(
    *,
    shots_per_setting: int,
    pair_counts: Sequence[int],
    fast_seconds: float,
    slow_seconds: float,
    process_variance: float,
    tau_seconds: float,
    extra_floor: float,
    replicates: int,
    seed: int,
    minimum_power: float,
    maximum_size: float,
) -> dict[str, Any]:
    floor = estimator_shot_floor(shots_per_setting)
    fast_mean = endpoint_mean(
        cadence_seconds=fast_seconds,
        shots_per_setting=shots_per_setting,
        process_variance=process_variance,
        tau_seconds=tau_seconds,
        extra_floor=extra_floor,
    )
    slow_mean = endpoint_mean(
        cadence_seconds=slow_seconds,
        shots_per_setting=shots_per_setting,
        process_variance=process_variance,
        tau_seconds=tau_seconds,
        extra_floor=extra_floor,
    )
    rows: list[dict[str, Any]] = []
    for pair_count in pair_counts:
        power = gate_claim_rate(
            pair_count=int(pair_count),
            fast_mean=fast_mean,
            slow_mean=slow_mean,
            replicates=replicates,
            seed=seed,
        )
        # The boundary null gives both cadences the same endpoint mean, which is exactly
        # what fixing the shots per setting produces when the injected drift is switched off.
        size = gate_claim_rate(
            pair_count=int(pair_count),
            fast_mean=fast_mean,
            slow_mean=fast_mean,
            replicates=replicates,
            seed=seed + 1,
        )
        rows.append({
            "pair_count": int(pair_count),
            "power": power,
            "boundary_size": size,
            "power_pass": bool(power >= minimum_power),
            "size_pass": bool(size <= maximum_size),
            "passed": bool(power >= minimum_power and size <= maximum_size),
        })
    return {
        "shots_per_setting": int(shots_per_setting),
        "estimator_shot_floor": floor,
        "extra_unmodelled_floor": float(extra_floor),
        "fast_endpoint_mean": fast_mean,
        "slow_endpoint_mean": slow_mean,
        "expected_ratio": fast_mean / slow_mean,
        "pure_drift_ratio": drift_endpoint_term(fast_seconds, process_variance, tau_seconds)
        / drift_endpoint_term(slow_seconds, process_variance, tau_seconds),
        "rows": rows,
    }


def evaluate_config(
    config: Mapping[str, Any],
    *,
    extra_floor: float = 0.0,
    replicates: int = 4000,
    seed: int = 20260815,
    pair_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reachability of the registered endpoint implied by a collection config itself.

    This is the check that para 286 asks for and that the previous synthetic-effect test
    could not perform: the shots per setting are read from the config, converted into the
    estimator noise floor they actually produce, and pushed through the frozen gate.
    """
    correction = config.get("collection_correction")
    if not isinstance(correction, Mapping):
        raise ValueError("config carries no collection correction to check")
    registered_total = int(correction["registered_cycle_pairs_total"])
    minimum_pairs = int(correction.get("minimum_adjudicated_cycle_pairs", registered_total))
    counts = list(pair_counts) if pair_counts is not None else sorted({minimum_pairs, registered_total})
    result = evaluate(
        shots_per_setting=int(correction["sensing_shots_per_setting"]),
        pair_counts=counts,
        fast_seconds=float(config["cadence"]["fast_seconds"]),
        slow_seconds=float(config["cadence"]["slow_seconds"]),
        process_variance=float(config["controlled_ou"]["stationary_process_variance"]),
        tau_seconds=float(config["controlled_ou"]["tau_seconds"]),
        extra_floor=extra_floor,
        replicates=replicates,
        seed=seed,
        minimum_power=float(correction.get("minimum_power", 0.8)),
        maximum_size=float(correction.get("maximum_boundary_size", 0.05)),
    )
    binding = next(row for row in result["rows"] if row["pair_count"] == minimum_pairs)
    return {
        "schema": SCHEMA,
        "registered_cycle_pairs_total": registered_total,
        "minimum_adjudicated_cycle_pairs": minimum_pairs,
        "minimum_power": float(correction.get("minimum_power", 0.8)),
        "maximum_boundary_size": float(correction.get("maximum_boundary_size", 0.05)),
        "replicates": int(replicates),
        "seed": int(seed),
        "calibration_arms": [dict(row) for row in CALIBRATION_ARMS],
        "shot_floor_constant": SHOT_FLOOR_CONSTANT,
        "binding_pair_count": minimum_pairs,
        "reachable": bool(binding["passed"]),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--extra-floor", type=float, default=0.0)
    parser.add_argument("--replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    config = json.loads(args.loop_config.read_text(encoding="utf-8"))
    report = evaluate_config(
        config,
        extra_floor=args.extra_floor,
        replicates=args.replicates,
        seed=args.seed,
    )
    report["loop_config"] = str(args.loop_config)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite evidence artifact: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["reachable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
