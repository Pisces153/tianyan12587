from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from src.adaptive import sensing_economics, task_metric_mirror


SCRIPT = Path(__file__).parents[1] / "scripts" / "simulate_t7_element3_dgp_v2.py"
spec = importlib.util.spec_from_file_location("t7_dgp_v2", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_gate_does_not_use_dgp_oracle() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    gate_source = source[source.index("def evaluate_sequence"):source.index("def feasibility")]
    assert "scenario ==" not in gate_source
    assert "dgp ==" not in gate_source
    assert "latent" not in gate_source
    assert "linear_slope_per_snapshot" not in gate_source


def test_observed_signal_estimator_has_no_oracle_argument() -> None:
    values = np.full(64, 0.1)
    floors = module.observed_floor(values, 16384)
    times = module.regular_schedule(module.utc("2026-01-01T00:00:00Z"), 8, 8)
    assert module.estimate_signal_snr(values, floors, times) < 1e-9


def test_new_gate_is_reachable_on_a_constructive_observed_sequence() -> None:
    start = module.utc("2026-01-01T00:00:00Z")
    times = module.regular_schedule(start, 28, 4)
    observed = np.linspace(0.08, 0.11, len(times))
    floors = module.observed_floor(observed, 16384)
    event_mask = np.zeros(len(times), dtype=bool)
    result = module.evaluate_sequence(observed, floors, times, event_mask, np.arange(len(times)), (40, 10, 1))
    assert result["fold_feasible"]
    assert "statistical_gate" in result


def test_denoised_persistence_undefined_is_explicit() -> None:
    errors = np.ones(20)
    sigma2 = np.ones(20)
    result = module.denoised_persistence_skill(errors, errors, sigma2)
    assert result["denoised_undefined"] is True


def test_all_fold_gate_requires_every_fold() -> None:
    metric = {
        name: {"raw": 0.1, "raw_ci_lower": 0.01, "denoised": 0.1, "denoised_ci_lower": 0.01, "denoised_undefined": False}
        for name in ("climatology", "persistence", "calendar")
    }
    base = {
        "fold_feasible": True, "n_folds": 1, "n_test_pairs": 10, "metrics": metric,
        "statistical_gate": True, "signal_snr_observed": 1.0,
        "discrimination": {"n_folds_with_events": 0, "trig_selected": float("nan"), "cal_selected": float("nan"), "trig_brier": float("nan"), "cal_brier": float("nan")},
    }
    rows = [dict(base), dict(base), dict(base)]
    rows[1]["statistical_gate"] = False
    assert module.aggregate_fold_evaluations(rows)["statistical_gate"] is False


def test_b4_sensing_gate_reachable_with_known_observed_ou_signal() -> None:
    rng = np.random.default_rng(20260804)
    times = np.arange(0.0, 43200.0, 23.0)
    tau_seconds = 900.0
    process_variance = 0.003
    latent = np.zeros(len(times))
    decay = np.exp(-np.diff(times) / tau_seconds)
    for index, coefficient in enumerate(decay, start=1):
        latent[index] = coefficient * latent[index - 1] + np.sqrt(process_variance * (1.0 - coefficient**2)) * rng.normal()
    probability = np.clip(0.3 + latent, 1e-4, 1.0 - 1e-4)
    observed = rng.binomial(16384, probability) / 16384.0
    result = sensing_economics.analyze_ou_sensing(
        values=observed,
        times_seconds=times,
        shots=16384,
        regime_ids=np.zeros(len(times), dtype=int),
        burst_flags=np.zeros(len(times), dtype=bool),
        lag_edges_seconds=np.geomspace(20.0, 10800.0, 16),
        effective_shots_per_second=245.0,
        maximum_interval_seconds=10800.0,
        interface_floor_seconds=23.0,
    )
    assert result["variance_gate"]["passed"] is True
    assert result["interior_optimum_claim"] is True
    assert result["worth_sensing"] is True
    assert result["t_star_seconds"] < 23.0
    assert result["t_star_ci_lower_seconds"] < 23.0


def test_b4_cadence_gate_reachable_and_null_not_forced_through() -> None:
    rng = np.random.default_rng(17)
    common = rng.lognormal(mean=-6.0, sigma=0.2, size=100)
    reached = sensing_economics.cadence_ratio_gate(common * 0.45, common)
    null = sensing_economics.cadence_ratio_gate(common, common)
    assert reached["passed"] is True
    assert null["passed"] is False


def test_b4_raw_mirror_loss_plus_cadence_gate_is_unreachable_at_nominal_effect() -> None:
    pair_noise = np.linspace(-1.0, 1.0, 24)
    pair_noise *= 0.017788 / np.std(pair_noise, ddof=1)
    slow = np.full(24, 0.6455)
    fast = slow - 0.000462 + pair_noise
    gate = sensing_economics.cadence_ratio_gate(fast, slow)
    standard_error = (gate["ci_upper"] - gate["ci_lower"]) / (2.0 * 1.959963984540054)
    minimum_detectable_effect = (1.959963984540054 + 0.8416212335729143) * standard_error
    expected_effect = 0.000462 / 0.6455
    assert gate["passed"] is False
    assert minimum_detectable_effect / expected_effect > 20.0


def test_b4_mirror_metric_gate_reachable_with_known_raw_count_signal() -> None:
    rows = []
    for index in range(40):
        common = {
            "pair_id": f"pair-{index:03d}",
            "backend_id": "tianyan-287",
            "time_window_id": f"window-{index:03d}",
            "shots": 1000,
            "task_family": "mirror_v1",
            "depth": 8,
            "task_duration_seconds": 1.0,
            "total_strategy_shots_in_window": 80000,
            "ideal_bitstring": "000000",
        }
        rows.append({**common, "strategy": "adaptive", "raw_counts": {"000000": 720, "111111": 280}})
        rows.append({**common, "strategy": "fixed", "raw_counts": {"000000": 600, "111111": 400}})
    report = task_metric_mirror.compare_strategies(rows, resamples=1000, seed=20260804)
    assert report["endpoint"]["paired_bootstrap_interval"][0] > 0.0
    assert report["endpoint"]["passed"] is True
