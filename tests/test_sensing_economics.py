from __future__ import annotations

from pathlib import Path

import numpy as np

from src.adaptive import sensing_economics as module


ROOT = Path(__file__).resolve().parents[1]


def test_binomial_variance_uses_unbiased_n_minus_one_formula() -> None:
    result = module.binomial_variance_unbiased([0.1, 0.8], 100)
    assert np.allclose(result, [0.1 * 0.9 / 99.0, 0.8 * 0.2 / 99.0])


def test_cross_regime_step_is_not_continuous_process_variance() -> None:
    values = np.r_[np.full(20, 0.1), np.full(20, 0.2)]
    regimes = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    gate = module.variance_component_gate(values, 16384, regimes)
    assert gate.passed is False
    rows = module.structure_function(
        values,
        np.arange(len(values), dtype=float),
        16384,
        regimes,
        np.zeros(len(values), dtype=bool),
        [0.5, 1.5, 4.5, 20.5, 50.0],
    )
    assert all(abs(float(row["sf_debiased"])) < 2e-5 for row in rows)


def test_burst_rows_and_cross_event_pairs_are_excluded() -> None:
    values = np.asarray([0.1, 0.1, 0.9, 0.2, 0.2])
    rows = module.structure_function(
        values,
        np.arange(5, dtype=float),
        16384,
        [0, 0, 0, 1, 1],
        [False, False, True, False, False],
        [0.5, 1.5, 5.0],
    )
    assert sum(int(row["n_pairs"]) for row in rows) == 2


def test_structure_function_never_pairs_across_instruments() -> None:
    values = np.asarray([0.1, 0.2, 0.1, 0.2])
    rows = module.structure_function(
        values,
        [0.0, 1.0, 2.0, 3.0],
        [1024, 16384, 1024, 16384],
        [0, 0, 0, 0],
        [False, False, False, False],
        [0.5, 2.5, 4.0],
        instrument_ids=["anchor_33", "probe_burst", "anchor_33", "probe_burst"],
    )
    assert sum(int(row["n_pairs"]) for row in rows) == 2


def test_analysis_path_has_no_scenario_branch_or_truth_argument() -> None:
    source = (ROOT / "src" / "adaptive" / "sensing_economics.py").read_text(encoding="utf-8")
    assert "scenario ==" not in source
    signature_source = source[source.index("def analyze_ou_sensing"):]
    assert "true_parameter" not in signature_source
    assert "dgp_label" not in signature_source


def test_nonparametric_curve_is_monotone_and_integrable() -> None:
    rows = [
        {"lag_mid_seconds": 10.0, "sf_debiased": 0.4, "n_effective_points": 10},
        {"lag_mid_seconds": 20.0, "sf_debiased": 0.2, "n_effective_points": 10},
        {"lag_mid_seconds": 40.0, "sf_debiased": 0.6, "n_effective_points": 10},
    ]
    curve = module.monotone_structure_curve(rows)
    assert np.all(np.diff(curve["sf"]) >= 0.0)
    residual = module.nonparametric_residual_variance(30.0, 0.1, 100.0, rows)
    assert np.isfinite(float(residual))


def test_parametric_bootstrap_reports_two_layers() -> None:
    rng = np.random.default_rng(9)
    times = np.arange(0.0, 7200.0, 120.0)
    latent = np.zeros(len(times))
    for index in range(1, len(times)):
        coefficient = np.exp(-120.0 / 900.0)
        latent[index] = coefficient * latent[index - 1] + np.sqrt(0.001 * (1.0 - coefficient**2)) * rng.normal()
    observed = rng.binomial(4096, np.clip(0.2 + latent, 1e-4, 1.0 - 1e-4)) / 4096.0
    analysis = module.analyze_ou_sensing(
        values=observed,
        times_seconds=times,
        shots=4096,
        regime_ids=np.zeros(len(times), dtype=int),
        burst_flags=np.zeros(len(times), dtype=bool),
        lag_edges_seconds=np.geomspace(100.0, 3600.0, 10),
        effective_shots_per_second=100.0,
        maximum_interval_seconds=3600.0,
        interface_floor_seconds=23.0,
        bootstrap_resamples=30,
        bootstrap_seed=12,
    )
    assert analysis["parametric_bootstrap"]["available"] is True
    assert "OU process layer plus binomial observation layer" in analysis["parametric_bootstrap"]["method"]


def test_ablation_structure_function_changes_only_requested_pair_membership() -> None:
    from scripts import analyze_sensing_economics as analysis_script

    values = np.asarray([0.1, 0.11, 0.2, 0.21])
    times = np.asarray([0.0, 10.0, 20.0, 30.0])
    regimes = np.asarray([0, 0, 1, 1])
    edges = np.asarray([1.0, 40.0])
    all_pairs, all_audit = analysis_script._ablation_structure_function(
        values=values,
        times_seconds=times,
        shots=1024,
        regimes=regimes,
        lag_edges_seconds=edges,
        unbiased_shot_noise=False,
        enforce_pairing_discipline=False,
        minimum_pairs_per_bin=2,
    )
    paired, paired_audit = analysis_script._ablation_structure_function(
        values=values,
        times_seconds=times,
        shots=1024,
        regimes=regimes,
        lag_edges_seconds=edges,
        unbiased_shot_noise=True,
        enforce_pairing_discipline=True,
        minimum_pairs_per_bin=2,
    )
    assert all_audit["eligible_pairs"] == 6
    assert paired_audit["eligible_pairs"] == 2
    assert paired_audit["excluded_cross_regime_pairs"] == 4
    assert paired[0]["sf_debiased"] < all_pairs[0]["sf_debiased"]
