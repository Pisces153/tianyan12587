"""Design-feasibility audit for the frozen v3 T7b decision-class gate."""

from __future__ import annotations

from math import comb

import pytest

from src.adaptive import task_metric_mirror


V3_WINDOW_LENGTH = 10
V3_EXCEEDANCE_RATE = 0.233
MINIMUM_FEASIBILITY_PROBABILITY = 0.80


def _probability_of_non_degenerate_window(*, exceedance_rate: float) -> float:
    """The frozen >=5 / >=5 rule in a length-10 window permits only 5 / 5."""
    return comb(V3_WINDOW_LENGTH, 5) * exceedance_rate**5 * (1.0 - exceedance_rate) ** 5


def test_v3_t7b_decision_class_gate_is_design_infeasible() -> None:
    """Three v3 windows have about 1e-4 chance to satisfy the frozen rule."""
    one_window = _probability_of_non_degenerate_window(exceedance_rate=V3_EXCEEDANCE_RATE)
    three_windows = one_window**3

    assert one_window == pytest.approx(
        _probability_of_non_degenerate_window(exceedance_rate=1.0 - V3_EXCEEDANCE_RATE)
    )
    assert three_windows < MINIMUM_FEASIBILITY_PROBABILITY
    print(
        "v3 invalid gate: T7b >=5 / >=5 at window=10 | "
        f"P(one window)={one_window:.4f}; P(three windows)={three_windows:.6f}; "
        f"required feasibility >= {MINIMUM_FEASIBILITY_PROBABILITY:.2f}"
    )


def test_b4_mirror_pair_budget_can_fit_daily_wallclock_envelope() -> None:
    budget = task_metric_mirror.budget_summary(
        pairs=4,
        shots_per_task=16384,
        throughput_shots_per_second=490.0,
        fixed_overhead_seconds_per_job=10.0,
    )
    assert budget["wallclock_seconds"] <= 600.0
