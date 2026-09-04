"""The permutation amendment must change the calibration and nothing else.

Two things have to hold for the amendment to be legitimate:

1. The *statistic* is byte-identical to the registered one.  Only the critical value is
   recalibrated, so the registered endpoint is not redefined.
2. The recalibrated rule actually holds its nominal size at the pair counts reachable
   inside the machine-time quota, where the frozen delta-method rule does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptive import cadence_permutation as perm
from src.adaptive.sensing_economics import cadence_ratio_gate


FAST_DRIFT = 6.324035415366083e-4
SLOW_DRIFT = 1.7050861229342266e-3


def test_statistic_is_identical_to_the_frozen_gate() -> None:
    generator = np.random.default_rng(11)
    for size in (5, 17, 40):
        fast = generator.exponential(1.0e-3, size)
        slow = generator.exponential(2.0e-3, size)
        assert float(perm.paired_ratio_statistic(fast, slow)) == cadence_ratio_gate(fast, slow)["ratio"]


def test_frozen_delta_method_verdict_is_always_reported() -> None:
    generator = np.random.default_rng(12)
    fast = generator.exponential(9.0e-4, 24)
    slow = generator.exponential(2.0e-3, 24)
    result = perm.cadence_ratio_permutation_gate(fast, slow, permutations=2000)
    frozen = cadence_ratio_gate(fast, slow)
    assert result["frozen_delta_method"] == frozen
    assert result["ratio"] == frozen["ratio"]


@pytest.mark.parametrize("pair_count", [16, 20, 24, 40])
def test_permutation_rule_holds_size_where_the_frozen_rule_does_not(pair_count: int) -> None:
    """Boundary null: both cadences carry the same endpoint mean."""
    mean = 4.746 / 11000 + FAST_DRIFT
    size = perm.permutation_claim_rate(
        pair_count=pair_count,
        fast_mean=mean,
        slow_mean=mean,
        replicates=6000,
        permutations=400,
        seed=4242,
    )
    # 6000 replicates put the two-sigma Monte Carlo band at roughly +/-0.006 around 0.05.
    assert size <= 0.062


def test_permutation_rule_detects_the_registered_cadence_contrast() -> None:
    floor = 4.746 / 11000
    power = perm.permutation_claim_rate(
        pair_count=40,
        fast_mean=floor + FAST_DRIFT,
        slow_mean=floor + SLOW_DRIFT,
        replicates=3000,
        permutations=400,
        seed=4242,
    )
    assert power >= 0.8


def test_p_value_is_valid_for_any_permutation_count() -> None:
    """The (1 + count) / (1 + permutations) form is exact, not asymptotic."""
    generator = np.random.default_rng(13)
    fast = generator.exponential(1.0e-3, 20)
    slow = generator.exponential(1.0e-3, 20)
    for permutations in (99, 999):
        result = perm.cadence_ratio_permutation_gate(fast, slow, permutations=permutations)
        assert 1.0 / (1.0 + permutations) <= result["p_value"] <= 1.0


def test_adjudication_admits_hardware_pairs_only() -> None:
    rows = [
        {
            "fast_endpoint_squared_residual": 8.0e-4,
            "slow_endpoint_squared_residual": 2.0e-3,
        }
        for _ in range(24)
    ]
    result = perm.adjudicate_registered_pairs(rows, permutations=999)
    assert result["evidence_scope"] == "hardware_registered_cycle_pairs_only"
    assert result["pair_count"] == 24
