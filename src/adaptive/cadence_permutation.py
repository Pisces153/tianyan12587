"""Exact within-pair permutation calibration for the frozen cadence ratio gate.

``cadence_ratio_gate`` is imported unchanged and is never reimplemented.  Its point
statistic ``ratio = mean(fast)/mean(slow)`` is kept exactly as registered; only the
*calibration of the critical value* changes.

Why this is needed.  The frozen gate builds its interval from a delta-method standard
error.  On the true hardware endpoint law -- the squared norm of a residual over two
controlled field components, i.e. Exponential with coefficient of variation 1 -- that
interval is anti-conservative at small pair counts.  Measured boundary size of the
frozen rule at 40000 replicates: 0.0726 at 24 pairs, 0.0527 at 56, 0.0514 at 64,
0.0498 at 72.  The gate's own ``size_pass = size <= 0.05`` criterion therefore fails
at every pair count reachable inside the available machine-time quota.

Under the boundary null the fast and slow members of a registered cycle pair are
independent draws with the same endpoint mean, so the cadence labels are exchangeable
*within* a pair.  Swapping labels within pairs is therefore an exact
randomisation-invariance transformation, and the resulting test holds its nominal size
at every pair count, including n = 20 and n = 40.

The frozen delta-method verdict is always computed and reported alongside as the
secondary readout, so the amendment adds a rule and hides nothing.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .sensing_economics import cadence_ratio_gate

SCHEMA = "cadence_pair_permutation_adjudication_v1"
DEFAULT_PERMUTATIONS = 20000
DEFAULT_SEED = 20260815


def paired_ratio_statistic(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """The frozen gate's point statistic, vectorised over leading axes.

    ``tests/test_cadence_permutation.py`` asserts this agrees with
    ``cadence_ratio_gate(...)["ratio"]`` exactly, so the registered statistic is
    unchanged by this module.
    """
    return fast.mean(axis=-1) / slow.mean(axis=-1)


def permutation_null_ratios(
    fast: Sequence[float],
    slow: Sequence[float],
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Within-pair label swaps of the registered statistic under the boundary null."""
    fast_array = np.asarray(fast, dtype=np.float64)
    slow_array = np.asarray(slow, dtype=np.float64)
    if fast_array.shape != slow_array.shape or fast_array.ndim != 1:
        raise ValueError("paired cadence losses need equal one-dimensional samples")
    generator = np.random.default_rng(seed)
    swap = generator.random((int(permutations), fast_array.size)) < 0.5
    permuted_fast = np.where(swap, slow_array, fast_array)
    permuted_slow = np.where(swap, fast_array, slow_array)
    return paired_ratio_statistic(permuted_fast, permuted_slow)


def cadence_ratio_permutation_gate(
    fast_squared_error: Sequence[float],
    slow_squared_error: Sequence[float],
    *,
    alpha: float = 0.05,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Primary adjudication: registered statistic, exact within-pair calibration.

    One-sided.  A fast-cadence benefit means a *smaller* ratio, so the test rejects in
    the lower tail.  The ``(1 + count) / (1 + permutations)`` p-value is exactly valid
    for any Monte Carlo permutation count, so the seed and count below are part of the
    registration rather than a tuning knob.
    """
    fast = np.asarray(fast_squared_error, dtype=np.float64)
    slow = np.asarray(slow_squared_error, dtype=np.float64)
    frozen = cadence_ratio_gate(fast, slow, alpha=alpha)
    observed = float(frozen["ratio"])
    if not np.isfinite(observed):
        return {
            "schema": SCHEMA,
            "pair_count": int(fast.size),
            "ratio": observed,
            "p_value": float("nan"),
            "critical_ratio": float("nan"),
            "passed": False,
            "frozen_delta_method": dict(frozen),
        }
    null = permutation_null_ratios(fast, slow, permutations=permutations, seed=seed)
    count = int(np.count_nonzero(null <= observed))
    p_value = (1.0 + count) / (1.0 + float(permutations))
    return {
        "schema": SCHEMA,
        "pair_count": int(fast.size),
        "permutations": int(permutations),
        "seed": int(seed),
        "alpha": float(alpha),
        "ratio": observed,
        "p_value": p_value,
        "critical_ratio": float(np.quantile(null, alpha, method="linear")),
        "passed": bool(p_value <= alpha),
        "frozen_delta_method": dict(frozen),
        "adjudication_note": (
            "primary endpoint; the frozen delta-method verdict is reported as the "
            "secondary readout and is not used to decide the endpoint"
        ),
    }


CLAIM_RATE_CHUNK_BYTES = 400_000_000


def permutation_claim_rate(
    *,
    pair_count: int,
    fast_mean: float,
    slow_mean: float,
    replicates: int,
    permutations: int = 800,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    chunk_bytes: int = CLAIM_RATE_CHUNK_BYTES,
) -> float:
    """Claim rate of the permutation rule on Exponential endpoint draws.

    Setting ``fast_mean == slow_mean`` gives the realised size, which is what the
    exactness argument above predicts should sit at or below ``alpha`` for every
    ``pair_count``.

    The permutation draw is a ``replicates x permutations x pair_count`` array, which at
    the replicate counts needed to resolve a design choice runs to tens of gigabytes, so
    it is generated in replicate blocks sized by ``chunk_bytes``.  The observed statistics
    are drawn in full first, so they do not depend on the chunking; the permutation stream
    does, which is why the chunk budget is a frozen constant rather than a caller's
    convenience.  This is a simulation utility -- the registered adjudication path is
    ``cadence_ratio_permutation_gate`` and is untouched by it.
    """
    generator = np.random.default_rng(seed)
    fast = generator.exponential(fast_mean, (replicates, pair_count))
    slow = generator.exponential(slow_mean, (replicates, pair_count))
    observed = paired_ratio_statistic(fast, slow)
    per_replicate_bytes = 8 * int(permutations) * int(pair_count)
    chunk = max(1, int(chunk_bytes) // max(1, per_replicate_bytes))
    claims = 0
    for start in range(0, int(replicates), chunk):
        stop = min(start + chunk, int(replicates))
        block_fast = fast[start:stop]
        block_slow = slow[start:stop]
        swap = generator.random((stop - start, permutations, pair_count)) < 0.5
        null = paired_ratio_statistic(
            np.where(swap, block_slow[:, None, :], block_fast[:, None, :]),
            np.where(swap, block_fast[:, None, :], block_slow[:, None, :]),
        )
        p_value = (
            1.0 + (null <= observed[start:stop, None]).sum(axis=1)
        ) / (1.0 + permutations)
        claims += int(np.count_nonzero(p_value <= alpha))
    return claims / float(replicates)


def permutation_claim_rate_under_block_offsets(
    *,
    pairs_per_session: int,
    sessions: int,
    fast_mean: float,
    slow_mean: float,
    block_one_offset: float,
    block_two_offset: float,
    replicates: int,
    permutations: int = 800,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    chunk_bytes: int = CLAIM_RATE_CHUNK_BYTES,
) -> float:
    """Claim rate when a shared-baseline offset loads the two cadence blocks unequally.

    This is the simulation behind the pre-registered baseline-drift sensitivity analysis.
    Sessions alternate their starting cadence, which is what the registered
    ``session_block_order`` encodes, so block one is the fast arm in even sessions and the
    slow arm in odd ones.  That alternation is the whole reason an asymmetric offset does
    not bias the pooled statistic, and running this with ``sessions=1`` shows what is lost
    without it.

    Setting ``fast_mean == slow_mean`` gives the realized size under the offset, which is
    the quantity that decides whether drift can manufacture a pass.
    """
    if int(sessions) < 1 or int(pairs_per_session) < 1:
        raise ValueError("need at least one session of at least one pair")
    generator = np.random.default_rng(seed)
    fast_blocks: list[np.ndarray] = []
    slow_blocks: list[np.ndarray] = []
    for session_index in range(int(sessions)):
        fast_first = session_index % 2 == 0
        fast_offset = block_one_offset if fast_first else block_two_offset
        slow_offset = block_two_offset if fast_first else block_one_offset
        fast_blocks.append(np.full(int(pairs_per_session), fast_mean + fast_offset))
        slow_blocks.append(np.full(int(pairs_per_session), slow_mean + slow_offset))
    fast_means = np.concatenate(fast_blocks)
    slow_means = np.concatenate(slow_blocks)
    pair_count = int(fast_means.size)
    fast = generator.exponential(np.broadcast_to(fast_means, (int(replicates), pair_count)))
    slow = generator.exponential(np.broadcast_to(slow_means, (int(replicates), pair_count)))
    observed = paired_ratio_statistic(fast, slow)
    per_replicate_bytes = 8 * int(permutations) * pair_count
    chunk = max(1, int(chunk_bytes) // max(1, per_replicate_bytes))
    claims = 0
    for start in range(0, int(replicates), chunk):
        stop = min(start + chunk, int(replicates))
        block_fast = fast[start:stop]
        block_slow = slow[start:stop]
        swap = generator.random((stop - start, permutations, pair_count)) < 0.5
        null = paired_ratio_statistic(
            np.where(swap, block_slow[:, None, :], block_fast[:, None, :]),
            np.where(swap, block_fast[:, None, :], block_slow[:, None, :]),
        )
        p_value = (
            1.0 + (null <= observed[start:stop, None]).sum(axis=1)
        ) / (1.0 + permutations)
        claims += int(np.count_nonzero(p_value <= alpha))
    return claims / float(replicates)


def adjudicate_registered_pairs(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.05,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Adjudicate complete registered cycle pairs, hardware records only.

    Simulated pairs are never admitted here; the simulation is reported in its own
    section and is never pooled into this endpoint.
    """
    fast = [float(row["fast_endpoint_squared_residual"]) for row in pair_rows]
    slow = [float(row["slow_endpoint_squared_residual"]) for row in pair_rows]
    result = cadence_ratio_permutation_gate(
        fast, slow, alpha=alpha, permutations=permutations, seed=seed
    )
    result["evidence_scope"] = "hardware_registered_cycle_pairs_only"
    return result
