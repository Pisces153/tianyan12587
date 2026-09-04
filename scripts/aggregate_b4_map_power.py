#!/usr/bin/env python3
"""Retired scalar-throughput B-4 aggregator.

The former implementation silently discarded fixed per-setting overhead and
could generate an invalid one-dimensional R lookup.  It is kept only as an
explicit migration guard; use ``finalize_b4_map_timing_lookup.py``.
"""

from __future__ import annotations


MESSAGE = (
    "scalar throughput aggregation is retired; run "
    "scripts/finalize_b4_map_timing_lookup.py on the (shot_rate, setting_overhead) grid"
)


def decision_segments(*_args, **_kwargs):
    raise RuntimeError(MESSAGE)


def aggregate(*_args, **_kwargs):
    raise RuntimeError(MESSAGE)


def main() -> int:
    raise RuntimeError(MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
