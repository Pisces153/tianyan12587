#!/usr/bin/env python3
"""Command-line entry point for XA-202609 simulation pretraining."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.engine import parse_args, run_training


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
