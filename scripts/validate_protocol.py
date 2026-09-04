#!/usr/bin/env python3
"""Validate the protocol and report whether platform submission is blocked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.protocol import load_json, submission_blockers, validate_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "config" / "protocol_v2.json"))
    parser.add_argument("--backends", default=str(PROJECT_ROOT / "config" / "backends_v1.json"))
    parser.add_argument("--g0", default=None)
    args = parser.parse_args()

    protocol = load_json(args.protocol)
    backends = load_json(args.backends)
    g0 = load_json(args.g0) if args.g0 else None
    contract_errors = validate_contract(protocol, backends)
    blockers = submission_blockers(protocol, backends, g0)
    report = {
        "contract_valid": not contract_errors,
        "submission_ready": not blockers,
        "contract_errors": contract_errors,
        "submission_blockers": blockers,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not contract_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
