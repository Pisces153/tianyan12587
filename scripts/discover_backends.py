#!/usr/bin/env python3
"""Read TianYan machine codes without creating or submitting an experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends.tianyan_discovery import query_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "artifacts" / "logs" / "backend_inventory.json"),
        help="Output JSON path. The file contains machine metadata but no credentials.",
    )
    args = parser.parse_args()
    try:
        inventory = query_inventory()
    except RuntimeError as exc:
        print(f"Backend discovery failed: {exc}", file=sys.stderr)
        return 2
    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved read-only machine inventory: {output_path}")
    for role, candidates in inventory["role_candidates"].items():
        print(f"{role}: {', '.join(candidates) if candidates else 'no exact display-name match'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
