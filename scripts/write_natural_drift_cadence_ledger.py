#!/usr/bin/env python3
"""Write non-overwriting cadence-deviation ledger from frozen campaign evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive.cadence_ledger import build_cadence_ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite cadence ledger: {args.out}")
    result = build_cadence_ledger(args.campaign_root / "snapshots.jsonl", args.campaign_root / "campaign_manifest.json", args.config)
    args.out.mkdir(parents=True)
    (args.out / "cadence_ledger.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
