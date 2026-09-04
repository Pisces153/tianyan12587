#!/usr/bin/env python3
"""Validate that simulation shards comply with the AEMTN hardware input contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pauli import XOBS_INDICES
from src.training.dataset import CompetitionDataset, discover_npz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    files = discover_npz([args.data])
    dataset = CompetitionDataset(files)
    errors: list[str] = []
    shard_report: list[dict[str, object]] = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            keys = set(data.files)
            required = {"xobs_model", "pauli15_qa", "counts_9x64", "t", "h1", "h2", "Jz"}
            missing = sorted(required.difference(keys))
            if missing:
                errors.append(f"{file}: missing {missing}")
            if {"x", "v"}.intersection(keys):
                errors.append(f"{file}: legacy input key present")
            if "xobs_model" in data and "pauli15_qa" in data:
                xobs = np.asarray(data["xobs_model"])
                pauli15 = np.asarray(data["pauli15_qa"])
                if xobs.shape[1:] != (6,) or pauli15.shape[1:] != (15,):
                    errors.append(f"{file}: unexpected feature shapes")
                elif not np.allclose(xobs, pauli15[:, XOBS_INDICES], atol=1e-7):
                    errors.append(f"{file}: xobs_model is not selected from pauli15_qa")
            if "counts_9x64" in data:
                counts = np.asarray(data["counts_9x64"])
                if counts.ndim != 3 or counts.shape[1:] != (9, 64):
                    errors.append(f"{file}: counts_9x64 has unexpected shape {counts.shape}")
                elif not np.all(counts.sum(axis=2) == 1024):
                    errors.append(f"{file}: counts do not sum to 1024 per basis")
            shard_report.append({"file": str(file), "keys": sorted(keys), "samples": int(data["xobs_model"].shape[0])})

    report = {
        "valid": not errors,
        "errors": errors,
        "dataset_samples": len(dataset),
        "auxiliary_names": list(dataset.auxiliary_names),
        "shards": shard_report,
    }
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
