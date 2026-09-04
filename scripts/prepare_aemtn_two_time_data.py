#!/usr/bin/env python3
"""Convert two-time Pauli trajectories into the strict AEMTN xobs6 + t contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pauli import select_xobs6


TARGETS = ("h1", "h2", "Jz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()

    rows: dict[str, list[np.ndarray]] = {name: [] for name in (*TARGETS, "xobs_model", "t", "split", "sample_id")}
    source_files = sorted(arguments.data.glob("*.npz"))
    if not source_files:
        raise FileNotFoundError(f"No trajectory shards found in {arguments.data}.")
    for path in source_files:
        with np.load(path, allow_pickle=False) as source:
            required = {"sample_id", "split", "times", "pauli15_trajectory", *TARGETS}
            missing = required.difference(source.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            pauli = np.asarray(source["pauli15_trajectory"], dtype=np.float32)
            times = np.asarray(source["times"], dtype=np.float32)
            if pauli.ndim != 3 or pauli.shape[2] != 15 or times.shape != pauli.shape[:2]:
                raise ValueError(f"{path} has an invalid trajectory layout.")
            sample_count, time_count = times.shape
            rows["xobs_model"].append(
                np.asarray([[select_xobs6(value) for value in trajectory] for trajectory in pauli], dtype=np.float32).reshape(-1, 6)
            )
            rows["t"].append(times.reshape(-1, 1))
            rows["split"].append(np.repeat(np.asarray(source["split"]).reshape(sample_count), time_count))
            identifiers = np.asarray(source["sample_id"]).reshape(sample_count).astype("U64")
            rows["sample_id"].append(
                np.asarray([f"{identifier}:t{time_index}" for identifier in identifiers for time_index in range(time_count)], dtype="U72")
            )
            for target in TARGETS:
                values = np.asarray(source[target], dtype=np.float32).reshape(sample_count, 1)
                rows[target].append(np.repeat(values, time_count, axis=0))

    output = arguments.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    merged = {name: np.concatenate(values, axis=0) for name, values in rows.items()}
    if not np.all(np.isfinite(merged["xobs_model"])) or np.max(np.abs(merged["xobs_model"])) > 1.0001:
        raise ValueError("xobs_model is outside the Pauli range.")
    np.savez_compressed(output, **merged)
    report = {
        "source_type": "simulator",
        "model_input": {"xobs_model": [6], "t": [1]},
        "forbidden_inputs": ["h1", "h2", "Jz", "gamma_sim", "preparation_depolarization", "known_controls"],
        "source_trajectory_files": len(source_files),
        "row_count": int(len(merged["t"])),
        "train_rows": int(np.sum(merged["split"].astype(str) == "sim_train")),
        "holdout_rows": int(np.sum(merged["split"].astype(str) == "sim_holdout")),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(output), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
