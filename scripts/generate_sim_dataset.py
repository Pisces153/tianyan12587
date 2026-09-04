#!/usr/bin/env python3
"""Generate leakage-free QuTiP simulation shards from sampled 1024-shot counts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pauli import BASIS_ORDER, counts_array_to_pauli15, sample_all_counts, select_xobs6
from src.physics.hamiltonian import (
    auxiliary_labels,
    evolve_density_matrix,
    evolution_time_and_gamma,
    sample_parameters,
    sample_preparation_depolarization,
)
from src.protocol import load_json, validate_contract


def generate_one(sample_index: int, seed: int, shots: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + sample_index)
    parameters = sample_parameters(rng)
    evolution_time, gamma = evolution_time_and_gamma(parameters, rng)
    preparation_depolarization = sample_preparation_depolarization(rng)
    final_density_matrix = evolve_density_matrix(
        parameters, evolution_time, gamma, preparation_depolarization
    )
    counts = sample_all_counts(final_density_matrix, shots, rng)
    pauli15 = counts_array_to_pauli15(counts, shots=shots)
    return {
        "sample_index": sample_index,
        "sample_id": f"XA26-SIM-{sample_index:06d}",
        "split": "sim_holdout" if sample_index % 10 == 0 else "sim_train",
        "counts_9x64": counts,
        "pauli15_qa": pauli15,
        "xobs_model": select_xobs6(pauli15),
        "t": evolution_time,
        "gamma_sim": gamma,
        "preparation_depolarization": preparation_depolarization,
        "parameters": parameters,
        "auxiliary": auxiliary_labels(final_density_matrix, parameters),
    }


def _worker(item: tuple[int, int, int]) -> dict[str, Any]:
    return generate_one(*item)


def write_chunk(path: Path, rows: list[dict[str, Any]]) -> None:
    parameter_keys = ("h1", "h2", "Jz", "Jx", "Jy", "Jxz", "Jzx", "D", "hy1", "hy2", "hz1", "hz2")
    auxiliary_keys = ("entropies", "inter_entropies", "target_fidelities", "phase_labels")
    arrays: dict[str, np.ndarray] = {
        "sample_index": np.asarray([row["sample_index"] for row in rows], dtype=np.int64),
        "sample_id": np.asarray([row["sample_id"] for row in rows], dtype="U32"),
        "split": np.asarray([row["split"] for row in rows], dtype="U16"),
        "counts_9x64": np.asarray([row["counts_9x64"] for row in rows], dtype=np.int32),
        "pauli15_qa": np.asarray([row["pauli15_qa"] for row in rows], dtype=np.float32),
        "xobs_model": np.asarray([row["xobs_model"] for row in rows], dtype=np.float32),
        "t": np.asarray([[row["t"]] for row in rows], dtype=np.float32),
        "gamma_sim": np.asarray([[row["gamma_sim"]] for row in rows], dtype=np.float32),
        "preparation_depolarization": np.asarray(
            [[row["preparation_depolarization"]] for row in rows], dtype=np.float32
        ),
    }
    for key in parameter_keys:
        arrays[key] = np.asarray([[row["parameters"][key]] for row in rows], dtype=np.float32)
    for key in auxiliary_keys:
        dtype = np.int64 if key == "phase_labels" else np.float32
        arrays[key] = np.asarray([[row["auxiliary"][key]] for row in rows], dtype=dtype)
    np.savez_compressed(path, **arrays)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for NPZ simulation shards")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "config" / "protocol_v2.json"))
    parser.add_argument("--backends", default=str(PROJECT_ROOT / "config" / "backends_v1.json"))
    args = parser.parse_args()
    if args.samples < 2 or args.chunk_size < 1 or args.shots < 1 or args.workers < 1:
        raise ValueError("samples >= 2 and chunk-size, shots, workers must be positive")

    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = load_json(protocol_path)
    contract_errors = validate_contract(protocol, load_json(args.backends))
    if contract_errors:
        raise RuntimeError(f"Protocol contract errors: {contract_errors}")
    if args.shots != protocol["measurement"]["shots_per_basis"]:
        raise ValueError("--shots must match protocol.measurement.shots_per_basis")

    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.npz"))
    if existing:
        raise FileExistsError(f"Output directory already contains NPZ shards: {output_dir}")

    all_indices = list(range(args.samples))
    worker_items = [(index, args.seed, args.shots) for index in all_indices]
    manifest_chunks: list[dict[str, Any]] = []
    if args.workers == 1:
        generated = map(_worker, worker_items)
        pool = None
    else:
        pool = get_context("spawn").Pool(processes=args.workers)
        generated = pool.imap(_worker, worker_items)

    try:
        current_chunk: list[dict[str, Any]] = []
        for row in generated:
            current_chunk.append(row)
            if len(current_chunk) == args.chunk_size:
                chunk_index = len(manifest_chunks)
                path = output_dir / f"sim_{chunk_index:04d}.npz"
                write_chunk(path, current_chunk)
                manifest_chunks.append(
                    {
                        "file": path.name,
                        "sha256": sha256(path),
                        "samples": len(current_chunk),
                        "first_sample_id": current_chunk[0]["sample_id"],
                        "last_sample_id": current_chunk[-1]["sample_id"],
                    }
                )
                print(f"Wrote {path.name}: {len(current_chunk)} samples", flush=True)
                current_chunk = []
        if current_chunk:
            chunk_index = len(manifest_chunks)
            path = output_dir / f"sim_{chunk_index:04d}.npz"
            write_chunk(path, current_chunk)
            manifest_chunks.append(
                {
                    "file": path.name,
                    "sha256": sha256(path),
                    "samples": len(current_chunk),
                    "first_sample_id": current_chunk[0]["sample_id"],
                    "last_sample_id": current_chunk[-1]["sample_id"],
                }
            )
            print(f"Wrote {path.name}: {len(current_chunk)} samples", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    protocol_sha256 = sha256(protocol_path)
    manifest = {
        "schema_version": "sim_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "QuTiP local-depolarization Lindblad evolution + sampled counts",
        "backend_role": "local_reference_for_primary_noisy_density_matrix",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "samples": args.samples,
        "shots_per_basis": args.shots,
        "preparation_depolarization_range": [0.0, 0.2],
        "lindblad_noise_model": "local_isotropic_depolarization",
        "basis_order": list(BASIS_ORDER),
        "bit_order": "q0_leftmost",
        "chunks": manifest_chunks,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Completed {args.samples} samples in {output_dir}")


if __name__ == "__main__":
    main()
