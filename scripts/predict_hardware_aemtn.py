#!/usr/bin/env python3
"""Run label-free AEMTN ensemble inference from a hardware count trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pauli import XOBS_ORDER, counts_array_to_pauli15, select_pauli_features
from src.models.aemtn_hardware import AEMTNHardware, ModelConfig
from src.training.dataset import NormalizationStats
from src.training.engine import load_checkpoint


def hardware_features(
    path: Path, feature_order: tuple[str, ...] = XOBS_ORDER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        counts = np.asarray(data["counts_trajectory"], dtype=np.int32)
        sample_id = np.asarray(data["sample_id"]).astype("U64")
        times = np.asarray(data["times"], dtype=np.float32)
    if counts.ndim != 4 or counts.shape[2:] != (9, 64):
        raise ValueError(f"Expected counts shape (samples, times, 9, 64); got {counts.shape}")
    if times.shape != counts.shape[:2]:
        raise ValueError("Time grid does not match the count trajectory.")
    xobs = np.asarray(
        [[select_pauli_features(counts_array_to_pauli15(counts[sample, time]), feature_order) for time in range(counts.shape[1])]
         for sample in range(counts.shape[0])],
        dtype=np.float32,
    )
    return sample_id, times, xobs


def predict_checkpoint(path: Path, xobs: np.ndarray, times: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    device = torch.device("cpu")
    checkpoint = load_checkpoint(path, device)
    contract = checkpoint.get("data_contract", {})
    if contract.get("xobs_key") != "xobs_model" or contract.get("control_key") != "t":
        raise ValueError(f"{path} is not an xobs_model + t checkpoint.")
    model = AEMTNHardware(ModelConfig.from_dict(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    normalization = NormalizationStats.from_dict(checkpoint["normalization"])
    flat_xobs = torch.from_numpy(xobs.reshape(-1, 6))
    flat_t = torch.from_numpy(times.reshape(-1, 1))
    with torch.inference_mode():
        output = model(flat_xobs, normalization.normalize_time(flat_t))
    means: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    for target in model.config.target_names:
        mean = normalization.denormalize_target(target, output["predictions"][target]).cpu().numpy()
        std = normalization.physical_std(target, output["log_variances"][target]).cpu().numpy()
        means[target] = mean.reshape(xobs.shape[:2])
        variances[target] = np.square(std.reshape(xobs.shape[:2]))
    return means, variances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.out.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.out}")
    contracts = [
        load_checkpoint(path, torch.device("cpu")).get("data_contract", {})
        for path in arguments.checkpoints
    ]
    feature_orders = [tuple(contract.get("xobs_order", XOBS_ORDER)) for contract in contracts]
    if len(set(feature_orders)) != 1:
        raise ValueError("Ensemble checkpoints use different xobs_order contracts.")
    sample_id, times, xobs = hardware_features(arguments.counts, feature_orders[0])
    ensemble = [predict_checkpoint(path, xobs, times) for path in arguments.checkpoints]
    targets = tuple(ensemble[0][0])
    predictions: dict[str, dict[str, object]] = {}
    for target in targets:
        means = np.stack([item[0][target] for item in ensemble], axis=0)
        variances = np.stack([item[1][target] for item in ensemble], axis=0)
        mean = means.mean(axis=0)
        total_variance = np.maximum(0.0, (variances + np.square(means)).mean(axis=0) - np.square(mean))
        predictions[target] = {"mean": mean.tolist(), "std": np.sqrt(total_variance).tolist()}
    payload = {
        "source_type": "hardware_label_free_inference",
        "training_forbidden": True,
        "model_input": {"xobs_model": list(feature_orders[0]), "t": [1]},
        "forbidden_inputs": ["h1", "h2", "Jz", "known_controls", "hardware_manifest_parameters"],
        "sample_id": sample_id.tolist(),
        "times": times.tolist(),
        "checkpoints": [str(path.resolve()) for path in arguments.checkpoints],
        "predictions": predictions,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(arguments.out), "sample_count": len(sample_id), "time_count": times.shape[1], "targets": list(targets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
