#!/usr/bin/env python3
"""Blind simulated closed-loop validation for the frozen v4 AEMTN ensemble."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.aemtn_hardware import AEMTNHardware, ModelConfig
from src.training.dataset import NormalizationStats
from src.training.engine import load_checkpoint


TARGETS = ("h1", "h2", "Jz")
AUTO_TARGETS = ("h1", "h2")


def load_generator():
    path = ROOT / "scripts" / "generate_aemtn_closed_loop_v4_data.py"
    spec = importlib.util.spec_from_file_location("generate_aemtn_closed_loop_v4_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the v4 data generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ensemble:
    def __init__(self, paths: list[Path]) -> None:
        self.members = []
        contracts = []
        for path in paths:
            checkpoint = load_checkpoint(path, torch.device("cpu"))
            config = ModelConfig.from_dict(checkpoint["model_config"])
            model = AEMTNHardware(config)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            self.members.append((model, NormalizationStats.from_dict(checkpoint["normalization"])))
            contracts.append(checkpoint["data_contract"])
        if not self.members:
            raise ValueError("At least one AEMTN checkpoint is required")
        if any(tuple(value["xobs_order"]) != ("X0", "Y0", "Z0", "X1", "Y1", "Z1") for value in contracts):
            raise ValueError("Every checkpoint must use the v4 local-six input contract")
        if any(tuple(value["target_names"]) != TARGETS for value in contracts):
            raise ValueError("Every checkpoint must predict h1, h2, and Jz")

    def predict(self, xobs: np.ndarray, times: tuple[float, ...]) -> dict[str, tuple[float, float]]:
        x = torch.from_numpy(np.asarray(xobs, dtype=np.float32))
        t = torch.tensor(times, dtype=torch.float32).reshape(-1, 1)
        member_means: dict[str, list[np.ndarray]] = {name: [] for name in TARGETS}
        member_variances: dict[str, list[np.ndarray]] = {name: [] for name in TARGETS}
        with torch.inference_mode():
            for model, normalization in self.members:
                output = model(x, normalization.normalize_time(t))
                for name in TARGETS:
                    mean = normalization.denormalize_target(
                        name, output["predictions"][name]
                    ).numpy().reshape(-1)
                    std = normalization.physical_std(
                        name, output["log_variances"][name]
                    ).numpy().reshape(-1)
                    member_means[name].append(mean)
                    member_variances[name].append(np.square(std))
        result = {}
        for name in TARGETS:
            means = np.stack(member_means[name])
            variances = np.stack(member_variances[name])
            mean_by_time = means.mean(axis=0)
            variance_by_time = np.maximum(
                (variances + np.square(means)).mean(axis=0) - np.square(mean_by_time),
                1e-8,
            )
            precision = np.reciprocal(variance_by_time)
            result[name] = (
                float(np.sum(mean_by_time * precision) / np.sum(precision)),
                float(np.sqrt(1.0 / np.sum(precision))),
            )
        return result


def features(parameters, contrast, bias, decay_rate, shots, rng, generator) -> np.ndarray:
    rows = []
    for evolution_time in generator.TIMES:
        circuit = generator.runner().build_logical_circuit(
            parameters,
            evolution_time,
            None,
            order=2,
            steps=1,
            measure=False,
        )
        state = Statevector.from_instruction(circuit)
        ideal = np.asarray(
            [float(np.real(state.expectation_value(pauli))) for pauli in generator.PAULIS]
        )
        rows.append(
            generator.noisy_expectations(
                ideal,
                contrast,
                bias,
                decay_rate * evolution_time,
                shots,
                rng,
            )
        )
    return np.asarray(rows, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.trials < 1 or args.shots < 1:
        parser.error("trials and shots must be positive")

    generator = load_generator()
    ensemble = Ensemble(args.checkpoints)
    rng = np.random.default_rng(args.seed)
    amplitudes = np.asarray((0.3, 0.5, 0.7))
    rows = []
    for index in range(args.trials):
        signs = rng.choice((-1.0, 1.0), size=2)
        truth = signs * rng.choice(amplitudes, size=2)
        target = json.loads(json.dumps(generator.NOMINAL))
        before = json.loads(json.dumps(target))
        before["hx"][0] += float(truth[0])
        before["hx"][1] += float(truth[1])
        contrast = rng.uniform(0.84, 1.0, size=6)
        bias = rng.uniform(-0.04, 0.04, size=6)
        decay_rate = float(rng.uniform(0.0, 0.35))
        target_xobs = features(target, contrast, bias, decay_rate, args.shots, rng, generator)
        before_xobs = features(before, contrast, bias, decay_rate, args.shots, rng, generator)
        reference_prediction = ensemble.predict(target_xobs, generator.TIMES)
        before_prediction = ensemble.predict(before_xobs, generator.TIMES)
        predicted = np.asarray(
            [before_prediction[name][0] - reference_prediction[name][0] for name in AUTO_TARGETS]
        )
        uncertainty = np.asarray(
            [
                np.hypot(before_prediction[name][1], reference_prediction[name][1])
                for name in AUTO_TARGETS
            ]
        )
        accepted = bool(
            np.all(np.abs(predicted) >= 0.12)
            and np.all(np.abs(predicted) / uncertainty >= 1.0)
            and np.all(uncertainty <= 0.35)
        )
        compensation = -np.clip(predicted, -0.8, 0.8) if accepted else np.zeros(2)
        after = json.loads(json.dumps(before))
        after["hx"][0] += float(compensation[0])
        after["hx"][1] += float(compensation[1])
        after_xobs = features(after, contrast, bias, decay_rate, args.shots, rng, generator)
        before_distance = float(np.mean(np.abs(before_xobs - target_xobs)))
        after_distance = float(np.mean(np.abs(after_xobs - target_xobs)))
        residual = truth + compensation
        before_residual = float(np.linalg.norm(truth))
        after_residual = float(np.linalg.norm(residual))
        success = bool(
            accepted
            and after_residual <= 0.8 * before_residual
            and after_distance < before_distance
        )
        rows.append(
            {
                "trial": index,
                "true_offset": truth.tolist(),
                "predicted_offset": predicted.tolist(),
                "uncertainty": uncertainty.tolist(),
                "accepted": accepted,
                "direction_correct": (np.sign(predicted) == np.sign(truth)).tolist(),
                "covered_1sigma": (np.abs(predicted - truth) <= uncertainty).tolist(),
                "residual_l2": {"before": before_residual, "after": after_residual},
                "xobs6_mae": {"before": before_distance, "after": after_distance},
                "success": success,
            }
        )

    accepted_rows = [row for row in rows if row["accepted"]]
    report = {
        "profile": "aemtn_h1h2_closed_loop_v4_blind_simulation",
        "seed": args.seed,
        "trial_count": len(rows),
        "checkpoint_paths": [str(path.resolve()) for path in args.checkpoints],
        "input_contract": {"xobs_model": list(generator.XOBS_ORDER), "t": [1]},
        "inference_estimator": "AEMTNHardware ensemble only",
        "analytic_inverse_used": False,
        "direction_accuracy": {
            name: float(np.mean([row["direction_correct"][index] for row in rows]))
            for index, name in enumerate(AUTO_TARGETS)
        },
        "coverage_1sigma": {
            name: float(np.mean([row["covered_1sigma"][index] for row in rows]))
            for index, name in enumerate(AUTO_TARGETS)
        },
        "accepted_fraction": float(len(accepted_rows) / len(rows)),
        "closed_loop_success_fraction": float(np.mean([row["success"] for row in rows])),
        "accepted_success_fraction": (
            float(np.mean([row["success"] for row in accepted_rows])) if accepted_rows else None
        ),
        "mean_parameter_residual_l2": {
            phase: float(np.mean([row["residual_l2"][phase] for row in rows]))
            for phase in ("before", "after")
        },
        "mean_xobs6_mae": {
            phase: float(np.mean([row["xobs6_mae"][phase] for row in rows]))
            for phase in ("before", "after")
        },
        "xobs6_improved_fraction": float(
            np.mean([row["xobs6_mae"]["after"] < row["xobs6_mae"]["before"] for row in rows])
        ),
        "Jz_policy": "always reject automatic compensation",
        "trials": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
