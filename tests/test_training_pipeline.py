from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.models.aemtn_hardware import ModelConfig, AEMTNHardware
from src.training.dataset import CompetitionDataset
from src.training.engine import TRAINING_LOSS_WEIGHTS, TrainConfig, fixed_loss_weights, load_checkpoint, run_training


def write_synthetic_shard(path: Path, count: int = 32, *, include_legacy: bool = False) -> None:
    rng = np.random.default_rng(20260727)
    h1 = rng.uniform(-1.0, 1.0, count).astype(np.float32)
    h2 = rng.uniform(-1.0, 1.0, count).astype(np.float32)
    jz = rng.uniform(-0.5, 1.5, count).astype(np.float32)
    t = rng.uniform(0.05, 2.0, count).astype(np.float32)
    noise = rng.normal(0.0, 0.01, (count, 6)).astype(np.float32)
    xobs = np.stack(
        (
            np.tanh(h1 * t),
            np.tanh(h2 * t),
            np.tanh(jz * t),
            np.tanh((h1 + h2) * t / 2.0),
            np.tanh((h2 + jz) * t / 2.0),
            np.tanh((h1 + jz) * t / 2.0),
        ),
        axis=1,
    ).astype(np.float32)
    xobs = np.clip(xobs + noise, -1.0, 1.0)
    entropy = np.clip(0.5 + 0.2 * np.abs(jz), 0.0, 2.0).astype(np.float32)
    inter_entropy = np.clip(entropy + 0.1 * np.abs(h1 - h2), 0.0, 2.0).astype(np.float32)
    fidelity = np.clip(1.0 - 0.1 * np.abs(jz), 0.0, 1.0).astype(np.float32)
    phase = np.digitize(h1 + h2 + jz, (-0.25, 0.75)).astype(np.int64)
    split = np.array(["sim_train"] * 24 + ["sim_holdout"] * (count - 24))
    fields = {
        "xobs_model": xobs,
        "t": t[:, None],
        "h1": h1[:, None],
        "h2": h2[:, None],
        "Jz": jz[:, None],
        "entropies": entropy[:, None],
        "inter_entropies": inter_entropy[:, None],
        "target_fidelities": fidelity[:, None],
        "phase_labels": phase[:, None],
        "split": split,
    }
    if include_legacy:
        fields["v"] = np.stack((t, h1, h2, jz), axis=1)
    np.savez(path, **fields)


class ModelContractTests(unittest.TestCase):
    def test_jz_loss_compensation_is_frozen_for_the_weak_signal(self) -> None:
        self.assertEqual(TRAINING_LOSS_WEIGHTS, {"h1": 1.0, "h2": 1.0, "jz": 80.0})
        weights = fixed_loss_weights(("h1", "h2", "Jz"), ())
        self.assertEqual(weights, {"target:h1": 1.0, "target:h2": 1.0, "target:Jz": 80.0})

    def test_model_uses_six_observables_and_one_time_value(self) -> None:
        model = AEMTNHardware(ModelConfig(r_dim=32, task_dim=8, num_subspaces=4))
        output = model(torch.zeros(4, 6), torch.ones(4, 1))
        self.assertEqual(tuple(model.backbone.x_branch[0].weight.shape), (32, 6))
        self.assertEqual(tuple(model.backbone.t_branch[0].weight.shape), (32, 1))
        self.assertEqual(set(output["predictions"]), {"h1", "h2", "Jz"})
        for name in ("h1", "h2", "Jz"):
            self.assertEqual(tuple(output["predictions"][name].shape), (4, 1))
            self.assertTrue(torch.isfinite(output["predictions"][name]).all())
            self.assertTrue(torch.isfinite(output["log_variances"][name]).all())

    def test_model_rejects_legacy_width(self) -> None:
        model = AEMTNHardware(ModelConfig(r_dim=32, task_dim=8, num_subspaces=4))
        with self.assertRaisesRegex(ValueError, "xobs must have shape"):
            model(torch.zeros(2, 8192), torch.ones(2, 1))
        with self.assertRaisesRegex(ValueError, "evolution_time must have shape"):
            model(torch.zeros(2, 6), torch.ones(2, 14))


class DatasetContractTests(unittest.TestCase):
    def test_split_selection_and_target_separation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "data.npz"
            write_synthetic_shard(shard)
            dataset = CompetitionDataset([shard], split_names={"sim_train"})
            self.assertEqual(len(dataset), 24)
            item = dataset[0]
            self.assertEqual(tuple(item["xobs_model"].shape), (6,))
            self.assertEqual(tuple(item["t"].shape), (1,))
            self.assertEqual(set(item["targets"]), {"h1", "h2", "Jz"})
            self.assertNotIn("h1", item)
            self.assertNotIn("Jz", item)

    def test_legacy_v_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "leaked.npz"
            write_synthetic_shard(shard, include_legacy=True)
            with self.assertRaisesRegex(ValueError, "forbidden legacy input keys"):
                CompetitionDataset([shard])


class TrainingSmokeTest(unittest.TestCase):
    def test_one_epoch_writes_a_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "data.npz"
            output_dir = root / "checkpoints"
            write_synthetic_shard(shard)
            config = TrainConfig(
                data=(str(shard),),
                output_dir=str(output_dir),
                seeds=(7,),
                epochs=1,
                batch_size=8,
                num_workers=0,
                device="cpu",
                amp=False,
                r_dim=32,
                task_dim=8,
                num_subspaces=4,
                early_stopping_patience=2,
            )
            summaries = run_training(config)
            self.assertEqual(len(summaries), 1)
            checkpoint_path = output_dir / "seed_7" / "best.pt"
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
            self.assertEqual(checkpoint["checkpoint_version"], 1)
            self.assertEqual(checkpoint["data_contract"]["x_dim"], 6)
            self.assertEqual(checkpoint["data_contract"]["control_dim"], 1)
            self.assertFalse(checkpoint["data_contract"]["legacy_checkpoint_compatible"])
            first_layer = checkpoint["model_state"]["backbone.x_branch.0.weight"]
            self.assertEqual(tuple(first_layer.shape), (32, 6))
            for key in (
                "model_state",
                "task_weighting_state",
                "optimizer_state",
                "scheduler_state",
                "scaler_state",
                "normalization",
                "rng_state",
            ):
                self.assertIn(key, checkpoint)


if __name__ == "__main__":
    unittest.main()
