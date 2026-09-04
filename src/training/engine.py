"""Reproducible training engine for the hardware-facing AEMTN model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.models.aemtn_hardware import AEMTNHardware, ModelConfig
from src.training.dataset import (
    CompetitionDataset,
    NormalizationStats,
    discover_split_names,
    split_dataset,
)


DEFAULT_SEEDS = (20260731, 20260801, 20260802)
TRAIN_SPLIT_NAMES = {"train", "sim_train"}
VALIDATION_SPLIT_NAMES = {"val", "valid", "validation", "holdout", "sim_holdout"}
# Frozen manuscript loss initialization.  Local fields are naturally stronger
# signals; Jz is deliberately upweighted to counter information shielding.
TRAINING_LOSS_WEIGHTS = {"h1": 1.0, "h2": 1.0, "jz": 80.0}


@dataclass(frozen=True)
class TrainConfig:
    data: tuple[str, ...]
    output_dir: str
    validation_data: tuple[str, ...] = ()
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    target_names: tuple[str, ...] = ("h1", "h2", "Jz")
    xobs_order: tuple[str, ...] = ("X0", "Y0", "Z0", "X0X1", "Y0Y1", "Z0Z1")
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    split_seed: int = 20260727
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    early_stopping_patience: int = 10
    device: str = "auto"
    amp: bool = True
    deterministic: bool = True
    r_dim: int = 320
    task_dim: int = 6
    num_subspaces: int = 4
    dropout: float = 0.1
    resume: str | None = None

    def validate(self) -> None:
        if not self.data:
            raise ValueError("At least one training data path is required.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid optimizer settings.")
        if self.resume and len(self.seeds) != 1:
            raise ValueError("--resume requires exactly one seed.")
        if len(self.xobs_order) != 6 or len(set(self.xobs_order)) != 6:
            raise ValueError("--xobs-order requires six unique observable names.")


class TaskUncertaintyWeighting(nn.Module):
    """Learned homoscedastic weights with fixed physics priorities."""

    def __init__(self, names: Sequence[str], fixed_weights: Mapping[str, float]) -> None:
        super().__init__()
        self.names = tuple(names)
        self.name_to_index = {name: index for index, name in enumerate(self.names)}
        weights = [float(fixed_weights.get(name, 1.0)) for name in self.names]
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("All fixed loss weights must be positive.")
        self.register_buffer("fixed_weights", torch.tensor(weights, dtype=torch.float32))
        self.log_scales = nn.Parameter(torch.zeros(len(self.names), dtype=torch.float32))

    def forward(self, losses: Mapping[str, Tensor]) -> Tensor:
        if set(losses) != set(self.names):
            raise ValueError(
                f"Loss components changed: got {sorted(losses)}, expected {sorted(self.names)}"
            )
        total = torch.zeros((), device=self.log_scales.device)
        for name in self.names:
            index = self.name_to_index[name]
            log_scale = self.log_scales[index].clamp(min=-5.0, max=5.0)
            total = total + (
                torch.exp(-log_scale) * self.fixed_weights[index] * losses[name]
                + log_scale
            )
        return total

    def effective_weights(self) -> dict[str, float]:
        values = torch.exp(-self.log_scales.detach().clamp(-5.0, 5.0)) * self.fixed_weights
        return {name: float(values[index].cpu()) for index, name in enumerate(self.names)}


def fixed_loss_weights(target_names: Sequence[str], auxiliary_names: Iterable[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in target_names:
        result[f"target:{name}"] = TRAINING_LOSS_WEIGHTS.get(name.lower(), 1.0)
    auxiliary_weights = {
        "entropies": 0.1,
        "inter_entropies": 5.0,
        "target_fidelities": 1.0,
        "phase_labels": 1.0,
    }
    for name in auxiliary_names:
        result[f"aux:{name}"] = auxiliary_weights[name]
    return result


def set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _worker_seed(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, torch.Generator]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_seed if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )
    return loader, generator


def build_datasets(config: TrainConfig) -> tuple[Dataset, Dataset, tuple[str, ...]]:
    if config.validation_data:
        train_dataset = CompetitionDataset(config.data, config.target_names)
        validation_dataset = CompetitionDataset(config.validation_data, config.target_names)
        if train_dataset.auxiliary_names != validation_dataset.auxiliary_names:
            raise ValueError("Training and validation auxiliary labels differ.")
        return train_dataset, validation_dataset, train_dataset.auxiliary_names

    split_names = discover_split_names(config.data)
    train_splits = split_names.intersection(TRAIN_SPLIT_NAMES)
    validation_splits = split_names.intersection(VALIDATION_SPLIT_NAMES)
    if train_splits and validation_splits:
        unknown = split_names.difference(TRAIN_SPLIT_NAMES | VALIDATION_SPLIT_NAMES)
        if unknown:
            raise ValueError(
                f"Unrecognized split names {sorted(unknown)}. Refusing to mix them into training."
            )
        train_dataset = CompetitionDataset(
            config.data, config.target_names, split_names=train_splits
        )
        validation_dataset = CompetitionDataset(
            config.data, config.target_names, split_names=validation_splits
        )
        if train_dataset.auxiliary_names != validation_dataset.auxiliary_names:
            raise ValueError("Training and validation auxiliary labels differ.")
        return train_dataset, validation_dataset, train_dataset.auxiliary_names
    if split_names:
        raise ValueError(
            "Split metadata exists but does not contain both a recognized train and validation "
            f"split. Found: {sorted(split_names)}"
        )

    full_dataset = CompetitionDataset(config.data, config.target_names)
    train_dataset, validation_dataset = split_dataset(
        full_dataset,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    return train_dataset, validation_dataset, full_dataset.auxiliary_names


def move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        "xobs_model": batch["xobs_model"].to(device, non_blocking=True),
        "t": batch["t"].to(device, non_blocking=True),
        "targets": {
            name: value.to(device, non_blocking=True)
            for name, value in batch["targets"].items()
        },
        "auxiliary": {
            name: value.to(device, non_blocking=True)
            for name, value in batch["auxiliary"].items()
        },
    }


def compute_components(
    output: Mapping[str, object],
    batch: Mapping[str, object],
    normalization: NormalizationStats,
    target_names: Sequence[str],
    auxiliary_names: Sequence[str],
) -> dict[str, Tensor]:
    components: dict[str, Tensor] = {}
    predictions = output["predictions"]
    log_variances = output["log_variances"]
    for name in target_names:
        target = normalization.normalize_target(name, batch["targets"][name])
        mean = predictions[name]
        log_variance = log_variances[name]
        squared_error = (mean - target).square()
        # The constant keeps the bounded-log-variance NLL non-negative for task weighting.
        components[f"target:{name}"] = 0.5 * (
            torch.exp(-log_variance) * squared_error + log_variance + 8.0
        ).mean()

    auxiliary_output = output["auxiliary"]
    for name in auxiliary_names:
        target = batch["auxiliary"][name]
        if name == "phase_labels":
            components[f"aux:{name}"] = F.cross_entropy(
                auxiliary_output["phase_logits"], target.reshape(-1).long()
            )
        else:
            components[f"aux:{name}"] = F.mse_loss(auxiliary_output[name], target.float())
    return components


def _new_metric_state(target_names: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "squared": 0.0,
            "absolute": 0.0,
            "target_sum": 0.0,
            "target_squared_sum": 0.0,
            "inside_1sigma": 0.0,
            "inside_2sigma": 0.0,
        }
        for name in target_names
    }


@torch.no_grad()
def evaluate(
    model: AEMTNHardware,
    task_weighting: TaskUncertaintyWeighting,
    loader: DataLoader,
    normalization: NormalizationStats,
    target_names: Sequence[str],
    auxiliary_names: Sequence[str],
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    model.eval()
    task_weighting.eval()
    total_loss = 0.0
    sample_count = 0
    metric_state = _new_metric_state(target_names)

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        batch_size = int(batch["xobs_model"].shape[0])
        normalized_time = normalization.normalize_time(batch["t"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(batch["xobs_model"], normalized_time)
            components = compute_components(
                output, batch, normalization, target_names, auxiliary_names
            )
            loss = task_weighting(components)
        if not torch.isfinite(loss):
            raise FloatingPointError("Validation loss became NaN or Inf.")
        total_loss += float(loss) * batch_size
        sample_count += batch_size

        for name in target_names:
            prediction = normalization.denormalize_target(
                name, output["predictions"][name].float()
            )
            uncertainty = normalization.physical_std(
                name, output["log_variances"][name].float()
            )
            error = prediction - batch["targets"][name]
            target = batch["targets"][name]
            metric_state[name]["squared"] += float(error.square().sum())
            metric_state[name]["absolute"] += float(error.abs().sum())
            metric_state[name]["target_sum"] += float(target.sum())
            metric_state[name]["target_squared_sum"] += float(target.square().sum())
            metric_state[name]["inside_1sigma"] += float(
                (error.abs() <= uncertainty).sum()
            )
            metric_state[name]["inside_2sigma"] += float(
                (error.abs() <= 2.0 * uncertainty).sum()
            )

    if sample_count == 0:
        raise ValueError("Validation loader is empty.")
    targets: dict[str, dict[str, float]] = {}
    normalized_rmse_sum = 0.0
    for name in target_names:
        state = metric_state[name]
        rmse = math.sqrt(state["squared"] / sample_count)
        target_sst = state["target_squared_sum"] - state["target_sum"] ** 2 / sample_count
        targets[name] = {
            "rmse": rmse,
            "mae": state["absolute"] / sample_count,
            "r2": 1.0 - state["squared"] / max(target_sst, 1e-12),
            "coverage_1sigma": state["inside_1sigma"] / sample_count,
            "coverage_2sigma": state["inside_2sigma"] / sample_count,
        }
        normalized_rmse_sum += rmse / normalization.target_std[name]
    return {
        "loss": total_loss / sample_count,
        "selection_metric": normalized_rmse_sum / len(target_names),
        "sample_count": sample_count,
        "targets": targets,
    }


def _make_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def _rng_state(loader_generator: torch.Generator) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, object], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_payload(
    *,
    epoch: int,
    seed: int,
    best_metric: float,
    epochs_without_improvement: int,
    model: AEMTNHardware,
    task_weighting: TaskUncertaintyWeighting,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: object,
    normalization: NormalizationStats,
    loader_generator: torch.Generator,
    config: TrainConfig,
    auxiliary_names: Sequence[str],
) -> dict[str, object]:
    return {
        "checkpoint_version": 1,
        "epoch": epoch,
        "seed": seed,
        "best_metric": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "model_state": model.state_dict(),
        "task_weighting_state": task_weighting.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": model.config.to_dict(),
        "normalization": normalization.to_dict(),
        "train_config": asdict(config),
        "data_contract": {
            "xobs_key": "xobs_model",
            "xobs_order": list(model.config.xobs_order),
            "x_dim": 6,
            "control_key": "t",
            "control_dim": 1,
            "target_names": list(config.target_names),
            "auxiliary_names": list(auxiliary_names),
            "legacy_checkpoint_compatible": False,
        },
        "rng_state": _rng_state(loader_generator),
    }


def write_history(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _flatten_validation_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    result = {
        "val_loss": float(metrics["loss"]),
        "selection_metric": float(metrics["selection_metric"]),
    }
    for name, values in metrics["targets"].items():
        for metric_name, value in values.items():
            result[f"{name}_{metric_name}"] = float(value)
    return result


def train_one_seed(
    config: TrainConfig,
    seed: int,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    auxiliary_names: Sequence[str],
) -> dict[str, object]:
    set_global_seed(seed, config.deterministic)
    device = resolve_device(config.device)
    use_amp = bool(config.amp and device.type == "cuda")
    run_dir = Path(config.output_dir).expanduser().resolve() / f"seed_{seed}"
    if not config.resume and any(
        (run_dir / name).exists() for name in ("history.csv", "best.pt", "last.pt")
    ):
        raise FileExistsError(
            f"Run directory already contains training artifacts: {run_dir}. "
            "Choose a new --out directory or resume explicitly."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    normalization = NormalizationStats.fit(train_dataset, config.target_names)
    model_config = ModelConfig(
        target_names=config.target_names,
        xobs_order=config.xobs_order,
        r_dim=config.r_dim,
        task_dim=config.task_dim,
        num_subspaces=config.num_subspaces,
        dropout=config.dropout,
    )
    model = AEMTNHardware(model_config).to(device)
    component_names = [f"target:{name}" for name in config.target_names]
    component_names.extend(f"aux:{name}" for name in auxiliary_names)
    loss_weights = fixed_loss_weights(config.target_names, auxiliary_names)
    task_weighting = TaskUncertaintyWeighting(component_names, loss_weights).to(device)
    optimizer = AdamW(
        [
            {"params": model.parameters(), "weight_decay": config.weight_decay},
            {"params": task_weighting.parameters(), "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=1e-6)
    scaler = _make_scaler(use_amp)

    train_loader, train_generator = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        seed=seed,
    )
    validation_loader, _ = make_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        seed=config.split_seed,
    )

    start_epoch = 1
    best_metric = math.inf
    epochs_without_improvement = 0
    if config.resume:
        checkpoint = load_checkpoint(config.resume, device)
        if int(checkpoint["seed"]) != seed:
            raise ValueError("Resume checkpoint seed does not match --seeds.")
        if ModelConfig.from_dict(checkpoint["model_config"]) != model_config:
            raise ValueError("Resume checkpoint model configuration does not match this run.")
        checkpoint_normalization = NormalizationStats.from_dict(checkpoint["normalization"])
        if checkpoint_normalization != normalization:
            raise ValueError("Resume data normalization changed; refusing a non-reproducible resume.")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        task_weighting.load_state_dict(checkpoint["task_weighting_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        _restore_rng_state(checkpoint["rng_state"], train_generator)

    run_metadata = {
        "seed": seed,
        "device": str(device),
        "amp_enabled": use_amp,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "model_config": model_config.to_dict(),
        "normalization": normalization.to_dict(),
        "fixed_loss_weights": loss_weights,
        "training_loss_weights": loss_weights,
        "auxiliary_names": list(auxiliary_names),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2, ensure_ascii=False)

    print(
        f"[seed {seed}] device={device} amp={use_amp} "
        f"train={len(train_dataset)} validation={len(validation_dataset)}"
    )
    last_validation: dict[str, object] | None = None
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        task_weighting.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_samples = 0
        start_time = time.perf_counter()

        for step, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            batch_size = int(batch["xobs_model"].shape[0])
            normalized_time = normalization.normalize_time(batch["t"])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch["xobs_model"], normalized_time)
                components = compute_components(
                    output, batch, normalization, config.target_names, auxiliary_names
                )
                full_loss = task_weighting(components)
                loss = full_loss / config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Training loss became NaN or Inf at epoch {epoch}, step {step}."
                )
            scaler.scale(loss).backward()

            update_now = (
                step % config.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if update_now:
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    [*model.parameters(), *task_weighting.parameters()],
                    config.max_gradient_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(full_loss.detach()) * batch_size
            epoch_samples += batch_size

        scheduler.step()
        train_loss = epoch_loss / max(1, epoch_samples)
        last_validation = evaluate(
            model,
            task_weighting,
            validation_loader,
            normalization,
            config.target_names,
            auxiliary_names,
            device,
            use_amp,
        )
        selection_metric = float(last_validation["selection_metric"])
        improved = selection_metric < best_metric
        if improved:
            best_metric = selection_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            epoch=epoch,
            seed=seed,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            model=model,
            task_weighting=task_weighting,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            normalization=normalization,
            loader_generator=train_generator,
            config=config,
            auxiliary_names=auxiliary_names,
        )
        atomic_torch_save(payload, run_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, run_dir / "best.pt")

        elapsed = time.perf_counter() - start_time
        history_row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            **_flatten_validation_metrics(last_validation),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        write_history(run_dir / "history.csv", history_row)
        target_summary = " ".join(
            f"{name}_rmse={last_validation['targets'][name]['rmse']:.6f}"
            for name in config.target_names
        )
        print(
            f"[seed {seed}] epoch={epoch:03d} train={train_loss:.6f} "
            f"val={last_validation['loss']:.6f} score={selection_metric:.6f} "
            f"{target_summary}"
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"[seed {seed}] early stopping after epoch {epoch}")
            break

    if last_validation is None:
        raise RuntimeError("No epoch ran. Increase --epochs or use a checkpoint from an earlier epoch.")
    best_checkpoint = load_checkpoint(run_dir / "best.pt", device)
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    task_weighting.load_state_dict(best_checkpoint["task_weighting_state"], strict=True)
    best_validation = evaluate(
        model,
        task_weighting,
        validation_loader,
        normalization,
        config.target_names,
        auxiliary_names,
        device,
        use_amp,
    )
    summary = {
        "seed": seed,
        "best_selection_metric": best_metric,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation": best_validation,
        "last_validation": last_validation,
        "effective_loss_weights": task_weighting.effective_weights(),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def run_training(config: TrainConfig) -> list[dict[str, object]]:
    config.validate()
    train_dataset, validation_dataset, auxiliary_names = build_datasets(config)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        train_one_seed(
            config,
            seed,
            train_dataset,
            validation_dataset,
            auxiliary_names,
        )
        for seed in config.seeds
    ]
    aggregate_targets: dict[str, dict[str, dict[str, float]]] = {}
    for target in config.target_names:
        aggregate_targets[target] = {}
        metric_names = summaries[0]["best_validation"]["targets"][target].keys()
        for metric_name in metric_names:
            values = np.asarray(
                [
                    summary["best_validation"]["targets"][target][metric_name]
                    for summary in summaries
                ],
                dtype=np.float64,
            )
            aggregate_targets[target][metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    aggregate = {
        "seeds": list(config.seeds),
        "mean_best_selection_metric": float(
            np.mean([summary["best_selection_metric"] for summary in summaries])
        ),
        "target_metrics": aggregate_targets,
        "runs": summaries,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2, ensure_ascii=False)
    return summaries


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train the leakage-free XA-202609 AEMTN model."
    )
    parser.add_argument("--data", nargs="+", required=True, help="Training NPZ files/directories")
    parser.add_argument("--val-data", nargs="*", default=(), help="Optional validation data")
    parser.add_argument("--out", required=True, help="Output checkpoint directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--targets", nargs="+", default=["h1", "h2", "Jz"])
    parser.add_argument(
        "--xobs-order",
        nargs=6,
        default=["X0", "Y0", "Z0", "X0X1", "Y0Y1", "Z0Z1"],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--r-dim", type=int, default=320)
    parser.add_argument("--task-dim", type=int, default=6)
    parser.add_argument("--num-subspaces", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    return TrainConfig(
        data=tuple(args.data),
        validation_data=tuple(args.val_data),
        output_dir=args.out,
        seeds=tuple(args.seeds),
        target_names=tuple(args.targets),
        xobs_order=tuple(args.xobs_order),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_gradient_norm=args.max_gradient_norm,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        amp=args.amp,
        deterministic=args.deterministic,
        r_dim=args.r_dim,
        task_dim=args.task_dim,
        num_subspaces=args.num_subspaces,
        dropout=args.dropout,
        resume=args.resume,
    )
