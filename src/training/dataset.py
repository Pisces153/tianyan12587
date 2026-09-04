"""Strict NPZ data contract for simulation pretraining and hardware fine-tuning."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


XOBS_KEY = "xobs_model"
TIME_KEYS = ("t", "t_evolve")
AUXILIARY_KEYS = (
    "entropies",
    "inter_entropies",
    "target_fidelities",
    "phase_labels",
)
LEGACY_INPUT_KEYS = ("x", "v")


@dataclass(frozen=True)
class ShardInfo:
    path: str
    count: int
    time_key: str


@dataclass(frozen=True)
class NormalizationStats:
    time_mean: float
    time_std: float
    target_mean: dict[str, float]
    target_std: dict[str, float]

    def normalize_time(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.time_mean) / self.time_std

    def normalize_target(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return (value - self.target_mean[name]) / self.target_std[name]

    def denormalize_target(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return value * self.target_std[name] + self.target_mean[name]

    def physical_std(self, name: str, log_variance: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * log_variance) * self.target_std[name]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "NormalizationStats":
        return cls(
            time_mean=float(value["time_mean"]),
            time_std=float(value["time_std"]),
            target_mean={k: float(v) for k, v in value["target_mean"].items()},
            target_std={k: float(v) for k, v in value["target_std"].items()},
        )

    @classmethod
    def fit(
        cls,
        dataset: Dataset,
        target_names: Sequence[str],
    ) -> "NormalizationStats":
        time_sum = 0.0
        time_sq_sum = 0.0
        target_sum = {name: 0.0 for name in target_names}
        target_sq_sum = {name: 0.0 for name in target_names}
        count = 0

        for item in dataset:
            time_value = float(item["t"].reshape(-1)[0])
            time_sum += time_value
            time_sq_sum += time_value * time_value
            for name in target_names:
                value = float(item["targets"][name].reshape(-1)[0])
                target_sum[name] += value
                target_sq_sum[name] += value * value
            count += 1

        if count < 2:
            raise ValueError("At least two training samples are required to fit normalization stats.")

        def stable_std(total: float, total_sq: float) -> tuple[float, float]:
            mean = total / count
            variance = max(total_sq / count - mean * mean, 1e-12)
            return mean, max(variance ** 0.5, 1e-6)

        time_mean, time_std = stable_std(time_sum, time_sq_sum)
        target_mean: dict[str, float] = {}
        target_std: dict[str, float] = {}
        for name in target_names:
            target_mean[name], target_std[name] = stable_std(
                target_sum[name], target_sq_sum[name]
            )
        return cls(time_mean, time_std, target_mean, target_std)


def discover_npz(paths: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".npz":
            files.add(path)
        elif path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob("*.npz"))
        else:
            raise FileNotFoundError(f"Data path does not exist or is not an NPZ file: {path}")
    if not files:
        raise ValueError("No .npz shards were found.")
    return sorted(files)


def discover_split_names(paths: Iterable[str | Path]) -> set[str]:
    names: set[str] = set()
    for path in discover_npz(paths):
        with np.load(path, allow_pickle=False) as data:
            if "split" not in data.files:
                continue
            values = np.asarray(data["split"])
            if values.ndim == 0:
                names.add(_normalize_split_value(values.item()))
            else:
                names.update(_normalize_split_value(value) for value in values.reshape(-1))
    return names


def _normalize_split_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class CompetitionDataset(Dataset):
    """Lazy shard dataset that refuses the legacy leaked input schema."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        target_names: Sequence[str] = ("h1", "h2", "Jz"),
        *,
        split_names: Iterable[str] | None = None,
        cache_size: int = 2,
        validate_ranges: bool = True,
    ) -> None:
        self.target_names = tuple(target_names)
        self.split_names = set(split_names) if split_names else None
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.shards: list[ShardInfo] = []
        self.records: list[tuple[int, int]] = []
        common_auxiliary: set[str] | None = None

        for path in discover_npz(paths):
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                legacy = keys.intersection(LEGACY_INPUT_KEYS)
                if legacy:
                    raise ValueError(
                        f"{path} contains forbidden legacy input keys {sorted(legacy)}. "
                        "Competition shards must expose only xobs_model and t as model inputs."
                    )
                missing = {XOBS_KEY, *self.target_names}.difference(keys)
                if missing:
                    raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
                time_keys = [key for key in TIME_KEYS if key in keys]
                if len(time_keys) != 1:
                    raise ValueError(
                        f"{path} must contain exactly one time key from {TIME_KEYS}; "
                        f"found {time_keys}"
                    )
                time_key = time_keys[0]
                xobs = np.asarray(data[XOBS_KEY])
                if xobs.ndim != 2 or xobs.shape[1] != 6:
                    raise ValueError(
                        f"{path}:{XOBS_KEY} must have shape (N, 6); got {xobs.shape}"
                    )
                count = int(xobs.shape[0])
                if count == 0:
                    raise ValueError(f"{path} contains no samples.")
                self._validate_scalar_column(path, time_key, data[time_key], count)
                for name in self.target_names:
                    self._validate_scalar_column(path, name, data[name], count)

                if not np.isfinite(xobs).all():
                    raise ValueError(f"{path}:{XOBS_KEY} contains NaN or Inf.")
                if validate_ranges and np.max(np.abs(xobs)) > 1.0001:
                    raise ValueError(
                        f"{path}:{XOBS_KEY} contains values outside the Pauli range [-1, 1]."
                    )
                time_values = np.asarray(data[time_key]).reshape(count, -1)[:, 0]
                if not np.isfinite(time_values).all() or np.any(time_values <= 0.0):
                    raise ValueError(f"{path}:{time_key} must contain finite positive times.")

                present_auxiliary = set(AUXILIARY_KEYS).intersection(keys)
                for name in present_auxiliary:
                    self._validate_scalar_column(path, name, data[name], count)
                if common_auxiliary is None:
                    common_auxiliary = present_auxiliary
                elif common_auxiliary != present_auxiliary:
                    raise ValueError(
                        "All shards must expose the same auxiliary labels; "
                        f"{path} has {sorted(present_auxiliary)}, expected {sorted(common_auxiliary)}"
                    )

                local_indices = np.arange(count, dtype=np.int64)
                if self.split_names is not None:
                    if "split" not in keys:
                        raise ValueError(
                            f"{path} has no split column, but split_names were requested."
                        )
                    raw_split = np.asarray(data["split"])
                    if raw_split.ndim == 0:
                        split_values = np.repeat(raw_split.item(), count)
                    else:
                        if raw_split.shape[0] != count:
                            raise ValueError(f"{path}:split length does not match xobs_model.")
                        split_values = raw_split.reshape(count, -1)[:, 0]
                    mask = np.array(
                        [
                            _normalize_split_value(value) in self.split_names
                            for value in split_values
                        ],
                        dtype=bool,
                    )
                    local_indices = local_indices[mask]

                shard_index = len(self.shards)
                self.shards.append(ShardInfo(str(path), count, time_key))
                self.records.extend((shard_index, int(index)) for index in local_indices)

        self.auxiliary_names = tuple(sorted(common_auxiliary or set()))
        if not self.records:
            requested = sorted(self.split_names) if self.split_names else "all"
            raise ValueError(f"No samples matched requested splits: {requested}")

    @staticmethod
    def _validate_scalar_column(
        path: Path, name: str, value: np.ndarray, expected_count: int
    ) -> None:
        array = np.asarray(value)
        if array.ndim not in (1, 2) or array.shape[0] != expected_count:
            raise ValueError(
                f"{path}:{name} must have shape (N,) or (N, 1); got {array.shape}"
            )
        if array.ndim == 2 and array.shape[1] != 1:
            raise ValueError(f"{path}:{name} must be scalar per sample; got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}:{name} contains NaN or Inf.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if shard_index in self._cache:
            self._cache.move_to_end(shard_index)
            return self._cache[shard_index]

        info = self.shards[shard_index]
        keys = {
            XOBS_KEY,
            info.time_key,
            *self.target_names,
            *self.auxiliary_names,
        }
        with np.load(info.path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in keys}
        if self.cache_size:
            self._cache[shard_index] = arrays
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    @staticmethod
    def _scalar(array: np.ndarray, index: int, *, dtype: torch.dtype) -> torch.Tensor:
        value = np.asarray(array[index]).reshape(-1)[0]
        return torch.tensor([value], dtype=dtype)

    def __getitem__(self, index: int) -> dict[str, object]:
        shard_index, local_index = self.records[index]
        info = self.shards[shard_index]
        arrays = self._load_shard(shard_index)
        auxiliary: dict[str, torch.Tensor] = {}
        for name in self.auxiliary_names:
            dtype = torch.long if name == "phase_labels" else torch.float32
            auxiliary[name] = self._scalar(arrays[name], local_index, dtype=dtype)
        return {
            "xobs_model": torch.as_tensor(
                np.array(arrays[XOBS_KEY][local_index], dtype=np.float32, copy=True)
            ),
            "t": self._scalar(arrays[info.time_key], local_index, dtype=torch.float32),
            "targets": {
                name: self._scalar(arrays[name], local_index, dtype=torch.float32)
                for name in self.target_names
            },
            "auxiliary": auxiliary,
        }

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state


def split_dataset(
    dataset: Dataset,
    *,
    validation_fraction: float = 0.1,
    seed: int = 20260727,
) -> tuple[Subset, Subset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("At least two samples are required for a train/validation split.")
    validation_size = max(1, int(round(len(dataset) * validation_fraction)))
    validation_size = min(validation_size, len(dataset) - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)
