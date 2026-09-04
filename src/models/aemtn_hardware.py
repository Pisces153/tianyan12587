"""Hardware-facing AEMTN model.

The competition contract is intentionally narrow: six measured Pauli features and
one known evolution time enter the network. Hamiltonian labels never enter an
input branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    x_dim: int = 6
    t_dim: int = 1
    r_dim: int = 320
    task_dim: int = 6
    num_subspaces: int = 4
    dropout: float = 0.1
    target_names: tuple[str, ...] = ("h1", "h2", "Jz")
    xobs_order: tuple[str, ...] = ("X0", "Y0", "Z0", "X0X1", "Y0Y1", "Z0Z1")
    inter_adapter_alpha: float = 0.8
    entropy_bridge_scale: float = 1.0
    detach_entropy_bridge: bool = True
    ham_guidance_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.x_dim != 6:
            raise ValueError("The XA-202609 model requires x_dim=6 (xobs_model).")
        if self.t_dim != 1:
            raise ValueError("The XA-202609 model requires t_dim=1 (evolution time only).")
        if self.r_dim % self.num_subspaces:
            raise ValueError("r_dim must be divisible by num_subspaces.")
        if not self.target_names:
            raise ValueError("At least one Hamiltonian target is required.")
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target_names must be unique.")
        if len(self.xobs_order) != self.x_dim or len(set(self.xobs_order)) != self.x_dim:
            raise ValueError("xobs_order must contain x_dim unique observable names.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ModelConfig":
        data = dict(value)
        if "target_names" in data:
            data["target_names"] = tuple(data["target_names"])
        if "xobs_order" in data:
            data["xobs_order"] = tuple(data["xobs_order"])
        return cls(**data)


class HardwareBackbone(nn.Module):
    """Separate encoders prevent the control scalar from being mixed with labels."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        r_dim = config.r_dim
        self.x_dim = config.x_dim
        self.t_dim = config.t_dim
        self.x_branch = nn.Sequential(
            nn.Linear(config.x_dim, r_dim),
            nn.LayerNorm(r_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.t_branch = nn.Sequential(
            nn.Linear(config.t_dim, r_dim),
            nn.LayerNorm(r_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * r_dim, r_dim),
            nn.LayerNorm(r_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(r_dim, r_dim),
            nn.LayerNorm(r_dim),
        )

    def forward(self, xobs: Tensor, evolution_time: Tensor) -> Tensor:
        if xobs.ndim != 2 or xobs.shape[-1] != self.x_dim:
            raise ValueError(f"xobs must have shape (batch, {self.x_dim}); got {tuple(xobs.shape)}")
        if evolution_time.ndim == 1:
            evolution_time = evolution_time.unsqueeze(-1)
        if evolution_time.ndim != 2 or evolution_time.shape[-1] != self.t_dim:
            raise ValueError(
                f"evolution_time must have shape (batch, {self.t_dim}); "
                f"got {tuple(evolution_time.shape)}"
            )
        if xobs.shape[0] != evolution_time.shape[0]:
            raise ValueError("xobs and evolution_time batch sizes differ.")

        rx = self.x_branch(xobs)
        rt = self.t_branch(evolution_time)
        return self.fuse(torch.cat((rx, rt), dim=-1)) + 0.5 * (rx + rt)


class TaskWiseRouting(nn.Module):
    """Task-conditioned attention over learned representation subspaces."""

    def __init__(self, config: ModelConfig, num_tasks: int) -> None:
        super().__init__()
        self.num_subspaces = config.num_subspaces
        self.sub_dim = config.r_dim // config.num_subspaces
        self.sub_projections = nn.ModuleList(
            nn.Linear(config.r_dim, self.sub_dim) for _ in range(config.num_subspaces)
        )
        self.task_embedding = nn.Embedding(num_tasks, config.task_dim)
        self.task_query = nn.Linear(config.task_dim, self.sub_dim)
        self.subspace_key = nn.Linear(self.sub_dim, self.sub_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(self.sub_dim),
            nn.Linear(self.sub_dim, config.r_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.r_dim, config.r_dim),
        )
        self.output_norm = nn.LayerNorm(config.r_dim)

    def forward(self, shared: Tensor, task_id: int) -> tuple[Tensor, Tensor]:
        batch_size = shared.shape[0]
        task_ids = torch.full(
            (batch_size,), task_id, dtype=torch.long, device=shared.device
        )
        tokens = torch.stack(
            [F.gelu(projection(shared)) for projection in self.sub_projections], dim=1
        )
        query = self.task_query(self.task_embedding(task_ids)).unsqueeze(1)
        keys = self.subspace_key(tokens)
        logits = (query * keys).sum(dim=-1) * (self.sub_dim ** -0.5)
        attention = torch.softmax(logits, dim=-1)
        routed = (attention.unsqueeze(-1) * tokens).sum(dim=1)
        return self.output_norm(shared + self.output(routed)), attention


class GaussianRegressionHead(nn.Module):
    def __init__(self, r_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = max(32, r_dim // 2)
        self.trunk = nn.Sequential(
            nn.Linear(r_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_variance = nn.Linear(hidden, 1)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.trunk(value)
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden).clamp(min=-8.0, max=5.0)
        return mean, log_variance


class ScalarHead(nn.Module):
    def __init__(self, r_dim: int, dropout: float, sigmoid: bool = False) -> None:
        super().__init__()
        hidden = max(32, r_dim // 2)
        layers: list[nn.Module] = [
            nn.Linear(r_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class AEMTNHardware(nn.Module):
    """AEMTN adapted to measurable hardware features without target leakage."""

    AUX_TASKS = ("entropy", "inter_entropy", "fidelity")

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        task_names = (*config.target_names, *self.AUX_TASKS)
        self.task_to_id = {name: index for index, name in enumerate(task_names)}

        self.backbone = HardwareBackbone(config)
        self.routing = TaskWiseRouting(config, num_tasks=len(task_names))
        self.inter_adapter = nn.Sequential(
            nn.Linear(config.r_dim, config.r_dim),
            nn.GELU(),
            nn.Linear(config.r_dim, config.r_dim),
            nn.GELU(),
            nn.Linear(config.r_dim, config.r_dim),
            nn.LayerNorm(config.r_dim),
        )
        self.entropy_bridge = nn.Sequential(
            nn.Linear(config.r_dim, config.r_dim),
            nn.Tanh(),
        )
        self.ham_guidance = nn.Sequential(
            nn.Linear(config.r_dim, config.r_dim // 2),
            nn.GELU(),
            nn.Linear(config.r_dim // 2, config.r_dim),
            nn.Tanh(),
        )

        self.target_heads = nn.ModuleDict(
            {
                name: GaussianRegressionHead(config.r_dim, config.dropout)
                for name in config.target_names
            }
        )
        self.entropy_head = ScalarHead(config.r_dim, config.dropout)
        self.inter_entropy_head = ScalarHead(config.r_dim, config.dropout)
        self.fidelity_head = ScalarHead(config.r_dim, config.dropout, sigmoid=True)
        self.phase_head = nn.Sequential(
            nn.Linear(config.r_dim, config.r_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.r_dim // 2, 3),
        )

    def forward(self, xobs_model: Tensor, evolution_time: Tensor) -> Dict[str, object]:
        shared = self.backbone(xobs_model, evolution_time)

        entropy_feature, entropy_attention = self.routing(
            shared, self.task_to_id["entropy"]
        )
        inter_feature, inter_attention = self.routing(
            shared, self.task_to_id["inter_entropy"]
        )
        bridge_source = entropy_feature.detach() if self.config.detach_entropy_bridge else entropy_feature
        inter_feature = inter_feature + self.config.entropy_bridge_scale * self.entropy_bridge(bridge_source)
        mapped_inter = self.inter_adapter(inter_feature)
        alpha = self.config.inter_adapter_alpha
        inter_feature = (1.0 - alpha) * inter_feature + alpha * mapped_inter
        guidance = self.config.ham_guidance_scale * self.ham_guidance(inter_feature)

        predictions: Dict[str, Tensor] = {}
        log_variances: Dict[str, Tensor] = {}
        routing_attention: Dict[str, Tensor] = {
            "entropy": entropy_attention,
            "inter_entropy": inter_attention,
        }
        for name in self.config.target_names:
            feature, attention = self.routing(shared, self.task_to_id[name])
            if name.lower() == "jz":
                feature = feature + guidance
            mean, log_variance = self.target_heads[name](feature)
            predictions[name] = mean
            log_variances[name] = log_variance
            routing_attention[name] = attention

        fidelity_feature, fidelity_attention = self.routing(
            shared, self.task_to_id["fidelity"]
        )
        routing_attention["fidelity"] = fidelity_attention
        return {
            "predictions": predictions,
            "log_variances": log_variances,
            "auxiliary": {
                "entropies": self.entropy_head(entropy_feature),
                "inter_entropies": self.inter_entropy_head(inter_feature),
                "target_fidelities": self.fidelity_head(fidelity_feature),
                "phase_logits": self.phase_head(shared),
            },
            "routing_attention": routing_attention,
        }


def build_aemtn_hardware(
    *,
    target_names: Sequence[str] = ("h1", "h2", "Jz"),
    r_dim: int = 320,
    task_dim: int = 6,
    num_subspaces: int = 4,
    dropout: float = 0.1,
    xobs_order: Sequence[str] = ("X0", "Y0", "Z0", "X0X1", "Y0Y1", "Z0Z1"),
) -> AEMTNHardware:
    config = ModelConfig(
        target_names=tuple(target_names),
        r_dim=r_dim,
        task_dim=task_dim,
        num_subspaces=num_subspaces,
        dropout=dropout,
        xobs_order=tuple(xobs_order),
    )
    return AEMTNHardware(config)
