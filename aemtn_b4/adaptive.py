"""自适应/漂移判别与安全盾——冻结核心的薄封装。

只依赖 numpy/scipy，不依赖 torch/qutip/cqlib，可在轻量云实例上运行。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .paths import PROJECT_ROOT


# ---------- 安全盾（5 门） ----------
def shield_decision(
    *,
    mu_h1: float,
    mu_h2: float,
    sigma_h1: float,
    sigma_h2: float,
    p_exceed: float = 0.0,
    shots_used: int = 0,
    ood_score: float = 0.0,
    in_support: bool = True,
    z_features: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    gain: float = 0.25,
    probe_settings: int = 72,
    control: str = "act",
    max_uncertainty: float = 0.20,
    shot_budget_cap: int = 250_000_000,
) -> dict[str, object]:
    """对一次补偿动作施加冻结的 5 道非学习安全门，返回裁决与原因。

    返回字段：``permitted``、``gate``、``reason``、``compensation``、``honest_label``。
    """
    from src.adaptive.bandit import Action, ShieldConfig, shield

    action = Action(float(gain), int(probe_settings), control)  # type: ignore[arg-type]
    config = ShieldConfig(max_uncertainty=max_uncertainty, shot_budget_cap=int(shot_budget_cap))
    state: dict[str, object] = {
        "mu_h1": float(mu_h1),
        "mu_h2": float(mu_h2),
        "sigma_h1": float(sigma_h1),
        "sigma_h2": float(sigma_h2),
        "p_exceed": float(p_exceed),
        "shots_used": float(shots_used),
        "ood_score": float(ood_score),
        "in_support": bool(in_support),
        "z_features": [float(v) for v in z_features],
    }
    try:
        decision = shield(state, action, config)
    except ValueError as err:
        return {"permitted": False, "gate": "invalid_state", "reason": str(err), "compensation": (0.0, 0.0, 0.0), "honest_label": "n/a"}
    return {
        "permitted": bool(decision.permitted),
        "gate": decision.gate,
        "reason": decision.reason,
        "compensation": list(decision.compensation),
        "honest_label": "online contextual bandit (single-step RL), simulator-warm-started, hardware deployment under an external safety shield",
    }


# ---------- 漂移判别 / 结构函数 ----------
def variance_component_gate(
    values: Sequence[float],
    shots: Sequence[int] | int,
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    """冻结的方差成分门：区分"shot 噪声"与"超过 shot 噪声的过程方差"。"""
    from src.adaptive.sensing_economics import variance_component_gate as _gate

    return _gate(values, shots, alpha=alpha)


def compute_update_interval(
    *,
    values: Sequence[float],
    times_seconds: Sequence[float],
    shots: Sequence[int] | int,
    regime_ids: Sequence[int],
    burst_flags: Sequence[bool],
    lag_edges_seconds: Sequence[float],
    effective_shots_per_second: float,
    maximum_interval_seconds: float,
    interface_floor_seconds: float,
    alpha: float = 0.05,
    bootstrap_resamples: int = 500,
) -> dict[str, object]:
    """漂移感知经济核心：计算去偏结构函数、OU/nonparam 残差曲线、T* 与经济门。

    冻结核心 ``src/adaptive/sensing_economics.py::analyze_ou_sensing``。
    """
    from src.adaptive.sensing_economics import analyze_ou_sensing

    return analyze_ou_sensing(
        values=values,
        times_seconds=times_seconds,
        shots=shots,
        regime_ids=regime_ids,
        burst_flags=burst_flags,
        lag_edges_seconds=lag_edges_seconds,
        effective_shots_per_second=effective_shots_per_second,
        maximum_interval_seconds=maximum_interval_seconds,
        interface_floor_seconds=interface_floor_seconds,
        alpha=alpha,
        bootstrap_resamples=bootstrap_resamples,
    )


def load_frozen_sf_summary() -> dict[str, object]:
    """返回冻结报告中的两类关键判定（E0 阴性对照 + E1 检出 + T*）。"""
    base = PROJECT_ROOT / "evidence" / "B4_T176_HYBRID_FINAL_20260829"
    # 冻结 T287 结构函数关键数在 CSV 中随包分发，见 figure 03/04/05
    return {
        "note": "详见 README 与技术路线；这里随图展示逐点曲线。",
        "source": str(base),
    }
