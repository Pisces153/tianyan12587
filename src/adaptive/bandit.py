"""Small-sample linear contextual bandits with an external five-gate shield.

No hardware executor exists in this module.  It can score or update a policy
only after all declared evidence prerequisites are true; shielded decisions do
not update posterior state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Mapping, Sequence

import numpy as np


Control = Literal["act", "abstain", "escalate"]
HONEST_LABEL = "online contextual bandit (single-step RL), simulator-warm-started, hardware deployment under an external safety shield"
REQUIRED_PREREQUISITES = (
    "T3_sigma_calibrated",
    "T3_sim_to_real_conservative_prior",
    "T4_twin_gate_passed",
    "T6_features",
    "T7_forecast",
)
FORBIDDEN_STATE_TERMS = ("true_value", "injected", "task_id", "hardware_hidden", "label")


@dataclass(frozen=True)
class Action:
    gain: float
    probe_settings: int
    control: Control
    gain_jz: float = 0.0


@dataclass(frozen=True)
class ShieldConfig:
    max_uncertainty: float = 0.20
    h1_range: tuple[float, float] = (-1.0, 1.0)
    h2_range: tuple[float, float] = (-1.0, 1.0)
    max_action_amplitude: float = 0.70
    shot_budget_cap: int = 250_000_000


@dataclass(frozen=True)
class ShieldDecision:
    action: Action
    permitted: bool
    gate: str | None
    reason: str | None
    compensation: tuple[float, float, float]


def action_space() -> tuple[Action, ...]:
    return tuple(Action(float(gain), int(probe), control) for gain, probe, control in product((0.0, 0.25, 0.5, 0.75, 1.0), (18, 36, 72), ("act", "abstain", "escalate")))


def validate_prerequisites(prerequisites: Mapping[str, bool]) -> None:
    missing = [name for name in REQUIRED_PREREQUISITES if prerequisites.get(name) is not True]
    if missing:
        raise RuntimeError(f"Bandit execution blocked by unmet prerequisites: {', '.join(missing)}")


def validate_state(state: Mapping[str, Any]) -> None:
    serialized_keys = " ".join(str(key).lower() for key in state)
    illegal = [term for term in FORBIDDEN_STATE_TERMS if term in serialized_keys]
    if illegal:
        raise ValueError(f"Forbidden state input fields: {illegal}")
    required = {"mu_h1", "mu_h2", "sigma_h1", "sigma_h2", "z_features", "p_exceed", "shots_used", "ood_score", "in_support"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Bandit state is missing required fields: {missing}")
    z_features = state["z_features"]
    if not isinstance(z_features, Sequence) or isinstance(z_features, (str, bytes)) or len(z_features) > 4:
        raise ValueError("z_features must contain at most four observable proxy values")
    numeric = ("mu_h1", "mu_h2", "sigma_h1", "sigma_h2", "p_exceed", "shots_used", "ood_score", *[f"z_features[{index}]" for index in range(len(z_features))])
    values = [state["mu_h1"], state["mu_h2"], state["sigma_h1"], state["sigma_h2"], state["p_exceed"], state["shots_used"], state["ood_score"], *z_features]
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError(f"Bandit state has non-finite numeric values in {numeric}")
    if not isinstance(state["in_support"], bool):
        raise ValueError("in_support must be bool")


def proposed_compensation(state: Mapping[str, Any], action: Action) -> tuple[float, float, float]:
    if action.control != "act":
        return (0.0, 0.0, 0.0)
    return (-action.gain * float(state["mu_h1"]), -action.gain * float(state["mu_h2"]), action.gain_jz)


def shield(state: Mapping[str, Any], action: Action, config: ShieldConfig) -> ShieldDecision:
    """Apply five non-learned safety gates before any action is exposed."""
    validate_state(state)
    compensation = proposed_compensation(state, action)
    abstain = Action(0.0, action.probe_settings, "abstain", 0.0)
    # Gate 1: confidence and support.
    if not bool(state["in_support"]) or max(float(state["sigma_h1"]), float(state["sigma_h2"])) > config.max_uncertainty:
        return ShieldDecision(abstain, False, "confidence", "posterior uncertainty or support gate", (0.0, 0.0, 0.0))
    # Gate 2: proposed effective fields stay in registered physical bounds.
    proposed_h1 = float(state["mu_h1"]) + compensation[0]
    proposed_h2 = float(state["mu_h2"]) + compensation[1]
    if not (config.h1_range[0] <= proposed_h1 <= config.h1_range[1] and config.h2_range[0] <= proposed_h2 <= config.h2_range[1]):
        return ShieldDecision(abstain, False, "physical_range", "proposed h1/h2 outside frozen range", (0.0, 0.0, 0.0))
    # Gate 3: Jz compensation is never permitted.
    if action.gain_jz != 0.0 or compensation[2] != 0.0:
        return ShieldDecision(abstain, False, "jz_reject", "gain_Jz must remain zero", (0.0, 0.0, 0.0))
    # Gate 4: bounded action amplitude.
    if max(abs(value) for value in compensation) > config.max_action_amplitude:
        return ShieldDecision(abstain, False, "action_amplitude", "compensation exceeds frozen infinity-norm bound", (0.0, 0.0, 0.0))
    # Gate 5: no action can overspend the campaign budget.
    if float(state["shots_used"]) + action.probe_settings > config.shot_budget_cap:
        return ShieldDecision(abstain, False, "budget", "next probe would exceed shot budget", (0.0, 0.0, 0.0))
    return ShieldDecision(action, True, None, None, compensation)


def action_features(state: Mapping[str, Any], action: Action) -> np.ndarray:
    validate_state(state)
    z = np.zeros(4, dtype=np.float64)
    z[:len(state["z_features"])] = np.asarray(state["z_features"], dtype=np.float64)
    control_value = {"act": 1.0, "abstain": 0.0, "escalate": -1.0}[action.control]
    features = np.asarray([
        1.0, float(state["mu_h1"]), float(state["mu_h2"]), float(state["sigma_h1"]), float(state["sigma_h2"]),
        *z, float(state["p_exceed"]), float(state["shots_used"]) / 1_000_000.0, float(state["ood_score"]),
        action.gain, action.probe_settings / 72.0, control_value,
    ], dtype=np.float64)
    if features.size > 20:
        raise AssertionError("Feature map exceeds frozen small-sample dimensionality")
    return features


class LinearContextualBandit:
    """Shared analytic Gaussian posterior for LinTS and LinUCB scoring."""

    def __init__(self, *, mode: Literal["lints", "linucb"], feature_dim: int = 15, prior_precision: float = 1.0, observation_variance: float = 1.0, seed: int = 202609) -> None:
        if mode not in {"lints", "linucb"}:
            raise ValueError("mode must be lints or linucb")
        self.mode = mode
        self.precision = np.eye(feature_dim, dtype=np.float64) * prior_precision
        self.response = np.zeros(feature_dim, dtype=np.float64)
        self.observation_variance = float(observation_variance)
        self.random = np.random.default_rng(seed)

    @property
    def posterior_mean(self) -> np.ndarray:
        return np.linalg.solve(self.precision, self.response)

    def select(self, state: Mapping[str, Any], *, config: ShieldConfig, ucb_alpha: float = 1.0) -> ShieldDecision:
        candidates = [shield(state, action, config) for action in action_space()]
        permitted = [decision for decision in candidates if decision.permitted]
        if not permitted:
            return candidates[0]
        covariance = self.observation_variance * np.linalg.pinv(self.precision)
        weights = self.random.multivariate_normal(self.posterior_mean, covariance) if self.mode == "lints" else self.posterior_mean
        scores: list[float] = []
        for decision in permitted:
            vector = action_features(state, decision.action)
            exploit = float(vector @ weights)
            bonus = ucb_alpha * float(np.sqrt(vector @ covariance @ vector)) if self.mode == "linucb" else 0.0
            scores.append(exploit + bonus)
        return permitted[int(np.argmax(scores))]

    def update(self, state: Mapping[str, Any], action: Action, reward: float) -> None:
        vector = action_features(state, action)
        self.precision += np.outer(vector, vector) / self.observation_variance
        self.response += vector * float(reward) / self.observation_variance

    def artifact(self, prerequisites: Mapping[str, bool]) -> dict[str, Any]:
        validate_prerequisites(prerequisites)
        return {
            "honest_label": HONEST_LABEL,
            "mode": self.mode,
            "action_space_size": len(action_space()),
            "feature_map_dim": int(self.precision.shape[0]),
            "posterior_precision": self.precision.tolist(),
            "posterior_response": self.response.tolist(),
            "prerequisites": {name: bool(prerequisites[name]) for name in REQUIRED_PREREQUISITES},
            "forbidden_state_inputs_asserted": True,
        }
