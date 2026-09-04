from __future__ import annotations

import pytest

from src.adaptive.bandit import Action, LinearContextualBandit, ShieldConfig, action_space, shield, validate_prerequisites


def _state(**updates):
    state = {"mu_h1": 0.2, "mu_h2": -0.2, "sigma_h1": 0.05, "sigma_h2": 0.05, "z_features": [0.1, 0.2], "p_exceed": 0.1, "shots_used": 0, "ood_score": 0.0, "in_support": True}
    state.update(updates)
    return state


def test_action_space_and_five_safety_gates() -> None:
    assert len(action_space()) == 45
    config = ShieldConfig(shot_budget_cap=100)
    assert shield(_state(sigma_h1=1.0), Action(0.25, 18, "act"), config).gate == "confidence"
    assert shield(_state(mu_h1=1.1), Action(0.0, 18, "act"), config).gate == "physical_range"
    assert shield(_state(), Action(0.25, 18, "act", gain_jz=0.1), config).gate == "jz_reject"
    assert shield(_state(mu_h1=0.9), Action(1.0, 18, "act"), ShieldConfig(h1_range=(-2, 2), h2_range=(-2, 2), max_action_amplitude=0.7, shot_budget_cap=100)).gate == "action_amplitude"
    assert shield(_state(shots_used=90), Action(0.25, 18, "act"), config).gate == "budget"


def test_bandit_refuses_unmet_prerequisites_and_can_score_safe_action() -> None:
    with pytest.raises(RuntimeError, match="T3_sigma_calibrated"):
        validate_prerequisites({})
    with pytest.raises(RuntimeError, match="T3_sim_to_real_conservative_prior"):
        validate_prerequisites({"T3_sigma_calibrated": True, "T4_twin_gate_passed": True, "T6_features": True, "T7_forecast": True})
    bandit = LinearContextualBandit(mode="lints")
    decision = bandit.select(_state(), config=ShieldConfig(shot_budget_cap=100))
    assert decision.permitted
    bandit.update(_state(), decision.action, reward=0.5)
    artifact = bandit.artifact({"T3_sigma_calibrated": True, "T3_sim_to_real_conservative_prior": True, "T4_twin_gate_passed": True, "T6_features": True, "T7_forecast": True})
    assert artifact["action_space_size"] == 45
