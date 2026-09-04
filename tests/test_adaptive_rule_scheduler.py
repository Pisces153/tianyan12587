from __future__ import annotations

from src.adaptive.bandit import ShieldConfig
from src.adaptive.rule_scheduler import decide_fixed_response, fallback_artifact


def _state(**updates):
    state = {"mu_h1": 0.2, "mu_h2": -0.2, "sigma_h1": 0.05, "sigma_h2": 0.05, "z_features": [0.1], "p_exceed": 0.1, "shots_used": 0, "ood_score": 0.0, "in_support": True}
    state.update(updates)
    return state


def test_fixed_rule_abstains_below_observed_trigger() -> None:
    decision = decide_fixed_response(_state(), shield_config=ShieldConfig(shot_budget_cap=100))
    assert decision.action.control == "abstain"
    assert decision.permitted


def test_fixed_rule_response_still_requires_external_shield() -> None:
    decision = decide_fixed_response(_state(p_exceed=0.5, sigma_h1=0.5), shield_config=ShieldConfig(shot_budget_cap=100))
    assert decision.action.control == "abstain"
    assert decision.permitted is False
    assert decision.gate == "confidence"


def test_fallback_artifact_cannot_be_mislabeled_as_rl() -> None:
    artifact = fallback_artifact()
    assert artifact["bandit_or_rl"] is False
    assert artifact["hardware_executor_present"] is False
    assert artifact["rule"]["gain_jz"] == 0.0
