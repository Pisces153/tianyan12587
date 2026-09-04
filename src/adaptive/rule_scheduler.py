"""Fixed, non-learning fallback for the preregistered T7-no-skill branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.adaptive.bandit import Action, ShieldConfig, ShieldDecision, shield


@dataclass(frozen=True)
class RuleScheduleConfig:
    """Frozen conservative response.  This is neither forecast nor bandit policy."""

    exceedance_trigger: float = 0.50
    response_gain: float = 0.25
    abstain_probe_settings: int = 18
    response_probe_settings: int = 72


def decide_fixed_response(
    state: Mapping[str, Any],
    *,
    shield_config: ShieldConfig,
    rule_config: RuleScheduleConfig = RuleScheduleConfig(),
) -> ShieldDecision:
    """Choose one predeclared action, then apply existing external shield.

    The trigger uses an observed current-state exceedance probability.  No model
    score, posterior update, reward, or hardware executor is present.
    """
    if not 0.0 <= rule_config.exceedance_trigger <= 1.0:
        raise ValueError("exceedance_trigger must be in [0, 1]")
    if not 0.0 <= rule_config.response_gain <= 1.0:
        raise ValueError("response_gain must be in [0, 1]")
    trigger = float(state["p_exceed"]) >= rule_config.exceedance_trigger
    action = Action(
        rule_config.response_gain if trigger else 0.0,
        rule_config.response_probe_settings if trigger else rule_config.abstain_probe_settings,
        "act" if trigger else "abstain",
    )
    return shield(state, action, shield_config)


def fallback_artifact() -> dict[str, Any]:
    """Stable declaration for the T7-fail deliverable."""
    config = RuleScheduleConfig()
    return {
        "task": "T7_no_skill_rule_scheduler",
        "honest_label": "fixed rule scheduler with drift characterization and safety response",
        "t7_forecast_required": False,
        "bandit_or_rl": False,
        "hardware_executor_present": False,
        "observed_state_only": True,
        "rule": {
            "exceedance_trigger": config.exceedance_trigger,
            "response_gain": config.response_gain,
            "abstain_probe_settings": config.abstain_probe_settings,
            "response_probe_settings": config.response_probe_settings,
            "gain_jz": 0.0,
        },
        "shield_required": True,
    }
