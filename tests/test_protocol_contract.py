from __future__ import annotations

from pathlib import Path
import unittest

from src.models.aemtn_hardware import ModelConfig
from src.protocol import load_json, submission_blockers, validate_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_json(PROJECT_ROOT / "config" / "protocol_v2.json")
        cls.backends = load_json(PROJECT_ROOT / "config" / "backends_v1.json")

    def test_protocol_contract_is_internally_consistent(self) -> None:
        self.assertEqual(validate_contract(self.protocol, self.backends), [])

    def test_training_defaults_match_protocol_targets(self) -> None:
        targets = tuple(self.protocol["model_data_contract"]["supervised_targets"])
        self.assertEqual(targets, ModelConfig().target_names)
        self.assertEqual(
            self.protocol["model_data_contract"]["training_loss_weights"],
            {"h1": 1.0, "h2": 1.0, "Jz": 80.0},
        )

    def test_backend_roles_are_frozen(self) -> None:
        roles = self.backends["roles"]
        self.assertEqual(roles["primary_noisy_simulator"]["display_name"], "密度矩阵带噪声")
        self.assertEqual(roles["ideal_reference_simulator"]["display_name"], "全振幅")
        self.assertEqual(roles["hardware"]["display_name"], "tianyan-287")

    def test_submission_is_blocked_before_machine_code_and_g0(self) -> None:
        blockers = submission_blockers(self.protocol, self.backends)
        self.assertTrue(any("machine_code" in blocker for blocker in blockers))
        self.assertIn("protocol is not locked_at_g0", blockers)
        self.assertIn("G0 approval is missing", blockers)


if __name__ == "__main__":
    unittest.main()
