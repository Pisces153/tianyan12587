from __future__ import annotations

import unittest

from src.backends.tianyan_discovery import build_inventory, candidate_codes


class TianYanDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.machines = [
            {"code": "sim-dm-noise", "name": "密度矩阵带噪声", "status": 0},
            {"code": "sim-sv", "name": "全振幅", "status": 0},
            {"code": "tianyan-287", "alias": "tianyan-287", "status": 1},
            {"code": "private", "access_token": "must-not-leak"},
        ]

    def test_candidate_matching_preserves_platform_codes(self) -> None:
        self.assertEqual(candidate_codes(self.machines, "密度矩阵带噪声"), ["sim-dm-noise"])
        self.assertEqual(candidate_codes(self.machines, "全振幅"), ["sim-sv"])
        self.assertEqual(candidate_codes(self.machines, "tianyan-287"), ["tianyan-287"])

    def test_inventory_redacts_credential_fields(self) -> None:
        inventory = build_inventory(self.machines, cqlib_version="1.3.11")
        self.assertTrue(inventory["read_only"])
        self.assertEqual(inventory["role_candidates"]["hardware"], ["tianyan-287"])
        self.assertNotIn("access_token", inventory["machines"][-1])


if __name__ == "__main__":
    unittest.main()
