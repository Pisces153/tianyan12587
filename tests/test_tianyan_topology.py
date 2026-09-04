from __future__ import annotations

import unittest

from src.backends.tianyan_topology import select_six_qubit_chain


class TianYanTopologyTests(unittest.TestCase):
    def test_selects_active_six_qubit_path(self) -> None:
        configuration = {
            "disabledQubits": "Q4",
            "disabledCouplers": "",
            "overview": {
                "coupler_map": {
                    "G1": ["Q1", "Q2"], "G2": ["Q2", "Q3"], "G3": ["Q3", "Q4"],
                    "G4": ["Q5", "Q6"], "G5": ["Q6", "Q7"], "G6": ["Q7", "Q8"],
                    "G7": ["Q8", "Q9"], "G8": ["Q9", "Q10"],
                }
            },
            "twoQubitGate": {"czGate": {"gate error": {"qubit_used": [f"G{index}" for index in range(1, 9)], "param_list": [0.2] * 8}}},
            "readout": {"readoutArray": {"Readout Error": {"qubit_used": [f"Q{index}" for index in range(1, 11)], "param_list": [1.0] * 10}}},
        }
        chain = select_six_qubit_chain(configuration)
        self.assertEqual(chain.physical_labels, ("Q5", "Q6", "Q7", "Q8", "Q9", "Q10"))
        self.assertEqual(chain.circuit_qubits, (4, 5, 6, 7, 8, 9))

    def test_preserves_zero_indexed_machine_labels(self) -> None:
        configuration = {
            "disabledQubits": "",
            "disabledCouplers": "",
            "overview": {
                "qubits": [f"Q{index}" for index in range(6)],
                "coupler_map": {f"G{index}": [f"Q{index}", f"Q{index + 1}"] for index in range(5)},
            },
            "twoQubitGate": {"czGate": {"gate error": {"qubit_used": [f"G{index}" for index in range(5)], "param_list": [0.2] * 5}}},
            "readout": {"readoutArray": {"Readout Error": {"qubit_used": [f"Q{index}" for index in range(6)], "param_list": [1.0] * 6}}},
        }
        chain = select_six_qubit_chain(configuration)
        self.assertEqual(chain.physical_labels, ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5"))
        self.assertEqual(chain.circuit_qubits, (0, 1, 2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
