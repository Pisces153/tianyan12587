"""AEMTN-B4 公共 API 包。

面向评审与第三方复用的最薄上层封装：把冻结的 ``src/`` 科研核心与
``scripts/`` CLI 暴露成干净的 Python 函数与数据对象。

设计边界
--------
* **零改写**：本包只 import 冻结核心，不改 ``src/``、``scripts/``、``tools/`` 任何一行。
* **懒加载**：torch/qutip/cqlib 等重量依赖在真正用到时才 import，保证
  ``pip install aemtn-b4`` 只带 numpy/scipy/pandas 也能拿到全部冻结结果。
* **诚实标注**：所有可供对外主张的数字都带 ``claim_boundary`` 标记，提醒
  哪些是 ``B4_PRESERVED_SIMULATION_ASSISTED``、哪些仍
  ``INCONCLUSIVE_MISSING_HARDWARE_SESSION1``。
"""

from __future__ import annotations

from .paths import PROJECT_ROOT, app_assets_dir, data_dir, evidence_dir, manifest_dir, models_dir
from .result import FinalReport, HybridResult, load_final_report
from .reproduce import reproduce_final
from .physics import pauli_features_from_counts
from .adaptive import shield_decision, compute_update_interval

__all__ = [
    "PROJECT_ROOT",
    "app_assets_dir",
    "data_dir",
    "evidence_dir",
    "manifest_dir",
    "models_dir",
    "FinalReport",
    "HybridResult",
    "load_final_report",
    "reproduce_final",
    "pauli_features_from_counts",
    "shield_decision",
    "compute_update_interval",
]

__version__ = "1.0.0"
