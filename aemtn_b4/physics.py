"""量子测量特征恢复——对外暴露的薄封装。

冻结核心是 ``src/features/pauli.py``：把九测量基 counts 恢复成 Pauli-15 与
local6 特征。本模块只做两件事：(1) 懒加载 qutip；(2) 提供不依赖 qutip 的
纯 numpy 快速入口（这是评审交互最常用的路径）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# local6 特征顺序（与冻结 xobs 契约一致）
XOB = (0,)
YOB = (1,)
ZOB = (2,)
XX = (3,)
YY = (4,)
ZZ = (5,)


def _pauli15_from_counts(counts: np.ndarray) -> np.ndarray:
    """把 (n, 9, 64) 的 counts 批量恢复成 (n, 15) Pauli-15（纯 numpy）。"""
    from src.features.pauli import counts_to_pauli15

    if counts.ndim != 3:
        raise ValueError(f"expected (n,9,64), got {counts.shape}")
    out = np.empty((counts.shape[0], 15), dtype=np.float64)
    for i in range(counts.shape[0]):
        out[i] = np.asarray(counts_to_pauli15(counts[i].astype(np.int64)))
    return out


def pauli_features_from_counts(
    counts: Sequence[int] | np.ndarray,
    *,
    shots: int = 1024,
    local6: bool = True,
) -> np.ndarray:
    """从单个样本的 raw counts 恢复 Pauli 特征。

    Args:
        counts: 形状 (9, 64) 的九基 counts 数组（或已展平的 576 长度序列）。
        shots: 每基 shots 数（默认 1024，与冻结协议一致）。
        local6: True 返回 local6（6 维，AEMTN 输入）；False 返回 Pauli-15。

    Returns:
        np.ndarray，形状 (6,) 或 (15,)。
    """
    from src.features.pauli import counts_array_to_pauli15, select_pauli_features

    arr = np.asarray(counts, dtype=np.int64)
    if arr.ndim == 1 and arr.size == 9 * 64:
        arr = arr.reshape(9, 64)
    if arr.ndim != 2 or arr.shape != (9, 64):
        raise ValueError(f"expected (9,64), got {arr.shape}")
    if arr.sum(axis=1).min() == 0:
        raise ValueError("每个测量基的 counts 总和必须为 shots(默认 1024)，不能为 0")
    p15 = counts_array_to_pauli15(arr, shots=shots)
    if local6:
        return select_pauli_features(p15, order=("X0", "Y0", "Z0", "X0X1", "Y0Y1", "Z0Z1"))
    return p15


def basis_order() -> tuple[str, ...]:
    from src.features.pauli import BASIS_ORDER

    return tuple(BASIS_ORDER)
