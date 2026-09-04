"""第 6 章：模型与数据契约。

只读展示随包分发的冻结资产生成契约：数据集 schema、配置、特征契约与三 seed
模型权重文件。不触碰任何私有 raw counts / NPZ；仅依赖随包公开的部分。
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from aemtn_b4 import PROJECT_ROOT
from app.common import PALETTE, read_json


def render() -> None:
    st.header("模型与数据")

    st.subheader("数据集契约")
    st.markdown(
        "训练数据是从 Lindblad 局域退极化主方程演化 + 采样 counts 生成的参考集。"
        "关键参数如下（来源 `data/sim_v3_paper_contract/dataset_manifest.json`）。"
    )
    manifest = read_json(PROJECT_ROOT / "data" / "sim_v3_paper_contract" / "dataset_manifest.json")
    if manifest:
        cols = st.columns(4)
        cols[0].metric("样本数", f"{manifest.get('samples')}", "2000")
        cols[1].metric("每基 shots", f"{manifest.get('shots_per_basis')}", "1024")
        cols[2].metric("测量基", f"{len(manifest.get('basis_order', []))}", "XX..ZZ 9 基")
        cols[3].metric("比特序", str(manifest.get("bit_order", "")), "q0_leftmost")

        st.markdown("**噪声与生成参数**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "**制备退极化范围**\n\n"
                f"`{manifest.get('preparation_depolarization_range')}`"
            )
        with c2:
            st.markdown(
                "**Lindblad 噪声模型**\n\n"
                f"`{manifest.get('lindblad_noise_model')}`"
            )
        with c3:
            st.markdown(
                "**生成器**\n\n"
                f"`{manifest.get('generator')}`"
            )
        st.caption("2000 个样本分 20 个 chunk，每 chunk 100 个。完整 sha256 见 manifest，可离线校验。")
    else:
        st.info("缺少 dataset manifest。")

    st.divider()

    st.subheader("量子测量特征契约")
    st.markdown(
        "9 个测量基的 raw counts → Pauli-15（15 维），再选 6 个 local 相关项组成 AEMTN 输入。"
        "**输入是 local6 = (X0, Y0, Z0, X0X1, Y0Y1, Z0Z1)，加 1 维时间，共 7 维。**"
    )
    from aemtn_b4 import pauli_features_from_counts

    st.code(
        "from aemtn_b4 import pauli_features_from_counts\n"
        "# counts 形状 (9,64)，每基 shots 和 = 1024\n"
        "feat_local6 = pauli_features_from_counts(counts, shots=1024, local6=True)  # (6,)\n"
        "feat_pauli15 = pauli_features_from_counts(counts, shots=1024, local6=False)  # (15,)", language="python"
    )

    st.divider()

    st.subheader("AEMTN 模型资产")
    st.markdown(
        "三 seed（20260731 / 20260801 / 20260802）的冻结权重，每份 17.3 MB。"
        "项目取最优 seed 的 `best.pt` 作为闭环推理用的冻结模型。"
    )
    models_dir = PROJECT_ROOT / "models" / "sim_pretrained_paper_contract"
    seed_rows = []
    if models_dir.is_dir():
        for seed_dir in sorted(models_dir.glob("seed_*")):
            pt = seed_dir / "best.pt"
            if pt.is_file():
                seed_rows.append(
                    {
                        "seed": seed_dir.name.replace("seed_", ""),
                        "文件": "best.pt",
                        "大小 (MB)": round(pt.stat().st_size / 1e6, 1),
                        "sha256": _sha256(pt)[:16] + "…",
                    }
                )
    if seed_rows:
        st.dataframe(seed_rows, width="stretch")
    else:
        st.info("未找到种子模型权重文件。")
    st.caption(
        "权重文件在库中随包分发；若你要在 GitHub 上发布，可用 Git LFS 或 Release 附件托管（见 DEPLOY.md）。"
    )

    st.divider()

    st.subheader("训练管线（离线可验证）")
    st.markdown(
        "冻结训练管线位于 `scripts/` 与 `tools/`。公开指标：模型约 1.43M 参数，"
        "`r_dim=320`，4 子空间多任务头。目标为 h1 / h2 / Jz，带高斯不确定度头。"
    )
    st.info(
        "**诚实声明：** h2 的 R²≈0、Jz 的 R²≈0.32，说明这两个目标在当前数据上"
        "学习信号弱。平台把它作为**已识别瓶颈**而非回避项；详见第 7 章边界声明。"
    )


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
