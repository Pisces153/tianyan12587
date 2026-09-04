"""技术路线与 AI 框架页：物理模型、测量方案、AEMTN 架构、选型依据。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.common import read_json

RepoRoot = Path(__file__).resolve().parents[2]


def render() -> None:
    root = RepoRoot
    st.header("技术路线与 AI 框架")

    st.subheader("物理模型：开放边界 DM 哈密顿量（6 比特）")
    st.latex(
        r"""\begin{aligned}
H &= \sum_{i=1}^{N-1}\Big[\,J_x\,\sigma_i^{x}\sigma_{i+1}^{x}
   + J_y\,\sigma_i^{y}\sigma_{i+1}^{y}
   + J_z\,\sigma_i^{z}\sigma_{i+1}^{z}
   + J_{xz}\,\sigma_i^{x}\sigma_{i+1}^{z}
   + J_{zx}\,\sigma_i^{z}\sigma_{i+1}^{x} \\
  &\qquad\; + D\left(\sigma_i^{x}\sigma_{i+1}^{y}-\sigma_i^{y}\sigma_{i+1}^{x}\right)\Big]
   + \sum_{i=1}^{N}\left(h_i^{x}\sigma_i^{x}+h_i^{y}\sigma_i^{y}+h_i^{z}\sigma_i^{z}\right),
   \qquad N=6
\end{aligned}"""
    )
    st.markdown(
        "- **耦合：** $J_x=J(1+\\varepsilon),\\;J_y=J(1-\\varepsilon)$，$\\varepsilon\\in[-0.12,0.12]$；"
        "$D$ 为沿 $z$ 轴的 DM 相互作用 $\\mathbf D\\cdot(\\boldsymbol\\sigma_i\\times\\boldsymbol\\sigma_{i+1})$；"
        "$J_{xz},J_{zx}$ 为交叉耦合\n"
        "- **反演目标：** $h_1\\equiv h_1^{x}$、$h_2\\equiv h_2^{x}$、$J_z$；其余参数为生成时随机采样的干扰项\n"
        "- **初态：** Néel 态 $|010101\\rangle$，开放边界\n"
        "- **噪声：** 局域各向同性退极化 $L_k^{a}=\\sqrt{\\gamma/3}\\,\\sigma_k^{a}$，Lindblad 主方程演化\n"
        "- **测量：** 9 个测量基 × 1024 shots，q0 为最左比特；由计数恢复 Pauli-15 与 local6 特征"
    )
    st.caption("与冻结代码 `src/physics/hamiltonian.py::build_hamiltonian` 逐项一致。")

    st.subheader("AEMTN：多任务硬件兼容网络")
    cfg = read_json(root / "models" / "sim_pretrained_paper_contract" / "seed_20260731" / "run_config.json")
    if cfg:
        mc = cfg.get("model_config", {})
        nparams = _param_count(root / "models" / "sim_pretrained_paper_contract" / "seed_20260731" / "best.pt")
        st.markdown(
            f"""
- **输入特征：** `{mc.get('x_dim')}` 维 local6 + `{mc.get('t_dim')}` 维演化时间
- **共享表示：** `r_dim={mc.get('r_dim')}`，`{mc.get('num_subspaces')}` 个纠缠子空间
- **任务头：** 目标 `{mc.get('target_names')}`，各带高斯不确定性输出；辅助熵/互信息/相位标签/fidelity
- **损失权重：** Jz 权重 80（观测弱、需加重），熵/互信息桥接
- **参数量：** `{nparams:,}`
- **训练：** 3 个确定性 seed，`{cfg.get('train_samples')}` 训练 / `{cfg.get('validation_samples')}` 验证
"""
        )

        st.markdown("##### 训练摘要")
        summary = read_json(root / "models" / "sim_pretrained_paper_contract" / "training_summary.json")
        if summary:
            rows = []
            for tgt, m in summary.get("target_metrics", {}).items():
                rows.append({
                    "目标": tgt,
                    "RMSE": round(m["rmse"]["mean"], 4),
                    "R²": round(m["r2"]["mean"], 4),
                    "1σ覆盖率": round(m["coverage_1sigma"]["mean"], 3),
                    "2σ覆盖率": round(m["coverage_2sigma"]["mean"], 3),
                })
            st.table(rows)
            st.warning(
                "**模型限制：** Jz 留出基准较弱（R²≈0.32），h2 观测性最差（R²≈0）。"
                "这符合项目文档所述：Jz 只作为输出记录，**不进入自动补偿主张**；"
                "h2 作为反演目标受可辨识性限制。"
            )

    st.subheader("AI 框架选型依据")
    st.markdown(
        """
| 模块 | 方法 | 选型依据 |
|---|---|---|
| 哈密顿量反演 | AEMTN 深度多任务（共享表示 + 任务路由 + 高斯头） | 6 比特规模小、样本 2000，需共享表示抑制过拟合 |
| 漂移判别 | 去偏结构函数 + 方差成分门 | E0 阴性对照 p=0.65 不触发、E1 p≈2.8e-289 触发，区分"噪声波动 vs 过程漂移" |
| 更新周期 | OU/非参数残差曲线 + 经济门 | 从数据反推最优再校准间隔，而非拍脑袋 |
| 决策安全 | 5 道非学习 shield + 规则调度 | 在线 RL 证据不足时代之以可解释、可审计的安全回退 |
| 预测 | 滚动起点预报 + 技能门 | 严格避免未来信息泄漏 |
"""
    )

    st.subheader("技术路线要点")
    st.markdown(
        """
- **量子技术路线** → 超导量子（天衍 T176/T287），6 比特开放边界 DM 模型
- **AI 技术框架** → 深度学习（AEMTN）+ 统计决策（结构函数/经济门）+ 安全 shield/规则调度
- **选型依据** → 见上表。在线 RL、跨设备迁移、大语言模型均未在真机部署，边界见第 7 章
"""
    )


def _param_count(best_pt: Path) -> int:
    """返回模型参数量。优先从冻结 checkpoint 实时计算；torch 不可用时
    回退到随包冻结的常量（1,429,328），确保精简云环境也能显示真实数量。"""
    try:
        import torch

        ck = torch.load(str(best_pt), map_location="cpu", weights_only=False)
        from src.models.aemtn_hardware import AEMTNHardware, ModelConfig

        model = AEMTNHardware(ModelConfig.from_dict(ck["model_config"]))
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 1_429_328
