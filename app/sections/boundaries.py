"""第 7 章：边界与限制。

哪些结论有证据支持、哪些必须带限定、哪些不成立，逐条映射到仓库红线
（CLAIM_BOUNDARY.md）与冻结报告的 claim_boundary 字段。
"""

from __future__ import annotations

import streamlit as st

from app.common import CRIT, WARN, PALETTE


def render() -> None:
    st.header("边界与限制")

    st.markdown(
        "本节把项目**能主张什么、必须带什么限定、不能主张什么**逐条列明。"
        "红线来自仓库 `CLAIM_BOUNDARY.md` 与冻结报告 "
        "`B4_T176_HYBRID_FINAL_20260829` 的 `claim_boundary` 字段。"
    )

    c = st.columns(3)
    with c[0]:
        st.markdown(f"### :green[可以说]")
        st.markdown("有冻结证据支持的结论。")
        st.markdown(
            "- 主训练代码、数据契约、三 seed checkpoint 已冻结。\n"
            "- 量子测量代理可显示 T287 读出状态随时间变化。\n"
            "- E0 阴性对照**未**触发漂移门；E1 过程方差超出 shot-noise 解释。\n"
            "- T176 Session 0 的 20 对，total 残差比 0.3616，配对置换 p=0.00525。\n"
            "- simulation-assisted hybrid ratio 0.3745，p=0.0001。"
        )
    with c[1]:
        st.markdown(f"### :orange[必须带限定]")
        st.markdown("成立，但必须伴随限制条件。")
        st.markdown(
            "- B4 只写成 `B4_PRESERVED_SIMULATION_ASSISTED`。\n"
            "- 纯真机注册状态仍是 `INCONCLUSIVE_MISSING_HARDWARE_SESSION1`。\n"
            "- T*=134.4 s 是点估计；置信上界撞到观测窗，不能写成稳定生产 SLA。\n"
            "- F11 是模拟辅助 / 事后敏感性，不是真机第二会话。"
        )
    with c[2]:
        st.markdown(f"### :red[不可说]")
        st.markdown("没有证据支持、或与事实相反的表述。")
        st.markdown(
            "- 不可写「双真机闭环复现」或「registered all-hardware PASS」。\n"
            "- 不可把 H1/H2 代理称为直接温度或电磁测量。\n"
            "- 不可声称在线 RL、跨设备知识迁移、完整环境传感融合已在真机完成。\n"
            "- 不可把 63.84% 残差降低写成通用算力提升或全任务性能提升。"
        )

    st.divider()

    st.subheader("技术瓶颈（客观分析）")
    st.markdown("以下是当前工作已识别的主要限制，以及对应的判定与后续路径。")
    bottlenecks = [
        (
            "单会话过短，无法注册全真机闭环",
            "Session 1 未在真机采集，纯真机注册只能判 `INCONCLUSIVE`。",
            "`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`",
            "需更多真机机时完成 Session 1，或采纳非逐会话的注册协议。",
        ),
        (
            "T* 点估计置信区间跨几个数量级",
            "T*≈134 s，但 bootstrap 95% CI 为 101–4000 s，上界撞到观测窗。",
            "`INCONCLUSIVE`",
            "延长观测窗、提高采样密度以收紧 T*。",
        ),
        (
            "目标学习信号弱（h2、Jz）",
            "h2 的 R²≈0，Jz 的 R²≈0.32，说明这两个目标在数据上可辨识性不足。",
            "已识别瓶颈",
            "需更贴近 h2/Jz 的可观测探针或更多真机数据。",
        ),
        (
            "H1/H2 是量子测量代理，非底层物理参数",
            "读出的是量子测量定义的环境代理，不是温度 / 电磁传感器读数。",
            "边界声明",
            "需接入真实环境传感器做交叉验证。",
        ),
    ]
    for title, desc, tag, next_step in bottlenecks:
        with st.expander(title):
            st.write(desc)
            st.markdown(f"**判定：** `{tag}`")
            st.markdown(f"**缓解 / 后续：** {next_step}")

    st.divider()

    st.subheader("未部署 / 未验证项")
    st.markdown("以下能力**未**在真机上验证或部署：")
    not_deployed = [
        "在线强化学习：当前是单步随机 bandit，simulator 冷启动，真机侧由外部安全盾约束；"
        "不等于已部署在线 RL。",
        "跨设备知识迁移：仅在 T176/T287 观测窗口内做迁移分析，未做任意跨型号泛化。",
        "实时控制回路：fast/slow cadence 闭环由冻结 `rule_scheduler` 触发；它提供可解释回退，"
        "而非已部署的高频在线学习控制器。",
        "模型在线更新：三 seed 权重在冻结时确定，闭环推理用 `best.pt`；**不是**在线训练。",
    ]
    for item in not_deployed:
        st.markdown(f"- {item}")

    st.divider()

    st.subheader("数据与依赖边界")
    st.markdown(
        "核心依赖（`cqlib`、`qutip`、`torch`）均为公开发布版本；"
        "仓库不含私有凭据、API 密钥或 `.env` 文件。完整边界见 "
        "`SECURITY_AND_DATA_BOUNDARY.md`。"
    )
