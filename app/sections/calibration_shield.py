"""校准决策与安全 shield 页：交互式演示五门安全逻辑，在线可跑（仅 numpy）。"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.common import CRIT, GOOD, PALETTE, WARN


def render() -> None:
    st.header("校准决策与安全 shield")

    st.subheader("校准更新周期：go/no-go 经济门")
    st.markdown(
        """
从漂移检出进入校准更新间隔经济学：给定可观测的漂移强度、shot 预算和接口延迟，
算出**值不值得感知、值不值得再校准、目标间隔应多长**。冻结核心
`src/adaptive/sensing_economics.py` 实现去偏结构函数、OU/非参数残差曲线、T\* 与经济门。
"""
    )

    st.subheader("交互演示：安全 shield 五道门")
    st.markdown(
        "下面是**作用于真实冻结逻辑**的在线演示（`src/adaptive/bandit.py`），"
        "可拖动滑块看一个补偿动作被哪道门放行或拦下。",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mu_h1 = st.slider("估计 h1", -1.0, 1.0, 0.30, 0.01)
        mu_h2 = st.slider("估计 h2", -1.0, 1.0, -0.20, 0.01)
    with c2:
        sigma_h1 = st.slider("h1 不确定性", 0.05, 0.50, 0.10, 0.01)
        sigma_h2 = st.slider("h2 不确定性", 0.05, 0.50, 0.10, 0.01)
    with c3:
        p_ex = st.slider("超出阈值概率", 0.0, 1.0, 0.30, 0.05)
        shots = st.slider("已用 shots (百万)", 0, 250, 100, 1) * 1_000_000

    gain = st.slider("补偿增益", 0.0, 1.0, 0.25, 0.05)
    control = st.selectbox("动作", ["act", "abstain", "escalate"])

    from aemtn_b4 import shield_decision  # 懒加载：仅 numpy

    sd = shield_decision(
        mu_h1=mu_h1, mu_h2=mu_h2, sigma_h1=sigma_h1, sigma_h2=sigma_h2,
        p_exceed=p_ex, shots_used=int(shots), gain=gain, control=control,
    )
    permitted = bool(sd["permitted"])

    color = GOOD if permitted else CRIT
    verdict = "**放行**" if permitted else f"**拦截**（门：`{sd['gate']}`）"
    st.markdown(f"### <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
    st.write(f"原因：{sd['reason']}")
    st.write(f"补偿量 (Δh1, Δh2, ΔJz)：`{sd['compensation']}`")

    # 门状态条：展示 5 道门中哪道卡住
    st.markdown("##### 五道安全门状态")
    gates = [
        ("confidence", "置信与支持度"),
        ("physical_range", "物理范围"),
        ("jz_reject", "Jz 补偿禁行"),
        ("action_amplitude", "动作幅度"),
        ("budget", "shot 预算"),
    ]
    gate_cols = st.columns(5)
    for i, (key, label) in enumerate(gates):
        ok = key != sd["gate"]
        with gate_cols[i]:
            st.markdown(f"{'🟢' if ok else '🔴'} **{label}**")

    st.info(
        "**说明：** shield 是外部、非学习、可审计的安全层。在线 RL 证据不足时，"
        "项目以 `rule_scheduler` 提供可解释回退（固定阈值触发、无模型评分、无硬件执行器）。"
        "两者都不等于\"已部署在线强化学习\"。"
    )

    # 展示五门如何随不确定性变化
    st.markdown("##### 不确定性与预置预算对放行的影响")
    xs = [0.05 + 0.01 * i for i in range(46)]
    ys = []
    for s in xs:
        r = shield_decision(
            mu_h1=mu_h1, mu_h2=mu_h2, sigma_h1=s, sigma_h2=s,
            p_exceed=p_ex, shots_used=int(shots), gain=gain, control=control,
        )
        ys.append(1 if bool(r["permitted"]) else 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="允许",
        line={"color": PALETTE["blue"], "width": 2}, marker={"size": 4},
    ))
    fig.add_hline(y=1, line_dash="dot", line_color=GOOD)
    fig.add_hline(y=0, line_dash="dot", line_color=CRIT)
    fig.add_vline(x=0.20, line_dash="dash", line_color=WARN)
    fig.update_layout(
        xaxis_title="假设的不确定性 σ", yaxis_title="是否放行 (1/0)",
        yaxis={"tickvals": [0, 1]}, template="simple_white",
        title="五门对不确定性的决策边界（σ=0.20 为冻结阈值）",
    )
    st.plotly_chart(fig, width="stretch")
