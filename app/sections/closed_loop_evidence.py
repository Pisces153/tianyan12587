"""第 5 章：真机闭环终测证据。

展示冻结的 T176 hybrid final（Session 0 诊断 + 混合闭环 primary + 反事实
Session 1），以及 12 张成果图里直接相关的 07–10/11。提供一个**在线复算**按钮：
在浏览器里用纯 Python 重跑 20000 次置换门，并与冻结值比对（matches_frozen）。
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.common import CRIT, GOOD, PALETTE, WARN, load_source_csv


def render() -> None:
    st.header("真机闭环终测证据")

    from aemtn_b4 import load_final_report

    report = load_final_report()

    st.subheader("证据状态")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "**项目级判定**\n\n"
            f"`{report.project_status}`\n\n"
            "只在明确的事后一致性检验中成立。"
        )
    with c2:
        st.markdown(
            "**纯真机注册**\n\n"
            f"`{report.registered_hardware_status}`\n\n"
            "Session 1 未上真机，故不构成全真机独立确认。"
        )
    with c3:
        st.markdown(
            "**Session 0 证据等级**\n\n"
            f"`{report.confidence}`\n\n"
            "描述性支持，非统计学确认。"
        )

    st.divider()

    st.subheader("三类证据层（冻结统计）")
    st.markdown(
        "来自 `evidence/B4_T176_HYBRID_FINAL_20260829` 的冻结判定。"
        "**比值是 fast/slow 端点平方残差之比，越小越好；通过标准是比值低于临界值。**"
    )
    header = st.columns([2, 1, 1, 1, 1, 1])
    for col, label in zip(header, ["证据层", "pairs", "比值", "相对降低", "p 值", "结论"]):
        col.markdown(f"**{label}**")
    rows = report.layers
    _LAYER_NOTE = {
        "真机 Session 0（诊断）": "fast 显著更优，但属描述性证据",
        "混合闭环（primary）": "合并 20 真机 + 20 模拟往返，事后一致性",
        "模拟 Session 1（反事实）": "未上真机，仅模拟反事实对照",
    }
    for L in rows:
        col = st.columns([2, 1, 1, 1, 1, 1])
        col[0].write(L.layer)
        col[1].write(L.pairs)
        col[2].write(f"{L.ratio:.4f}")
        col[3].write(f"{100.0 * L.relative_reduction:.2f}%")
        col[4].write(f"{L.p_value:.4g}")
        passed = L.passed
        col[5].markdown("🟢 通过" if passed else "🔴 未通过")
        st.caption(_LAYER_NOTE.get(L.layer, ""))

    st.divider()

    st.subheader("真机配对证据（图 07–09）")
    st.markdown("20 个真机往返 of Session 0 的配对残差。")
    _paired_scatter()
    _cumulative_step()

    st.subheader("置换检验直方图（图 10）")
    _permutation_histogram()

    st.subheader("基线敏感性（图 11）")
    _baseline_dumbbell()

    st.subheader("在线复算：冻结结果的独立重跑")
    st.markdown(
        "点击下方按钮，在**当前浏览器环境**用纯 Python 对三个证据层各跑 "
        "20,000 次配对置换，并与冻结值比对。这验证结果可独立复现。",
        unsafe_allow_html=True,
    )
    if st.button("▶ 重跑 20,000×3 次置换门", type="primary"):
        from aemtn_b4 import reproduce_final

        with st.spinner("正在重跑置换检验……"):
            res = reproduce_final()
        matches = all(v["matches_frozen"] for v in res.values())
        st.success(
            "复算完成：三个证据层全部与冻结值一致（matches_frozen=True）。"
            if matches
            else "⚠️ 复算结果与冻结值不一致，请检查数据完整性。"
        )
        # 用表格展示复算过程
        data = {
            "证据层": ["真机 Session 0", "混合闭环 primary", "模拟 Session 1"],
            "pairs": [v["pairs"] for v in res.values()],
            "重跑比值": [f"{v['ratio']:.4f}" for v in res.values()],
            "重跑 p": [f"{v['p_value']:.4g}" for v in res.values()],
            "与冻结一致": ["✅" if v["matches_frozen"] else "❌" for v in res.values()],
        }
        st.dataframe(data, width="stretch")

    st.divider()

    st.warning(
        "**结论必须被限制在：** B4 只在明确的事后、模拟辅助的一致性检验中被保存。"
        "注册的全真机端点仍为 `INCONCLUSIVE_MISSING_HARDWARE_SESSION1`，"
        "因为 Session 1 未在真机上采集。不能声称 p 值为独立全真机确认，"
        "也不能移除或重标模拟来源标签。"
    )


def _paired_scatter() -> None:
    rows = load_source_csv("07_t176_hardware_paired_scatter")
    if not rows:
        st.info("缺少图 07 source CSV。")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    for c in ("fast_endpoint_squared_residual", "slow_endpoint_squared_residual"):
        df[c] = df[c].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["fast_endpoint_squared_residual"], y=df["slow_endpoint_squared_residual"],
        mode="markers", marker={"color": PALETTE["blue"], "size": 10},
        text=df["analysis_pair_id"], name="pair",
    ))
    maxv = max(df["fast_endpoint_squared_residual"].max(), df["slow_endpoint_squared_residual"].max()) * 1.1
    fig.add_trace(go.Scatter(x=[0, maxv], y=[0, maxv], mode="lines", line={"color": PALETTE["muted"], "dash": "dash"}, name="y=x"))
    fig.update_layout(
        xaxis_title="fast 端点平方残差", yaxis_title="slow 端点平方残差", height=520,
        template="simple_white",
        title="真机配对：fast vs slow 端点平方残差（点越靠 y=x 下方越有利）",
    )
    fig.update_yaxes(type="log", range=[-6, 0])
    fig.update_xaxes(type="log", range=[-6, 0])
    st.plotly_chart(fig, width="stretch")
    st.caption("20 个真机往返；方格内 y=x 虚线是残差相等线。多数点落在下侧，即 fast 更优。")


def _cumulative_step() -> None:
    rows = load_source_csv("09_t176_cumulative_ratio_step")
    if not rows:
        st.info("缺少图 09 source CSV。")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    df["cumulative_fast_over_slow_ratio"] = df["cumulative_fast_over_slow_ratio"].astype(float)
    df["pair_index"] = df["pair_index"].astype(int)
    df = df.sort_values("pair_index")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["pair_index"], y=df["cumulative_fast_over_slow_ratio"],
        mode="lines+markers", line={"color": PALETTE["aqua"], "width": 2},
        name="累计 fast/slow 比值",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=CRIT)
    fig.update_layout(
        xaxis_title="按 block 顺序的配对序号", yaxis_title="累计 fast/slow 残差比",
        template="simple_white", height=420,
        title="累计比值阶梯：比值逐步稳定在 1 以下",
    )
    st.plotly_chart(fig, width="stretch")


def _permutation_histogram() -> None:
    rows = load_source_csv("10_t176_permutation_histogram")
    if not rows:
        st.info("缺少图 10 source CSV。")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    df["null_ratio"] = df["null_ratio"].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["null_ratio"], nbinsx=60, marker={"color": PALETTE["blue"], "opacity": 0.75},
        name="null 分布",
    ))
    fig.update_layout(
        xaxis_title="随机置换下的比值", yaxis_title="频数", template="simple_white",
        height=420, bargap=0.02,
        title="20,000 次置换 null 直方图（观测比值落在左侧则显著）",
    )
    st.plotly_chart(fig, width="stretch")


def _baseline_dumbbell() -> None:
    rows = load_source_csv("11_baseline_sensitivity_dumbbell")
    if not rows:
        st.info("缺少图 11 source CSV。")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    for c in ("ratio", "critical_ratio", "p_value", "margin_to_critical", "ratio_over_critical"):
        df[c] = df[c].astype(float)
    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["ratio"], r["critical_ratio"]],
            y=[r["analysis"], r["analysis"]],
            mode="lines+markers",
            line={"color": PALETTE["orange"], "width": 3},
            marker={"size": 12, "color": [PALETTE["blue"], CRIT]},
            showlegend=False,
        ))
    fig.update_layout(
        xaxis_title="比值", template="simple_white", height=360,
        title="基线敏感性：观测比值 vs 临界值（比值越左越好）",
    )
    st.plotly_chart(fig, width="stretch")
