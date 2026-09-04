"""环境漂移感知与判别页：可观测环境代理 → 漂移判别。"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import PALETTE, FIG_DIR, load_source_csv


def render() -> None:

    st.subheader("第 1 层：环境状态可观测")
    st.markdown(
        "**读出一个事实：** 云 API 用户位能否从量子测量看到随时间变化的状态？"
        "H1/H2 是量子测量定义的环境代理，**不是**温度、电磁传感器或底层脉冲参数。"
    )
    rows = load_source_csv("01_t287_readout_state_heatmap")
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        for col in ("readout_all_zero_error", "readout_all_one_error"):
            df[col] = df[col].astype(float)
        fig = px.scatter(
            df,
            x="hours_from_first_snapshot",
            y=["readout_all_zero_error", "readout_all_one_error"],
            labels={"value": "读出错误概率", "hours_from_first_snapshot": "距首快照小时"},
            color_discrete_sequence=[PALETTE["blue"], PALETTE["magenta"]],
            title="T287 探针读出状态时间序列（78 个真机快照）",
        )
        fig.update_layout(legend_title_text="通道")
        st.plotly_chart(fig, width="stretch")
        st.caption("78 个快照全部保留，无排除。")
    else:
        st.info("缺少图 01 source CSV。")

    rows = load_source_csv("02_t287_effective_field_phase_space")
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        for c in ("effective_h1", "effective_h2", "combined_shot_sigma"):
            df[c] = df[c].astype(float)
        fig = px.scatter(
            df,
            x="effective_h1",
            y="effective_h2",
            color="hours_from_first_snapshot",
            error_x="effective_h1_shot_sigma",
            error_y="effective_h2_shot_sigma",
            color_continuous_scale="Blues",
            title="测量定义的有效场状态（H1/H2，误差棒为 shot uncertainty）",
        )
        st.plotly_chart(fig, width="stretch")
        st.info(
            "**边界声明：** H1/H2 是量子测量环境代理，不冒充温度/电磁传感器读数，"
            "也不等同于底层脉冲标定参数。"
        )

    st.subheader("第 2 层：区分普通波动与可检出漂移")
    st.markdown(
        "成对证据：**E0 阴性对照**应不触发漂移门，**E1** 通道应检出超过 shot-noise 的过程方差。"
        "这一步回答\"不是所有时间波动都叫漂移\"。"
    )
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**E0 阴性对照（应不触发）**")
        rows = load_source_csv("03_t287_e0_negative_control_forest")
        _forest_strip(rows, "E0", "应停在 shot-noise floor")
        st.caption("冻结方差门 n=30，p=0.6497，**未触发**漂移判定。")
    with c4:
        st.markdown("**E1 信号通道（应触发）**")
        rows = load_source_csv("04_t287_e1_drift_forest")
        _forest_strip(rows, "E1", "应超过 shot-noise floor")
        st.caption("冻结方差门 n=30，p≈2.8e-289，**检测到**超过 shot-noise 的过程方差。")

    st.subheader("第 3 层：漂移→校准更新间隔经济学")
    st.markdown(
        "残差—更新间隔经济曲线：从数据反推最优再校准间隔 T*，而非拍脑袋。"
    )
    rows = load_source_csv("05_t287_e1_interval_residual_heatmap")
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        for c in ("interval_seconds", "ou_point_residual_variance", "nonparametric_residual_variance", "no_sensing_process_variance"):
            df[c] = df[c].astype(float)
        figh = go.Figure()
        for col, name, color in [
            ("ou_point_residual_variance", "OU 点估计", PALETTE["blue"]),
            ("nonparametric_residual_variance", "非参数敏感性", PALETTE["orange"]),
            ("no_sensing_process_variance", "不感知（shot-only）", PALETTE["muted"]),
        ]:
            figh.add_trace(go.Scatter(x=df["interval_seconds"], y=df[col], name=name, line={"color": color, "width": 2}))
        figh.update_layout(
            xaxis_title="校准更新间隔 (s)", yaxis_title="端点平方残差",
            hovermode="x unified", template="simple_white",
            title="E1 通道残差—更新间隔经济曲线",
        )
        st.plotly_chart(figh, width="stretch")
        st.warning(
            "**T\* 结论：** OU 点估计 T\*≈134 s，但 bootstrap 95% CI 为 101–4000 s，"
            "**上界撞到观测窗**，因此冻结裁决为 `INCONCLUSIVE`。不能把 134 s 当作已识别、"
            "可部署的唯一最优策略。"
        )

    st.subheader("第 4 层：T\* 与云 API 控制 floor 对比")
    rows = load_source_csv("06_t287_control_floor_interval_strip")
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df["point_seconds"] = df["point_seconds"].astype(float)
        df["log"] = df["point_seconds"].apply(lambda x: max(x, 0.1))
        figg = px.bar(
            df, x="quantity", y="point_seconds", log_y=True,
            color="origin", color_discrete_sequence=[PALETTE["blue"], PALETTE["orange"]],
            title="云 API 接口延迟与协议可达 floor（对数秒）",
        )
        # 上下界有行缺省（interface/floor 无 bootstrap CI），用 to_numeric 容忍空串 -> NaN
        for col in ("ci_lower_seconds", "ci_upper_seconds"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        st.plotly_chart(figg, width="stretch")
        st.caption(
            "接口 P50≈12.4 s、P90≈14.7 s、协议可达 floor=60 s，"
            "与 T\* 的 bootstrap 95% CI 101–4000 s 比较。"
        )
        st.info(
            "**结论边界：** 平台能够响应**不等于**最优间隔已被识别。当前只能客观报告 `INCONCLUSIVE`。"
        )


def _forest_strip(rows, label: str, note: str) -> None:
    if not rows:
        st.info(f"缺少 {label} 图 source CSV。")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    for c in ("lag_mid_seconds", "sf_debiased", "sf_ci_lower", "sf_ci_upper", "shot_floor"):
        if c in df:
            df[c] = df[c].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["lag_mid_seconds"], y=df["sf_debiased"], mode="markers",
        marker={"color": PALETTE["orange"]}, name="去偏 SF", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df["lag_mid_seconds"], y=df["shot_floor"], mode="lines",
        line={"color": PALETTE["muted"], "dash": "dash"}, name="shot-floor",
    ))
    fig.update_layout(
        xaxis_title="滞后间隔 (s)", yaxis_title="结构函数",
        template="simple_white", title=f"{label} 结构函数 vs shot-floor",
    )
    st.plotly_chart(fig, width="stretch")
