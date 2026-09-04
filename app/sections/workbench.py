"""漂移诊断工作台：上传自己的端点时序 → 冻结核心真算 → 导出报告。

诚实边界
--------
* 漂移判别（方差成分门、去偏结构函数、OU 拟合、T*、感知经济门）由冻结核心
  ``src/adaptive/sensing_economics.py::analyze_ou_sensing`` **对上传数据实时计算**。
* 哈密顿量反演 / 噪声画像属 AEMTN 网络推理（需 torch + 天衍真机探针），本工作台
  **不对上传数据反演**，只展示冻结的 T176 参考值。
* 只依赖 numpy / scipy / pandas，可在 Streamlit Community Cloud 免费实例运行。
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import FIG_DIR, PALETTE

_DEMO_SHOTS = 1024
_MAX_ROWS = 5000


# ---------------------------------------------------------------- 数据
def _demo_frame() -> pd.DataFrame:
    """内置示例：冻结 T287 真机 78 个读出快照（all-one 通道，E1 漂移检出通道）。"""
    path = FIG_DIR / "01_t287_readout_state_heatmap.source.csv"
    raw = pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame({
        "time_seconds": raw["hours_from_first_snapshot"].astype(float) * 3600.0,
        "value": raw["readout_all_one_error"].astype(float),
        "shots": _DEMO_SHOTS,
    })


def _parse_upload(file) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
    except Exception as err:  # noqa: BLE001
        return None, f"CSV 解析失败：{err}"
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["time_seconds", "value"]
    missing = [c for c in need if c not in cols]
    if missing:
        return None, f"缺少必需列：{', '.join(missing)}（现有列：{', '.join(df.columns)}）"
    out = pd.DataFrame({
        "time_seconds": pd.to_numeric(df[cols["time_seconds"]], errors="coerce"),
        "value": pd.to_numeric(df[cols["value"]], errors="coerce"),
    })
    out["shots"] = pd.to_numeric(df[cols["shots"]], errors="coerce") if "shots" in cols else np.nan
    for opt in ("regime_id", "burst_flag"):
        if opt in cols:
            out[opt] = df[cols[opt]]
    out = out.dropna(subset=["time_seconds", "value"])
    if len(out) > _MAX_ROWS:
        return None, f"行数 {len(out)} 超过上限 {_MAX_ROWS}"
    if len(out) < 3:
        return None, "至少需要 3 条有效观测"
    if ((out["value"] < 0) | (out["value"] > 1)).any():
        return None, "value 必须是 [0,1] 内的概率（如读出错误率、激发布居）"
    out = out.sort_values("time_seconds").reset_index(drop=True)
    out["time_seconds"] = out["time_seconds"] - out["time_seconds"].iloc[0]
    return out, None


# ---------------------------------------------------------------- 计算
def _lag_edges(times: np.ndarray, n_bins: int) -> list[float]:
    """按 pair-lag 分位数取边，保证每 bin 有对；与冻结分析同源思路（对数等分）。"""
    left, right = np.triu_indices(len(times), k=1)
    lags = times[right] - times[left]
    lags = lags[lags > 0]
    lo, hi = float(lags.min()), float(lags.max()) * 1.0001
    if hi <= lo:
        return [lo, lo + 1.0]
    return [float(v) for v in np.geomspace(lo, hi, n_bins + 1)]


def _run(df: pd.DataFrame, *, shots_default: int, n_bins: int, rate: float, floor: float,
         window: float, bootstrap: int) -> dict:
    from src.adaptive.sensing_economics import analyze_ou_sensing

    times = df["time_seconds"].to_numpy(float)
    values = df["value"].to_numpy(float)
    shots = df["shots"].fillna(shots_default).to_numpy(int)
    regimes = df["regime_id"].astype(str).to_numpy() if "regime_id" in df else None
    bursts = df["burst_flag"].astype(str).str.lower().isin(["1", "true", "yes"]).to_numpy() if "burst_flag" in df else None
    return analyze_ou_sensing(
        values=values,
        times_seconds=times,
        shots=shots,
        regime_ids=regimes,
        burst_flags=bursts,
        lag_edges_seconds=_lag_edges(times, n_bins),
        effective_shots_per_second=rate,
        maximum_interval_seconds=window,
        interface_floor_seconds=floor,
        bootstrap_resamples=bootstrap,
    )


# ---------------------------------------------------------------- 绘图
def _dark(fig: go.Figure, title: str, x: str, y: str) -> go.Figure:
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=y,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e6e6"), margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.2),
        yaxis_title_standoff=8,
    )
    fig.update_yaxes(title_font=dict(size=12))
    fig.update_xaxes(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a")
    fig.update_yaxes(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a")
    return fig


def _plot_series(df: pd.DataFrame, shots_default: int) -> go.Figure:
    p = df["value"].to_numpy(float)
    n = df["shots"].fillna(shots_default).to_numpy(float)
    sigma = np.sqrt(p * (1 - p) / n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_seconds"], y=p, mode="markers+lines", name="观测值",
        error_y=dict(type="data", array=sigma, visible=True, color="#777"),
        line=dict(color=PALETTE["blue"], width=1), marker=dict(size=5),
    ))
    return _dark(fig, "上传的端点时序（误差棒 = shot-noise σ）", "时间 / s", "概率")


def _plot_sf(res: dict) -> go.Figure:
    rows = res["structure_function"]
    fig = go.Figure()
    if rows:
        lag = [r["lag_mid_seconds"] for r in rows]
        sf = [r["sf_debiased"] for r in rows]
        lo = [r["sf_ci_lower"] for r in rows]
        hi = [r["sf_ci_upper"] for r in rows]
        fig.add_trace(go.Scatter(
            x=lag, y=sf, mode="markers", name="去偏结构函数 ± 95% CI",
            error_y=dict(type="data", symmetric=False,
                         array=np.subtract(hi, sf), arrayminus=np.subtract(sf, lo)),
            marker=dict(color=PALETTE["orange"], size=8),
        ))
        fig.add_trace(go.Scatter(
            x=lag, y=[r["shot_floor"] for r in rows], mode="lines", name="shot floor",
            line=dict(color="#666", dash="dot"),
        ))
    fit = res["ou_fit"]
    if fit["ok"]:
        from src.adaptive.sensing_economics import ou_structure_value

        t = np.geomspace(max(min(lag) / 3, 1e-3), max(lag) * 2, 200)
        fig.add_trace(go.Scatter(
            x=t, y=ou_structure_value(t, fit["process_variance"], fit["tau_seconds"]),
            mode="lines", name=f"OU 拟合 τ={fit['tau_seconds']:.3g}s",
            line=dict(color=PALETTE["aqua"], width=2),
        ))
    fig.update_xaxes(type="log")
    return _dark(fig, "漂移结构函数 SF(Δt) 与 OU 拟合", "时滞 Δt / s（对数）", "SF")


def _plot_residual(res: dict, mean_p: float, rate: float, floor: float, window: float) -> go.Figure | None:
    fit = res["ou_fit"]
    if not fit["ok"]:
        return None
    from src.adaptive.sensing_economics import ou_residual_variance

    t = np.geomspace(max(window / 1e4, 1e-3), window, 300)
    r = ou_residual_variance(t, mean_p, rate, fit["process_variance"], fit["tau_seconds"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=r, mode="lines", name="预测残差方差", line=dict(color=PALETTE["blue"], width=2)))
    pv = res["variance_gate"]["process_variance"]
    fig.add_hline(y=pv, line=dict(color=PALETTE["red"], dash="dash"), annotation_text="不感知（过程方差）", annotation_font_color="#ccc")
    fig.add_vline(x=floor, line=dict(color="#888", dash="dot"), annotation_text="接口下限 T_floor", annotation_font_color="#ccc")
    if res["t_star_seconds"]:
        fig.add_vline(x=res["t_star_seconds"], line=dict(color=PALETTE["aqua"]), annotation_text="T*", annotation_font_color="#ccc")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _dark(fig, "更新周期 vs 残差方差（感知经济曲线）", "更新周期 T / s（对数）", "残差方差（对数）")


# ---------------------------------------------------------------- 报告
def _verdict(res: dict) -> tuple[str, str, str]:
    vg = res["variance_gate"]
    if not vg["passed"]:
        return ("crit", "未检出超 shot-noise 的过程方差",
                f"方差成分门未通过（p={vg['p_value']:.3g}，过程方差 CI 下界={vg['process_variance_ci_lower']:.3g}）。"
                "观测涨落与 shot noise 相容，无法主张漂移；与冻结 E0 阴性对照同型。")
    if not res["ou_fit"]["ok"]:
        return ("warn", "检出过程方差，但 OU 结构不可拟合",
                "方差门通过，但结构函数 bin 不足或拟合失败；请增加观测点或调整 lag bin 数。")
    if res["worth_sensing"]:
        return ("good", "检出漂移，且值得感知（go）",
                f"T* = {res['t_star_seconds']:.3g} s（CI {res['t_star_ci_lower_seconds']:.3g}–{res['t_star_ci_upper_seconds']:.3g}），"
                f"经济分离裕度 {res['economic_separation_margin']:.3g}。以 T* 周期重标定优于不感知。")
    return ("warn", "检出漂移，但不值得按此吞吐感知（no-go）",
            f"T* = {res['t_star_seconds']:.3g} s，但最差角残差 {res['worst_corner_residual_variance']:.3g} ≥ "
            f"过程方差 CI 下界 {res['process_variance_ci_lower_for_economic_gate']:.3g}；请提高有效吞吐或放宽接口下限。")


def _report(res: dict, df: pd.DataFrame, params: dict, verdict: tuple[str, str, str]) -> bytes:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": "AEMTN-B4 drift-diagnosis workbench",
        "frozen_core": "src/adaptive/sensing_economics.py::analyze_ou_sensing",
        "claim_boundary": "drift analysis computed on uploaded data; Hamiltonian inference NOT performed (frozen T176 reference only)",
        "input_summary": {"n_observations": int(len(df)), "span_seconds": float(df["time_seconds"].iloc[-1]),
                          "mean_value": float(df["value"].mean())},
        "parameters": params,
        "verdict": {"level": verdict[0], "headline": verdict[1], "detail": verdict[2]},
        "analysis": clean(res),
    }
    return json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")


# ---------------------------------------------------------------- 页面
def render() -> None:
    from app import theme

    st.markdown(
        '<p style="max-width:52rem">上传<b style="color:#fff">你自己的</b>量子端点时序'
        "（任意可重复测量的概率型观测：读出错误率、激发布居、Ramsey 端点……），"
        "平台用<b style=\"color:#fff\">冻结核心</b>实时判别：是否存在超 shot-noise 的环境漂移、"
        "漂移相关时间 τ、最优重标定周期 T*、是否值得感知。</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # 01 数据
    with theme.row("01", "数据"):
        src = st.radio("数据来源", ["内置示例（T287 真机）", "上传 CSV"], horizontal=True, label_visibility="collapsed")
        df: pd.DataFrame | None = None
        if src.startswith("上传"):
            up = st.file_uploader("选择 CSV", type=["csv"], label_visibility="collapsed")
            if up is not None:
                df, err = _parse_upload(up)
                if err:
                    st.error(err)
                    return
        else:
            df = _demo_frame()
        with st.expander("输入格式与示例文件"):
            st.markdown(
                """
| 列 | 必需 | 含义 |
|---|---|---|
| `time_seconds` | 是 | 观测时刻（秒，任意零点） |
| `value` | 是 | 概率型观测值 ∈ [0,1] |
| `shots` | 否 | 每点 shot 数（缺省用参数区默认值） |
| `regime_id` | 否 | 标定区段 id；跨区段不配对 |
| `burst_flag` | 否 | 1/true 表示突发点，主估计中剔除 |
"""
            )
            st.download_button("下载示例 CSV · 冻结 T287 真机 78 快照", _demo_frame().to_csv(index=False).encode("utf-8"),
                               "aemtn_demo_t287_readout_all_one.csv", "text/csv")
        if df is None:
            st.caption("等待上传。")
            return

    st.markdown("---")
    # 02 参数
    with theme.row("02", "参数"):
        span = float(df["time_seconds"].iloc[-1]) if len(df) else 3600.0
        p1, p2, p3 = st.columns(3)
        with p1:
            shots_default = st.number_input("默认 shots / 点", 2, 1_000_000, _DEMO_SHOTS, step=256)
            n_bins = st.slider("lag bin 数", 3, 12, 6)
        with p2:
            rate = st.number_input("有效吞吐 shots/s", 1.0, 1e7, 100.0, step=50.0, format="%.1f")
            floor = st.number_input("接口下限 T_floor / s", 0.001, 1e6, 60.0, format="%.3f")
        with p3:
            window = st.number_input("最大更新周期 / s", 0.01, 1e8, max(span, 1.0), format="%.1f")
            bootstrap = st.slider("参数 bootstrap 次数", 0, 500, 0, step=50, help="0 = 不做；>0 会显著变慢")

    st.markdown("---")
    # 03 时序
    with theme.row("03", "时序"):
        c1, c2, c3 = st.columns(3)
        c1.metric("观测点数", len(df))
        c2.metric("时间跨度", f"{span:,.0f} s")
        c3.metric("均值", f"{df['value'].mean():.4f}")
        st.plotly_chart(_plot_series(df, int(shots_default)), width="stretch")
        go_ = st.button("▶ 执行漂移诊断", type="primary")

    if not go_:
        return

    params = dict(shots_default=int(shots_default), n_lag_bins=int(n_bins), effective_shots_per_second=float(rate),
                  interface_floor_seconds=float(floor), maximum_interval_seconds=float(window), bootstrap_resamples=int(bootstrap))
    with st.spinner("冻结核心计算中……"):
        try:
            res = _run(df, shots_default=int(shots_default), n_bins=int(n_bins), rate=float(rate),
                       floor=float(floor), window=float(window), bootstrap=int(bootstrap))
        except ValueError as err:
            st.error(f"核心拒绝该输入：{err}")
            return

    level, headline, detail = _verdict(res)
    vg, fit = res["variance_gate"], res["ou_fit"]

    st.markdown("---")
    # 04 裁决
    with theme.row("04", "裁决"):
        {"good": st.success, "warn": st.warning, "crit": st.error}[level](f"**{headline}**  \n{detail}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("方差门 p 值", f"{vg['p_value']:.2e}", "通过" if vg["passed"] else "未通过")
        m2.metric("过程方差", f"{vg['process_variance']:.3g}")
        m3.metric("相关时间 τ", f"{fit['tau_seconds']:.3g} s" if fit["ok"] else "—")
        m4.metric("最优周期 T*", f"{res['t_star_seconds']:.3g} s" if res["t_star_seconds"] else "—")

    st.markdown("---")
    # 05 结构
    with theme.row("05", "结构函数"):
        st.plotly_chart(_plot_sf(res), width="stretch")
        rfig = _plot_residual(res, float(df["value"].mean()), float(rate), float(floor), float(window))
        if rfig is not None:
            st.plotly_chart(rfig, width="stretch")
        if res["structure_function"]:
            with st.expander("逐 bin 数值"):
                st.dataframe(pd.DataFrame(res["structure_function"]).round(9), width="stretch", hide_index=True)

    st.markdown("---")
    # 06 参考
    with theme.row("06", "冻结参考"):
        st.markdown("**哈密顿量 / 噪声画像**（不对上传数据反演）")
        st.caption(
            "AEMTN 反演需 torch 网络 + 天衍真机 Pauli 探针，超出本云实例范围。"
            "以下为冻结 T176 Session 0 参考值，详见「证据」页。"
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("终测 ratio", "0.3616", "fast/slow 残差比")
        r2.metric("置换检验 p", "0.0052", "20 000 × 3")
        r3.metric("参考标签", "B4_PRESERVED", "SIMULATION_ASSISTED")

    st.markdown("---")
    # 07 导出
    with theme.row("07", "导出"):
        st.download_button(
            "⬇ 导出诊断报告 (JSON)",
            _report(res, df, params, (level, headline, detail)),
            f"aemtn_drift_report_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json",
            "application/json",
        )
