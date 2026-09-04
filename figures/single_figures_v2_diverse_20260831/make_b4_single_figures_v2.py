#!/usr/bin/env python3
"""Build the B4 v2 gallery with one diverse chart per scientific question.

The script consumes the frozen v1 source CSV files, never the rendered figures.
For every v2 figure it exports SVG/PDF/PNG/TIFF plus a CSV containing both the
raw values and any derived visual encodings (z scores, log10 values, plot order,
or decision margins).  Hardware, model-derived, sensitivity, and workload
evidence remain explicitly separated.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "single_figures_main_20260831"

COLORS = {
    "ink": "#202124",
    "muted": "#697077",
    "grid": "#D7DBDF",
    "blue": "#1F5A94",
    "blue_soft": "#DCEAF6",
    "teal": "#16837A",
    "teal_soft": "#D9EFEC",
    "violet": "#6E5DA8",
    "amber": "#C68A13",
    "red": "#B6423E",
    "red_soft": "#F5DEDC",
    "green": "#2D8259",
    "green_soft": "#DCEFE4",
    "grey": "#9AA0A6",
    "grey_soft": "#EEF1F3",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"Non-finite values in {field}")
    return values


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> Path:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_chart(
    title: str,
    subtitle: str,
    *,
    figsize: tuple[float, float] = (7.2, 4.6),
    left: float = 0.13,
    right: float = 0.96,
    bottom: float = 0.16,
    top: float = 0.78,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.text(0.055, 0.965, title, ha="left", va="top", fontsize=12.0, fontweight="bold", color=COLORS["ink"])
    fig.text(0.055, 0.898, subtitle, ha="left", va="top", fontsize=7.8, color=COLORS["muted"])
    fig.text(0.97, 0.022, "v2 独立主图｜同名 source.csv 为逐点数据", ha="right", va="bottom", fontsize=6.3, color=COLORS["muted"])
    ax.set_axisbelow(True)
    return fig, ax


def fig_legend(fig: mpl.figure.Figure, ax: mpl.axes.Axes, *, ncol: int = 2, y: float = 0.832) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, y), ncol=ncol, frameon=False, handlelength=2.1, columnspacing=1.5)


def save_all(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        path = stem.with_suffix(suffix)
        fig.savefig(path, facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def add_grid(ax: mpl.axes.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.65, alpha=0.9)


def zscore(values: np.ndarray) -> np.ndarray:
    sd = float(values.std(ddof=1))
    if sd <= 0:
        raise ValueError("Cannot standardize a constant series")
    return (values - float(values.mean())) / sd


def figure_01(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) != 78:
        raise ValueError("F01 requires exactly 78 hardware snapshots")
    e0 = as_float(rows, "readout_all_zero_error")
    e1 = as_float(rows, "readout_all_one_error")
    z0, z1 = zscore(e0), zscore(e1)
    matrix = np.vstack([z0, z1])
    bound = float(np.max(np.abs(matrix)))
    fig, ax = new_chart(
        "01｜T287 读出状态快照热图",
        "78 个真机快照，0 exclusions；按快照顺序排列，颜色为各通道内 z-score，缺口不做时间插值",
        figsize=(7.2, 3.65),
        left=0.16,
        right=0.89,
        bottom=0.23,
        top=0.72,
    )
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax.set_yticks([0, 1], ["E0 all-zero", "E1 all-one"])
    ticks = np.unique(np.linspace(0, len(rows) - 1, 9).round().astype(int))
    ax.set_xticks(ticks, [str(int(i) + 1) for i in ticks])
    ax.set_xlabel("快照序号（UTC 与距首次时间见数据表）")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["grid"])
    cbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label("通道内 z-score", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.0)
    derived: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["heatmap_column"] = i + 1
        item["readout_all_zero_zscore"] = float(z0[i])
        item["readout_all_one_zscore"] = float(z1[i])
        derived.append(item)
    source = write_csv(HERE / "01_t287_readout_state_heatmap.source.csv", derived)
    files = save_all(fig, HERE / "01_t287_readout_state_heatmap")
    return meta(1, "temporal heatmap", files, [source], len(rows), "T287 hardware", "两个读出通道的标准化状态在 78 个快照间呈现明显静态偏离。", "展示时间变异，不单独证明环境原因或底层脉冲参数漂移。", "F01_ReadoutHeatmap")


def figure_02(rows: list[dict[str, str]]) -> dict[str, Any]:
    h1 = as_float(rows, "effective_h1")
    h2 = as_float(rows, "effective_h2")
    s1 = as_float(rows, "effective_h1_shot_sigma")
    s2 = as_float(rows, "effective_h2_shot_sigma")
    time_h = as_float(rows, "hours_from_first_snapshot")
    fig, ax = new_chart(
        "02｜T287 有效场相空间散点图",
        "每个点是一次三时点 Y/Z 量子测量反演；颜色表示距首次快照小时数，浅灰误差棒是 shot sigma",
        figsize=(7.2, 5.15),
        left=0.13,
        right=0.86,
        bottom=0.14,
        top=0.79,
    )
    ax.errorbar(h1, h2, xerr=s1, yerr=s2, fmt="none", ecolor=COLORS["grey"], elinewidth=0.55, alpha=0.18, zorder=1)
    points = ax.scatter(h1, h2, c=time_h, cmap="viridis", s=28, edgecolors="white", linewidths=0.45, alpha=0.92, zorder=3)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.axhline(0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("Effective H1（a.u.）")
    ax.set_ylabel("Effective H2（a.u.）")
    add_grid(ax, "both")
    cbar = fig.colorbar(points, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label("距首个快照时间（h）", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.0)
    derived: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["phase_space_point_order"] = i + 1
        item["combined_shot_sigma"] = float(math.hypot(s1[i], s2[i]))
        derived.append(item)
    source = write_csv(HERE / "02_t287_effective_field_phase_space.source.csv", derived)
    files = save_all(fig, HERE / "02_t287_effective_field_phase_space")
    return meta(2, "uncertainty-aware phase-space scatter", files, [source], len(rows), "T287 hardware", "量子测量定义的 H1/H2 状态在观测窗内占据非单点区域。", "H1/H2 是量子测量代理，不是温度计、电磁传感器或底层脉冲参数。", "F02_PhaseSpace")


def forest_structure(rows: list[dict[str, str]], number: int, stem: str, title: str, subtitle: str, color: str, claim: str, boundary: str, sheet: str) -> dict[str, Any]:
    lag = as_float(rows, "lag_mid_seconds")
    point = as_float(rows, "sf_debiased")
    lower = as_float(rows, "sf_ci_lower")
    upper = as_float(rows, "sf_ci_upper")
    floor = as_float(rows, "shot_floor")
    y = np.arange(len(rows))
    fig, ax = new_chart(title, subtitle, figsize=(7.2, 5.75), left=0.21, right=0.96, bottom=0.13, top=0.80)
    for yi in y[::2]:
        ax.axhspan(yi - 0.5, yi + 0.5, color=COLORS["grey_soft"], alpha=0.65, zorder=0)
    xerr = np.vstack([point - lower, upper - point])
    ax.errorbar(point, y, xerr=xerr, fmt="o", markersize=4.8, capsize=2.3, color=color, ecolor=color, elinewidth=1.05, label="去偏结构函数及 95% CI", zorder=3)
    ax.scatter(floor, y, marker="D", s=22, color=COLORS["muted"], label="Shot-noise floor", zorder=4)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.85)
    ax.set_xscale("symlog", linthresh=1e-5, linscale=0.8)
    ax.set_yticks(y, [f"{v:.0f} s" if v < 1000 else f"{v/3600:.2g} h" for v in lag])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlabel("去偏结构函数（symlog）")
    ax.set_ylabel("滞后时间箱中心")
    add_grid(ax, "x")
    fig_legend(fig, ax, ncol=2, y=0.848)
    source = write_csv(HERE / f"{stem}.source.csv", rows)
    files = save_all(fig, HERE / stem)
    return meta(number, "horizontal dot-whisker / forest plot", files, [source], len(rows), "T287 hardware", claim, boundary, sheet)


def figure_03(rows: list[dict[str, str]]) -> dict[str, Any]:
    p = float(rows[0]["variance_gate_p_value"])
    return forest_structure(rows, 3, "03_t287_e0_negative_control_forest", "03｜阴性对照 E0：各滞后箱点估计与置信区间", f"n=30，p={p:.4f}，Drift gate = FAIL；用横向森林图保留每个 lag bin 的不确定性", COLORS["teal"], "E0 阴性对照未通过漂移方差门，方法没有把所有波动都归为漂移。", "FAIL 是对照成功，不是证明 E0 在任意尺度绝对不漂移。", "F03_E0Forest")


def figure_04(rows: list[dict[str, str]]) -> dict[str, Any]:
    p = float(rows[0]["variance_gate_p_value"])
    return forest_structure(rows, 4, "04_t287_e1_drift_forest", "04｜E1 漂移通道：各滞后箱点估计与置信区间", f"n=30，p={p:.3e}，Drift gate = PASS；菱形是同箱 shot-noise floor，圆点是去偏估计", COLORS["blue"], "E1 的过程方差超出 shot-noise floor，识别到可定量的静态之外变化。", "证据限于 E1 读出代理通道，不直接定位温度或电磁扰动源。", "F04_E1Forest")


def geometric_edges(x: np.ndarray) -> np.ndarray:
    if np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Expected a positive, strictly increasing grid")
    mids = np.sqrt(x[:-1] * x[1:])
    first = x[0] ** 2 / mids[0]
    last = x[-1] ** 2 / mids[-1]
    return np.concatenate([[first], mids, [last]])


def figure_05(rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> dict[str, Any]:
    interval = as_float(rows, "interval_seconds")
    ou = as_float(rows, "ou_point_residual_variance")
    nonparam = as_float(rows, "nonparametric_residual_variance")
    baseline = as_float(rows, "no_sensing_process_variance")
    if np.any(ou <= 0) or np.any(nonparam <= 0) or np.any(baseline <= 0):
        raise ValueError("F05 residual variances must be positive")
    log_matrix = np.vstack([np.log10(ou), np.log10(nonparam), np.log10(baseline)])
    edges = geometric_edges(interval)
    y_edges = np.arange(4) - 0.5
    info = summary_rows[0]
    tstar = float(info["intrinsic_t_star_seconds"])
    fig, ax = new_chart(
        "05｜E1 更新间隔经济性残差热图",
        "400 个间隔网格；颜色为 log10(残差方差)；点估计有最优点，但 bootstrap 上界碰到观测窗，结论仍为 INCONCLUSIVE",
        figsize=(7.2, 4.25),
        left=0.22,
        right=0.87,
        bottom=0.21,
        top=0.73,
    )
    mesh = ax.pcolormesh(edges, y_edges, log_matrix, shading="flat", cmap="magma_r")
    ax.set_xscale("log")
    ax.set_yticks([0, 1, 2], ["OU point", "Nonparametric", "No sensing"])
    ax.set_ylim(2.5, -0.5)
    ax.axvline(60.0, color=COLORS["teal"], linewidth=1.35, label="Protocol floor = 60 s")
    ax.axvline(tstar, color=COLORS["blue"], linewidth=1.35, linestyle="--", label=f"点估计 T*={tstar:.1f} s")
    ax.scatter([interval[int(np.argmin(ou))], interval[int(np.argmin(nonparam))]], [0, 1], marker="*", s=70, color=COLORS["white"], edgecolor=COLORS["ink"], linewidth=0.55, zorder=5, label="行内最小值")
    ax.set_xlabel("校准更新间隔（s，log）")
    ax.tick_params(axis="y", length=0)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label("log10(残差方差)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.0)
    fig_legend(fig, ax, ncol=3, y=0.812)
    derived: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["grid_index"] = i + 1
        item["log10_ou_point_residual_variance"] = float(np.log10(ou[i]))
        item["log10_nonparametric_residual_variance"] = float(np.log10(nonparam[i]))
        item["log10_no_sensing_process_variance"] = float(np.log10(baseline[i]))
        item["is_ou_grid_minimum"] = i == int(np.argmin(ou))
        item["is_nonparametric_grid_minimum"] = i == int(np.argmin(nonparam))
        derived.append(item)
    source = write_csv(HERE / "05_t287_e1_interval_residual_heatmap.source.csv", derived)
    summary = write_csv(HERE / "05_t287_e1_interval_residual_heatmap.summary.csv", summary_rows)
    files = save_all(fig, HERE / "05_t287_e1_interval_residual_heatmap")
    return meta(5, "three-row residual heatmap", files, [source, summary], len(rows), "T287 hardware-derived model", "点曲线存在最优更新间隔，但 T* 的置信上界碰到观测窗。", "T*=134.4 s 不能当作已精确识别的生产策略；当前裁决为 INCONCLUSIVE。", "F05_ResidualMap")


def figure_06(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = {
        "interface_P50": "API P50",
        "interface_P90": "API P90",
        "protocol_reachable": "协议可达底线",
        "intrinsic_T_star": "内秉 T* 点估计",
    }
    y = np.arange(len(rows))
    point = as_float(rows, "point_seconds")
    fig, ax = new_chart(
        "06｜控制底线与内秉 T* 的时标尺",
        "云 API 用户位 P50=12.44 s、P90=14.70 s；协议可达底线 60 s；T* 95% CI=[101.34, 4000.21] s",
        figsize=(7.2, 4.55),
        left=0.24,
        right=0.96,
        bottom=0.18,
        top=0.76,
    )
    ax.axvspan(0.1, 60.0, color=COLORS["grey_soft"], alpha=0.82, label="协议不可达或接口开销区")
    for i, row in enumerate(rows):
        color = COLORS["teal"] if row["origin"] == "reachable_control_floor" else COLORS["blue"]
        lower = row.get("ci_lower_seconds", "")
        upper = row.get("ci_upper_seconds", "")
        if lower and upper:
            lo, hi = float(lower), float(upper)
            ax.hlines(i, lo, hi, color=color, linewidth=5.5, alpha=0.32, zorder=2)
            ax.vlines([lo, hi], i - 0.10, i + 0.10, color=color, linewidth=1.0, zorder=3)
        ax.scatter(point[i], i, s=48, color=color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.axvline(60.0, color=COLORS["teal"], linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y, [labels[row["quantity"]] for row in rows])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlabel("时间（s，log）")
    add_grid(ax, "x")
    fig_legend(fig, ax, ncol=1, y=0.82)
    source = write_csv(HERE / "06_t287_control_floor_interval_strip.source.csv", rows)
    files = save_all(fig, HERE / "06_t287_control_floor_interval_strip")
    return meta(6, "log-scale interval strip / calibration clock", files, [source], len(rows), "T287 hardware economics", "云 API 接口延迟和协议可达底线均已量化。", "T* 置信区间过宽且上界碰窗，不宣称已找到稳定的最优生产间隔。", "F06_ControlClock")


def figure_07(rows: list[dict[str, str]]) -> dict[str, Any]:
    fast = as_float(rows, "fast_endpoint_squared_residual")
    slow = as_float(rows, "slow_endpoint_squared_residual")
    if np.any(fast <= 0) or np.any(slow <= 0):
        raise ValueError("F07 log axes require positive residuals")
    better = np.asarray([bool_value(row["fast_better"]) for row in rows])
    fig, ax = new_chart(
        "07｜T176 Session 0 快/慢 cadence 配对散点图",
        "20 个 pair-complete 真机对，0 exclusions；对角线下方表示 fast 端点平方残差更小（14/20）",
        figsize=(7.2, 5.25),
        left=0.15,
        right=0.96,
        bottom=0.14,
        top=0.79,
    )
    lo = float(min(fast.min(), slow.min()) * 0.65)
    hi = float(max(fast.max(), slow.max()) * 1.45)
    ax.plot([lo, hi], [lo, hi], color=COLORS["muted"], linewidth=1.0, linestyle="--", label="Fast = slow")
    ax.scatter(slow[better], fast[better], s=42, color=COLORS["green"], edgecolor="white", linewidth=0.6, label="Fast 更优", zorder=3)
    ax.scatter(slow[~better], fast[~better], s=42, color=COLORS["red"], edgecolor="white", linewidth=0.6, label="Slow 更优", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("slow cadence 端点平方残差")
    ax.set_ylabel("fast cadence 端点平方残差")
    add_grid(ax, "both")
    fig_legend(fig, ax, ncol=3, y=0.835)
    derived: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["scatter_region"] = "below_identity_fast_better" if bool_value(row["fast_better"]) else "above_identity_slow_better"
        derived.append(item)
    source = write_csv(HERE / "07_t176_hardware_paired_scatter.source.csv", derived)
    files = save_all(fig, HERE / "07_t176_hardware_paired_scatter")
    return meta(7, "paired log-log scatter", files, [source], len(rows), "T176 hardware Session 0 only", "Session 0 中 14/20 个配对的 fast cadence 端点残差更小。", "Hardware Session 1 缺失；不宣称已完成注册双会话 all-hardware PASS。", "F07_PairedScatter")


def figure_08(rows: list[dict[str, str]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["log2_slow_over_fast"]))
    effect = as_float(ordered, "log2_slow_over_fast")
    y = np.arange(len(ordered))
    positive = effect > 0
    fig, ax = new_chart(
        "08｜T176 配对效应的发散棒棒糖图",
        "log2(slow/fast)>0 代表 fast cadence 更优；排序只用于展示异质性，不改变原 pair_index",
        figsize=(7.2, 6.15),
        left=0.18,
        right=0.96,
        bottom=0.12,
        top=0.81,
    )
    for yi, value, pos in zip(y, effect, positive):
        color = COLORS["green"] if pos else COLORS["red"]
        ax.hlines(yi, 0, value, color=color, linewidth=1.25, alpha=0.75)
        ax.scatter(value, yi, s=36, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.9)
    ax.set_yticks(y, [f"Pair {int(row['pair_index']):02d}" for row in ordered])
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_xlabel("log2(slow 残差 / fast 残差)　← slow 更优｜fast 更优 →")
    ax.set_ylabel("按效应量排序的配对")
    add_grid(ax, "x")
    derived: list[dict[str, Any]] = []
    for order, row in enumerate(ordered, start=1):
        item = dict(row)
        item["plot_order"] = order
        item["benefit_direction"] = "fast_better" if float(row["log2_slow_over_fast"]) > 0 else "slow_better"
        derived.append(item)
    source = write_csv(HERE / "08_t176_pair_benefit_lollipop.source.csv", derived)
    files = save_all(fig, HERE / "08_t176_pair_benefit_lollipop")
    return meta(8, "sorted diverging lollipop", files, [source], len(rows), "T176 hardware Session 0", "配对效应具有明显异质性，但方向上 14/20 倾向 fast cadence。", "排序图不是时序图，不用于推断效应随时间增长。", "F08_BenefitRank")


def figure_09(rows: list[dict[str, str]], null_summary: list[dict[str, str]]) -> dict[str, Any]:
    pair = as_float(rows, "pair_index")
    ratio = as_float(rows, "cumulative_fast_over_slow_ratio")
    critical = float(null_summary[0]["critical_ratio_5pct"])
    observed = float(null_summary[0]["observed_ratio"])
    fig, ax = new_chart(
        "09｜T176 Session 0 累计快/慢残差比",
        "这是唯一保留的序列折线：它诊断累计比是否被单一配对驱动；终点 ratio=0.3616",
        figsize=(7.2, 4.75),
        left=0.13,
        right=0.96,
        bottom=0.16,
        top=0.78,
    )
    ax.axhspan(0, critical, color=COLORS["green_soft"], alpha=0.72, label=f"5% permutation 临界区（≤{critical:.3f}）")
    ax.axhline(1.0, color=COLORS["muted"], linestyle="--", linewidth=1.0, label="No-effect ratio = 1")
    ax.step(pair, ratio, where="mid", color=COLORS["blue"], linewidth=1.8, label="累计 fast/slow ratio")
    ax.scatter(pair, ratio, s=22, color=COLORS["blue"], edgecolor="white", linewidth=0.45, zorder=3)
    ax.scatter([pair[-1]], [observed], s=58, marker="D", color=COLORS["amber"], edgecolor="white", linewidth=0.6, zorder=4, label="最终观测比")
    ax.set_xlim(0.4, len(rows) + 0.6)
    ax.set_ylim(0, max(1.08, float(ratio.max()) * 1.12))
    ax.set_xticks(np.arange(1, len(rows) + 1, 2))
    ax.set_xlabel("按执行顺序累积的 complete pair 数")
    ax.set_ylabel("累计 fast / slow 平方残差比")
    add_grid(ax, "y")
    fig_legend(fig, ax, ncol=2, y=0.84)
    source = write_csv(HERE / "09_t176_cumulative_ratio_step.source.csv", rows)
    files = save_all(fig, HERE / "09_t176_cumulative_ratio_step")
    return meta(9, "cumulative step diagnostic", files, [source], len(rows), "T176 hardware Session 0", "累计比在加入 20 对后结束于 0.3616，并非仅最后单对造成。", "序列诊断仍属 Session 0 描述性证据，不补足缺失的 Hardware Session 1。", "F09_Cumulative")


def figure_10(rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> dict[str, Any]:
    null = as_float(rows, "null_ratio")
    info = summary_rows[0]
    observed = float(info["observed_ratio"])
    critical = float(info["critical_ratio_5pct"])
    p_value = float(info["permutation_p_value"])
    fig, ax = new_chart(
        "10｜T176 Session 0 配对置换零分布",
        f"20,000 次固定种子置换（seed 20260815）；观测 ratio={observed:.4f}，5% 临界值={critical:.4f}，p={p_value:.4f}",
        figsize=(7.2, 4.8),
        left=0.13,
        right=0.96,
        bottom=0.16,
        top=0.78,
    )
    ax.hist(null, bins=58, color=COLORS["blue_soft"], edgecolor=COLORS["blue"], linewidth=0.45, alpha=0.95, label="置换零分布")
    ax.axvline(critical, color=COLORS["amber"], linewidth=1.45, linestyle="--", label="5% 临界值")
    ax.axvline(observed, color=COLORS["red"], linewidth=1.8, label="观测比")
    ax.set_xlabel("置换后 fast / slow 平方残差比")
    ax.set_ylabel("置换次数")
    add_grid(ax, "y")
    fig_legend(fig, ax, ncol=3, y=0.84)
    source = write_csv(HERE / "10_t176_permutation_histogram.source.csv", rows)
    summary = write_csv(HERE / "10_t176_permutation_histogram.summary.csv", summary_rows)
    files = save_all(fig, HERE / "10_t176_permutation_histogram")
    return meta(10, "permutation histogram", files, [source, summary], len(rows), "T176 hardware Session 0", "在 Session 0 配对置换零假设下，观测比位于低尾，p=0.00525。", "p 值只针对已冻结的 Session 0 配对置换检验；不是跨会话或跨设备复现证据。", "F10_Permutation")


def figure_11(rows: list[dict[str, str]]) -> dict[str, Any]:
    ratio = as_float(rows, "ratio")
    critical = as_float(rows, "critical_ratio")
    y = np.arange(len(rows))
    label_map = {
        "primary_hybrid": "Primary hybrid",
        "early_saturating_transient": "Early saturation",
        "linear_ramp": "Linear ramp",
        "step_at_block_boundary": "Block step",
    }
    fig, ax = new_chart(
        "11｜基线漂移形状敏感性哑铃图",
        "每行左端为观测 ratio，右端为该情景临界 ratio；四种漂移形状均 ratio < critical",
        figsize=(7.2, 4.6),
        left=0.22,
        right=0.92,
        bottom=0.17,
        top=0.76,
    )
    for i, row in enumerate(rows):
        ax.hlines(i, ratio[i], critical[i], color=COLORS["grid"], linewidth=4.5, zorder=1)
        ax.scatter(ratio[i], i, s=48, color=COLORS["green"], edgecolor="white", linewidth=0.6, zorder=3, label="观测 ratio" if i == 0 else None)
        ax.scatter(critical[i], i, s=48, facecolor="white", edgecolor=COLORS["ink"], linewidth=1.1, zorder=3, label="临界 ratio" if i == 0 else None)
        ax.text(critical[i] + 0.018, i, f"p={float(row['p_value']):.4g}", ha="left", va="center", fontsize=7.0, color=COLORS["muted"])
    ax.set_yticks(y, [label_map[row["analysis"]] for row in rows])
    ax.set_ylim(len(rows) - 0.55, -0.55)
    ax.set_xlim(0, max(critical) + 0.17)
    ax.set_xlabel("fast / slow 残差比（越小越支持 fast cadence）")
    ax.grid(False)
    fig_legend(fig, ax, ncol=2, y=0.83)
    derived: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["margin_to_critical"] = float(row["critical_ratio"]) - float(row["ratio"])
        item["ratio_over_critical"] = float(row["ratio"]) / float(row["critical_ratio"])
        derived.append(item)
    source = write_csv(HERE / "11_baseline_sensitivity_dumbbell.source.csv", derived)
    files = save_all(fig, HERE / "11_baseline_sensitivity_dumbbell")
    return meta(11, "scenario dumbbell", files, [source], len(rows), "simulation-assisted / post-hoc sensitivity", "四种预设漂移形状下观测比均低于相应临界值。", "这是模拟辅助、事后敏感性证据，不是缺失的真机 Hardware Session 1。", "F11_Sensitivity")


def figure_12(rows: list[dict[str, str]]) -> dict[str, Any]:
    lookup = {row["stage"]: row for row in rows}
    stages = ["submitted_jobs", "unique_query_ids", "completed_cycles", "complete_pairs"]
    labels = {
        "submitted_jobs": "提交\n44 jobs",
        "unique_query_ids": "平台返回\n88 tasks",
        "completed_cycles": "完成\n40 cycles",
        "complete_pairs": "冻结分析\n20 pairs",
    }
    fig, ax = new_chart(
        "12｜T176 真机负载与 pair-complete 数据流",
        "计数单位不同，因此用流程图而非伪比例 Sankey；44 jobs → 88 tasks → 40 cycles → 20 complete pairs，0 excluded pairs",
        figsize=(7.2, 4.35),
        left=0.05,
        right=0.98,
        bottom=0.08,
        top=0.76,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = np.linspace(0.13, 0.82, 4)
    box_w, box_h = 0.17, 0.23
    colors = [COLORS["blue_soft"], COLORS["teal_soft"], "#F3EAD6", COLORS["green_soft"]]
    for idx, (stage, x) in enumerate(zip(stages, centers)):
        rect = FancyBboxPatch((x - box_w / 2, 0.52 - box_h / 2), box_w, box_h, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=colors[idx], edgecolor=COLORS["ink"], linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x, 0.52, labels[stage], ha="center", va="center", fontsize=9.0, fontweight="bold", color=COLORS["ink"])
        if idx < 3:
            arrow = FancyArrowPatch((x + box_w / 2 + 0.012, 0.52), (centers[idx + 1] - box_w / 2 - 0.012, 0.52), arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=COLORS["muted"])
            ax.add_patch(arrow)
    excluded = int(float(lookup["excluded_pairs"]["count"]))
    small = FancyBboxPatch((0.865, 0.17), 0.115, 0.16, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=COLORS["grey_soft"], edgecolor=COLORS["muted"], linewidth=0.8)
    ax.add_patch(small)
    ax.text(0.9225, 0.25, f"排除\n{excluded} pairs", ha="center", va="center", fontsize=8.2, color=COLORS["ink"])
    branch = FancyArrowPatch((centers[-1] + 0.02, 0.395), (0.885, 0.33), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=COLORS["muted"], connectionstyle="arc3,rad=0.18")
    ax.add_patch(branch)
    ax.text(0.5, 0.10, "数据完整性：20/20 分析对保留；但仅有 Hardware Session 0", ha="center", va="center", fontsize=8.0, color=COLORS["muted"])
    source = write_csv(HERE / "12_t176_hardware_workload_flow.source.csv", rows)
    files = save_all(fig, HERE / "12_t176_hardware_workload_flow")
    return meta(12, "unit-aware process flow", files, [source], len(rows), "T176 hardware Session 0 engineering accounting", "真机工作负载、平台任务、完成周期和分析对数已逐层对账，0 pair exclusions。", "工作负载计数证明工程完整性，不等于算法效果或跨设备复现。", "F12_WorkloadFlow")


def meta(number: int, chart_type: str, files: Sequence[Path], sources: Sequence[Path], rows: int, origin: str, conclusion: str, boundary: str, workbook_sheet: str) -> dict[str, Any]:
    return {
        "number": number,
        "chart_type": chart_type,
        "files": [{"path": str(path), "sha256": file_hash(path)} for path in files],
        "source_data": [{"path": str(path), "sha256": file_hash(path), "rows": sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1} for path in sources],
        "primary_rows": rows,
        "evidence_origin": origin,
        "conclusion": conclusion,
        "claim_boundary": boundary,
        "workbook_sheet": workbook_sheet,
    }


def write_readme(figures: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# B4 v2 独立主图索引（多图形语言版）",
        "",
        "本目录不覆盖 v1。12 张图一图一问，仅 F09 因累计顺序诊断保留折线/阶梯线；其余使用热图、相空间散点、森林图、区间条、配对散点、发散棒棒糖、置换直方图、哑铃图和流程图。",
        "",
        "| 图 | 图形类型 | 数据表 | 证据起源 |",
        "|---:|---|---|---|",
    ]
    for item in figures:
        source_name = Path(item["source_data"][0]["path"]).name
        lines.append(f"| {int(item['number']):02d} | {item['chart_type']} | `{source_name}` / `{item['workbook_sheet']}` | {item['evidence_origin']} |")
    lines.extend(
        [
            "",
            "## 严格边界",
            "",
            "- T287 和 T176 证据不混为“双真机闭环复现”。",
            "- T176 仅 Hardware Session 0；Hardware Session 1 缺失，不宣称注册 all-hardware PASS。",
            "- F05/F06 的 T* 上界碰到观测窗，因此为 INCONCLUSIVE。",
            "- F11 是模拟辅助/事后敏感性，不是第二次真机会话。",
            "- 所有源 CSV 包含图中使用的派生编码列；完整数据还将汇总到单一 Excel 数据簿。",
            "",
        ]
    )
    (HERE / "README_v2_逐图数据索引.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    required = {
        "f01": V1 / "01_t287_readout_proxy_timeseries.source.csv",
        "f02": V1 / "02_t287_effective_field_state.source.csv",
        "f03": V1 / "03_t287_e0_negative_control_structure_function.source.csv",
        "f04": V1 / "04_t287_e1_drift_structure_function.source.csv",
        "f05": V1 / "05_t287_e1_interval_economics.source.csv",
        "f05s": V1 / "05_t287_e1_interval_economics.summary.csv",
        "f06": V1 / "06_t287_tstar_vs_control_floor.source.csv",
        "f07": V1 / "07_t176_hardware_pair_slopegraph.source.csv",
        "f08": V1 / "08_t176_pair_benefit_distribution.source.csv",
        "f09": V1 / "09_t176_cumulative_ratio.source.csv",
        "f10": V1 / "10_t176_hardware_permutation_null.source.csv",
        "f10s": V1 / "10_t176_hardware_permutation_null.summary.csv",
        "f11": V1 / "11_baseline_drift_sensitivity.source.csv",
        "f12": V1 / "12_t176_hardware_workload_flow.source.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen source files:\n" + "\n".join(missing))
    data = {key: read_csv(path) for key, path in required.items()}
    figures = [
        figure_01(data["f01"]),
        figure_02(data["f02"]),
        figure_03(data["f03"]),
        figure_04(data["f04"]),
        figure_05(data["f05"], data["f05s"]),
        figure_06(data["f06"]),
        figure_07(data["f07"]),
        figure_08(data["f08"]),
        figure_09(data["f09"], data["f10s"]),
        figure_10(data["f10"], data["f10s"]),
        figure_11(data["f11"]),
        figure_12(data["f12"]),
    ]
    manifest = {
        "gallery": "B4_v2_diverse_single_figures",
        "figure_count": len(figures),
        "one_chart_per_canvas": True,
        "only_sequence_line_figure": 9,
        "inputs": {key: {"path": str(path), "sha256": file_hash(path)} for key, path in required.items()},
        "figures": figures,
    }
    (HERE / "single_figure_v2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(figures)
    print(json.dumps({"output": str(HERE), "figures": len(figures), "formats_per_figure": 4, "source_tables": sum(len(x["source_data"]) for x in figures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
