#!/usr/bin/env python3
"""Build the B4 main-report gallery as twelve independent, single-chart figures.

Each canvas answers one question.  Hardware, simulation, sensitivity, and
engineering-accounting evidence are not mixed within a canvas.  Every figure
is exported in vector and 600-dpi raster formats with a same-number source CSV.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


COLORS = {
    "ink": "#252525",
    "muted": "#6D7278",
    "grid": "#D6DBE0",
    "blue": "#0F4D92",
    "blue_soft": "#DCE8F5",
    "teal": "#248A83",
    "teal_soft": "#DDF1EE",
    "violet": "#6F63A8",
    "violet_soft": "#E9E5F5",
    "amber": "#C58B19",
    "amber_soft": "#F6EBCF",
    "red": "#B64342",
    "green": "#2E8B57",
    "green_soft": "#DDF0E5",
    "grey_soft": "#EEF0F2",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> Path:
    if not rows:
        raise ValueError(f"Refusing to write an empty source-data file: {path}")
    if fields is None:
        fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_float(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"Non-finite values in {field}")
    return values


def require_positive(*arrays: np.ndarray) -> None:
    for values in arrays:
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Logarithmic chart received a non-positive or non-finite value")


def new_chart(
    title: str,
    subtitle: str,
    *,
    left: float = 0.12,
    right: float = 0.97,
    bottom: float = 0.16,
    top: float = 0.78,
    height: float = 4.35,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=(7.2, height), dpi=600)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    fig.text(0.055, 0.965, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=COLORS["ink"])
    fig.text(0.055, 0.895, subtitle, ha="left", va="top", fontsize=8.0, color=COLORS["muted"])
    fig.text(0.97, 0.025, "冻结结构化产物；同名 CSV 为逐点源数据", ha="right", va="bottom", fontsize=6.5, color=COLORS["muted"])
    ax.set_axisbelow(True)
    return fig, ax


def figure_legend(fig: mpl.figure.Figure, ax: mpl.axes.Axes, *, ncol: int, anchor_y: float = 0.835) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, anchor_y),
        ncol=ncol,
        handlelength=2.1,
        columnspacing=1.6,
        frameon=False,
    )


def save_figure(fig: mpl.figure.Figure, out_base: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix, options in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        path = out_base.with_suffix(suffix)
        fig.savefig(path, facecolor="white", **options)
        outputs.append(path)
    plt.close(fig)
    return outputs


def source_subset(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> list[dict[str, str]]:
    return [{field: row[field] for field in fields} for row in rows]


def build_01(observable: list[dict[str, str]], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    if len(observable) != 78:
        raise ValueError("Figure 01 requires all 78 T287 snapshots")
    x = to_float(observable, "hours_from_first_snapshot")
    e0 = to_float(observable, "readout_all_zero_error")
    e1 = to_float(observable, "readout_all_one_error")
    s0 = to_float(observable, "readout_all_zero_shot_noise_floor")
    s1 = to_float(observable, "readout_all_one_shot_noise_floor")
    fig, ax = new_chart(
        "01｜T287 量子探针直接看到读出状态随时间变化",
        "78 个真机快照，0 exclusions；阴影保留 shot-noise floor，连线只连接已观测点",
    )
    ax.plot(x, e0, marker="o", markersize=2.8, color=COLORS["teal"], label="E0 all-zero error")
    ax.fill_between(x, e0 - s0, e0 + s0, color=COLORS["teal_soft"], alpha=0.9, linewidth=0)
    ax.plot(x, e1, marker="o", markersize=2.8, color=COLORS["blue"], label="E1 all-one error")
    ax.fill_between(x, e1 - s1, e1 + s1, color=COLORS["blue_soft"], alpha=0.9, linewidth=0)
    ax.set_xlabel("距首个快照时间（h）")
    ax.set_ylabel("测得读出错误概率")
    figure_legend(fig, ax, ncol=2)
    fields = [
        "snapshot_index",
        "scheduled_utc",
        "hours_from_first_snapshot",
        "readout_all_zero_error",
        "readout_all_zero_shot_noise_floor",
        "readout_all_one_error",
        "readout_all_one_shot_noise_floor",
    ]
    source = write_csv(out_dir / "01_t287_readout_proxy_timeseries.source.csv", source_subset(observable, fields))
    base = out_dir / "01_t287_readout_proxy_timeseries"
    return save_figure(fig, base), [source], {"claim": "T287 readout-state proxies vary over time beyond a static single-value description.", "origin": "T287 hardware", "n": 78, "exclusions": 0}


def build_02(observable: list[dict[str, str]], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    x = to_float(observable, "hours_from_first_snapshot")
    h1 = to_float(observable, "effective_h1")
    h2 = to_float(observable, "effective_h2")
    s1 = to_float(observable, "effective_h1_shot_sigma")
    s2 = to_float(observable, "effective_h2_shot_sigma")
    fig, ax = new_chart(
        "02｜T287 三时点 Y/Z 反演形成可观测有效场状态",
        "H1/H2 是量子测量定义的环境代理，不是温度计或电磁传感器，也不等同于底层脉冲参数",
    )
    ax.errorbar(x, h1, yerr=s1, fmt="o-", markersize=2.6, capsize=0, color=COLORS["violet"], alpha=0.90, label="Effective H1")
    ax.errorbar(x, h2, yerr=s2, fmt="o-", markersize=2.6, capsize=0, color=COLORS["amber"], alpha=0.88, label="Effective H2")
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("距首个快照时间（h）")
    ax.set_ylabel("测量定义的有效场（a.u.）")
    figure_legend(fig, ax, ncol=2)
    fields = ["snapshot_index", "scheduled_utc", "hours_from_first_snapshot", "effective_h1", "effective_h1_shot_sigma", "effective_h2", "effective_h2_shot_sigma"]
    source = write_csv(out_dir / "02_t287_effective_field_state.source.csv", source_subset(observable, fields))
    base = out_dir / "02_t287_effective_field_state"
    return save_figure(fig, base), [source], {"claim": "Three-time-point quantum measurements yield a time-resolved effective-field state.", "origin": "T287 hardware", "n": 78}


def plot_structure_function(
    rows: list[dict[str, str]],
    *,
    title: str,
    subtitle: str,
    color: str,
    out_base: Path,
    source_path: Path,
    gate: Mapping[str, Any],
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    x = to_float(rows, "lag_mid_seconds")
    y = to_float(rows, "sf_debiased")
    lower = to_float(rows, "sf_ci_lower")
    upper = to_float(rows, "sf_ci_upper")
    floor = to_float(rows, "shot_floor")
    require_positive(x, floor)
    fig, ax = new_chart(title, subtitle)
    ax.errorbar(x, y, yerr=np.vstack([y - lower, upper - y]), fmt="o-", markersize=3.5, capsize=2.5, color=color, label="Debiased structure function")
    ax.plot(x, floor, marker=".", linestyle="--", color=COLORS["muted"], label="Shot-noise floor")
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
    ax.set_xlabel("滞后时间（s）")
    ax.set_ylabel("去偏结构函数")
    figure_legend(fig, ax, ncol=2)
    source_rows = []
    for row in rows:
        item = dict(row)
        item.update({"observation_count": gate["observation_count"], "variance_gate_p_value": gate["p_value"], "variance_gate_passed": gate["passed"]})
        source_rows.append(item)
    source = write_csv(source_path, source_rows)
    return save_figure(fig, out_base), [source], {"origin": "T287 hardware", "n": int(gate["observation_count"]), "p_value": float(gate["p_value"]), "gate_passed": bool(gate["passed"])}


def build_03(sf_rows: list[dict[str, str]], sf_report: dict[str, Any], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    rows = [row for row in sf_rows if row["channel"] == "e0_readout_all_zero"]
    channel = sf_report["channels"]["e0_readout_all_zero"]
    gate = {"observation_count": channel["observation_count"], **channel["variance_gate"]}
    return plot_structure_function(
        rows,
        title="03｜阴性对照 E0：漂移门没有被误触发",
        subtitle=f"n={int(gate['observation_count'])}；p={float(gate['p_value']):.4f}；Drift gate = FAIL——说明方法没有把所有波动都叫作漂移",
        color=COLORS["teal"],
        out_base=out_dir / "03_t287_e0_negative_control_structure_function",
        source_path=out_dir / "03_t287_e0_negative_control_structure_function.source.csv",
        gate=gate,
    )


def build_04(sf_rows: list[dict[str, str]], sf_report: dict[str, Any], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    rows = [row for row in sf_rows if row["channel"] == "e1_readout_all_one"]
    channel = sf_report["channels"]["e1_readout_all_one"]
    gate = {"observation_count": channel["observation_count"], **channel["variance_gate"]}
    outputs, sources, meta = plot_structure_function(
        rows,
        title="04｜信号通道 E1：检测到超过 shot-noise 的过程方差",
        subtitle=f"n={int(gate['observation_count'])}；p={float(gate['p_value']):.2e}；Drift gate = PASS——这是更新间隔经济学的进入条件",
        color=COLORS["blue"],
        out_base=out_dir / "04_t287_e1_drift_structure_function",
        source_path=out_dir / "04_t287_e1_drift_structure_function.source.csv",
        gate=gate,
    )
    meta["claim"] = "E1 shows process variance beyond the shot-noise floor."
    return outputs, sources, meta


def build_05(curve_rows: list[dict[str, str]], sensing_report: dict[str, Any], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    rows = [row for row in curve_rows if row["channel"] == "e1_readout_all_one"]
    if len(rows) != 400:
        raise ValueError("Figure 05 requires all 400 E1 interval-grid rows")
    interval = to_float(rows, "interval_seconds")
    point = to_float(rows, "ou_point_residual_variance")
    sensitivity = to_float(rows, "nonparametric_residual_variance")
    no_sensing = to_float(rows, "no_sensing_process_variance")
    require_positive(interval, point, sensitivity, no_sensing)
    channel = sensing_report["channels"]["e1_readout_all_one"]
    t_star = float(channel["map"][0]["intrinsic_t_star_seconds"])
    t_low, t_high = map(float, channel["bootstrap"]["t_star_seconds_interval"])
    mask = interval >= 8.0
    fig, ax = new_chart(
        "05｜E1 更新间隔存在点估计最优，但不确定性尚未收敛",
        f"OU 点估计 T*={t_star:.0f} s；bootstrap 95%={t_low:.0f}–{t_high:.0f} s，上界撞 4000 s 观测窗",
    )
    ax.plot(interval[mask], point[mask], color=COLORS["blue"], label="OU point curve")
    ax.plot(interval[mask], sensitivity[mask], color=COLORS["violet"], linestyle="-.", label="Nonparametric sensitivity")
    ax.plot(interval[mask], no_sensing[mask], color=COLORS["ink"], linestyle="--", label="No-sensing variance")
    ax.axvspan(8, 60, color=COLORS["grey_soft"], alpha=0.95)
    ax.axvline(60, color=COLORS["muted"], linestyle=":", linewidth=1.1)
    nearest = int(np.argmin(np.abs(interval - t_star)))
    ax.scatter([t_star], [point[nearest]], s=44, color=COLORS["amber"], edgecolor="white", linewidth=0.8, zorder=5, label="Point T*")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8, 5000)
    ax.set_ylim(1e-7, 0.35)
    ax.set_xlabel("校准更新间隔（s）")
    ax.set_ylabel("预测残差方差")
    figure_legend(fig, ax, ncol=4)
    source = write_csv(out_dir / "05_t287_e1_interval_economics.source.csv", rows)
    summary = write_csv(
        out_dir / "05_t287_e1_interval_economics.summary.csv",
        [{"intrinsic_t_star_seconds": t_star, "bootstrap_ci95_lower_seconds": t_low, "bootstrap_ci95_upper_seconds": t_high, "verdict": "INCONCLUSIVE", "upper_bound_hits_observation_window": True}],
    )
    base = out_dir / "05_t287_e1_interval_economics"
    return save_figure(fig, base), [source, summary], {"claim": "The point curve has an optimum, but T* is not identified.", "origin": "T287 hardware-derived model", "grid_rows": 400, "verdict": "INCONCLUSIVE"}


def build_06(sensing_report: dict[str, Any], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    channel = sensing_report["channels"]["e1_readout_all_one"]
    t_star = float(channel["map"][0]["intrinsic_t_star_seconds"])
    t_low, t_high = map(float, channel["bootstrap"]["t_star_seconds_interval"])
    floors = sensing_report["hardware_economics"]["floor_definitions"]
    rows: list[dict[str, Any]] = []
    for floor in floors:
        rows.append({"quantity": floor["label"], "point_seconds": float(floor["seconds"]), "ci_lower_seconds": "", "ci_upper_seconds": "", "primary": floor["primary"], "origin": "reachable_control_floor"})
    rows.append({"quantity": "intrinsic_T_star", "point_seconds": t_star, "ci_lower_seconds": t_low, "ci_upper_seconds": t_high, "primary": False, "origin": "E1_bootstrap"})
    fig, ax = new_chart(
        "06｜可达控制 floor 与 T* 置信区间：目前只能判 INCONCLUSIVE",
        f"API P50={float(floors[0]['seconds']):.1f} s；P90={float(floors[1]['seconds']):.1f} s；协议 floor={float(floors[2]['seconds']):.0f} s；T* 95%={t_low:.0f}–{t_high:.0f} s",
        left=0.20,
    )
    labels = ["API P50", "API P90", "Protocol floor", "Intrinsic T* (95% CI)"]
    y = np.arange(4)
    floor_values = [float(row["seconds"]) for row in floors]
    for index, value in enumerate(floor_values):
        color = COLORS["blue"] if floors[index]["primary"] else COLORS["muted"]
        ax.scatter([value], [index], s=52, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(value * 1.18, index, f"{value:.1f} s", va="center", fontsize=7.8, color=color)
    ax.hlines(3, t_low, t_high, color=COLORS["amber"], linewidth=7.0, alpha=0.58)
    ax.scatter([t_star], [3], s=58, color=COLORS["amber"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(8, 6000)
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks(y, labels)
    ax.set_xlabel("秒（对数轴）")
    source = write_csv(out_dir / "06_t287_tstar_vs_control_floor.source.csv", rows)
    base = out_dir / "06_t287_tstar_vs_control_floor"
    return save_figure(fig, base), [source], {"claim": "Reachable control floors are measurable, but the intrinsic optimum remains weakly identified.", "origin": "T287 hardware economics", "verdict": "INCONCLUSIVE"}


def build_07(pairs: list[dict[str, str]], summary: dict[str, str], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    if len(pairs) != 20:
        raise ValueError("Figure 07 requires all 20 T176 hardware pairs")
    fast = to_float(pairs, "fast_endpoint_squared_residual")
    slow = to_float(pairs, "slow_endpoint_squared_residual")
    require_positive(fast, slow)
    fig, ax = new_chart(
        "07｜T176 真机 Session 0：20 对任务逐对比较",
        f"14/20 pairs 中 fast 更低；均值比={float(summary['fast_over_slow_ratio']):.5f}，即描述性降低 {100*float(summary['relative_reduction']):.1f}%",
    )
    for index in range(20):
        color = COLORS["green"] if fast[index] < slow[index] else COLORS["red"]
        ax.plot([0, 1], [slow[index], fast[index]], color=color, alpha=0.56, linewidth=1.2)
        ax.scatter([0, 1], [slow[index], fast[index]], color=color, s=20, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks([0, 1], ["Slow cadence", "Fast cadence"])
    ax.set_ylabel("端点平方残差（对数轴）")
    source = write_csv(out_dir / "07_t176_hardware_pair_slopegraph.source.csv", pairs)
    base = out_dir / "07_t176_hardware_pair_slopegraph"
    return save_figure(fig, base), [source], {"claim": "T176 hardware Session 0 shows a large pair-complete descriptive cadence benefit.", "origin": "T176 hardware Session 0 only", "n": 20, "exclusions": 0}


def build_08(pairs: list[dict[str, str]], summary: dict[str, str], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    values = to_float(pairs, "log2_slow_over_fast")
    colors = [COLORS["green"] if value > 0 else COLORS["red"] for value in values]
    fig, ax = new_chart(
        "08｜T176 真机 pair 改善幅度有异质性，但多数方向一致",
        f"正值表示 fast 优于 slow；14/20 为正；最终均值比={float(summary['fast_over_slow_ratio']):.5f}",
    )
    ax.bar(np.arange(1, 21), values, color=colors, width=0.80)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.9)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("注册 pair 顺序")
    ax.set_ylabel("log2(slow / fast)")
    source = write_csv(out_dir / "08_t176_pair_benefit_distribution.source.csv", pairs)
    base = out_dir / "08_t176_pair_benefit_distribution"
    return save_figure(fig, base), [source], {"claim": "Pair-level effects are heterogeneous but directionally favor fast cadence in most pairs.", "origin": "T176 hardware Session 0", "positive_pairs": 14, "n": 20}


def build_09(pairs: list[dict[str, str]], summary: dict[str, str], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    cumulative = to_float(pairs, "cumulative_fast_over_slow_ratio")
    critical = float(summary["permutation_critical_ratio"])
    observed = float(summary["fast_over_slow_ratio"])
    x = np.arange(1, 21)
    fig, ax = new_chart(
        "09｜T176 顺序诊断：最终比值低于冻结临界比",
        f"最终 n=20：ratio={observed:.5f}；critical={critical:.5f}；p={float(summary['permutation_p_value']):.4f}；中间点仅为 post-hoc 诊断",
    )
    ax.plot(x, cumulative, marker="o", markersize=3.8, color=COLORS["blue"], label="Cumulative ratio")
    ax.axhline(critical, color=COLORS["amber"], linestyle="--", linewidth=1.3, label="Frozen critical ratio")
    ax.axhline(1.0, color=COLORS["ink"], linestyle=":", linewidth=1.0, label="No improvement")
    ax.scatter([20], [observed], s=48, color=COLORS["blue"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("累计纳入 pair 数")
    ax.set_ylabel("累计 mean(fast) / mean(slow)")
    figure_legend(fig, ax, ncol=3)
    source = write_csv(out_dir / "09_t176_cumulative_ratio.source.csv", pairs)
    base = out_dir / "09_t176_cumulative_ratio"
    return save_figure(fig, base), [source], {"claim": "The final registered pair set lies below the frozen permutation critical ratio.", "origin": "T176 hardware Session 0", "n": 20, "p_value": float(summary["permutation_p_value"])}


def build_10(null_rows: list[dict[str, str]], summary: dict[str, str], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    rows = [row for row in null_rows if row["evidence_origin"] == "hardware_session0"]
    if len(rows) != 20000:
        raise ValueError("Figure 10 requires all 20,000 hardware permutation ratios")
    null = to_float(rows, "null_ratio")
    observed = float(summary["fast_over_slow_ratio"])
    critical = float(summary["permutation_critical_ratio"])
    lower = min(float(np.quantile(null, 0.002)), observed)
    upper = max(float(np.quantile(null, 0.998)), critical)
    fig, ax = new_chart(
        "10｜T176 真机 S0：冻结 pair 内置换检验",
        f"20,000 次标签交换；observed={observed:.5f}；critical={critical:.5f}；p={float(summary['permutation_p_value']):.4f}",
    )
    ax.hist(null, bins=70, range=(lower, upper), density=True, color=COLORS["blue"], alpha=0.72, label="Permutation null")
    ax.axvline(critical, color=COLORS["amber"], linestyle="--", linewidth=1.4, label="Critical ratio (5%)")
    ax.axvline(observed, color=COLORS["red"], linewidth=1.6, label="Observed ratio")
    ax.set_xlim(lower, upper)
    ax.set_xlabel("置换 mean(fast) / mean(slow)")
    ax.set_ylabel("密度")
    figure_legend(fig, ax, ncol=3)
    source = write_csv(out_dir / "10_t176_hardware_permutation_null.source.csv", rows)
    summary_source = write_csv(out_dir / "10_t176_hardware_permutation_null.summary.csv", [{"observed_ratio": observed, "critical_ratio_5pct": critical, "permutation_p_value": summary["permutation_p_value"], "permutations": 20000, "seed": 20260815}])
    base = out_dir / "10_t176_hardware_permutation_null"
    return save_figure(fig, base), [source, summary_source], {"claim": "The exact frozen permutation result is reproducible from all 20 hardware pairs.", "origin": "T176 hardware Session 0", "permutations": 20000, "seed": 20260815}


def build_11(sensitivity: list[dict[str, str]], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    if len(sensitivity) != 4:
        raise ValueError("Figure 11 requires the primary hybrid result plus three drift-shape sensitivities")
    labels = [
        "Primary hybrid",
        "Early saturating transient",
        "Linear ramp",
        "Step at block boundary",
    ]
    ratios = to_float(sensitivity, "ratio")
    critical = to_float(sensitivity, "critical_ratio")
    passed = [row["passed"].lower() == "true" for row in sensitivity]
    fig, ax = new_chart(
        "11｜基线漂移形状敏感性：4/4 点估计均在临界线左侧",
        "这是 simulation-assisted / post-hoc 稳健性证据；它不把缺失的 hardware Session 1 变成真机数据",
        left=0.30,
    )
    y = np.arange(4)[::-1]
    for index in range(4):
        ax.hlines(y[index], ratios[index], critical[index], color=COLORS["grid"], linewidth=5.0)
    ax.scatter(ratios, y, color=COLORS["blue"], s=52, edgecolor="white", linewidth=0.8, zorder=4, label="Observed ratio")
    ax.scatter(critical, y, color=COLORS["amber"], marker="|", s=190, linewidth=2.3, zorder=4, label="Critical ratio")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("fast / slow（观测点在临界值左侧即通过）")
    figure_legend(fig, ax, ncol=2)
    source = write_csv(out_dir / "11_baseline_drift_sensitivity.source.csv", sensitivity)
    base = out_dir / "11_baseline_drift_sensitivity"
    return save_figure(fig, base), [source], {"claim": "All declared baseline-shape sensitivities preserve the descriptive hybrid conclusion.", "origin": "post-hoc simulation-assisted sensitivity", "passed": int(sum(passed)), "total": 4, "registered_all_hardware_status": "missing_hardware_session1"}


def build_12(workload: list[dict[str, str]], out_dir: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    lookup = {row["stage"]: int(row["count"]) for row in workload}
    expected = {"submitted_jobs": 44, "unique_query_ids": 88, "completed_cycles": 40, "complete_pairs": 20, "excluded_pairs": 0}
    if lookup != expected:
        raise ValueError(f"Figure 12 workload counts changed: {lookup}")
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=600)
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.22, top=0.74)
    fig.text(0.055, 0.95, "12｜T176 final Session 0：真机工作量完整、0 exclusions", ha="left", va="top", fontsize=12.5, fontweight="bold", color=COLORS["ink"])
    fig.text(0.055, 0.865, "44 jobs → 88 platform tasks → 40 closed-loop cycles → 20 complete pairs", ha="left", va="top", fontsize=8.0, color=COLORS["muted"])
    ax.set_axis_off()
    stages = [
        ("44", "真机 jobs", COLORS["blue_soft"], COLORS["blue"]),
        ("88", "平台 tasks", COLORS["violet_soft"], COLORS["violet"]),
        ("40", "闭环 cycles", COLORS["teal_soft"], COLORS["teal"]),
        ("20", "完整 pairs", COLORS["green_soft"], COLORS["green"]),
    ]
    for index, (number, label, face, edge) in enumerate(stages):
        x0 = 0.01 + index * 0.25
        box = FancyBboxPatch((x0, 0.25), 0.19, 0.48, boxstyle="round,pad=0.015,rounding_size=0.025", transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x0 + 0.095, 0.56, number, transform=ax.transAxes, ha="center", va="center", fontsize=20, fontweight="bold", color=edge)
        ax.text(x0 + 0.095, 0.36, label, transform=ax.transAxes, ha="center", va="center", fontsize=8.2, color=COLORS["ink"])
        if index < 3:
            ax.annotate("", xy=(x0 + 0.245, 0.49), xytext=(x0 + 0.205, 0.49), xycoords=ax.transAxes, arrowprops={"arrowstyle": "-|>", "color": COLORS["muted"], "linewidth": 1.2})
    fig.text(0.055, 0.115, "边界：注册全真机端点仍缺 hardware Session 1；不以模拟补写真机工作量。", ha="left", va="center", fontsize=8.0, color=COLORS["red"])
    fig.text(0.97, 0.025, "冻结结构化产物；同名 CSV 为工作量台账", ha="right", va="bottom", fontsize=6.5, color=COLORS["muted"])
    source = write_csv(out_dir / "12_t176_hardware_workload_flow.source.csv", workload)
    base = out_dir / "12_t176_hardware_workload_flow"
    return save_figure(fig, base), [source], {"claim": "T176 Session-0 hardware workload is complete and exclusion-free.", "origin": "T176 hardware Session 0", **expected, "registered_all_hardware_status": "missing_hardware_session1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(r"C:\Users\Mercu\Documents\天衍\成果展示_20260829\figures_b4_core_20260830"))
    parser.add_argument("--artifact-root", type=Path, default=Path(r"E:\TianYan\XA-202609"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    inputs = {
        "observable": source_dir / "source_data_03_t287_observable_state.csv",
        "sf": source_dir / "source_data_04_t287_structure_function.csv",
        "curve": source_dir / "source_data_04_t287_e1_sensing_curve.csv",
        "pairs": source_dir / "source_data_05_t176_hardware_pairs.csv",
        "hardware_summary": source_dir / "source_data_05_t176_hardware_summary.csv",
        "nulls": source_dir / "source_data_06_permutation_null_ratios.csv",
        "sensitivity": source_dir / "source_data_06_drift_sensitivity.csv",
        "workload": source_dir / "source_data_08_t176_final_workload.csv",
        "sf_report": artifact_root / "artifacts" / "analysis" / "B4_B9_T287_SF_20260815_r2" / "t287_sf_report.json",
        "sensing_report": artifact_root / "artifacts" / "analysis" / "B4_B9_T287_SENSING_MAP_20260815_r5" / "t287_sensing_map_report.json",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing single-figure input files: {missing}")

    observable = read_csv(inputs["observable"])
    sf_rows = read_csv(inputs["sf"])
    curve_rows = read_csv(inputs["curve"])
    pairs = read_csv(inputs["pairs"])
    hardware_summary_rows = read_csv(inputs["hardware_summary"])
    if len(hardware_summary_rows) != 1:
        raise ValueError("Hardware summary must contain exactly one row")
    hardware_summary = hardware_summary_rows[0]
    null_rows = read_csv(inputs["nulls"])
    sensitivity = read_csv(inputs["sensitivity"])
    workload = read_csv(inputs["workload"])
    sf_report = json.loads(inputs["sf_report"].read_text(encoding="utf-8"))
    sensing_report = json.loads(inputs["sensing_report"].read_text(encoding="utf-8"))

    builders = [
        lambda: build_01(observable, out_dir),
        lambda: build_02(observable, out_dir),
        lambda: build_03(sf_rows, sf_report, out_dir),
        lambda: build_04(sf_rows, sf_report, out_dir),
        lambda: build_05(curve_rows, sensing_report, out_dir),
        lambda: build_06(sensing_report, out_dir),
        lambda: build_07(pairs, hardware_summary, out_dir),
        lambda: build_08(pairs, hardware_summary, out_dir),
        lambda: build_09(pairs, hardware_summary, out_dir),
        lambda: build_10(null_rows, hardware_summary, out_dir),
        lambda: build_11(sensitivity, out_dir),
        lambda: build_12(workload, out_dir),
    ]
    figures: list[dict[str, Any]] = []
    for number, builder in enumerate(builders, start=1):
        outputs, source_files, meta = builder()
        figures.append(
            {
                "number": number,
                "files": [{"path": str(path), "sha256": file_sha256(path)} for path in outputs],
                "source_data": [{"path": str(path), "sha256": file_sha256(path)} for path in source_files],
                **meta,
            }
        )
        print(f"built single Figure {number:02d}: {outputs[0].stem}")

    manifest = {
        "gallery": "B4_main_report_single_figures",
        "figure_count": len(figures),
        "one_chart_per_canvas": True,
        "inputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in inputs.items()},
        "figures": figures,
        "claim_boundary": {
            "supported": [
                "T287 observable state and E1 drift detection",
                "update-interval point economics with an inconclusive T-star interval",
                "T176 hardware Session-0 pair-complete descriptive benefit",
                "exact frozen permutation reproduction",
                "simulation-assisted baseline-shape sensitivity",
            ],
            "not_supported": [
                "registered all-hardware PASS",
                "hardware Session-1 collection",
                "deployed online reinforcement learning",
                "learned cross-device transfer",
                "universal pulse-level calibration",
            ],
        },
    }
    manifest_path = out_dir / "single_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "figures": len(figures), "exports": sum(len(item["files"]) for item in figures), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
