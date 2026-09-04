"""应用共享：路径、调色板、安全读到并渲染冻结图。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# 数据可视化调色板（与 dataviz 规范一致）
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "green": "#008300",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "ink": "#0b0b0b",
    "muted": "#898781",
    "grid": "#e1e0d9",
}
GOOD = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"
SERIOUS = "#ec835a"

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "single_figures_v2_diverse_20260831"
MAIN_FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "single_figures_main_20260831"


def load_source_csv(name: str) -> list[dict[str, object]]:
    """安全读取 v2 图的 source CSV；缺失则返回空表。"""
    import csv

    path = FIG_DIR / f"{name}.source.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def read_json(path: Path) -> dict | None:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fig_md(name: str) -> tuple[str | None, Path]:
    """给出一张图的 (说明, png 路径)。"""
    png = FIG_DIR / f"{name}.png"
    if not png.is_file():
        return None, png
    return name, png


def show_figure(name: str, caption: str | None = None) -> None:
    png = FIG_DIR / f"{name}.png"
    if png.is_file():
        st.image(str(png), width="stretch")
    if caption:
        st.caption(caption)


def status_badge(text: str, kind: str = "good") -> None:
    color = {"good": GOOD, "warn": WARN, "crit": CRIT, "info": PALETTE["blue"]}.get(kind, GOOD)
    st.markdown(f":{'green' if kind=='good' else 'orange' if kind=='warn' else 'red'}bold[{text}]")
