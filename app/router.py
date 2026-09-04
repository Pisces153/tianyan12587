"""页面路由：3 个产品区（平台 / 工作台 / 证据），每个是一个可调用页函数。

供 `streamlit_app.py` 构造 `st.navigation`，也供回归测试直接调用。
"""

from __future__ import annotations

import streamlit as st

from app import theme

PAGE_SPECS = [
    # key, url_path, title, icon
    ("home", "", "平台", ":material/home:"),
    ("workbench", "workbench", "工作台", ":material/science:"),
    ("evidence", "evidence", "证据", ":material/verified:"),
]

PAGE_URLS = {key: "/" + url for key, url, _, _ in PAGE_SPECS}

_PAGES: dict = {}


def build_pages() -> dict:
    """构造 st.Page 对象并缓存到 _PAGES，供页函数内部做 page_link。"""
    _PAGES.clear()
    for key, url, title, icon in PAGE_SPECS:
        fn = PAGE_FUNCS[key]
        _PAGES[key] = st.Page(fn, title=title, icon=icon, url_path=url or None, default=(key == "home"))
    return _PAGES


def page_home() -> None:
    from app.sections import home

    home.render(_PAGES)


def page_workbench() -> None:
    theme.page_title("工作台", "漂移诊断与决策安全盾", "两个在线可用的工具，均直接调用冻结核心，不做任何离线预算。")
    tab_wb, tab_sh = st.tabs(["漂移诊断", "决策安全盾"])
    with tab_wb:
        from app.sections import workbench

        workbench.render()
    with tab_sh:
        from app.sections import calibration_shield

        calibration_shield.render()


def page_evidence() -> None:
    theme.page_title("证据", "真机数据与闭环终测", "T287 漂移感知（78 个真机快照）与 T176 闭环终测（20 对真机端点），含在线复算。")
    tab_a, tab_b = st.tabs(["T287 漂移感知与判别", "T176 闭环终测"])
    with tab_a:
        from app.sections import drift_sensing

        drift_sensing.render()
    with tab_b:
        from app.sections import closed_loop_evidence

        closed_loop_evidence.render()


PAGE_FUNCS = {
    "home": page_home,
    "workbench": page_workbench,
    "evidence": page_evidence,
}
