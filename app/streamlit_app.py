"""AEMTN-B4 在线平台（Streamlit，顶部导航多页应用）。

平台 / 工作台 / 证据 / 方法 / 边界与复现 五个产品区。工作台对上传数据调用
冻结核心实时诊断；其余页面渲染随包分发的冻结图与公开派生 CSV，并提供
两项轻量在线交互（终测复算、安全盾演示）。不触碰任何私有证据。

运行:
    本地:  streamlit run app/streamlit_app.py
    云端:  Streamlit Community Cloud 指向本文件即可。
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

st.set_page_config(
    page_title="AEMTN-B4 · 开放量子系统辨识与 AI 自校准平台",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from app import router, theme

theme.inject()
st.logo(str(APP_DIR / "assets" / "logo.svg"), size="medium")

pages = router.build_pages()
nav = st.navigation(list(pages.values()), position="top")
nav.run()
