"""AEMTN-B4 在线仪表盘（Streamlit 多页应用）。

第 0 章为在线工作台（上传数据 → 冻结核心实时漂移诊断 → 导出报告），
其余章节为冻结结果展示 + 两项轻量在线交互（终测复算、安全盾演示）。
不触碰任何私有证据；仅渲染随包分发的冻结图与公开派生 CSV。

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
    initial_sidebar_state="expanded",
)


def _heading() -> None:
    st.title("面向量子云真机的开放量子系统辨识与人工智能自校准平台")
    st.caption(
        "AEMTN-B4 · 天衍超导真机 · 6 比特 · 仿真-真机混合闭环 · 冻结结果 2026-08-31"
    )


# 侧边栏导航
_heading()
with st.sidebar:
    st.markdown("### 平台导航")
    page = st.radio(
        "章节",
        [
            "0. 漂移诊断工作台",
            "1. 项目总览",
            "2. 技术路线与 AI 框架",
            "3. 环境漂移感知与判别",
            "4. 校准决策与安全 shield",
            "5. 真机闭环终测证据",
            "6. 模型与数据",
            "7. 边界与限制（诚实声明）",
            "8. 复现与部署",
        ],
        index=0,
    )
    st.divider()
    st.caption(
        "报告状态：`B4_PRESERVED_SIMULATION_ASSISTED`\n\n"
        "纯真机注册：`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`"
    )


_REPO = REPO_ROOT


def main() -> None:
    if page.startswith("0."):
        from app.pages import workbench

        workbench.render()
    elif page.startswith("1."):
        from app.pages import overview

        overview.render()
    elif page.startswith("2."):
        from app.pages import technical_route

        technical_route.render()
    elif page.startswith("3."):
        from app.pages import drift_sensing

        drift_sensing.render()
    elif page.startswith("4."):
        from app.pages import calibration_shield

        calibration_shield.render()
    elif page.startswith("5."):
        from app.pages import closed_loop_evidence

        closed_loop_evidence.render()
    elif page.startswith("6."):
        from app.pages import model_data

        model_data.render()
    elif page.startswith("7."):
        from app.pages import boundaries

        boundaries.render()
    else:
        from app.pages import reproduce

        reproduce.render()


if __name__ == "__main__":
    main()
