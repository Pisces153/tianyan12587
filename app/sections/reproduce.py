"""复现与部署。如何在本地复现冻结结果、校验完整性；在线部署说明。"""

from __future__ import annotations

import streamlit as st


def render() -> None:

    st.subheader("本地复现")
    st.markdown(
        "所有冻结结果都可用纯净依赖独立复现。核心 `aemtn_b4` 只依赖 "
        "numpy / scipy / pandas，不强制 torch / qutip / cqlib。"
    )
    st.code(
        "pip install -e .                # 安装 aemtn-b4（含核心依赖）\n"
        "aemtn verify                    # 校验冻结核心字节与 sha256\n"
        "aemtn reproduce-final           # 重算 3×20000 次置换门\n"
        "aemtn final-report              # 打印结构化终测结果\n"
        "aemtn dashboard                 # 启动 Streamlit 仪表盘（需 app 依赖）",
        language="bash",
    )

    st.markdown(
        "`aemtn verify` 会做三层校验：**L0** 冻结核心必须与 manifest 的字节与 "
        "sha256 完全一致；**L1** 允许新增的 `aemtn_b4` / `app` 包装层；**L2** 忽略 "
        "`.venv` / `__pycache__` 等工作区产物。"
    )
    st.subheader("验证测试套件")
    st.code(
        "pip install -e .[dev]\n"
        "python -m pytest -q -k \"not (scheduler_path_runs_end_to_end or non_overwriting_and_hardware_mode or legacy_control_is_five_endpoint)\"\n"
        "# 预期：288 passed",
        language="bash",
    )
    st.caption("排除的 3 项测试依赖未随包分发的真机接口计时原始工件，CI 配置同样排除。")

    st.divider()

    st.subheader("在线部署")
    st.markdown(
        "本站运行于 Streamlit Community Cloud，入口 `app/streamlit_app.py`，"
        "依赖仅 numpy / scipy / pandas / streamlit / plotly。"
        "torch / qutip / cqlib 只在离线训练与真机对接时需要，不影响在线展示、复算与漂移诊断工作台。"
    )

    st.divider()

    st.subheader("完整性与边界文档")
    st.markdown(
        "* `manifest/PACKAGE_FILES.csv` — 333 文件的字节数与 sha256\n"
        "* `CLAIM_BOUNDARY.md` — 结论红线\n"
        "* `SECURITY_AND_DATA_BOUNDARY.md` — 数据与凭据边界（仓库不含 API 密钥或 `.env`）"
    )
