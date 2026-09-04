"""第 8 章：复现与部署。

面向两种读者：
* 评审 —— 如何在本地复现冻结结果、校验完整性。
* 朋友/发布者 —— 3 步把仓库发到 GitHub 并部署到 Streamlit Cloud。
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("复现与部署")

    st.subheader("评审复现路径")
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
    st.info(
        "**注意：** 仓库含 3 个测试需要作者机器上的私有工件 "
        "(`E:\\TianYan\\...\\all_endpoints_timing_grid.json`)，该路径不随包分发。"
        "因此官方 CI 里会剔除这 3 个测试；本地如无该文件，这 3 个测试也会失败。"
        "其余 288 个测试在所有环境都通过。"
    )

    st.subheader("验证测试套件")
    st.code(
        "pip install -e .[dev]\n"
        "python -m pytest -q -k \"not (scheduler_path_runs_end_to_end or non_overwriting_and_hardware_mode or legacy_control_is_five_endpoint)\"\n"
        "# 预期：288 passed",
        language="bash",
    )

    st.divider()

    st.subheader("3 步发布到 GitHub 并部署（给发布者）")
    st.markdown(
        "如果你要把这个仓库发到 GitHub 并让评审通过公网访问仪表盘，按下面 3 步。"
        "完整版见 `DEPLOY.md`。"
    )
    st.markdown(
        "**1. 推送仓库**\n\n"
        "在提交前，确认 `git-lfs` 已安装（3 个 17.3 MB 的 `best.pt` 建议走 LFS），"
        "然后 `git init && git add . && git commit -m 'release' && git push`。"
    )
    st.markdown(
        "**2. 上传模型**\n\n"
        "如果仓库较大（含模型），可在 GitHub Release 里上传模型附件，"
        "或确认 `.gitattributes` 已把 `.pt` 标记为二进制后用 LFS。"
    )
    st.markdown(
        "**3. 部署到 Streamlit Cloud**\n\n"
        "Streamlit Community Cloud 支持公开或私有仓库（私有应用可按邮箱设 viewer 白名单）。连接仓库后，"
        "入口文件指向 `app/streamlit_app.py`，依赖用根目录 `requirements.txt`。"
        "**注意：** Cloud 上不安装 torch / qutip / cqlib（重量依赖），"
        "仪表盘的核心展示与在线复算只依赖 numpy / scipy / pandas + streamlit / plotly。"
    )

    st.subheader("Streamlit Cloud 依赖（requirements.txt）")
    st.code(
        "numpy\n"
        "scipy\n"
        "pandas\n"
        "streamlit\n"
        "plotly\n"
        "# 不需要：torch, qutip, cqlib —— 它们只在离线训练/真机对接时才需要",
        language="text",
    )

    st.divider()

    st.subheader("可信度与安全")
    st.markdown(
        "仓库随包分发：\n"
        "* `manifest/PACKAGE_FILES.csv`（333 文件 + 字节 + sha256）\n"
        "* `SECURITY_AND_DATA_BOUNDARY.md`（数据与凭据边界）\n"
        "* `CLAIM_BOUNDARY.md`（结论红线）"
    )
    st.warning(
        "**评审/发布者须知：** 包内不含 API 密钥或 `.env`。但 `manifest/ORIGIN.json` "
        "与 `provenance/` 保留了作者机器的本地路径（如 `C:\\Users\\Mercu\\...`），"
        "发布前请确认是否需要脱敏。平台凭据轮换确认状态为 "
        "`ROTATION_CONFIRMATION_PENDING`，在首次个人真机对接前请完成轮换。"
    )
