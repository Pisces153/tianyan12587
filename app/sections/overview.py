"""项目总览页：标题、闭环链路、核心数字、评审对照。"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("项目总览")

    st.markdown(
        """
**一句话：** 量子云真机易受器件无序与环境退相干影响，真实哈密顿量长期偏离
设计模型。本项目用 **AI 反演 + 自校准闭环**，从可测的量子态测量数据学习多体
关联与退相干特征，反推哈密顿量与噪声通道，形成器件级噪声画像，再反馈任务
调度与误差缓解。全部结果由冻结代码与冻结证据支撑，逐条可复算。
"""
    )

    # 核心数字磁贴
    st.subheader("核心数字")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("真机数据层", "6 比特", "天衍 T176/T287")
    c2.metric("闭环链路", "14 环节", "感知→决策→补偿→再测量")
    c3.metric("终测 ratio", "0.3616", "真机 Session 0")
    c4.metric("描述性降低", "63.84%", "fast vs slow cadence")
    c5.metric("置换检验 p", "0.0052", "2 万次 pair 内交换")

    # 闭环链路
    st.subheader("全工程闭环链路")
    stages = [
        "仿真数据生成", "AEMTN 三seed 训练", "冻结 best.pt 推理",
        "量子态测量特征提取", "环境/设备漂移代理", "去偏结构函数",
        "误差预测与残差曲线", "最优更新周期 T*", "go/no-go 感知经济门",
        "安全 shield 与调度", "fast/slow cadence 闭环", "T176 Session 0 真机诊断",
        "Session 1 模拟应急", "simulation-assisted hybrid final", "12 张成果图与数据簿",
    ]
    for i, s in enumerate(stages, 1):
        st.markdown(f"`{i:02d}` {s}")
    st.caption("链路与冻结 `PIPELINE.md` 一致；每环节均有回归测试与冻结证据背书。")

    # 与评审要求对照
    st.subheader("与评审关注点的对照")
    rows = [
        ("技术路线", "超导量子（天衍 T176/T287）+ 深度学习 AEMTN 网络"),
        ("AI 框架", "深度多任务学习 + 不确定性建模 + 安全 shield + 规则调度"),
        ("关键参数与验证", "真机 20 对端点、置换检验、12 张主图、288 项测试"),
        ("客观瓶颈分析", "6 比特规模、Jz 弱、T* 置信上界撞窗、硬件 Session1 缺失"),
    ]
    st.table([{"维度": a, "对应": b} for a, b in rows])
    st.caption("全部结论均遵守 `CLAIM_BOUNDARY.md`：不可宣称在线RL/跨设备迁移已在真机完成。")

    st.subheader("证据边界（评审必须知晓）")
    st.info(
        "**项目层状态：** `B4_PRESERVED_SIMULATION_ASSISTED`\n\n"
        "**纯真机注册状态：** `INCONCLUSIVE_MISSING_HARDWARE_SESSION1`\n\n"
        "T176 真机 Session 0 提供 20 对；缺失的真机 Session 1 由独立冻结的"
        "模拟应急工件补作一致性分析，**不进入纯真机注册裁决**。"
    )
