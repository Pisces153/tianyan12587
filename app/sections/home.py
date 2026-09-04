"""平台首页：产品定位、可用能力入口、核心数字、闭环链路、证据状态。"""

from __future__ import annotations

import streamlit as st

from app import theme
from app.router import PAGE_URLS


def render(pages: dict) -> None:
    theme.hero(
        "AEMTN-B4 · 天衍超导真机 · 6 qubits · 冻结 2026-08-31",
        "面向量子云真机的<br>开放量子系统辨识与<br><em>AI 自校准</em>平台",
        "量子云真机受器件无序与环境退相干影响，真实哈密顿量长期偏离设计模型。"
        "本平台从可测的量子态测量数据学习漂移与退相干特征，反推哈密顿量与噪声通道，"
        "给出何时、以何种周期重标定的可执行裁决。全部数字由冻结代码与冻结证据支撑，逐条可复算。",
    )
    c1, c2, _ = st.columns([1.8, 1.4, 3.6])
    with c1:
        with st.container(key="cta_primary"):
            st.page_link(pages["workbench"], label="进入漂移诊断工作台", icon=":material/arrow_forward:")
    with c2:
        st.page_link(pages["evidence"], label="查看真机证据", icon=":material/verified:")

    theme.stats([
        ("真机数据层", "6 qubits", "天衍 T176 / T287"),
        ("终测 ratio", "0.3616", "真机 Session 0，fast/slow 残差比"),
        ("描述性降低", "63.84%", "fast vs slow cadence"),
        ("置换检验 p", "0.0052", "2 万次 pair 内交换"),
        ("闭环链路", "14 环节", "感知 · 决策 · 补偿 · 再测量"),
    ])

    st.subheader("平台能力")
    cols = st.columns(3, gap="small")
    cards = [
        ("01", "漂移诊断", "上传你自己的端点时序，冻结核心实时判别是否存在超 shot-noise 的漂移，给出相关时间 τ、最优重标定周期 T* 与 go/no-go 裁决，导出 JSON 报告。", "在线可用", "live", "workbench", "打开工作台"),
        ("02", "决策安全盾", "五道非学习安全门作用于真实冻结逻辑。拖动估计值、不确定性与预算，实时看到一个补偿动作被哪道门放行或拦下。", "在线可用", "live", "workbench", "进入工作台"),
        ("03", "证据复算", "在浏览器里用纯 Python 重跑三个证据层各 20,000 次配对置换，与冻结值逐项比对，验证结果可独立复现。", "在线可用", "live", "evidence", "去复算"),
    ]
    for col, (n, t, b, tag, kind, target, link) in zip(cols, cards):
        with col:
            theme.card(n, t, b, tag, kind, href=PAGE_URLS[target], link=link)

    st.subheader("全工程闭环链路")
    theme.pipeline([
        "仿真数据生成", "AEMTN 三 seed 训练", "冻结 best.pt 推理",
        "量子态测量特征提取", "环境 / 设备漂移代理", "去偏结构函数",
        "误差预测与残差曲线", "最优更新周期 T*", "go/no-go 感知经济门",
        "安全 shield 与调度", "fast/slow cadence 闭环", "T176 Session 0 真机诊断",
        "Session 1 模拟应急", "simulation-assisted hybrid final", "12 张成果图与数据簿",
    ])
    st.caption("链路与冻结 `PIPELINE.md` 一致；每环节均有回归测试与冻结证据背书。")

    st.subheader("技术要点")
    rows = [
        ("技术路线", "超导量子（天衍 T176/T287）+ 深度学习 AEMTN 网络"),
        ("AI 框架", "深度多任务学习 + 不确定性建模 + 安全 shield + 规则调度"),
        ("关键参数与验证", "真机 20 对端点、置换检验、12 张主图、288 项测试"),
        ("客观瓶颈", "6 比特规模、Jz 弱、T* 置信上界撞窗、硬件 Session 1 缺失"),
    ]
    st.table([{"维度": a, "对应": b} for a, b in rows])

    st.subheader("证据状态")
    theme.status([
        ("项目层状态", "B4_PRESERVED_SIMULATION_ASSISTED", "只在明确的事后、模拟辅助的一致性检验中成立。"),
        ("纯真机注册状态", "INCONCLUSIVE_MISSING_HARDWARE_SESSION1", "T176 真机 Session 0 提供 20 对；缺失的 Session 1 由独立冻结的模拟应急工件补作一致性分析，不进入纯真机注册裁决。"),
        ("结论红线", "CLAIM_BOUNDARY.md", "全部结论遵守仓库红线；能说、必须带限定、不可说三类分界见仓库 `CLAIM_BOUNDARY.md` 与冻结报告 `claim_boundary` 字段。"),
    ])
