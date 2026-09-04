#!/usr/bin/env python3
"""Build the audited AEMTN/B4 closed-loop code release.

This builder copies only the curated effective chain, preserves source hashes,
excludes credentials/raw counts/query IDs, generates a per-file inventory, runs
an internal secret scan, and creates a ZIP64 archive.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable
import zipfile


DEFAULT_REPO = Path(r"C:\Users\Mercu\Desktop\aemtn\competition_xa202609")
DEFAULT_WORKSPACE = Path(r"C:\Users\Mercu\Documents\天衍")
DEFAULT_EVIDENCE = Path(r"E:\TianYan\XA-202609")
DEFAULT_OUT = DEFAULT_WORKSPACE / "AEMTN_B4_CLOSED_LOOP_CODE_PACKAGE_20260831"


@dataclass(frozen=True)
class Item:
    source: Path
    destination: str
    category: str
    what: str
    purpose: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def repo_items(repo: Path) -> list[Item]:
    entries: list[tuple[str, str, str, str]] = [
        # Package and input contract.
        ("src/__init__.py", "01 核心包", "Python包标记", "保留原工程导入布局，使src.*模块可直接导入。"),
        ("src/protocol.py", "01 核心包", "协议契约校验器", "加载并校验训练/硬件协议，阻止缺字段或越界配置进入运行链。"),
        ("src/features/__init__.py", "01 核心包", "特征子包标记", "暴露量子测量特征模块。"),
        ("src/features/pauli.py", "01 核心包", "Pauli特征提取核心", "把九测量基counts恢复为Pauli-15与local6，锁定AEMTN输入顺序。"),
        ("src/physics/__init__.py", "01 核心包", "物理子包标记", "暴露哈密顿量和仿真函数。"),
        ("src/physics/hamiltonian.py", "01 核心包", "六比特物理仿真核心", "生成哈密顿量、Lindblad演化、初态、参数与shot采样所需物理量。"),
        ("src/circuits/__init__.py", "01 核心包", "电路子包标记", "暴露Trotter电路构建模块。"),
        ("src/circuits/trotter.py", "01 核心包", "Trotter电路生成器", "把冻结哈密顿量协议转换为测量电路并提供电路复杂度/端序检查。"),
        ("src/models/__init__.py", "02 AEMTN训练", "模型子包标记", "暴露AEMTN硬件兼容模型。"),
        ("src/models/aemtn_hardware.py", "02 AEMTN训练", "AEMTN模型定义", "实现local6+t输入、共享表示、任务路由、不确定性与h1/h2/Jz预测头。"),
        ("src/training/__init__.py", "02 AEMTN训练", "训练子包标记", "暴露数据集与训练引擎。"),
        ("src/training/dataset.py", "02 AEMTN训练", "训练数据契约", "发现NPZ、执行train/holdout隔离、拒绝遗留泄漏输入并保存归一化统计。"),
        ("src/training/engine.py", "02 AEMTN训练", "训练与checkpoint引擎", "运行多任务优化、确定性种子、早停和完整checkpoint保存/加载。"),
        ("src/backends/__init__.py", "03 天衍适配", "后端子包标记", "暴露天衍发现、拓扑和QCIS适配器。"),
        ("src/backends/tianyan_discovery.py", "03 天衍适配", "后端发现器", "只读查询可用机器及能力，不提交量子任务。"),
        ("src/backends/tianyan_topology.py", "03 天衍适配", "拓扑选择器", "从后端耦合图选择满足协议的六比特链。"),
        ("src/backends/tianyan_native.py", "03 天衍适配", "原生QCIS校验器", "检查原生门集合和门计数，防止不可执行电路进入平台。"),
        ("src/backends/tianyan_v8_entangling.py", "03 天衍适配", "天衍纠缠电路族", "生成固定低CZ量子探针、测量旋转和Pauli期望恢复。"),
        ("src/baselines/__init__.py", "03 天衍适配", "硬件基线子包标记", "暴露冻结天衍名义基线，支持低CZ探针偏移估计。"),
        ("src/baselines/tianyan_v8_nominal.py", "03 天衍适配", "天衍v8名义基线", "用低CZ Pauli观测拟合名义偏移，并生成local6基线预测。"),
        # Adaptive layer.
        ("src/adaptive/__init__.py", "04 漂移感知", "自适应子包标记", "暴露感知、判据、shield和cadence模块。"),
        ("src/adaptive/environment_proxy.py", "04 漂移感知", "环境/设备状态代理", "从量子counts与平台字段提取readout、effective-field和shot-noise代理。"),
        ("src/adaptive/effective_field_diagnostics.py", "04 漂移感知", "有效场诊断", "审计H1/H2反演的可辨识性、分支和不确定性。"),
        ("src/adaptive/forecast.py", "04 漂移感知", "在线滚动预测基线", "按时间顺序运行rolling-origin预测、概率评分和技能门，避免未来信息泄漏。"),
        ("src/adaptive/sensing_economics.py", "05 校准决策", "B4数学核心", "实现精确去偏结构函数、OU/非参残差曲线、T*、经济门与cadence ratio gate。"),
        ("src/adaptive/shared_baseline_sensing.py", "05 校准决策", "共享基线差分感知", "用session首尾基线控制慢漂移并生成敏感性偏移。"),
        ("src/adaptive/cadence_permutation.py", "05 校准决策", "配对置换裁决", "对冻结fast/slow比值做pair内标签置换，输出临界比和p值。"),
        ("src/adaptive/cadence_ledger.py", "05 校准决策", "校准时钟账本", "把观测时间、延迟、更新周期和新鲜度写成可审计ledger。"),
        ("src/adaptive/task_metric_mirror.py", "06 闭环控制", "量子任务镜像指标", "构建mirror电路并从raw counts计算success probability与配对区间。"),
        ("src/adaptive/bandit.py", "06 闭环控制", "安全shield五门", "根据幅度、置信度、预算与物理约束决定执行、降幅或弃权。"),
        ("src/adaptive/rule_scheduler.py", "06 闭环控制", "规则调度器", "在RL证据不足时提供可解释、可审计的安全调度回退。"),
        # Training and inference scripts.
        ("scripts/generate_sim_dataset.py", "02 AEMTN训练", "主训练数据生成入口", "按论文修正版物理协议生成带shot采样的local6+t仿真训练集。"),
        ("scripts/train_sim.py", "02 AEMTN训练", "主训练CLI", "调用训练引擎完成三seed仿真预训练并输出best/last/history/summary。"),
        ("scripts/check_no_leakage.py", "02 AEMTN训练", "标签泄漏审计", "确认h1/h2/Jz目标没有被拼回AEMTN输入。"),
        ("scripts/validate_protocol.py", "02 AEMTN训练", "协议预检CLI", "在生成数据或提交硬件前验证JSON协议与阻断条件。"),
        ("scripts/predict_hardware_aemtn.py", "02 AEMTN训练", "冻结checkpoint推理", "加载模型和归一化统计，对硬件counts派生特征做h1/h2/Jz推理。"),
        ("scripts/validate_aemtn_closed_loop_v4.py", "02 AEMTN训练", "仿真闭环验收", "验证注入、检测、建议、补偿和再测量恢复的离线能力链。"),
        ("scripts/prepare_aemtn_two_time_data.py", "02 AEMTN训练", "两时点数据桥", "把硬件/仿真两时点Pauli数据转换成冻结AEMTN输入格式。"),
        ("scripts/scan_trotter_reference.py", "02 AEMTN训练", "电路参考扫描", "核对Trotter电路与理想物理演化的端序、符号和复杂度。"),
        # B0 and hardware measurement.
        ("scripts/discover_backends.py", "03 天衍适配", "后端清单CLI", "读取天衍机器列表并保存只读能力清单。"),
        ("scripts/poll_platform_config.py", "04 漂移感知", "平台元数据轮询器", "零机时记录calibrationTime与平台误差字段，形成重校事件旁证。"),
        ("scripts/run_platform_config_poll_supervisor.bat", "04 漂移感知", "轮询守护脚本", "轮询异常退出后按冻结策略恢复，不提交量子电路。"),
        ("scripts/audit_b4_poll_acceptance.py", "04 漂移感知", "轮询验收审计", "检查24小时覆盖、空窗、双后端文件增长和重启恢复。"),
        ("scripts/drift_campaign.py", "03 天衍适配", "通用漂移采集底座", "封装登录、提交、收集、raw落盘、hash链和平台时间字段。"),
        ("scripts/drift_campaign_v4.py", "03 天衍适配", "B4 v4采集器", "实现双后端、burst、regime、探针拆绑和真实lag记录。"),
        ("scripts/b4_dry_run_common.py", "03 天衍适配", "B4 dry-run公共层", "统一接口测时、探针构建、结果校验和dry-run工件结构。"),
        ("scripts/measure_b4_interface_floor.py", "03 天衍适配", "接口延迟测量", "测量云API往返P50/P90并确定协议时间底线。"),
        ("scripts/measure_b4_throughput.py", "03 天衍适配", "吞吐/固定开销分解", "用不同shots任务估计每shot速率R与每setting固定开销c。"),
        ("scripts/scan_b4_mirror_depth.py", "03 天衍适配", "Mirror深度扫描", "选择success probability落在可分辨区间的任务深度。"),
        ("scripts/verify_b4_t176_probe.py", "03 天衍适配", "T176探针验收", "验证账号、后端字段、物理链、shots和落盘链，不扩大结论。"),
        # Scientific design and sensing analyses.
        ("scripts/analyze_crossover_feasibility.py", "04 漂移感知", "漂移时标可行性分析", "从旧快照估计job内/跨snapshot方差和OU相容时标。"),
        ("scripts/analyze_sensing_economics.py", "05 校准决策", "通用感知经济分析CLI", "调用同一sensing_economics核心生成SF、残差曲线、T*与判据表。"),
        ("scripts/simulate_t7_element3_dgp_v2.py", "05 校准决策", "注入式T7 element-3仿真", "生成已知零假设与漂移信号，验证结构函数和经济门具备可达性。"),
        ("scripts/simulate_b4_design_power.py", "05 校准决策", "B4功效与size仿真", "在null/OU/pink/step DGP下检验门的假阳性、功效和T*恢复。"),
        ("scripts/aggregate_b4_map_power.py", "05 校准决策", "功效网格聚合", "合并分片仿真结果并生成map power/size总表。"),
        ("scripts/regrid_b4_endpoints.py", "05 校准决策", "终点再网格化", "按冻结R与固定开销c重算真实可达端点网格。"),
        ("scripts/refine_b4_map_power.py", "05 校准决策", "地图功效加密", "对边界timing cell追加重复以稳定map裁决。"),
        ("scripts/refine_b4_4day_gap_size.py", "05 校准决策", "跨日gap size复核", "验证四日空窗和lag结构不会抬高假阳性。"),
        ("scripts/refine_b4_tag_decision_size.py", "05 校准决策", "标签裁决size复核", "检验最终tag/branch规则在零假设下的错误率。"),
        ("scripts/diagnose_b4_map_timing.py", "05 校准决策", "map时序根因诊断", "拆解吞吐与固定开销对非单调map power的贡献。"),
        ("scripts/diagnose_b4_regrid_mechanism.py", "05 校准决策", "再网格机制诊断", "核对重采样、锚点和判据边界的机械原因。"),
        ("scripts/diagnose_b4_map_root_cause_contrast.py", "05 校准决策", "map根因对照", "用A/B对照区分模型、时序和离散网格造成的差异。"),
        ("scripts/evaluate_b4_timing_endpoints.py", "05 校准决策", "时序终点评估", "在候选R/c组合上评估size、power和可达性。"),
        ("scripts/finalize_b4_map_timing_lookup.py", "05 校准决策", "冻结时序查表", "把通过条件的timing cells固化成分析/采集查表。"),
        ("scripts/refine_b4_borderline_timing_cells.py", "05 校准决策", "临界timing加密", "增加临界cells的Monte Carlo样本，避免边界随机翻转。"),
        ("scripts/run_b4_exact_anchor_leverage.py", "05 校准决策", "精确锚点杠杆分析", "测量锚点布局对map可辨识度的贡献。"),
        ("scripts/run_b4_map_root_cause_ab.py", "05 校准决策", "map根因A/B实验", "冻结改变一个机制，其余保持不变，定位功效异常来源。"),
        ("scripts/run_b4_map_root_cause_bisection.py", "05 校准决策", "map根因二分", "在机制组件间二分定位造成裁决变化的最小集合。"),
        ("scripts/simulate_b4_cadence_endpoint_power.py", "06 闭环控制", "cadence端点功效仿真", "验证逐cycle residual²端点的size、power和配对置换可达性。"),
        ("scripts/design_b4_cadence_from_timing.py", "06 闭环控制", "cadence计划求解", "依据实测R/c与接口底线选择fast/slow周期、pair数和预测区间。"),
        ("scripts/build_b4_platform_time_ledger.py", "04 漂移感知", "平台时钟总账", "把提交、返回、执行和快照时间统一成可审计时标。"),
        ("scripts/analyze_b4_t287_sf.py", "04 漂移感知", "T287结构函数分析", "对冻结T287语料复算E0/E1去偏SF、阴性门与事件排除。"),
        ("scripts/analyze_b4_t287_sensing_map.py", "05 校准决策", "T287感知值得性地图", "输出逐通道Var_proc、T*、残差、接口底线和INCONCLUSIVE边界。"),
        ("scripts/audit_b4_b9_inputs.py", "05 校准决策", "B4/B9输入审计", "验证来源hash、regime排除、字段白名单和分析前置条件。"),
        ("scripts/analyze_natural_drift.py", "04 漂移感知", "自然漂移基础分析", "从campaign中提取环境代理并运行早期滚动预测基线。"),
        ("scripts/refresh_natural_drift_analysis.py", "04 漂移感知", "自然漂移刷新入口", "在新增快照后重建派生结果且不覆盖raw。"),
        ("scripts/write_natural_drift_cadence_ledger.py", "04 漂移感知", "自然漂移cadence账本", "把真实观测间隔与空窗写成独立可审计表。"),
        # Closed-loop execution.
        ("scripts/run_mirror_metric.py", "06 闭环控制", "Mirror指标CLI", "独立运行mirror任务指标与配对bootstrap。"),
        ("scripts/run_cadence_pair_loop.py", "06 闭环控制", "闭环调度核心", "执行感知→shield→补偿→mirror循环，并维护fast/slow配对状态。"),
        ("scripts/run_cadence_pair_hardware_smoke.py", "06 闭环控制", "真机闭环smoke", "用最小真实任务验证闭环电路、接口、raw counts与任务指标。"),
        ("scripts/freeze_b4_cadence_v4_start_only_baseline.py", "06 闭环控制", "v4计划冻结器", "构建start-only baseline与完整pair计划并锁定hash。"),
        ("scripts/preflight_b4_cadence_collection_correction.py", "06 闭环控制", "修正采集预飞", "离线回放全部周期、预算、同session配对和停止规则。"),
        ("scripts/run_b4_cadence_pair_hardware.py", "06 闭环控制", "T176正式硬件runner", "按冻结计划提交baseline/loop/mirror任务，追加journal并执行at-most-once保护。"),
        ("scripts/build_b4_cadence_continuation_plan.py", "06 闭环控制", "Session 1续采计划构建器", "在不改原plan的前提下验证Session 0并冻结独立续采计划。"),
        ("scripts/run_b4_cadence_continuation.py", "06 闭环控制", "Session 1续采监督器", "等待后端连续running后再启动续采，并保留安全失败分类。"),
        ("scripts/analyze_b4_t287_cadence_residual_curve.py", "07 结果分析", "T287 cadence残差分析", "用逐cycle residual²重算fast/slow曲线、置换门和诊断图。"),
        # Final contingency and analysis.
        ("scripts/run_b4_session1_simulation_contingency.py", "07 结果分析", "独立模拟Session 1", "按冻结相反顺序生成20对模拟应急数据，明确不写入硬件journal。"),
        ("scripts/analyze_b4_t176_hybrid_final.py", "07 结果分析", "B4 hybrid终测分析器", "验证hash链、提取20+20 pairs、运行置换/敏感性/谬误审计并生成最终报告。"),
    ]
    return [
        Item(repo / rel, rel, category, what, purpose)
        for rel, category, what, purpose in entries
    ]


def config_items(repo: Path) -> list[Item]:
    entries = [
        ("config/protocol_v2.json", "08 配置", "训练/测量主协议", "冻结local6+t、shots、物理初态和数据隔离契约。"),
        ("config/backends_v1.json", "08 配置", "后端角色配置", "定义仿真、参考和天衍硬件角色；不含登录凭据。"),
        ("config/b4_mirror_metric_v1.json", "08 配置", "Mirror指标配置", "冻结mirror电路族、深度、shots、seed和主指标。"),
        ("config/b4_drift_campaign_v4_tianyan287.json", "08 配置", "T287 B4采集配置", "冻结T287 burst、anchors、shots、regime和调度。"),
        ("config/b4_drift_campaign_v4_tianyan176.json", "08 配置", "T176 B4采集配置", "冻结T176同构采集参数和后端差异。"),
        ("config/b4_cadence_pair_loop_v1.json", "08 配置", "原始cadence仿真配置", "保留早期OU漂移与三档注入幅度设计；供cadence单元测试和方法谱系审计。"),
        ("config/b4_cadence_pair_loop_cycle_paired_v2.json", "08 配置", "cycle-paired v2配置", "保留三session、每session八对的修正版设计；验证硬件runner历史行为。"),
        ("config/b4_cadence_pair_loop_cycle_paired_v3.json", "08 配置", "cycle-paired v3配置", "保留deadline slack修订版；审计平台延迟约束修正。"),
        ("config/b4_cadence_pair_loop_cycle_paired_v4.json", "08 配置", "最终cadence配对配置", "冻结90/360秒、逐cycle residual²、pair数、置换seed和门。"),
        ("config/tianyan_h1h2_dual_backend_drift_campaign_v1.json", "08 配置", "双后端漂移v1配置", "保留33-setting原始交错协议、预算门和断点续采行为；供漂移采集谱系测试。"),
        ("config/tianyan_h1h2_dual_backend_drift_campaign_v2.json", "08 配置", "双后端漂移历史配置", "复现T287旧快照与双后端采集字段契约。"),
        ("config/trajectory_reproduction_v1.json", "08 配置", "轨迹复现配置", "冻结两时点硬件/仿真对齐实验设置。"),
        ("requirements-platform.txt", "08 配置", "平台依赖声明", "记录原工程天衍SDK依赖入口。"),
    ]
    return [Item(repo / rel, rel, category, what, purpose) for rel, category, what, purpose in entries]


TEST_PURPOSES = {
    "test_counts_features.py": "验证counts端序、九测量基、Pauli-15与local6恢复。",
    "test_physics_contract.py": "验证哈密顿量、初态、噪声和采样物理契约。",
    "test_training_pipeline.py": "验证数据隔离、训练、checkpoint与确定性恢复。",
    "test_protocol_contract.py": "验证协议缺字段、泄漏字段和阻断规则。",
    "test_trotter_circuits.py": "验证Trotter电路结构、测量旋转和复杂度。",
    "test_tianyan_discovery.py": "验证只读机器发现与字段解析。",
    "test_tianyan_topology.py": "验证六比特链选择和拓扑拒绝。",
    "test_tianyan_v8_entangling_contract.py": "验证天衍低CZ电路和Pauli期望契约。",
    "test_adaptive_bandit.py": "验证安全shield允许、降幅和弃权分支。",
    "test_adaptive_environment_proxy.py": "验证readout/effective-field代理和shot噪声。",
    "test_adaptive_rule_scheduler.py": "验证可解释规则调度及安全回退。",
    "test_effective_field_diagnostics.py": "验证H1/H2诊断与不确定性输出。",
    "test_sensing_economics.py": "验证去偏SF、OU/非参积分、T*和经济门。",
    "test_shared_baseline_sensing.py": "验证共享基线差分和漂移敏感性偏移。",
    "test_cadence_permutation.py": "验证配对置换ratio、p值、临界值与seed复现。",
    "test_task_metric_mirror.py": "验证mirror构建、success probability和bootstrap。",
    "test_drift_campaign.py": "验证通用采集、hash链、raw落盘和失败恢复。",
    "test_drift_campaign_v4.py": "验证v4 burst、regime、角色和非T287/非1024配置。",
    "test_poll_platform_config.py": "验证零机时轮询和断线重试。",
    "test_gate_reachability.py": "验证注入已知信号时统计门能够通过。",
    "test_gate_feasibility.py": "验证闭环任务预算与平台时间约束。",
    "test_simulate_b4_design_power.py": "验证B4 DGP、size/power字段和无oracle泄漏。",
    "test_b4_dry_run.py": "验证接口底线、吞吐和镜像深度dry-run公共逻辑。",
    "test_build_b4_platform_time_ledger.py": "验证平台时间字段、时区、匿名化和总账。",
    "test_analyze_b4_t287_sf.py": "验证T287 SF样本筛选、事件排除与E0/E1门。",
    "test_analyze_b4_t287_sensing_map.py": "验证T*、CI碰窗和INCONCLUSIVE地图口径。",
    "test_audit_b4_b9_inputs.py": "验证B4/B9输入hash、字段和regime完整性。",
    "test_cadence_pair_loop.py": "验证感知→shield→补偿→mirror状态机。",
    "test_cadence_pair_hardware_smoke.py": "验证真机smoke的任务角色与counts处理。",
    "test_b4_cadence_endpoint_reachability.py": "验证逐cycle residual²端点的可达性。",
    "test_b4_cadence_v4_forty_minute_design.py": "验证v4 40-cycle/20-pair冻结设计。",
    "test_freeze_b4_cadence_v4_start_only_baseline.py": "验证计划冻结、baseline和hash不变性。",
    "test_preflight_b4_cadence_collection_correction.py": "验证修正采集的完整离线回放。",
    "test_run_b4_cadence_pair_hardware.py": "验证正式runner的at-most-once、journal和失败分类。",
    "test_b4_cadence_continuation.py": "验证独立Session 1计划和恢复监督器。",
    "test_b4_session1_simulation_contingency.py": "验证模拟Session 1不提交硬件且来源隔离。",
    "test_analyze_b4_t176_hybrid_final.py": "验证终测hash链、pair提取、置换、边界和确定性签名。",
    "test_analyze_b4_t287_cadence_residual_curve.py": "验证T287逐cycle残差曲线与置换诊断。",
    "test_natural_drift_cadence_ledger.py": "验证自然漂移观测间隔和空窗账本。",
    "test_analyze_natural_drift.py": "验证早期自然漂移代理分析不会混入标签。",
    "test_refresh_natural_drift_analysis.py": "验证刷新分析不覆盖raw数据。",
    "test_diagnose_b4_map_root_cause_contrast.py": "验证map根因A/B对照机械一致性。",
    "test_diagnose_b4_regrid_mechanism.py": "验证再网格机制诊断。",
    "test_refine_b4_4day_gap_size.py": "验证四日gap size加密。",
    "test_refine_b4_tag_decision_size.py": "验证最终tag裁决size。",
    "test_regrid_b4_endpoints.py": "验证R/c再网格端点。",
    "test_run_b4_exact_anchor_leverage.py": "验证精确锚点杠杆分析。",
    "test_run_b4_map_root_cause_ab.py": "验证map根因A/B实验。",
    "test_run_b4_map_root_cause_bisection.py": "验证map根因二分。",
}


def test_items(repo: Path) -> list[Item]:
    items = [Item(repo / "tests/__init__.py", "tests/__init__.py", "09 测试", "测试包标记", "支持pytest导入测试辅助模块。")]
    for name, purpose in TEST_PURPOSES.items():
        items.append(Item(repo / "tests" / name, f"tests/{name}", "09 测试", f"{name}回归测试", purpose))
    return items


def documentation_items(repo: Path, workspace: Path) -> list[Item]:
    repo_entries = [
        ("README.md", "10 文档", "原工程README", "说明AEMTN主训练输入、命令、数据边界与硬件角色。"),
        ("PROJECT_NOTES.md", "10 文档", "项目事实与红线", "记录模型谱系、历史审计、结论边界和最终停点。"),
        ("docs/aemtn_model_lineage.md", "10 文档", "模型谱系表", "防止不同checkpoint和硬件任务被错误混用。"),
        ("docs/B4_WORK_ORDER_20260804.md", "10 文档", "B4原始工单", "保存B0-B9设计、验收标准、统计纪律和措辞红线。"),
        ("docs/B4_T176_HYBRID_FINAL_RESULT_20260829.md", "10 文档", "B4最终结果说明", "给出simulation-assisted终测结果、完整性和严格边界。"),
        ("docs/B4_T176_HYBRID_FINAL_MANIFEST_20260829.json", "10 文档", "B4最终机器清单", "记录最终数值、工件路径、SHA256和验证状态。"),
        ("docs/B4_T176_MIGRATION_CURATED_MANIFEST_20260823.json", "10 文档", "T176迁移精选清单", "记录迁移到T176时允许使用的runner/config/manifest。"),
        ("docs/B4_B5_T176_BACKEND_MIGRATION_AMENDMENT_20260823.md", "10 文档", "后端迁移修正案", "说明T176后端差异与不能改变的冻结端点。"),
        ("docs/B4_B5_T176_MIGRATION_HANDOFF_20260823.md", "10 文档", "T176迁移交接", "记录真机采集前置、输出目录和安全操作。"),
        ("docs/B4_T176_20260817_TIMING_QUERY_MANIFEST.json", "10 文档", "接口测时来源清单", "追溯P50/P90、R和c的原始查询证据。"),
        ("docs/NATURAL_DRIFT_CAMPAIGN_PREREGISTRATION_v1.md", "10 文档", "自然漂移预注册", "冻结33-setting交错采集、预算、停止规则和来源哈希；通用漂移runner运行时校验此文件。"),
        ("docs/NATURAL_DRIFT_P1_IMPLEMENTATION_v1.md", "10 文档", "自然漂移实现说明", "解释environment proxy、预测与边界。"),
    ]
    items = [Item(repo / rel, rel, cat, what, purpose) for rel, cat, what, purpose in repo_entries]
    workspace_entries = [
        ("HANDOFF.md", "docs/HANDOFF_20260829.md", "最终交接主档", "保存2026-08-29状态覆盖、权威入口和禁止扩张的结论。"),
        ("天衍_全项目流程与聊天迁移总档.md", "docs/天衍_全项目流程与聊天迁移总档.md", "全工程时间线", "把训练、IBM、天衍、自然漂移、B4旧端点和终测串成一条谱系。"),
    ]
    for rel, dest, what, purpose in workspace_entries:
        items.append(Item(workspace / rel, dest, "10 文档", what, purpose))
    return items


def ensure_items(items: Iterable[Item]) -> None:
    missing = [str(item.source) for item in items if not item.source.is_file()]
    if missing:
        raise FileNotFoundError("Missing curated inputs:\n" + "\n".join(missing))


def copy_item(item: Item, output: Path) -> dict[str, object]:
    destination = output / Path(item.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.source, destination)
    return {
        **asdict(item),
        "source": str(item.source),
        "destination": item.destination.replace("\\", "/"),
        "source_sha256": sha256_file(item.source),
        "packaged_sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def copy_tree(
    source: Path,
    destination: Path,
    *,
    package_root: Path,
    include=None,
) -> list[dict[str, object]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    records: list[dict[str, object]] = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source)
        if include is not None and not include(path, relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        records.append(
            {
                "source": str(path),
                "destination": target.relative_to(package_root).as_posix(),
                "source_sha256": sha256_file(path),
                "packaged_sha256": sha256_file(target),
                "bytes": target.stat().st_size,
                "category": "11 数据/证据",
                "what": "数据或证据工件",
                "purpose": "支持离线复现、结果核验或图表再生成。",
            }
        )
    return records


README = r"""# AEMTN / B4 AI驱动量子校准闭环代码包

版本：2026-08-31。这个包从主训练仓、冻结模型、B4分析/控制代码和最终派生证据反向整理而成，目标是为下一步API、桌面工具或SaaS封装提供一个可审计基线。

## 闭环主线

`仿真数据生成 → AEMTN训练/推理 → 量子测量代理 → 去偏结构函数 → 更新周期/T* → shield → fast/slow cadence → T176 Session 0 → 模拟Session 1 → hybrid final → 图表/数据簿`

## 最快使用

```powershell
python .\b4ctl.py inventory
python .\b4ctl.py verify
python .\b4ctl.py reproduce-final
python .\b4ctl.py test --tier core
```

训练入口示例：

```powershell
python .\b4ctl.py train --data .\data\sim_v3_paper_contract --seeds 20260731 --out .\runs\training_smoke
```

真机runner被安全门保护，只有显式传入 `--allow-hardware` 才会转交原runner；本包构建和验收过程中不会连接平台、不会提交量子任务。

## 目录

- `src/`：训练、特征、物理、后端和自适应核心模块。
- `scripts/`：从数据生成到hybrid final的原始有效CLI。
- `config/`：冻结协议和B4 v4配置；不含凭据。
- `tests/`：与所列代码逐项对应的回归测试。
- `models/`：三seed有效best checkpoint及训练记录；未重复打包last.pt。
- `data/`：完整sim_v3 paper-contract训练集。
- `evidence/`：最终公开派生pair表、模拟应急工件、计划和验证报告。
- `figures/`：12张独立主图、逐图CSV、矢量/位图版本和总数据簿。
- `manifest/`：逐文件来源、SHA256、完整包文件表和外部私有证据索引。
- `tools/`：完整性验证、公开派生结果复算和本发布构建器。

## 证据边界

- 项目层：`B4_PRESERVED_SIMULATION_ASSISTED`。
- 注册纯真机层：`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`。
- T287和T176承担不同证据角色，不能写成双真机端到端复制。
- 模拟Session 1不能写成硬件Session 1。
- 当前代码包含安全shield和规则调度，但不等于在线RL已在真机部署。

详见 `CODE_INVENTORY.md`、`PIPELINE.md`、`CLAIM_BOUNDARY.md` 和 `SECURITY_AND_DATA_BOUNDARY.md`。
"""


PIPELINE = r"""# 闭环工程链

## 1. 数据与物理层

`src/physics/hamiltonian.py`定义六比特系统、初态和噪声；`scripts/generate_sim_dataset.py`加入有限shots并生成严格分割的local6+t数据。

## 2. AEMTN训练层

`src/models/aemtn_hardware.py`、`src/training/dataset.py`、`src/training/engine.py`组成主训练；三个冻结best checkpoint随包提供。Jz留出性能弱，只作为输出记录，不进入自动补偿主张。

## 3. 量子测量与环境代理层

Pauli counts经`src/features/pauli.py`恢复为local6；`environment_proxy.py`与`effective_field_diagnostics.py`产生readout和H1/H2量子侧代理。它们不是温度计或电磁传感器。

## 4. 漂移判别层

`sensing_economics.py`计算去偏结构函数；E0阴性对照不通过漂移门，E1通过，建立“不是所有波动都叫漂移”的特异性。

## 5. 更新周期决策层

同一核心计算OU/非参数残差曲线、T*和go/no-go经济门。T*点估计约134.4秒，但95%区间上界碰观测窗，裁决为INCONCLUSIVE。

## 6. 控制执行层

`run_cadence_pair_loop.py`组织感知→shield→补偿→mirror；`run_b4_cadence_pair_hardware.py`按冻结v4计划执行40 cycles/20 pairs，并用append-only journal与at-most-once保护采集。

## 7. 终测层

T176真机Session 0提供20对；缺失的真机Session 1由独立模拟应急工件补作一致性分析，但不进入注册纯真机裁决。`analyze_b4_t176_hybrid_final.py`完成20+20 hybrid、置换检验、漂移形状敏感性和11项统计谬误审计。

## 8. 展示层

`figures/single_figures_v2_diverse_20260831`包含12张一图一问主图、逐图CSV和多格式导出；总数据簿集中保存字段字典与claim boundary。
"""


CLAIM_BOUNDARY = r"""# 结论与主张边界

## 可以说

- AEMTN主训练代码、数据契约和三seed checkpoint已冻结。
- 量子测量代理能够显示T287读出状态的时间变化。
- E0阴性对照未通过漂移门；E1过程方差超出shot-noise解释范围。
- 更新周期残差地图、接口底线和T*点估计已经计算。
- T176 Hardware Session 0的20对中，fast/slow累计平方残差比为0.361649790，配对置换p=0.005249738。
- simulation-assisted hybrid ratio为0.374481312，p=0.000099995。

## 必须带限定

- B4只写成`B4_PRESERVED_SIMULATION_ASSISTED`。
- 纯真机注册状态仍是`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`。
- T*=134.4秒是点估计；置信上界碰到观测窗，不能写成稳定生产SLA。
- F11是模拟辅助/事后敏感性，不是真机第二会话。

## 不可以说

- 不可写“双真机闭环复现”或“registered all-hardware PASS”。
- 不可把H1/H2代理称为直接温度或电磁测量。
- 不可声称在线强化学习、跨设备知识迁移或完整环境传感融合已在真机完成。
- 不可把63.84%残差降低写成通用算力提升或所有任务的性能提升。
"""


SECURITY = r"""# 安全与数据边界

- 包内没有API key、天衍OpenID、token、密码或`.env`。
- 所有平台凭据必须通过进程环境变量注入；不得写入config、命令历史或manifest。
- 原T176 `snapshots.jsonl`未收入ZIP，因为包含88个query ID、raw结果路径和本机路径引用。
- 原raw counts与NPZ未收入ZIP；它们的来源路径和SHA256保存在`manifest/EXTERNAL_PRIVATE_EVIDENCE.json`。
- ZIP包含公开派生的cycle endpoints和hybrid pair rows，因此可以复算核心比值与置换检验。
- 三seed模型只收入`best.pt`，不重复收入内容角色相近的`last.pt`；排除决定不会改变代码或已报告结果。
- 任何真机命令默认阻断，统一入口要求显式`--allow-hardware`。
"""


B4CTL = r'''#!/usr/bin/env python3
"""Unified, safety-gated facade over the frozen AEMTN/B4 scripts."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

CORE_TESTS = [
    "tests/test_counts_features.py",
    "tests/test_training_pipeline.py",
    "tests/test_sensing_economics.py",
    "tests/test_cadence_permutation.py",
    "tests/test_task_metric_mirror.py",
    "tests/test_drift_campaign_v4.py",
    "tests/test_cadence_pair_loop.py",
    "tests/test_run_b4_cadence_pair_hardware.py",
    "tests/test_b4_session1_simulation_contingency.py",
    "tests/test_analyze_b4_t176_hybrid_final.py",
]


def run(command: list[str]) -> int:
    return subprocess.call(command, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    sub.add_parser("pipeline")
    sub.add_parser("verify")
    sub.add_parser("reproduce-final")
    test = sub.add_parser("test")
    test.add_argument("--tier", choices=("core", "full"), default="core")
    train = sub.add_parser("train")
    train.add_argument("args", nargs=argparse.REMAINDER)
    generate = sub.add_parser("generate-data")
    generate.add_argument("args", nargs=argparse.REMAINDER)
    hardware = sub.add_parser("hardware-run")
    hardware.add_argument("--allow-hardware", action="store_true")
    hardware.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command == "inventory":
        print((ROOT / "CODE_INVENTORY.md").read_text(encoding="utf-8"))
        return 0
    if args.command == "pipeline":
        print((ROOT / "PIPELINE.md").read_text(encoding="utf-8"))
        return 0
    if args.command == "verify":
        return run([sys.executable, "tools/verify_package.py"])
    if args.command == "reproduce-final":
        return run([sys.executable, "tools/reproduce_public_final.py"])
    if args.command == "test":
        targets = CORE_TESTS if args.tier == "core" else ["tests"]
        return run([sys.executable, "-m", "pytest", "-q", *targets])
    if args.command == "train":
        return run([sys.executable, "scripts/train_sim.py", *args.args])
    if args.command == "generate-data":
        return run([sys.executable, "scripts/generate_sim_dataset.py", *args.args])
    if args.command == "hardware-run":
        if not args.allow_hardware:
            print("REFUSED: hardware submission requires explicit --allow-hardware", file=sys.stderr)
            return 2
        return run([sys.executable, "scripts/run_b4_cadence_pair_hardware.py", *args.args])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


VERIFY_TOOL = r'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest" / "PACKAGE_FILES.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig", newline="")))
    expected = {row["path"]: row for row in rows}
    failures = []
    for relative, row in expected.items():
        path = ROOT / Path(relative)
        if not path.is_file():
            failures.append(f"MISSING {relative}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"SIZE {relative}")
        if digest(path) != row["sha256"]:
            failures.append(f"HASH {relative}")
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts
    }
    extras = sorted(actual - set(expected))
    failures.extend(f"EXTRA {path}" for path in extras)
    if failures:
        print("PACKAGE VERIFY FAILED")
        print("\n".join(failures))
        return 1
    print(f"PACKAGE VERIFY PASS: {len(rows)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


REPRODUCE_TOOL = r'''#!/usr/bin/env python3
"""Recompute the public, derived B4 final ratios and permutation diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive.cadence_permutation import cadence_ratio_permutation_gate


def close(left: float, right: float, tolerance: float = 1e-14) -> bool:
    return bool(abs(left - right) <= tolerance)


def gate(rows):
    fast = np.asarray([float(row["fast_endpoint_squared_residual"]) for row in rows])
    slow = np.asarray([float(row["slow_endpoint_squared_residual"]) for row in rows])
    return cadence_ratio_permutation_gate(fast, slow, permutations=20000, seed=20260815)


def main() -> int:
    evidence = ROOT / "evidence" / "B4_T176_HYBRID_FINAL_20260829"
    rows = list(csv.DictReader((evidence / "hybrid_pair_rows.csv").open("r", encoding="utf-8-sig", newline="")))
    report = json.loads((evidence / "hybrid_final_report.json").read_text(encoding="utf-8"))
    groups = {
        "hardware_session0_diagnostic": [row for row in rows if row["evidence_origin"] == "hardware_session0"],
        "simulation_session1_diagnostic": [row for row in rows if row["evidence_origin"] == "simulation_session1"],
        "hybrid_primary": rows,
    }
    results = {name: gate(group) for name, group in groups.items()}
    expected = report["statistical_findings"]
    failures = []
    for name, result in results.items():
        for key in ("ratio", "p_value", "critical_ratio"):
            if not close(float(result[key]), float(expected[name][key])):
                failures.append(f"{name}.{key}")
    payload = {
        name: {
            "pairs": int(result["pair_count"]),
            "ratio": float(result["ratio"]),
            "relative_reduction": 1.0 - float(result["ratio"]),
            "critical_ratio": float(result["critical_ratio"]),
            "p_value": float(result["p_value"]),
            "passed": bool(result["passed"]),
        }
        for name, result in results.items()
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        print("PUBLIC FINAL REPRODUCTION FAILED: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("PUBLIC FINAL REPRODUCTION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_inventory(output: Path, records: list[dict[str, object]]) -> None:
    lines = [
        "# 有效代码逐文件清单",
        "",
        "下面逐个列出本ZIP中的代码、配置、测试与关键文档。数据、模型、图像和证据的每一个实际文件另见 `manifest/PACKAGE_FILES.csv`。",
        "",
    ]
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["category"]), []).append(record)
    for category in sorted(grouped):
        lines.extend([f"## {category}", "", "| 文件 | 是什么 | 干什么用 |", "|---|---|---|"])
        for record in sorted(grouped[category], key=lambda item: str(item["destination"])):
            destination = str(record["destination"]).replace("|", "\\|")
            what = str(record["what"]).replace("|", "\\|")
            purpose = str(record["purpose"]).replace("|", "\\|")
            lines.append(f"| `{destination}` | {what} | {purpose} |")
        lines.append("")
    write_text(output / "CODE_INVENTORY.md", "\n".join(lines))


def write_generated_files(output: Path) -> None:
    write_text(output / "README.md", README)
    write_text(output / "PIPELINE.md", PIPELINE)
    write_text(output / "CLAIM_BOUNDARY.md", CLAIM_BOUNDARY)
    write_text(output / "SECURITY_AND_DATA_BOUNDARY.md", SECURITY)
    write_text(output / "b4ctl.py", B4CTL)
    write_text(output / "tools/verify_package.py", VERIFY_TOOL)
    write_text(output / "tools/reproduce_public_final.py", REPRODUCE_TOOL)
    write_text(
        output / "requirements-core.txt",
        "\n".join(
            [
                "numpy==2.3.2",
                "scipy==1.16.1",
                "pandas==2.3.2",
                "matplotlib==3.10.6",
                "torch==2.7.1+cu126",
                "qutip==5.2.2",
                "scikit-learn==1.7.1",
                "joblib==1.5.2",
                "pytest==9.0.0",
            ]
        ),
    )


def secret_scan(output: Path) -> list[dict[str, object]]:
    text_extensions = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".bat", ".ps1", ".ini", ".cfg"}
    strong_patterns = [
        ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
        ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
        ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
        ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}")),
    ]
    assignment = re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|login[_-]?key|openid|password|secret)\b\s*[:=]\s*['\"]([^'\"]{12,})['\"]")
    # Require a base64-specific character (+, /, or =) so ordinary hexadecimal
    # SHA256 values and analysis signatures are not misclassified as secrets.
    base64ish = re.compile(
        r"(?<![A-Za-z0-9+/])(?=[A-Za-z0-9+/=]{50,}(?![A-Za-z0-9+/=]))"
        r"(?=[A-Za-z0-9+/]*[+/=])[A-Za-z0-9+/]{50,}={0,2}(?![A-Za-z0-9+/])"
    )
    findings: list[dict[str, object]] = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.suffix.lower() in text_extensions):
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            for kind, pattern in strong_patterns:
                if pattern.search(line):
                    findings.append({"path": path.relative_to(output).as_posix(), "line": number, "kind": kind})
            match = assignment.search(line)
            if match:
                value = match.group(2).lower()
                if not any(marker in value for marker in ("<", "your", "env", "pending", "none", "null", "example", "replace")):
                    findings.append({"path": path.relative_to(output).as_posix(), "line": number, "kind": "credential_assignment"})
            if base64ish.search(line) and not any(marker in line.lower() for marker in ("sha", "hash", "digest", "signature", "record_", "analysis_signature")):
                findings.append({"path": path.relative_to(output).as_posix(), "line": number, "kind": "high_entropy_base64"})
    return findings


def package_files(output: Path) -> list[dict[str, object]]:
    manifest = output / "manifest" / "PACKAGE_FILES.csv"
    return [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(p for p in output.rglob("*") if p.is_file() and p != manifest and "__pycache__" not in p.parts)
    ]


def write_package_manifest(output: Path) -> list[dict[str, object]]:
    rows = package_files(output)
    path = output / "manifest" / "PACKAGE_FILES.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    repo = args.repo.resolve()
    workspace = args.workspace.resolve()
    evidence = args.evidence.resolve()
    output = args.output.resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {output} / {archive}")

    items = repo_items(repo) + config_items(repo) + test_items(repo) + documentation_items(repo, workspace)
    ensure_items(items)
    output.mkdir(parents=True)
    source_records = [copy_item(item, output) for item in items]

    # Exact simulation training data.
    source_records.extend(
        copy_tree(
            evidence / "data/interim/sim_v3_paper_contract",
            output / "data/sim_v3_paper_contract",
            package_root=output,
        )
    )

    # Effective best checkpoints only; last.pt duplicates deployment role and is excluded.
    checkpoint_root = evidence / "artifacts/checkpoints/sim_pretrained_paper_contract"
    checkpoint_files = [checkpoint_root / "training_summary.json"]
    for seed in ("seed_20260731", "seed_20260801", "seed_20260802"):
        checkpoint_files.extend(checkpoint_root / seed / name for name in ("best.pt", "history.csv", "run_config.json", "summary.json"))
    for path in checkpoint_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(checkpoint_root)
        target = output / "models/sim_pretrained_paper_contract" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_records.append({
            "source": str(path),
            "destination": target.relative_to(output).as_posix(),
            "source_sha256": sha256_file(path),
            "packaged_sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "category": "11 数据/证据",
            "what": "冻结AEMTN模型或训练记录",
            "purpose": "供后续推理封装、模型核验和训练谱系追溯。",
        })

    # Final derived evidence and simulation contingency. No raw counts or query IDs.
    source_records.extend(copy_tree(
        evidence / "artifacts/analysis/B4_T176_HYBRID_FINAL_20260829",
        output / "evidence/B4_T176_HYBRID_FINAL_20260829",
        package_root=output,
    ))
    source_records.extend(copy_tree(
        evidence / "artifacts/analysis/B4_T176_SESSION1_SIMULATION_CONTINGENCY_20260829",
        output / "evidence/B4_T176_SESSION1_SIMULATION_CONTINGENCY_20260829",
        package_root=output,
    ))
    plan_paths = [
        evidence / "artifacts/analysis/B4_T176_HYBRID_FINAL_20260829.plan.json",
        evidence / "artifacts/analysis/B4_T176_HYBRID_FINAL_20260829_v2.plan.json",
        evidence / "artifacts/analysis/B4_T176_SESSION1_SIMULATION_CONTINGENCY_20260829.plan.json",
        evidence / "quarantine/tianyan176/B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823/campaign_manifest.json",
        evidence / "quarantine/tianyan176/B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823/supplement_plan.json",
    ]
    for path in plan_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        target = output / "evidence/plans_and_manifests" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_records.append({
            "source": str(path),
            "destination": target.relative_to(output).as_posix(),
            "source_sha256": sha256_file(path),
            "packaged_sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "category": "11 数据/证据",
            "what": "冻结计划或manifest",
            "purpose": "追溯预注册、模拟应急、失败v1与修正v2分析谱系。",
        })

    # Figures and source data.
    v2_figures = workspace / "成果展示_20260829/single_figures_v2_diverse_20260831"
    source_records.extend(copy_tree(
        v2_figures,
        output / "figures/single_figures_v2_diverse_20260831",
        package_root=output,
    ))
    v1_figures = workspace / "成果展示_20260829/single_figures_main_20260831"
    allowed_v1_names = {"make_b4_single_figures.py", "README_主图索引.md", "逐图中文图注.md", "QA_REPORT.md", "single_figure_manifest.json"}
    source_records.extend(copy_tree(
        v1_figures,
        output / "figures/single_figures_main_20260831",
        package_root=output,
        include=lambda path, relative: path.name in allowed_v1_names or path.name.endswith(".source.csv") or path.name.endswith(".summary.csv"),
    ))
    workbook = workspace / "成果展示_20260829/outputs/01a04d7f-e622-71e2-b53b-c412b65a5b7a/B4_主图逐图完整数据.xlsx"
    workbook_target = output / "figures/B4_主图逐图完整数据.xlsx"
    workbook_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workbook, workbook_target)
    source_records.append({
        "source": str(workbook),
        "destination": workbook_target.relative_to(output).as_posix(),
        "source_sha256": sha256_file(workbook),
        "packaged_sha256": sha256_file(workbook_target),
        "bytes": workbook_target.stat().st_size,
        "category": "11 数据/证据",
        "what": "12图总数据簿",
        "purpose": "集中保存图索引、关键指标、逐图数据、字段字典和claim boundary。",
    })

    write_generated_files(output)
    shutil.copy2(Path(__file__).resolve(), output / "tools/build_release.py")

    figure_code = {
        "figures/single_figures_main_20260831/make_b4_single_figures.py": (
            "主图v1生成器",
            "从逐图source CSV生成12张独立科学图，并导出PNG/SVG/PDF/TIFF。",
        ),
        "figures/single_figures_v2_diverse_20260831/make_b4_single_figures_v2.py": (
            "多样化主图v2生成器",
            "用热图、森林图、雨云/分布、区间图等多种图型重建最终12图。",
        ),
    }
    for record in source_records:
        destination = str(record.get("destination", ""))
        if destination in figure_code:
            record["category"] = "12 包装与展示工具"
            record["what"], record["purpose"] = figure_code[destination]

    generated_code = [
        ("b4ctl.py", "统一控制入口", "提供清单、流程、哈希验证、最终复算、测试、训练、数据生成和受保护真机入口。"),
        ("tools/build_release.py", "可复现发布构建器", "从原工程、冻结模型、派生证据和图表数据重建同结构审计ZIP。"),
        ("tools/verify_package.py", "包完整性校验器", "逐文件核对manifest中的大小与SHA256，发现缺失、篡改或多余内容。"),
        ("tools/reproduce_public_final.py", "公开终测复算器", "从包内冻结pair数据重算硬件Session 0、模拟Session 1和hybrid统计量。"),
    ]
    builder_path = Path(__file__).resolve()
    for relative, what, purpose in generated_code:
        path = output / relative
        source_records.append({
            "source": str(builder_path) if relative == "tools/build_release.py" else f"generated_by:{builder_path}",
            "destination": relative,
            "source_sha256": sha256_file(path),
            "packaged_sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "category": "12 包装与展示工具",
            "what": what,
            "purpose": purpose,
        })

    write_inventory(output, [record for record in source_records if str(record.get("category", "")) != "11 数据/证据"])

    manifest_dir = output / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_text(manifest_dir / "SOURCE_MANIFEST.json", json.dumps(source_records, ensure_ascii=False, indent=2, default=str))

    hardware_root = evidence / "quarantine/tianyan176/B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823"
    external = {
        "schema": "aemtn_b4_external_private_evidence_v1",
        "reason_not_packaged": "contains platform query IDs, raw-result paths, or raw scientific payloads; no credentials were found, but default distributable ZIP excludes them",
        "files": [],
    }
    for name in ("snapshots.jsonl",):
        path = hardware_root / name
        external["files"].append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "included": False})
    write_text(manifest_dir / "EXTERNAL_PRIVATE_EVIDENCE.json", json.dumps(external, ensure_ascii=False, indent=2))

    provenance = output / "provenance"
    write_text(provenance / "git_status.txt", command_text(["git", "status", "--short"], cwd=repo))
    origin = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_python": sys.version,
        "source_repo": str(repo),
        "source_git_head": command_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "source_git_branch": command_text(["git", "branch", "--show-current"], cwd=repo),
        "workspace": str(workspace),
        "evidence_root": str(evidence),
        "claim_status": "B4_PRESERVED_SIMULATION_ASSISTED",
        "registered_hardware_status": "INCONCLUSIVE_MISSING_HARDWARE_SESSION1",
    }
    write_text(manifest_dir / "ORIGIN.json", json.dumps(origin, ensure_ascii=False, indent=2))
    write_text(
        provenance / "python_environment.txt",
        command_text([sys.executable, "-m", "pip", "freeze"]),
    )

    findings = secret_scan(output)
    write_text(manifest_dir / "SECURITY_SCAN.json", json.dumps({"passed": not findings, "findings": findings}, ensure_ascii=False, indent=2))
    if findings:
        raise RuntimeError(f"Secret scan failed with {len(findings)} finding(s); see manifest/SECURITY_SCAN.json")

    rows = write_package_manifest(output)
    # Verify before archiving.
    subprocess.run([sys.executable, str(output / "tools/verify_package.py")], cwd=output, check=True)
    subprocess.run([sys.executable, str(output / "tools/reproduce_public_final.py")], cwd=output, check=True)

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as handle:
        for path in sorted(p for p in output.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            arcname = Path(output.name) / path.relative_to(output)
            handle.write(path, arcname.as_posix())

    summary = {
        "output": str(output),
        "archive": str(archive),
        "files_verified": len(rows),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "source_records": len(source_records),
        "security_scan_passed": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output, archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
