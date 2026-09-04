# 有效代码逐文件清单

下面逐个列出本ZIP中的代码、配置、测试与关键文档。数据、模型、图像和证据的每一个实际文件另见 `manifest/PACKAGE_FILES.csv`。

## 01 核心包

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `src/__init__.py` | Python包标记 | 保留原工程导入布局，使src.*模块可直接导入。 |
| `src/circuits/__init__.py` | 电路子包标记 | 暴露Trotter电路构建模块。 |
| `src/circuits/trotter.py` | Trotter电路生成器 | 把冻结哈密顿量协议转换为测量电路并提供电路复杂度/端序检查。 |
| `src/features/__init__.py` | 特征子包标记 | 暴露量子测量特征模块。 |
| `src/features/pauli.py` | Pauli特征提取核心 | 把九测量基counts恢复为Pauli-15与local6，锁定AEMTN输入顺序。 |
| `src/physics/__init__.py` | 物理子包标记 | 暴露哈密顿量和仿真函数。 |
| `src/physics/hamiltonian.py` | 六比特物理仿真核心 | 生成哈密顿量、Lindblad演化、初态、参数与shot采样所需物理量。 |
| `src/protocol.py` | 协议契约校验器 | 加载并校验训练/硬件协议，阻止缺字段或越界配置进入运行链。 |

## 02 AEMTN训练

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `scripts/check_no_leakage.py` | 标签泄漏审计 | 确认h1/h2/Jz目标没有被拼回AEMTN输入。 |
| `scripts/generate_sim_dataset.py` | 主训练数据生成入口 | 按论文修正版物理协议生成带shot采样的local6+t仿真训练集。 |
| `scripts/predict_hardware_aemtn.py` | 冻结checkpoint推理 | 加载模型和归一化统计，对硬件counts派生特征做h1/h2/Jz推理。 |
| `scripts/prepare_aemtn_two_time_data.py` | 两时点数据桥 | 把硬件/仿真两时点Pauli数据转换成冻结AEMTN输入格式。 |
| `scripts/scan_trotter_reference.py` | 电路参考扫描 | 核对Trotter电路与理想物理演化的端序、符号和复杂度。 |
| `scripts/train_sim.py` | 主训练CLI | 调用训练引擎完成三seed仿真预训练并输出best/last/history/summary。 |
| `scripts/validate_aemtn_closed_loop_v4.py` | 仿真闭环验收 | 验证注入、检测、建议、补偿和再测量恢复的离线能力链。 |
| `scripts/validate_protocol.py` | 协议预检CLI | 在生成数据或提交硬件前验证JSON协议与阻断条件。 |
| `src/models/__init__.py` | 模型子包标记 | 暴露AEMTN硬件兼容模型。 |
| `src/models/aemtn_hardware.py` | AEMTN模型定义 | 实现local6+t输入、共享表示、任务路由、不确定性与h1/h2/Jz预测头。 |
| `src/training/__init__.py` | 训练子包标记 | 暴露数据集与训练引擎。 |
| `src/training/dataset.py` | 训练数据契约 | 发现NPZ、执行train/holdout隔离、拒绝遗留泄漏输入并保存归一化统计。 |
| `src/training/engine.py` | 训练与checkpoint引擎 | 运行多任务优化、确定性种子、早停和完整checkpoint保存/加载。 |

## 03 天衍适配

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `scripts/b4_dry_run_common.py` | B4 dry-run公共层 | 统一接口测时、探针构建、结果校验和dry-run工件结构。 |
| `scripts/discover_backends.py` | 后端清单CLI | 读取天衍机器列表并保存只读能力清单。 |
| `scripts/drift_campaign.py` | 通用漂移采集底座 | 封装登录、提交、收集、raw落盘、hash链和平台时间字段。 |
| `scripts/drift_campaign_v4.py` | B4 v4采集器 | 实现双后端、burst、regime、探针拆绑和真实lag记录。 |
| `scripts/measure_b4_interface_floor.py` | 接口延迟测量 | 测量云API往返P50/P90并确定协议时间底线。 |
| `scripts/measure_b4_throughput.py` | 吞吐/固定开销分解 | 用不同shots任务估计每shot速率R与每setting固定开销c。 |
| `scripts/scan_b4_mirror_depth.py` | Mirror深度扫描 | 选择success probability落在可分辨区间的任务深度。 |
| `scripts/verify_b4_t176_probe.py` | T176探针验收 | 验证账号、后端字段、物理链、shots和落盘链，不扩大结论。 |
| `src/backends/__init__.py` | 后端子包标记 | 暴露天衍发现、拓扑和QCIS适配器。 |
| `src/backends/tianyan_discovery.py` | 后端发现器 | 只读查询可用机器及能力，不提交量子任务。 |
| `src/backends/tianyan_native.py` | 原生QCIS校验器 | 检查原生门集合和门计数，防止不可执行电路进入平台。 |
| `src/backends/tianyan_topology.py` | 拓扑选择器 | 从后端耦合图选择满足协议的六比特链。 |
| `src/backends/tianyan_v8_entangling.py` | 天衍纠缠电路族 | 生成固定低CZ量子探针、测量旋转和Pauli期望恢复。 |
| `src/baselines/__init__.py` | 硬件基线子包标记 | 暴露冻结天衍名义基线，支持低CZ探针偏移估计。 |
| `src/baselines/tianyan_v8_nominal.py` | 天衍v8名义基线 | 用低CZ Pauli观测拟合名义偏移，并生成local6基线预测。 |

## 04 漂移感知

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `scripts/analyze_b4_t287_sf.py` | T287结构函数分析 | 对冻结T287语料复算E0/E1去偏SF、阴性门与事件排除。 |
| `scripts/analyze_crossover_feasibility.py` | 漂移时标可行性分析 | 从旧快照估计job内/跨snapshot方差和OU相容时标。 |
| `scripts/analyze_natural_drift.py` | 自然漂移基础分析 | 从campaign中提取环境代理并运行早期滚动预测基线。 |
| `scripts/audit_b4_poll_acceptance.py` | 轮询验收审计 | 检查24小时覆盖、空窗、双后端文件增长和重启恢复。 |
| `scripts/build_b4_platform_time_ledger.py` | 平台时钟总账 | 把提交、返回、执行和快照时间统一成可审计时标。 |
| `scripts/poll_platform_config.py` | 平台元数据轮询器 | 零机时记录calibrationTime与平台误差字段，形成重校事件旁证。 |
| `scripts/refresh_natural_drift_analysis.py` | 自然漂移刷新入口 | 在新增快照后重建派生结果且不覆盖raw。 |
| `scripts/run_platform_config_poll_supervisor.bat` | 轮询守护脚本 | 轮询异常退出后按冻结策略恢复，不提交量子电路。 |
| `scripts/write_natural_drift_cadence_ledger.py` | 自然漂移cadence账本 | 把真实观测间隔与空窗写成独立可审计表。 |
| `src/adaptive/__init__.py` | 自适应子包标记 | 暴露感知、判据、shield和cadence模块。 |
| `src/adaptive/effective_field_diagnostics.py` | 有效场诊断 | 审计H1/H2反演的可辨识性、分支和不确定性。 |
| `src/adaptive/environment_proxy.py` | 环境/设备状态代理 | 从量子counts与平台字段提取readout、effective-field和shot-noise代理。 |
| `src/adaptive/forecast.py` | 在线滚动预测基线 | 按时间顺序运行rolling-origin预测、概率评分和技能门，避免未来信息泄漏。 |

## 05 校准决策

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `scripts/aggregate_b4_map_power.py` | 功效网格聚合 | 合并分片仿真结果并生成map power/size总表。 |
| `scripts/analyze_b4_t287_sensing_map.py` | T287感知值得性地图 | 输出逐通道Var_proc、T*、残差、接口底线和INCONCLUSIVE边界。 |
| `scripts/analyze_sensing_economics.py` | 通用感知经济分析CLI | 调用同一sensing_economics核心生成SF、残差曲线、T*与判据表。 |
| `scripts/audit_b4_b9_inputs.py` | B4/B9输入审计 | 验证来源hash、regime排除、字段白名单和分析前置条件。 |
| `scripts/diagnose_b4_map_root_cause_contrast.py` | map根因对照 | 用A/B对照区分模型、时序和离散网格造成的差异。 |
| `scripts/diagnose_b4_map_timing.py` | map时序根因诊断 | 拆解吞吐与固定开销对非单调map power的贡献。 |
| `scripts/diagnose_b4_regrid_mechanism.py` | 再网格机制诊断 | 核对重采样、锚点和判据边界的机械原因。 |
| `scripts/evaluate_b4_timing_endpoints.py` | 时序终点评估 | 在候选R/c组合上评估size、power和可达性。 |
| `scripts/finalize_b4_map_timing_lookup.py` | 冻结时序查表 | 把通过条件的timing cells固化成分析/采集查表。 |
| `scripts/refine_b4_4day_gap_size.py` | 跨日gap size复核 | 验证四日空窗和lag结构不会抬高假阳性。 |
| `scripts/refine_b4_borderline_timing_cells.py` | 临界timing加密 | 增加临界cells的Monte Carlo样本，避免边界随机翻转。 |
| `scripts/refine_b4_map_power.py` | 地图功效加密 | 对边界timing cell追加重复以稳定map裁决。 |
| `scripts/refine_b4_tag_decision_size.py` | 标签裁决size复核 | 检验最终tag/branch规则在零假设下的错误率。 |
| `scripts/regrid_b4_endpoints.py` | 终点再网格化 | 按冻结R与固定开销c重算真实可达端点网格。 |
| `scripts/run_b4_exact_anchor_leverage.py` | 精确锚点杠杆分析 | 测量锚点布局对map可辨识度的贡献。 |
| `scripts/run_b4_map_root_cause_ab.py` | map根因A/B实验 | 冻结改变一个机制，其余保持不变，定位功效异常来源。 |
| `scripts/run_b4_map_root_cause_bisection.py` | map根因二分 | 在机制组件间二分定位造成裁决变化的最小集合。 |
| `scripts/simulate_b4_design_power.py` | B4功效与size仿真 | 在null/OU/pink/step DGP下检验门的假阳性、功效和T*恢复。 |
| `scripts/simulate_t7_element3_dgp_v2.py` | 注入式T7 element-3仿真 | 生成已知零假设与漂移信号，验证结构函数和经济门具备可达性。 |
| `src/adaptive/cadence_ledger.py` | 校准时钟账本 | 把观测时间、延迟、更新周期和新鲜度写成可审计ledger。 |
| `src/adaptive/cadence_permutation.py` | 配对置换裁决 | 对冻结fast/slow比值做pair内标签置换，输出临界比和p值。 |
| `src/adaptive/sensing_economics.py` | B4数学核心 | 实现精确去偏结构函数、OU/非参残差曲线、T*、经济门与cadence ratio gate。 |
| `src/adaptive/shared_baseline_sensing.py` | 共享基线差分感知 | 用session首尾基线控制慢漂移并生成敏感性偏移。 |

## 06 闭环控制

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `scripts/build_b4_cadence_continuation_plan.py` | Session 1续采计划构建器 | 在不改原plan的前提下验证Session 0并冻结独立续采计划。 |
| `scripts/design_b4_cadence_from_timing.py` | cadence计划求解 | 依据实测R/c与接口底线选择fast/slow周期、pair数和预测区间。 |
| `scripts/freeze_b4_cadence_v4_start_only_baseline.py` | v4计划冻结器 | 构建start-only baseline与完整pair计划并锁定hash。 |
| `scripts/preflight_b4_cadence_collection_correction.py` | 修正采集预飞 | 离线回放全部周期、预算、同session配对和停止规则。 |
| `scripts/run_b4_cadence_continuation.py` | Session 1续采监督器 | 等待后端连续running后再启动续采，并保留安全失败分类。 |
| `scripts/run_b4_cadence_pair_hardware.py` | T176正式硬件runner | 按冻结计划提交baseline/loop/mirror任务，追加journal并执行at-most-once保护。 |
| `scripts/run_cadence_pair_hardware_smoke.py` | 真机闭环smoke | 用最小真实任务验证闭环电路、接口、raw counts与任务指标。 |
| `scripts/run_cadence_pair_loop.py` | 闭环调度核心 | 执行感知→shield→补偿→mirror循环，并维护fast/slow配对状态。 |
| `scripts/run_mirror_metric.py` | Mirror指标CLI | 独立运行mirror任务指标与配对bootstrap。 |
| `scripts/simulate_b4_cadence_endpoint_power.py` | cadence端点功效仿真 | 验证逐cycle residual²端点的size、power和配对置换可达性。 |
| `src/adaptive/bandit.py` | 安全shield五门 | 根据幅度、置信度、预算与物理约束决定执行、降幅或弃权。 |
| `src/adaptive/rule_scheduler.py` | 规则调度器 | 在RL证据不足时提供可解释、可审计的安全调度回退。 |
| `src/adaptive/task_metric_mirror.py` | 量子任务镜像指标 | 构建mirror电路并从raw counts计算success probability与配对区间。 |

## 07 结果分析

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `scripts/analyze_b4_t176_hybrid_final.py` | B4 hybrid终测分析器 | 验证hash链、提取20+20 pairs、运行置换/敏感性/谬误审计并生成最终报告。 |
| `scripts/analyze_b4_t287_cadence_residual_curve.py` | T287 cadence残差分析 | 用逐cycle residual²重算fast/slow曲线、置换门和诊断图。 |
| `scripts/run_b4_session1_simulation_contingency.py` | 独立模拟Session 1 | 按冻结相反顺序生成20对模拟应急数据，明确不写入硬件journal。 |

## 08 配置

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `config/b4_cadence_pair_loop_cycle_paired_v2.json` | cycle-paired v2配置 | 保留三session、每session八对的修正版设计；验证硬件runner历史行为。 |
| `config/b4_cadence_pair_loop_cycle_paired_v3.json` | cycle-paired v3配置 | 保留deadline slack修订版；审计平台延迟约束修正。 |
| `config/b4_cadence_pair_loop_cycle_paired_v4.json` | 最终cadence配对配置 | 冻结90/360秒、逐cycle residual²、pair数、置换seed和门。 |
| `config/b4_cadence_pair_loop_v1.json` | 原始cadence仿真配置 | 保留早期OU漂移与三档注入幅度设计；供cadence单元测试和方法谱系审计。 |
| `config/b4_drift_campaign_v4_tianyan176.json` | T176 B4采集配置 | 冻结T176同构采集参数和后端差异。 |
| `config/b4_drift_campaign_v4_tianyan287.json` | T287 B4采集配置 | 冻结T287 burst、anchors、shots、regime和调度。 |
| `config/b4_mirror_metric_v1.json` | Mirror指标配置 | 冻结mirror电路族、深度、shots、seed和主指标。 |
| `config/backends_v1.json` | 后端角色配置 | 定义仿真、参考和天衍硬件角色；不含登录凭据。 |
| `config/protocol_v2.json` | 训练/测量主协议 | 冻结local6+t、shots、物理初态和数据隔离契约。 |
| `config/tianyan_h1h2_dual_backend_drift_campaign_v1.json` | 双后端漂移v1配置 | 保留33-setting原始交错协议、预算门和断点续采行为；供漂移采集谱系测试。 |
| `config/tianyan_h1h2_dual_backend_drift_campaign_v2.json` | 双后端漂移历史配置 | 复现T287旧快照与双后端采集字段契约。 |
| `config/trajectory_reproduction_v1.json` | 轨迹复现配置 | 冻结两时点硬件/仿真对齐实验设置。 |
| `requirements-platform.txt` | 平台依赖声明 | 记录原工程天衍SDK依赖入口。 |

## 09 测试

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `tests/__init__.py` | 测试包标记 | 支持pytest导入测试辅助模块。 |
| `tests/test_adaptive_bandit.py` | test_adaptive_bandit.py回归测试 | 验证安全shield允许、降幅和弃权分支。 |
| `tests/test_adaptive_environment_proxy.py` | test_adaptive_environment_proxy.py回归测试 | 验证readout/effective-field代理和shot噪声。 |
| `tests/test_adaptive_rule_scheduler.py` | test_adaptive_rule_scheduler.py回归测试 | 验证可解释规则调度及安全回退。 |
| `tests/test_analyze_b4_t176_hybrid_final.py` | test_analyze_b4_t176_hybrid_final.py回归测试 | 验证终测hash链、pair提取、置换、边界和确定性签名。 |
| `tests/test_analyze_b4_t287_cadence_residual_curve.py` | test_analyze_b4_t287_cadence_residual_curve.py回归测试 | 验证T287逐cycle残差曲线与置换诊断。 |
| `tests/test_analyze_b4_t287_sensing_map.py` | test_analyze_b4_t287_sensing_map.py回归测试 | 验证T*、CI碰窗和INCONCLUSIVE地图口径。 |
| `tests/test_analyze_b4_t287_sf.py` | test_analyze_b4_t287_sf.py回归测试 | 验证T287 SF样本筛选、事件排除与E0/E1门。 |
| `tests/test_analyze_natural_drift.py` | test_analyze_natural_drift.py回归测试 | 验证早期自然漂移代理分析不会混入标签。 |
| `tests/test_audit_b4_b9_inputs.py` | test_audit_b4_b9_inputs.py回归测试 | 验证B4/B9输入hash、字段和regime完整性。 |
| `tests/test_b4_cadence_continuation.py` | test_b4_cadence_continuation.py回归测试 | 验证独立Session 1计划和恢复监督器。 |
| `tests/test_b4_cadence_endpoint_reachability.py` | test_b4_cadence_endpoint_reachability.py回归测试 | 验证逐cycle residual²端点的可达性。 |
| `tests/test_b4_cadence_v4_forty_minute_design.py` | test_b4_cadence_v4_forty_minute_design.py回归测试 | 验证v4 40-cycle/20-pair冻结设计。 |
| `tests/test_b4_dry_run.py` | test_b4_dry_run.py回归测试 | 验证接口底线、吞吐和镜像深度dry-run公共逻辑。 |
| `tests/test_b4_session1_simulation_contingency.py` | test_b4_session1_simulation_contingency.py回归测试 | 验证模拟Session 1不提交硬件且来源隔离。 |
| `tests/test_build_b4_platform_time_ledger.py` | test_build_b4_platform_time_ledger.py回归测试 | 验证平台时间字段、时区、匿名化和总账。 |
| `tests/test_cadence_pair_hardware_smoke.py` | test_cadence_pair_hardware_smoke.py回归测试 | 验证真机smoke的任务角色与counts处理。 |
| `tests/test_cadence_pair_loop.py` | test_cadence_pair_loop.py回归测试 | 验证感知→shield→补偿→mirror状态机。 |
| `tests/test_cadence_permutation.py` | test_cadence_permutation.py回归测试 | 验证配对置换ratio、p值、临界值与seed复现。 |
| `tests/test_counts_features.py` | test_counts_features.py回归测试 | 验证counts端序、九测量基、Pauli-15与local6恢复。 |
| `tests/test_diagnose_b4_map_root_cause_contrast.py` | test_diagnose_b4_map_root_cause_contrast.py回归测试 | 验证map根因A/B对照机械一致性。 |
| `tests/test_diagnose_b4_regrid_mechanism.py` | test_diagnose_b4_regrid_mechanism.py回归测试 | 验证再网格机制诊断。 |
| `tests/test_drift_campaign.py` | test_drift_campaign.py回归测试 | 验证通用采集、hash链、raw落盘和失败恢复。 |
| `tests/test_drift_campaign_v4.py` | test_drift_campaign_v4.py回归测试 | 验证v4 burst、regime、角色和非T287/非1024配置。 |
| `tests/test_effective_field_diagnostics.py` | test_effective_field_diagnostics.py回归测试 | 验证H1/H2诊断与不确定性输出。 |
| `tests/test_freeze_b4_cadence_v4_start_only_baseline.py` | test_freeze_b4_cadence_v4_start_only_baseline.py回归测试 | 验证计划冻结、baseline和hash不变性。 |
| `tests/test_gate_feasibility.py` | test_gate_feasibility.py回归测试 | 验证闭环任务预算与平台时间约束。 |
| `tests/test_gate_reachability.py` | test_gate_reachability.py回归测试 | 验证注入已知信号时统计门能够通过。 |
| `tests/test_natural_drift_cadence_ledger.py` | test_natural_drift_cadence_ledger.py回归测试 | 验证自然漂移观测间隔和空窗账本。 |
| `tests/test_physics_contract.py` | test_physics_contract.py回归测试 | 验证哈密顿量、初态、噪声和采样物理契约。 |
| `tests/test_poll_platform_config.py` | test_poll_platform_config.py回归测试 | 验证零机时轮询和断线重试。 |
| `tests/test_preflight_b4_cadence_collection_correction.py` | test_preflight_b4_cadence_collection_correction.py回归测试 | 验证修正采集的完整离线回放。 |
| `tests/test_protocol_contract.py` | test_protocol_contract.py回归测试 | 验证协议缺字段、泄漏字段和阻断规则。 |
| `tests/test_refine_b4_4day_gap_size.py` | test_refine_b4_4day_gap_size.py回归测试 | 验证四日gap size加密。 |
| `tests/test_refine_b4_tag_decision_size.py` | test_refine_b4_tag_decision_size.py回归测试 | 验证最终tag裁决size。 |
| `tests/test_refresh_natural_drift_analysis.py` | test_refresh_natural_drift_analysis.py回归测试 | 验证刷新分析不覆盖raw数据。 |
| `tests/test_regrid_b4_endpoints.py` | test_regrid_b4_endpoints.py回归测试 | 验证R/c再网格端点。 |
| `tests/test_run_b4_cadence_pair_hardware.py` | test_run_b4_cadence_pair_hardware.py回归测试 | 验证正式runner的at-most-once、journal和失败分类。 |
| `tests/test_run_b4_exact_anchor_leverage.py` | test_run_b4_exact_anchor_leverage.py回归测试 | 验证精确锚点杠杆分析。 |
| `tests/test_run_b4_map_root_cause_ab.py` | test_run_b4_map_root_cause_ab.py回归测试 | 验证map根因A/B实验。 |
| `tests/test_run_b4_map_root_cause_bisection.py` | test_run_b4_map_root_cause_bisection.py回归测试 | 验证map根因二分。 |
| `tests/test_sensing_economics.py` | test_sensing_economics.py回归测试 | 验证去偏SF、OU/非参积分、T*和经济门。 |
| `tests/test_shared_baseline_sensing.py` | test_shared_baseline_sensing.py回归测试 | 验证共享基线差分和漂移敏感性偏移。 |
| `tests/test_simulate_b4_design_power.py` | test_simulate_b4_design_power.py回归测试 | 验证B4 DGP、size/power字段和无oracle泄漏。 |
| `tests/test_task_metric_mirror.py` | test_task_metric_mirror.py回归测试 | 验证mirror构建、success probability和bootstrap。 |
| `tests/test_tianyan_discovery.py` | test_tianyan_discovery.py回归测试 | 验证只读机器发现与字段解析。 |
| `tests/test_tianyan_topology.py` | test_tianyan_topology.py回归测试 | 验证六比特链选择和拓扑拒绝。 |
| `tests/test_tianyan_v8_entangling_contract.py` | test_tianyan_v8_entangling_contract.py回归测试 | 验证天衍低CZ电路和Pauli期望契约。 |
| `tests/test_training_pipeline.py` | test_training_pipeline.py回归测试 | 验证数据隔离、训练、checkpoint与确定性恢复。 |
| `tests/test_trotter_circuits.py` | test_trotter_circuits.py回归测试 | 验证Trotter电路结构、测量旋转和复杂度。 |

## 10 文档

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `PROJECT_NOTES.md` | 项目事实与红线 | 记录模型谱系、历史审计、结论边界和最终停点。 |
| `README.md` | 原工程README | 说明AEMTN主训练输入、命令、数据边界与硬件角色。 |
| `docs/B4_B5_T176_BACKEND_MIGRATION_AMENDMENT_20260823.md` | 后端迁移修正案 | 说明T176后端差异与不能改变的冻结端点。 |
| `docs/B4_B5_T176_MIGRATION_HANDOFF_20260823.md` | T176迁移交接 | 记录真机采集前置、输出目录和安全操作。 |
| `docs/B4_T176_20260817_TIMING_QUERY_MANIFEST.json` | 接口测时来源清单 | 追溯P50/P90、R和c的原始查询证据。 |
| `docs/B4_T176_HYBRID_FINAL_MANIFEST_20260829.json` | B4最终机器清单 | 记录最终数值、工件路径、SHA256和验证状态。 |
| `docs/B4_T176_HYBRID_FINAL_RESULT_20260829.md` | B4最终结果说明 | 给出simulation-assisted终测结果、完整性和严格边界。 |
| `docs/B4_T176_MIGRATION_CURATED_MANIFEST_20260823.json` | T176迁移精选清单 | 记录迁移到T176时允许使用的runner/config/manifest。 |
| `docs/B4_WORK_ORDER_20260804.md` | B4原始工单 | 保存B0-B9设计、验收标准、统计纪律和措辞红线。 |
| `docs/HANDOFF_20260829.md` | 最终交接主档 | 保存2026-08-29状态覆盖、权威入口和禁止扩张的结论。 |
| `docs/NATURAL_DRIFT_CAMPAIGN_PREREGISTRATION_v1.md` | 自然漂移预注册 | 冻结33-setting交错采集、预算、停止规则和来源哈希；通用漂移runner运行时校验此文件。 |
| `docs/NATURAL_DRIFT_P1_IMPLEMENTATION_v1.md` | 自然漂移实现说明 | 解释environment proxy、预测与边界。 |
| `docs/aemtn_model_lineage.md` | 模型谱系表 | 防止不同checkpoint和硬件任务被错误混用。 |
| `docs/天衍_全项目流程与聊天迁移总档.md` | 全工程时间线 | 把训练、IBM、天衍、自然漂移、B4旧端点和终测串成一条谱系。 |

## 12 包装与展示工具

| 文件 | 是什么 | 干什么用 |
|---|---|---|
| `b4ctl.py` | 统一控制入口 | 提供清单、流程、哈希验证、最终复算、测试、训练、数据生成和受保护真机入口。 |
| `figures/single_figures_main_20260831/make_b4_single_figures.py` | 主图v1生成器 | 从逐图source CSV生成12张独立科学图，并导出PNG/SVG/PDF/TIFF。 |
| `figures/single_figures_v2_diverse_20260831/make_b4_single_figures_v2.py` | 多样化主图v2生成器 | 用热图、森林图、雨云/分布、区间图等多种图型重建最终12图。 |
| `tools/build_release.py` | 可复现发布构建器 | 从原工程、冻结模型、派生证据和图表数据重建同结构审计ZIP。 |
| `tools/reproduce_public_final.py` | 公开终测复算器 | 从包内冻结pair数据重算硬件Session 0、模拟Session 1和hybrid统计量。 |
| `tools/verify_package.py` | 包完整性校验器 | 逐文件核对manifest中的大小与SHA256，发现缺失、篡改或多余内容。 |
