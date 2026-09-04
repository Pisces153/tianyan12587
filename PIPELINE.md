# 闭环工程链

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
