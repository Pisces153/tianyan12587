# B4 v2 独立主图索引（多图形语言版）

本目录不覆盖 v1。12 张图一图一问，仅 F09 因累计顺序诊断保留折线/阶梯线；其余使用热图、相空间散点、森林图、区间条、配对散点、发散棒棒糖、置换直方图、哑铃图和流程图。

| 图 | 图形类型 | 数据表 | 证据起源 |
|---:|---|---|---|
| 01 | temporal heatmap | `01_t287_readout_state_heatmap.source.csv` / `F01_ReadoutHeatmap` | T287 hardware |
| 02 | uncertainty-aware phase-space scatter | `02_t287_effective_field_phase_space.source.csv` / `F02_PhaseSpace` | T287 hardware |
| 03 | horizontal dot-whisker / forest plot | `03_t287_e0_negative_control_forest.source.csv` / `F03_E0Forest` | T287 hardware |
| 04 | horizontal dot-whisker / forest plot | `04_t287_e1_drift_forest.source.csv` / `F04_E1Forest` | T287 hardware |
| 05 | three-row residual heatmap | `05_t287_e1_interval_residual_heatmap.source.csv` / `F05_ResidualMap` | T287 hardware-derived model |
| 06 | log-scale interval strip / calibration clock | `06_t287_control_floor_interval_strip.source.csv` / `F06_ControlClock` | T287 hardware economics |
| 07 | paired log-log scatter | `07_t176_hardware_paired_scatter.source.csv` / `F07_PairedScatter` | T176 hardware Session 0 only |
| 08 | sorted diverging lollipop | `08_t176_pair_benefit_lollipop.source.csv` / `F08_BenefitRank` | T176 hardware Session 0 |
| 09 | cumulative step diagnostic | `09_t176_cumulative_ratio_step.source.csv` / `F09_Cumulative` | T176 hardware Session 0 |
| 10 | permutation histogram | `10_t176_permutation_histogram.source.csv` / `F10_Permutation` | T176 hardware Session 0 |
| 11 | scenario dumbbell | `11_baseline_sensitivity_dumbbell.source.csv` / `F11_Sensitivity` | simulation-assisted / post-hoc sensitivity |
| 12 | unit-aware process flow | `12_t176_hardware_workload_flow.source.csv` / `F12_WorkloadFlow` | T176 hardware Session 0 engineering accounting |

## 严格边界

- T287 和 T176 证据不混为“双真机闭环复现”。
- T176 仅 Hardware Session 0；Hardware Session 1 缺失，不宣称注册 all-hardware PASS。
- F05/F06 的 T* 上界碰到观测窗，因此为 INCONCLUSIVE。
- F11 是模拟辅助/事后敏感性，不是第二次真机会话。
- 所有源 CSV 包含图中使用的派生编码列；完整数据还将汇总到单一 Excel 数据簿。
