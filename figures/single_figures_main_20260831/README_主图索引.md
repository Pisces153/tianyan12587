# B4 主报告独立单图索引

本目录是 B4 成果展示的主报告图组。每个画布只回答一个问题，不使用四联或多面板拼图。每张图均同时提供：

- `PNG`：汇报、Word、网页直接使用；
- `SVG`：可编辑矢量版本；
- `PDF`：印刷与论文排版版本；
- `TIFF`：600 dpi 高分辨率投稿版本；
- 同编号 `source.csv`：逐点源数据；部分图另有 `summary.csv`；
- `collision-audit.json/pdf`：最终 PDF 的文本碰撞与裁切质检记录。

## 推荐报告顺序

### 第一层：环境状态可观测

1. `01_t287_readout_proxy_timeseries`：78 个 T287 真机快照中的 E0/E1 读出状态及 shot-noise floor。
2. `02_t287_effective_field_state`：三时点 Y/Z 反演得到的 H1/H2 有效场状态。

这一层只回答“云 API 用户能否从量子测量看到随时间变化的状态”。H1/H2 是量子测量代理，不冒充温度、电磁传感器或底层脉冲参数。

### 第二层：区分普通测量波动与可检出漂移

3. `03_t287_e0_negative_control_structure_function`：E0 阴性对照，n=30，p=0.6497，漂移门不触发。
4. `04_t287_e1_drift_structure_function`：E1 信号通道，n=30，p=2.79e-289，检测到超过 shot-noise 的过程方差。

这一层给出阴性对照与阳性信号的成对证据，避免把所有时间波动都解释为漂移。

### 第三层：从漂移检出进入校准更新间隔经济学

5. `05_t287_e1_interval_economics`：E1 残差—更新间隔曲线；OU 点估计 T*=134 s，但 bootstrap 95% CI 为 101–4000 s。
6. `06_t287_tstar_vs_control_floor`：API P50=12.4 s、P90=14.7 s、协议 floor=60 s 与 T* 置信区间的直接比较。

这一层支持“能够计算更新间隔经济曲线”，但当前冻结结论仍是 `INCONCLUSIVE`，不能把 134 s 当作已识别、可部署的唯一最优策略。

### 第四层：T176 真机 Session 0 闭环证据

7. `07_t176_hardware_pair_slopegraph`：20 个完整真机 pairs 的 slow/fast 逐对比较；14/20 改善。
8. `08_t176_pair_benefit_distribution`：逐 pair 的 log2(slow/fast)；完整展示 14 个正向和 6 个反向 pair。
9. `09_t176_cumulative_ratio`：冻结顺序下累计比值；最终 ratio=0.36165，critical=0.49341，p=0.0052。
10. `10_t176_hardware_permutation_null`：20,000 次冻结 pair 内标签交换的完整零分布。

这一层允许声称：T176 真机 Session 0 存在强、pair-complete 的描述性 cadence 收益。它不允许声称注册的全真机双 Session 端点已经通过。

### 第五层：稳健性与工作量边界

11. `11_baseline_drift_sensitivity`：primary hybrid 与三种预声明漂移形状敏感性均位于各自临界线左侧；证据明确标记为 simulation-assisted / post-hoc。
12. `12_t176_hardware_workload_flow`：44 真机 jobs、88 平台 tasks、40 闭环 cycles、20 完整 pairs、0 exclusions。

这一层说明结果并非由删 pair 获得，同时把缺口写清楚：hardware Session 1 尚未采集，模拟不补写真机工作量。

## 主图可支持的结论

- 在受限云 API 用户位，可以通过量子测量构造随时间变化的可观测环境状态；
- 阴性对照 E0 不触发漂移门，E1 检测到超过 shot-noise 的过程方差；
- 可以建立残差—更新间隔曲线，并将最优点与 API/协议控制 floor 放在同一经济框架中；
- T176 真机 Session 0 的 20 对 cadence 端点完整、0 exclusions，最终均值比为 0.36165；
- 冻结 pair 内置换检验可由 20,000 个零分布样本复核；
- 漂移形状敏感性支持 simulation-assisted 稳健性，但不替代缺失的 hardware Session 1。

## 主图不能支持的结论

- “注册全真机双 Session 端点 PASS”；
- “已经完成硬件 Session 1 采集”；
- “已经部署在线强化学习”；
- “已经学习出跨设备迁移模型”；
- “已经实现通用脉冲级自动标定”；
- “T*=134 s 已被高置信识别并可直接作为生产策略”。

## 复现入口

- 生成脚本：`make_b4_single_figures.py`
- 输入、输出和 SHA-256：`single_figure_manifest.json`
- 自动与人工质检摘要：`QA_REPORT.md`
- 可直接粘贴到报告的逐图图注：`逐图中文图注.md`

