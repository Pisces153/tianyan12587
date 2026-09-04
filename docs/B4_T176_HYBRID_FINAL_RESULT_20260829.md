# B4/T176 模拟辅助终测结果

**日期：** 2026-08-29  
**分析标签：** `POST_HOC_HYBRID_SIMULATION_ASSISTED`  
**项目层状态：** `B4_PRESERVED_SIMULATION_ASSISTED`  
**注册纯真机状态：** `INCONCLUSIVE_MISSING_HARDWARE_SESSION1`  
**验证状态：** `VERIFIED`；确定性复跑签名完全一致  
**信心等级：** `CAUTION`

## 1. 结论边界

B4 在明确标注的后验模拟辅助一致性分析中得到保留。分析结合：

- T176 真机 Session 0：20 个完整 cadence pairs；
- 独立冻结的模拟 Session 1：20 个完整 cadence pairs；
- 总 hybrid 分析单元：40 pairs。

该结果不能写成 `registered all-hardware PASS`。v4 注册设计要求两个相反起始顺序的真机 Session，并禁止模拟 pair 进入纯真机裁决；真机 Session 1 未采集。因此，项目层 simulation-assisted 状态与注册纯真机状态必须始终并列报告。

推荐正文：

> B4 在明确标注的后验模拟辅助一致性分析中得到保留。该分析结合了 20 对已完成的 T176 真机 Session 0 数据与 20 对独立冻结的 Session 1 模拟应急数据；由于真机 Session 1 未能采集，预注册的纯真机端点仍保持未决状态。

## 2. 冻结终测结果

| 证据层 | pairs | fast/slow ratio | 相对降低 | 20,000 次配对置换 p | 角色 |
|---|---:|---:|---:|---:|---|
| T176 真机 Session 0 | 20 | 0.361649790 | 63.84% | 0.005249738 | 强描述性真机支持 |
| 模拟 Session 1 | 20 | 0.381652296 | 61.83% | 0.002349883 | 模型限定反事实支持 |
| Hybrid | 40 | **0.374481312** | **62.55%** | **0.000099995** | 后验模拟辅助一致性终测 |

Hybrid 冻结临界比为 `0.635759870`；实测比值更低。次级 delta-method 区间为 `[0.210908744, 0.538053881]`。原 v4 比值预测区间 `[0.326055736, 0.788431436]` 命中；预测点绝对误差为 `0.132540050`。

来源分层方向一致，无 Simpson reversal：真机、模拟、hybrid 三层均显示 fast cadence 残差低于 slow cadence。模拟没有把方向相反的真机结果“救”为通过。

## 3. 完整性与复现

- 真机 Session 0：40 cycles、44 jobs、88 tasks、2 baselines、20 完整 pairs。
- 真机 query ID：88 个唯一值；重复数 0。
- 真机 journal：219 行；SHA 链逐条验证通过。
- 模拟 Session 1：40 cycles、20 完整 pairs、43 行独立记录链、硬件提交数 0。
- 原真机 manifest、plan、journal 未修改。
- 分析未读取 raw counts、NPZ 或 raw query 科学结果；只使用冻结白名单中的派生 endpoint 字段与两条基线记录。
- 三种注册漂移形状敏感性均与主结论一致。
- 统计谬误审计：11/11 checked；0 RED_FLAG；3 CAUTION。
- 相关回归：105 passed。
- 原分析签名与复跑签名均为 `1fe980cff5cb2db4be7424234f3ece18424135fa322598d850a042fd79943828`。

三项 CAUTION 来自同一边界：hybrid 方案为后验补救；模拟 Session 1 结果在 hybrid plan 冻结前已知；证据不能扩张成双 Session 真机因果确认。

## 4. v1 兼容性停止与 v2 erratum

首次 v1 plan 在读取 journal 后，于第 2 行 SHA 链检查安全停止。原因仅为十六进制 SHA 字符串大小写差异：原硬件 journal 使用大写，模拟 journal 使用小写；哈希内容一致。v1 未计算或输出科学统计。

v2 erratum 只把 SHA 比较改为大小写等价；端点、判据、seed、置换次数、预测区间、来源和 pair 数均未改变。旧 v1 plan 保留为失败证据；v2 plan 重新冻结后才运行终测。

## 5. 权威工件

机器可读路径与 SHA256 汇总见 `docs/B4_T176_HYBRID_FINAL_MANIFEST_20260829.json`。

- 完整验证报告：`E:\TianYan\XA-202609\artifacts\analysis\B4_T176_HYBRID_FINAL_20260829\B4_T176_HYBRID_FINAL_VALIDATION.md`
- 结构化结果：`E:\TianYan\XA-202609\artifacts\analysis\B4_T176_HYBRID_FINAL_20260829\hybrid_final_report.json`
- 40 对来源标记明细：`E:\TianYan\XA-202609\artifacts\analysis\B4_T176_HYBRID_FINAL_20260829\hybrid_pair_rows.csv`
- 复跑凭证：`E:\TianYan\XA-202609\artifacts\analysis\B4_T176_HYBRID_FINAL_20260829\reproducibility_verification.json`
- 冻结 v2 plan：`E:\TianYan\XA-202609\artifacts\analysis\B4_T176_HYBRID_FINAL_20260829_v2.plan.json`
- 保留 v1 plan：`E:\TianYan\XA-202609\artifacts\analysis\B4_T176_HYBRID_FINAL_20260829.plan.json`
- 分析器：`scripts/analyze_b4_t176_hybrid_final.py`
- 专门测试：`tests/test_analyze_b4_t176_hybrid_final.py`

## 6. 后续升级条件

若以后取得真实 Session 1，只新增全硬件报告版本；不删除或覆盖本模拟应急证据。真实 Session 1 应使用原同构计划，随后分别比较方向、预测区间命中、模型偏差与 calibration 前后 backend regime。只有两场真机满足原注册规则，注册状态才可从 `INCONCLUSIVE_MISSING_HARDWARE_SESSION1` 重新裁决。

## 7. 安全收口

本轮使用的平台凭据曾出现在聊天记录中；现有工件没有发现凭据落盘。终测完成后应在平台撤销或轮换该凭据。当前交接材料没有“已经轮换”的可核验证据，因此状态保持 `ROTATION_CONFIRMATION_PENDING`。任何后续真机运行必须使用新凭据，并只通过进程环境注入。
