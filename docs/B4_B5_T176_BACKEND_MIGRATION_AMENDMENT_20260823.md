# B4/B5 注册 cadence supplement 后端迁移修正案：T287 至 T176

**修正案日期：** 2026-08-23  
**状态：** 软件、治理与 timing 证据迁移已完成；`READY_FOR_OPERATOR_HARDWARE_CONFIRMATION`。仅改变后端与时间模型，不改变统计设计。  
**适用对象：** `config/b4_cadence_pair_loop_cycle_paired_v4.json` 所定义的 B-5 受控注入 cadence-pair hardware supplement。  
**不适用对象：** T176 natural-drift corpus、B9 Stage-2 迁移预测、既有 2026-08-17 capability probe 的科学结论。  
**硬件动作：** 本修正案不授权、不触发、也不执行任何硬件提交。实际 runner 仍须显式给出 `--confirm-hardware`。

## 1. 修正理由与生效条件

在本修正案生效前，v2、v3、v4 名下注册 cadence collection 的 hardware job 数均为 **0**。因此这是首个注册 cadence job 之前的前瞻性协议修正，不是见到注册终点数据后的改写。该事实的现有审计叙述见：

- `C:\Users\Mercu\Desktop\aemtn\competition_xa202609\docs\B4_B5_T176_MIGRATION_HANDOFF_20260823.md`，§1；
- `C:\Users\Mercu\Desktop\aemtn\competition_xa202609\config\b4_cadence_pair_loop_cycle_paired_v4.json`，`collection_correction.amendment_scope`。

原注册后端 `tianyan-287` 持续维护，平台改配 T176 时段。若仍等待 T287，注册 collection 很可能无法在项目时限内发生。故唯一后端替代为：

```text
registered_backend_id = "tianyan176"
superseded_backend_id = "tianyan-287"
reason = "T287 maintenance; platform-provided replacement backend"
```

后端 pin 仍是硬约束：runner 必须把 loop config 声明的 `registered_backend_id` 与实际 backend config 的 `backend.backend_id` 比较；不一致即停止。不得把 pin 删除或退化为“任意天衍后端”。

本修正案必须与下列软件门、正式 timing ledger、迁移增补清单及其 hash 一并通过，才可提交首个注册 cadence job。任一运行时复核失败，状态立即回退为 **NOT READY FOR HARDWARE**。

## 2. 不变的注册设计

本次迁移保留完整 40 pairs，不降为 32、36 或 38 pairs：

| 项目 | 冻结值 |
|---|---:|
| sessions | 2；每天 1 session |
| cadence | fast `90 s`；slow `360 s` |
| cycles per cadence per session | 20 |
| cycle pairs per session | 20 |
| registered cycle pairs total | 40 |
| minimum adjudicated cycle pairs | 30 |
| sensing jobs per session | 40 |
| baseline jobs per session | 2（session start/end） |
| mirror QC jobs per session | 2 |
| jobs/settings per session | 44 / 88 |
| shots per session / total | 621,920 / 1,243,840 |

### 2.1 Shot allocation

| role | jobs/session | settings/job | shots/setting | shots/session |
|---|---:|---:|---:|---:|
| SENSE | 40 | 2 | 6,186 | 494,880 |
| BASELINE | 2 | 2 | 27,664 | 110,656 |
| MIRROR QC | 2 | 2 | 4,096 | 16,384 |
| **合计** | **44** | **88 settings** | — | **621,920** |

下列结构全部保持：两 session 使用相反起始 cadence；同一 `session_index`、`cycle_index` 的 fast/slow 成对；session-start baseline 跨两个 cadence block 共用；session-end baseline 只作 drift QC；每个 cadence block 的首 cycle 保留一个 mirror QC job；缺 cycle 只丢对应 pair，缺 session-start baseline 才丢整个 session pair block；禁止 optional stopping、outlier exclusion 与 gate-module 修改。

## 3. 统计冻结：零改写

本修正案是纯 backend/timing 修正。`PER_CONDITION_FLOOR_CONSTANT = 2.373` 已由 T176 2026-08-17 数据直接复核为 `2.3639`，比注册值低约 0.38%，方向保守；该复核只扩充 provenance，不改变常数。来源：

`E:\TianYan\XA-202609\quarantine\tianyan176\B4_TB5_amplified_closed_loop_20260817_run2\amplified_closed_loop_report.json`

该文件当前 SHA256：`45c446463e6a77684a0e29342090d14462d83c9e13e136b4b43933bf870a9629`。

以下注册统计字段必须逐值保留：

| 字段 | 冻结值 |
|---|---:|
| `registered_endpoint` | `cadence_ratio_gate applied to paired per-cycle endpoint residual squared` |
| `primary_adjudication` | `cadence_ratio_permutation_gate` |
| `expected_fast_endpoint_mean` | `0.001101791041193009` |
| `expected_slow_endpoint_mean` | `0.002174473622590627` |
| `expected_ratio` | `0.5066932197965022` |
| `pure_drift_ratio_limit` | `0.3708924335436652` |
| `endpoint_shot_floor_per_cycle` | `0.00038360814742967997` |
| `endpoint_shot_floor_shared` | `0.00008577935222672066` |
| `endpoint_shot_floor_total` | `0.0004693874996564006` |
| `preregistered_ratio_prediction` | `0.5070213621785998` |
| `preregistered_ratio_interval` | `[0.3260557361133296, 0.7884314361673062]` |
| `preregistered_ratio_interval_mass` | `0.95` |
| `minimum_power` | `0.8` |
| `expected_power` | `0.90155` |
| `measured_boundary_size` | `0.048075` |
| `maximum_boundary_size` | `0.0533` |
| `gate_module_change_permitted` | `false` |
| `outlier_exclusion_permitted` | `false` |

旧 reachability 绑定继续有效，不生成新统计证据、不换 hash：

```text
path = E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_V4_PREREGISTRATION_20260815\forty_minute_prediction_start_only_baseline.json
registered_sha256 = 98d4c9b923eedb14b35882086401c21ad6ecb40e4b470fe57805e5dc39f18488
```

任何 timing freeze 工具若在默认模式下改变上述任一字段或 `reachability_evidence_sha256`，必须 fail closed。只有显式 `--rewrite-statistics` 才可进入统计重注册路径；本修正案禁止使用该选项。

## 4. T176 timing-only 证据

### 4.1 目标与来源

24 个目标 task 的分组清单：

`C:\Users\Mercu\t176_20260817_query_ids.json`  
SHA256：`e211fbdd5ef9b54cba889ac0df054cb7d55a1d21138dad2c4f3c6d9f6691138b`

命中 24/24 的平台响应是 page 27：

`C:\Users\Mercu\.codex\attachments\5c22a464-2ec0-4723-8f6e-d8315b042c28\pasted-text.txt`  
原始输入 SHA256：`4cb1cd3d52b62a80762c9d1fc49974f7ce86a103006c0671ac5ca0d16c56c4b7`

page 27 的 100 条记录全部为 `天衍-176`；目标清单 24 条全部命中，目标中 T287 为 0。

相邻 page 28：

`C:\Users\Mercu\.codex\attachments\26ebacfe-6d37-4d38-82c5-41fdf0b64dd6\pasted-text.txt`  
原始输入 SHA256：`26afc0e1b41cfd261aa9bd82ade11ebcb1fc30da2a5180de0f385f7ec9d34187`

page 28 混有 48 条 T176 与 52 条 T287，但目标 task 命中数为 0；故整页排除，不进入正式 ledger、拟合或预算。后端选择只按 query-ID manifest 精确连接，不按日期、页码或设备名近似筛选。

### 4.2 B9-safe metadata allowlist

正式 ledger 只能保留目标 task，并只保留下列标准化字段：

```text
id, backend, runStartTime, finishTime, startTime,
status, shots, difference, role
```

其中 `backend`、`shots`、`role` 分别由平台字段/manifest 标准化。`graphResult`、`result`、`resultJson`、`qcis`、`inputCode`、counts、概率、bitstring、mirror score、field estimate、endpoint residual 及其他结果内容不得复制进 ledger，也不得在 Stage-2 前读取或分析。原始响应不复制到正式 analysis artifact；正式 artifact 仅记录原始输入绝对路径与 SHA256，并写出 target-only allowlisted metadata extract。

### 4.3 观测结果

平台时间为 Asia/Shanghai；runtime 定义为 `finishTime - runStartTime`，排除 `startTime - runStartTime` 队列等待。

| role | targets | shots/setting | `runStartTime` | `finishTime` | runtime/task |
|---|---:|---:|---|---|---:|
| BASELINE | 2 | 27,664 | `2026-08-17 19:16:49.211` | `2026-08-17 19:17:07.946` | `18.735 s` |
| SENSE | 2 | 6,186 | `2026-08-17 19:17:45.920` | `2026-08-17 19:17:49.476` | `3.556 s` |
| MIRROR | 20 | 16,384 | `2026-08-17 19:18:32.337` | `2026-08-17 19:19:44.394` | `72.057 s` |

同一 job 内每个 setting 的 `runStartTime` 与 `finishTime` 完全相同。20-setting MIRROR 的 task-runtime 求和为 `20 × 72.057 = 1,441.140 s`，但整个 batch 的执行墙钟只有 `72.057 s`，且含队列的既有 job roundtrip 也只有 `133.4204007 s`。因此 settings 是 batch/parallel execution，不是串行 execution；`settings × fixed_overhead_seconds_per_setting` 模型失效。

## 5. 时间模型裁决与 40-pair 可达性

### 5.1 模型边界

不得再用一个通用 `shots/rate + settings×overhead` 模型跨 BASELINE、SENSE、MIRROR 三种 circuit role。MIRROR circuit 深度、任务类型、setting 数同时变化，不能用于 shot-rate 回归。

只在同为两-setting sensing family 的 BASELINE 与 SENSE 两点上，队列外斜率为：

```text
R_sensing = (27,664 - 6,186) / (18.735 - 3.556)
          = 1,414.98122405956 shots/s
intercept = -0.815789458981282 s/job
```

负 intercept 没有物理意义，且只有两个点；迁移 timing correction 采用 `overhead = 0 s`，不得把负值当作时间返还。`R_sensing` 只适用于该两-setting sensing family，不得乘总 settings、不得用于 MIRROR、不得声称是平台通用吞吐。

注册预算采用直接观测的 role envelope；对生产 MIRROR QC 的两个 2-setting jobs，保守地各充入整个 20-setting probe batch 的 `72.057 s`：

```text
wall/session
  = 40 × 3.556 + 2 × 18.735 + 2 × 72.057
  = 323.824 s

quota-upper-bound/session
  = 2 × wall/session
  = 647.648 s

quota-upper-bound/two sessions
  = 2 × 647.648
  = 1,295.296 s < 2,400 s
```

`wall/session` 按同一 job 的 settings 共享执行区间计费。`quota-upper-bound/session` 另取更保守假设：平台即使把每个两-setting job 的两个 task runtime 完整相加，也只收取两倍。两种解释都低于 `daily_window_seconds = 1,200`；两天保守总额也低于 `machine_time_ceiling_seconds = 2,400`。因此 full 40-pair design 保留，时长不再构成减 pair 理由。

该结论不依赖用户回忆的 quota decrement 数值。若后续获得准确扣减，只用于辨认平台计费口径，不得反向改变 endpoint、shot allocation、pair count 或统计字段。

## 6. B9 隔离边界裁决

### 6.1 原文证据

权威原文为：

1. `C:\Users\Mercu\Desktop\aemtn\competition_xa202609\docs\B4_WORK_ORDER_20260804.md:38` 明确把 T176 设计为“数据密封采集、两段式预注册后解封确认”。这同时说明 **T176 采集被允许**，否则不存在“密封采集”。
2. 同文件 `:226-227` 明确：“T176 原始数据落 `quarantine\tianyan176\`，落地即哈希登记，解封前禁止任何读取分析”；Stage-2 必须在 T287 分析完成后、T176 解封前写出并提交数值预测。
3. 同文件 `:240` 明确计划 T176 session；故禁令不是“Stage-2 前不得向 T176 提交任何 job”，而是“采集后结果必须密封，Stage-2 前不得结果分析”。
4. `C:\Users\Mercu\Desktop\aemtn\competition_xa202609\docs\B4_TB8_THREE_DAY_RUNBOOK_DRAFT_20260806.md:38` 再次规定 T176 原始 counts 密封；`:40-47` 同时把实际 timestamp、setting duration、lag 列入每日 operational QC。这给出 counts 与 timing metadata 的原始协议区分。

据此裁决：

- **允许：** 在本修正案与软件/哈希门全部生效后，按已冻结计划在 T176 采集注册 cadence supplement；读取 §4.2 allowlist 内 timing metadata，完成 task 身份、后端、shots、完成状态、队列外 runtime、并行性与配额上界审计。
- **必须密封：** T176 counts、概率、bitstring、mirror score、field estimate、shield 后的 count-derived values、endpoint residual、pair statistic、permutation verdict、prediction hit/miss 及任何科学结果。runner 为执行冻结闭环而进行的最小在线 count transform 可以发生，但所有输入和派生输出均直接落 quarantine；Stage-2 前不得人工检查、汇总、作图、调参或裁定。
- **禁止：** 把 T176 timing ledger 当自然漂移证据；把 2026-08-17 probe 或新 cadence counts 混入 T287 SF、sensing-economics map、cadence residual curve；在看到 T176 结果后改阈值、pair count、shot allocation、排除规则或代码。

本裁决解决的是 collection permission 与 metadata handling，不提前解封 T176 科学结果。

### 6.2 2026-08-17 probe 的角色保持

既有目录：

`E:\TianYan\XA-202609\quarantine\tianyan176\B4_TB5_amplified_closed_loop_20260817_run2\`

其冻结角色继续为：

```text
registered_endpoint_contribution = "none"
pooling_permitted = false
not_main_endpoint = true
capability_probe_only = true
```

本修正案只复用其 allowlisted task timing 和已登记的 backend-floor capability provenance；不把其 counts、20-setting mirror 曲线、suppression factor 或其他科学结果加入注册 cadence endpoint。它产生的 registered cadence pair 数仍为 0。

## 7. 固定输出路径与解封顺序

### 7.1 迁移前证据

| artifact | 固定路径 | SHA256 状态 |
|---|---|---|
| amendment | `C:\Users\Mercu\Desktop\aemtn\competition_xa202609\docs\B4_B5_T176_BACKEND_MIGRATION_AMENDMENT_20260823.md` | 提交/冻结时生成 |
| timing ledger | `E:\TianYan\XA-202609\artifacts\analysis\B4_T176_PLATFORM_TASK_TIME_LEDGER_20260823_r1\platform_task_time_ledger.json` | `D0089FE19A96DC8B0D95A07B53F4DE34A64AEE196BBBC21979005D9A7D673F10` |
| embedded timing analysis | 同一 ledger 的 `t176_timing_analysis` | `66AC2730A6FC2B088AAA139698C9CF0E8EF4C7902A4EB34C2A0F47AD97A7475B` |
| timing freeze manifest | `E:\TianYan\XA-202609\artifacts\analysis\B4_T176_PLATFORM_TASK_TIME_LEDGER_20260823_r1\freeze_manifest.json` | `8432EE328669898EA45D7958DCBD36C22C0C2B5FA2E9A96965BD4C92BE1DBEEC` |
| timing freeze sidecar | `E:\TianYan\XA-202609\artifacts\analysis\B4_T176_PLATFORM_TASK_TIME_LEDGER_20260823_r1\freeze_manifest.sha256` | 内容精确为 `8432EE328669898EA45D7958DCBD36C22C0C2B5FA2E9A96965BD4C92BE1DBEEC` |
| timing-only migration evidence | `E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_V4_T176_TIMING_MIGRATION_20260823\backend_migration_timing_evidence.json` | `F56B8B363914C7C50BA41DACC9B5DD250DDC158F98522AF554F4D6F2BC4B7F3F` |
| migration curated manifest | `C:\Users\Mercu\Desktop\aemtn\competition_xa202609\docs\B4_T176_MIGRATION_CURATED_MANIFEST_20260823.json` | 由 runner/preflight 在运行时逐文件验证；manifest 自身 hash 写入 plan/report |
| registered reachability | `E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_V4_PREREGISTRATION_20260815\forty_minute_prediction_start_only_baseline.json` | 注册 canonical-LF SHA `98D4C9B923EEDB14B35882086401C21AD6ECB40E4B470FE57805E5DC39F18488`，保持 |

注册 reachability 文件当前 Windows 存储字节使用 CRLF，storage SHA 为 `7FB70B8636E8B7BD66FDCFB172747F8ED8D1C08E3A55A70E3CA3D6CB7B714001`；仅将 CRLF 规范化为 LF 后，SHA 精确恢复注册值 `98D4...F18488`。preflight 只读核验此条件，记录 `crlf_storage_canonical_lf_match` 与 `statistical_content_drift_detected=false`，不改原 artifact；任何非换行差异均 fail closed。

### 7.2 新注册 collection quarantine

runner 的固定 `--output` 为：

`E:\TianYan\XA-202609\quarantine\tianyan176\B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823\`

目录必须新建且拒绝覆盖。计划、journal、raw query、counts、count-derived cycle rows 与最终未解封 report 均留在该目录；不得复制到 `artifacts\analysis`，直到 §7.3 解封门通过。

### 7.3 解封与分析顺序

严格顺序：

1. 完成并冻结 T287 SF；
2. 完成并冻结 sensing-economics map；
3. 完成并冻结 cadence residual curve；
4. 在未读取 T176 counts/结果的条件下，用冻结管线与 T287 后验写出 T176 Stage-2 数值预测（verdict 与区间），提交 git 并登记 SHA256；
5. 审计预测提交时间早于 T176 解封时间；
6. 解封 `E:\TianYan\XA-202609\quarantine\tianyan176\B4_CADENCE_PAIR_V4_T176_REGISTERED_20260823\`；
7. 用已冻结 analyzer 一次运行，不得基于结果迭代；
8. 非覆盖式写入 `E:\TianYan\XA-202609\artifacts\analysis\B4_CADENCE_PAIR_V4_T176_ADJUDICATION_20260823\`；
9. 最后才做复制判定与报告装配。

若 T287 前置分析因维护或缺数据未完成，T176 collection 可继续密封保存，但不得跳过第 1-5 步解封。

## 8. 首 job 前软件硬门

迁移实现必须完成并测试以下缺陷修复：

1. `scripts/run_b4_cadence_pair_hardware.py` 的 plan/report 写实际 `backend_config["backend"]["backend_id"]`；删除报告中的 T287 字面 provenance。
2. analyzer 接收独立 `expected_backend_id` 并与 artifact 比较；不得读取 artifact 自己声明的 backend 再自证。
3. backend pin 从 loop config 的注册字段读取；实际 backend 不一致即停止。
4. plan `source_hashes` 加入调用者实际 `backend_config`、`loop_config` 路径的 SHA256；reuse guard 同时验证二者，防止 T287 plan 被 T176 静默复用。
5. `scripts/preflight_b4_cadence_collection_correction.py` 默认指向 v4，不再指向 v2；READY status 与 quarantine 字段按实际 backend/阶段生成，不硬编码 T287/T176 假值。
6. `scripts/freeze_b4_cadence_v4_start_only_baseline.py` 增加 `--backend-config`；timing 与 statistics write-back 分离；`--rewrite-statistics` 默认 false；默认路径若任一统计字段变化即 fail closed。
7. backend timing provenance 明记：per-setting overhead 仅由 settings=2 的 T-B6 jobs 推出，未识别 per-job/per-setting 分解；T176 20-setting 同起止已证伪串行 per-setting 成本模型。
8. `scripts/build_b4_platform_time_ledger.py` 支持多页 JSON/HAR、query-ID manifest、24/24 完整性检查、T176-only backend assertion、target-only metadata allowlist、page provenance、queue-free runtime、batch parallelism 与预算复算；正式输出拒绝覆盖并可由 freeze manifest 重放验证。
9. runner、smoke 与 probes 的 `--confirm-hardware` 保持显式必需；任何 preflight、freeze、ledger、test 命令都不得隐式调用 hardware API。
10. 全部相关测试通过；生成首 job plan 时再次验证：40 pairs、2 sessions、90/360 cadence、shot allocation、统计字段与旧 reachability hash 逐值不变。

实施复核（2026-08-23）：上述十项均已落地；ledger/freeze 可重放验证；runner、preflight、analyzer、timing freeze、B9 audit 与设计合同合计 `111 passed`。preflight 使用 timing-only plan 构造，不重跑或重写注册 Monte Carlo；正常 runner 的 reachability 安全检查仍默认开启。

## 9. 最终治理结论

1. T287 维护构成后端替代的外部操作原因；由于迁移前注册 cadence job 为 0，本修正案可在首 job 前前瞻生效。
2. 后端改为且仅改为 `tianyan176`。完整 40-pair、两日、90/360 cadence、shot allocation、endpoint、阈值、预测区间、power、boundary size 与旧 reachability hash 全部保留。
3. 24-task timing-only 证据足以推翻串行 per-setting 成本模型。role envelope 为 `323.824 s/session`；最保守 task-runtime-sum quota 上界为 `647.648 s/session`，两天 `1,295.296 s < 2,400 s`。无需减 pair。
4. B9 原文允许 T176 密封采集；禁止的是 Stage-2 前读取/分析 T176 counts 与科学结果。§4.2 timing metadata allowlist 可用于操作审计，不构成解封。
5. 正式 ledger、embedded timing analysis、freeze manifest 与 timing-only migration evidence 均已生成、复核并回填；迁移增补清单钉住运行时软件与证据字节。当前状态仅为 **READY_FOR_OPERATOR_HARDWARE_CONFIRMATION**，不是硬件已提交；runner 仍要求操作者显式提供 `--confirm-hardware`。
