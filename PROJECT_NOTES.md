# PROJECT_NOTES — AEMTN 硬件线（XA-202609 / 天衍）

生成日期：2026-08-23；最新状态增补：2026-08-29
用途：Codex 账号从中转站（cubence）切到个人会员账号后，旧会话无法续聊（`encrypted_content` 只能由原后端解密）。本文件把旧会话里跟本项目相关的上下文沉淀进版本库，使新账号不依赖任何聊天历史即可接手。

本文件由直接读取 `C:\Users\Mercu\.codex\sessions\**\*.jsonl` 原始 rollout 汇总而成，不依赖 Codex 会话能否续聊。

---

## 0. 先读哪几份文件

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | 本文件 | 索引 + 旧会话里未落盘的坑 |
| 2 | `C:\Users\Mercu\Documents\天衍\天衍_全项目流程与聊天迁移总档.md` | 原 34 条历史会话 + B4 十七、十八轮增补的时间线、实验结论、artifact 路径。**权威主档** |
| 3 | `C:\Users\Mercu\Documents\天衍\HANDOFF.md` | 最近一次工作交接 |
| 4 | `C:\Users\Mercu\Documents\天衍\XA-202609 真机验证与参赛执行计划 V2.0.md` | 两周冲刺执行计划，14 张逐日任务卡、G0/G1/G2 放行条件 |
| 5 | `docs/aemtn_model_lineage.md`（本仓，未提交） | 模型谱系 |

**总档的覆盖边界**：原档从 Codex 项目任务列表构建，初版只覆盖注册在「天衍」项目下的 34 条会话；2026-08-29 已补 B4 十七、十八轮状态。经逐条比对 `sessions/` 目录，另有 8 条 cwd 为 `Documents\天衍` 的会话不在原索引内（`019fa1a0`、`019fa1a2`、`019fa1a3-16e8`、`019fa1a3-992b`、`019fa7b9`、`01a029b2`、`01a02a4b`、`01a02a5a`）。本文件第 3 节补的就是这批会话里、总档没有的内容。

---

## 1. 项目现状（停点）

- 主模型训练完成。留出集：`h1 R²=0.9653`、`h2 R²=0.9530`、`Jz R²=0.1478`。
- IBM 2-step 硬件验证完成，`xobs6` 平均绝对差 `0.2554 → 0.1113`。
- 天衍 Stage A 本地同协议仿真通过；远端 Stage A 历史记录口径冲突，**必须用 raw artifact 复核后才能声明真机通过**。
- B4/T176 真机 Session 0 已完成；独立模拟 Session 1 与后验 hybrid 终测也已完成。项目层为 `B4_PRESERVED_SIMULATION_ASSISTED`；注册纯真机端点因缺失真机 Session 1 仍为 `INCONCLUSIVE_MISSING_HARDWARE_SESSION1`。
- 截至 2026-08-29，当前 HEAD：`a59d24b docs: add PROJECT_NOTES.md for Codex account migration`。
- 工作树高度不干净，包含大量早期未跟踪文件与本轮 B4 分析器、测试和结果交接文件。不要批量 `git add .`；提交前必须按任务逐文件审计、凭据扫描和精选 staging。

## 2. 已确立的结论与红线

- h1/h2 闭环能力链有证据，但**不能**包装成全面 superiority、完整 RL 或强跨设备泛化。
- `Jz` 是弱可辨识参数，不能自动补偿。写论文/报告时必须带 Cramér–Rao 界的说明。
- T7 旧事件定义已被证伪：阈值 `0.02` 低于噪声底，四状态 OR 使事件近乎恒真。
- T7 v3 readout-only 清洗后仍无足够预报技巧证据；T7b/T8 强在线调度路径未解锁。
- B4/B5 根因：功效层用逐 cycle `residual²`，旧真机主端点用带大加性底座的 `raw mirror loss`，旧端点被稀释约 `22×`，结论 `NO-GO`。
- map 正式 headline 改为 `INCONCLUSIVE`，不再触发档位下降。
- 旧 B5 数据回收 `420/420 tasks`、`42 jobs`、`21 cycles`、`24 pairs`；旧 raw mirror 端点没有确认完整档 4，该历史结果继续保留。
- 新 B4/T176 结果只可写为 simulation-assisted preserved；不得偷换成双 Session 真机注册通过。
- B4 新证据不自动重评 AEMTN 其他线或总体竞赛档位；整体档位需另做跨模块综合裁决。

## 3. 代码仓审计结论（2026-07-27，总档未收录，优先级最高）

在写 V2.0 计划书前对 `C:\Users\Mercu\Desktop\aemtn` 做过一次逐行只读审计，结论是**论文叙述与代码实现存在结构性冲突**，被列为 G0 前置阻断项。这批结论没有进总档，也没有进任何 `docs/`，只存在于会话 `019fa1a0` 里，所以在此完整保留。

### 3.1 模型输入不是论文说的 `xobs`

| 项 | 代码事实 | 论文说法 |
|---|---|---|
| `x` | 代码中**没有** `xobs` 变量。实际 `x` = `Re(rho_init)` 的 4096 个 C-order 元素 + `Im(rho_init)` 的 4096 个元素，共 **8192 维**（`generator_node.py:225`）。训练器动态读维数（`trainer_active.py:64`），模型直接 `Linear(x_dim, r_dim)` 接收（`testmodel1.py:27`）。现存 5 个 checkpoint 首层均为 `(320, 8192)` | 论文称 `xobs` 是演化**后**约化态的局域观测输入 |
| 唯一明确的局域观测向量 | 按代码字段拼接恰为 **6 维**：`[<X0>, <Y0>, <Z0>, <X0X1>, <Y0Y1>, <Z0Z1>]`。算符与顺序见 `generator_node.py:41`，约化站点固定 `[0,1]` 见同文件 233 行，写盘顺序见 289、305 行。**这 6 项是监督输出标签，不是输入** | 论文写了所有 `α,β∈{x,y,z}`，且方向与代码相反 |
| 初态 | 5 个生成器均逐样本生成 6 个随机单比特纯态的直积态，加 `0.03 * rand_dm([2]*6)` 后 `.unit()`（`generator_node.py:203`）。不是固定态、不是基态、不是标准 depolarizing mixture | 论文称哈密顿量基态加 `p` 比例最大混态 |
| 边界条件 | 开边界 OBC，有向键 `(0,1),(1,2),(2,3),(3,4),(4,5)` | — |

**后果**：现有 checkpoint 不能直接接收真机 counts 或 6 维 `xobs`。G0 之前必须重构输入协议并重训。任何材料都不能写「现有 AEMTN 已可直接接收真机 `xobs`」。

### 3.2 三个 G0 阻断项

1. **标签泄漏**：`V` 的 14 维顺序为 `[t,gamma,h1,h2,Jz,Jx,Jy,Jxz,Jzx,D,hy1,hy2,hz1,hz2]`（`generator_node.py:56`），而训练时又把 `h1/h2/Jz` 作为预测目标（`trainer_active.py:207`）。
2. **没有 shot 采样**：仓内无 Born sampling、无 1024 shots、无 counts 恢复实现，代码直接算精确期望值。论文的 1024-shot 声明无代码支撑。
3. **训练入口指向错的生成器**：`auto_pilot.py:63` 调用的是振幅阻尼生成器，不是 `generator_node.py`。

### 3.3 生成器变体差异

仓内共 5 个 `generator*.py`。`generator_transverse_ising.py:84`：`H = -Jz ΣZZ - Σhx X`，`D=0`，仍用双向升降算符噪声。`generator_xxz.py:84`：`D=0`，**ZZ 实际系数是 `Jx*Jz/(abs(Jx)+1e-6)`**，`Jx<0` 时约为 `-Jz`，不能直接称 ZZ 系数恒为 `Jz`。

### 3.4 比赛协议的冻结决策

执行协议明确改为 **Néel `|010101>` + quench** 固定初态。这是新实现，**不能写成现有代码行为**。

## 4. 真机实验规程（V2.0 计划书收编的硬规范）

- 测量基旋转：X 基测量前加 `H`；Y 基测量前加 `S†H`；Z 基直接测。
- 两体期望由奇偶校验恢复：`<P⊗Q> = (n00+n11-n01-n10)/Nshots`；单体期望从同一联合 counts 边缘化。
- **原始 counts 永久保留**，缓解结果另存，禁止覆盖原始文件。raw / mitigated 双轨都要能重放。
- 用 `|00>`、`|++>`、Bell 态检查符号、端序、基旋转和边缘化。
- 数据隔离：`master_manifest.csv` 在 Day 2 前冻结；`H-ADAPT` 只能用于训练/校准/选策略，`H-TEST-*`/`H-DYN`/`H-UNI`/`H-ANCHOR` 不参与参数更新。失败样本不得直接删除，必须记录原因、重试次数和处置状态。
- 各集合参数哈希交集必须为 0；INJECT 配对靠 `pair_id` 显式表达，不能靠相邻行推断。
- shots 预算公式（可审计）：`N_sample = 230 + 25S + 10D`，`N_shots = 9216 × N_sample`，其中 `D` = 实际真机访问日数，`S∈{2,3}` = SCHED 子图数。**「约 300 万 shots」只在 D≈2–5 时成立**，长期 DRIFT 会线性增加（每个访问日 +10 样本 / +90 线路 / +92,160 shots）。
- 真机占用时长**不得虚构**天衍-287 速率，给公式 + 100/500/1000 shots/s 敏感性表，用 G1 的 92,160 shots 实测回填。
- G1 定义：10 样本、90 条线路、92,160 shots，10/10 均需具备 manifest、task_id、raw counts、xobs、预测和指标；每条线路 counts 总和必须等于 1024。
- G2：核对「计划/提交/成功/失败/重试/封板」六套计数；从 raw counts 重跑一次 `xobs→预测→指标`，与封板结果在设定浮点容差内一致。
- INJECT 必须**预先**冻结主指标、改善方向和 15 对配对检验方法，不能看到结果再选展示指标。
- 「推理约 3 ms/样本」必须记录设备、批大小、预热次数、重复次数和 P50/P95，不能沿用论文数字当比赛机器实测值。
- 术语：`γ/gamma` 只能作为模拟参数，真机字段一律写 `native hardware noise`。

## 5. 主训练完成后的既定顺序（不要跳步上真机）

1. 验收并冻结预训练基线：`sim_holdout` 上 3 个 seed 独立评估；确认输入层为 `(320,6)`、控制分支只接收 `t`、无 NaN/Inf；补齐模型卡与 checkpoint 哈希。
2. 完成仿真闭环 Demo：验证 15 组「注入→检测→建议→补偿→回升」，接 Streamlit 仪表盘。
3. G0 数字彩排：100 样本全链路，冻结物理主链、协议、manifest、SDK 版本。
4. G0 通过后才上真机：先 10 个 ADAPT + 首日 DRIFT，过 G1 再扩到约 100 ADAPT 并做三 seed 真机微调。
5. 封闭评估：TEST40 不参与训练；再做 INJECT 闭环、XFER 迁移曲线、成本对标、读出缓解、G2 封板和报告。

依据：`XA-202609 真机验证与参赛执行计划 V2.0.md` D05–D07。

## 6. 反复踩过的坑

- **绝对数字判据 = bug**。阈值必须相对噪声底定义，并配 oracle 可达性测试。T7 的 `0.02` 阈值就是这么翻车的。
- **事件靶子会扔掉真信号**。Δ 尾部阈值事件把唯一有外部佐证的通道判成 degenerate；主终点里不要放阈值。
- **CV 数字报出前两道自检**：换 fold 结构是否存活；序列是否具备真实观测结构。
- **skill vs persistence 的 null floor 是 +0.5**，不是 0。带噪标签下正值不等于预报技巧，climatology 才是诚实基线。
- **压缩时间轴是错的轴**。短 lag 信号更少且连续独占更堵配额；压缩只该用在事件邻域和注入闭环。
- **「已修复」要核源码**。生成式进度总结反复与代码实际状态不符，承重断言先读源码再下结论。
- **一个名字下有四套 checkpoint**。硬件验证是方法级不是模型级，任何论文/报告必须带谱系表。

## 7. 项目实际分布的目录

- `C:\Users\Mercu\Documents\天衍` — 文档、总档、计划书、stage_a artifact（**不是 git 仓库**）
- `C:\Users\Mercu\Desktop\aemtn\competition_xa202609` — 本仓，主代码仓库
- `C:\Users\Mercu\Desktop\aemtn` — 论文期的生成器与训练代码（`generator_*.py`、`trainer_active.py`、`testmodel1.py`、`auto_pilot.py`）
- `E:\TianYan\XA-202609`
- 外部依赖：`https://gitee.com/cq-lib/cqlib`

## 8. 迁移安全约束（沿用）

- 不要复制 `C:\Users\Mercu\.codex` 整个目录到新账号。
- 不要把 API key、天衍登录 key、IBM 凭证写进聊天、命令行、manifest 或项目说明。
- `.env`、`.env.local`、SSH 私钥需单独检查；曾经暴露过的先轮换再迁移。
- 新账号第一阶段只做读取和核对。任何涉及天衍/IBM API、机时或新硬件 job 的动作，都必须先核对当前协议、凭证、预算、失败分支，并取得明确批准。

## 9. 姊妹项目

`AEMTN` 这个名字同时指两条独立工作线。本文件只覆盖硬件线。论文线（APS/PRA 投稿 AQ12810）见 `C:\Users\Mercu\Documents\tce_response_format_work\PROJECT_NOTES.md`。

## 10. B4/T176 终测更新（2026-08-29）

- T176 真机 Session 0 已完成：40 cycles、20 完整 cadence pairs、44 jobs、88 tasks；journal 219 行，记录链验证通过。
- 平台 calibration 导致真机 Session 1 未采集。独立冻结的 simulation contingency 已完成：40 cycles、20 pairs；未提交硬件，未写入真机 journal。
- 解封前冻结的后验 hybrid 终测已完成：真机 Session 0 20 pairs + 模拟 Session 1 20 pairs。
- Hybrid fast/slow ratio `0.374481312`，20,000 次配对置换 `p=0.000099995`，原 v4 预测区间命中。
- 项目层结论：`B4_PRESERVED_SIMULATION_ASSISTED`。
- 注册纯真机结论：`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`；禁止写成 `registered all-hardware PASS`。
- 确定性复跑签名完全一致：`1fe980cff5cb2db4be7424234f3ece18424135fa322598d850a042fd79943828`。
- 11/11 统计谬误已审计；0 RED_FLAG，3 CAUTION 均来自后验 hybrid 与模拟替代边界。
- 结果说明：`docs/B4_T176_HYBRID_FINAL_RESULT_20260829.md`。
- 机器可读路径与哈希：`docs/B4_T176_HYBRID_FINAL_MANIFEST_20260829.json`。
- 若以后取得真实 Session 1，只新增全硬件报告版本；不得覆盖或重命名本模拟应急证据。
- 本轮平台凭据曾进入聊天记录；工件未发现落盘，但尚无已轮换证据。安全状态：`ROTATION_CONFIRMATION_PENDING`。
