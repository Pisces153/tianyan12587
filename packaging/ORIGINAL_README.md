# AEMTN / B4 AI驱动量子校准闭环代码包

版本：2026-08-31。这个包从主训练仓、冻结模型、B4分析/控制代码和最终派生证据反向整理而成，目标是为下一步API、桌面工具或SaaS封装提供一个可审计基线。

> **说明：** 本文件是 2026-08-31 原始发布的 README.md 的完整备份。现根目录 README.md
> 已重写为面向 GitHub 评审 / 复用的版本。若需追溯原始表述（含"最快使用"的
> `b4ctl.py` 入口），以此备份为准。

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
