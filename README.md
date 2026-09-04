# AEMTN-B4 · 面向量子云真机的开放量子系统辨识与人工智能自校准平台

> 6 比特 · 天衍超导真机（T176 / T287）· 仿真-真机混合闭环 · 冻结结果 2026-08-31
>
> **项目级判定**：`B4_PRESERVED_SIMULATION_ASSISTED` ｜
> **纯真机注册**：`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`

本项目构建一个 **AI 驱动、面向量子云真机** 的开放量子系统辨识与自校准平台：
用多任务 **AEMTN** 网络从量子测量恢复有效哈密顿参数，对环境漂移做**可检出的
感知与判别**，再经**去偏结构函数 / 更新周期经济门 / 非学习安全盾**驱动
fast/slow 校准闭环，并在天衍超导真机上完成 **Session 0 诊断 + 模拟 Session 1
反事实**的混合终测。

## 为什么做这个

超导量子真机运行时受温度、电磁干扰、噪声漂移等动态扰动，导致**参数漂移**。
传统静态校准范式无法实时响应。本项目突破静态范式，建立
**环境感知 → 误差预测 → 实时调优**的闭环进化框架，并给出可审计的证据边界，
而不是夸大不实的主张。

## 闭环主线

`仿真数据生成 → AEMTN 三 seed 训练 → 冻结 best.pt 推理 → 量子态测量特征提取 → 环境/设备漂移代理 → 去偏结构函数 → 误差预测与残差曲线 → 最优更新周期 T* → go/no-go 感知经济门 → 安全 shield 与调度 → fast/slow cadence 闭环 → T176 Session 0 真机诊断 → Session 1 模拟应急 → simulation-assisted hybrid final → 12 张独立成果图及完整数据`

## 关键技术参数与验证方案（评审速览）

| 项 | 值 / 结论 | 验证 |
|----|----------|------|
| 量子系统 | 6 比特开边界 DM 哈密顿量，Néel 初态 `010101` | 物理合约测试 |
| 测量协议 | 9 个测量基 × 1024 shots，q0-leftmost 位序 | `test_physics_contract` |
| 输入特征 | local6（X0,Y0,Z0,X0X1,Y0Y1,Z0Z1）+ 1 维时间 = 7 维 | `test_counts_features` |
| AEMTN 模型 | 约 1.43M 参数，r_dim=320，4 子空间多任务头，目标 h1/h2/Jz + 高斯不确定性头；三 seed 冻结 | `test_training_pipeline` |
| 漂移判别 | E0 阴性对照**未触发**（p=0.6497）；E1 通道**检出**过程方差（p≈2.8e-289） | `test_sensing_economics` |
| T* 更新间隔 | 点估计 ≈134 s，但 bootstrap 95% CI 101–4000 s（上界撞观测窗），判定 `INCONCLUSIVE` | 残差-间隔经济曲线 |
| 真机 Session 0 | 20 对，fast/slow 累计残差比 0.36165，配对置换 p=0.00525 | 置换检验 |
| 混合闭环 primary | 40 对，ratio 0.37448，p=0.00010 | 置换检验 |

## 五个维度的定位

1. **技术路线 / 技术框架**：量子测量代理（而非直接温度/电磁传感）+ 环境漂移判别
   + 更新周期经济门 + 非学习安全盾 + fast/slow cadence 闭环。
2. **关键参数与验证方案**：上表所有参数均由冻结测试逐项验证。
3. **客观分析技术瓶颈**：主动披露 h2 R²≈0、Jz R²≈0.32、T* CI 跨数量级、
   Session 1 未上真机等瓶颈（详见第 7 章）。
4. **原型系统落地 / 行业适配**：可部署的 Streamlit 仪表盘 + 从高斯不确定性到安全盾的
   完整调度回路，含真机诊断。
5. **理论方法突破 / 普惠化 / 生态协同**：非学习安全盾与规则回退设计，兼容
   第三方量子云（cqlib 为公开库），生态可扩展。

## 快速开始

```bash
# 核心 API（只依赖 numpy/scipy/pandas，可直接装）
pip install -e .
aemtn verify          # 冻结核心完整性（字节 + sha256）
aemtn reproduce-final # 重算 3×20000 次置换，校验与冻结值一致
aemtn final-report    # 打印结构化终测结果
aemtn dashboard       # 启动 Streamlit 仪表盘（需 app 依赖）
```

完整科研复现（训练 / 物理仿真 / 真机对接）需额外依赖：

```bash
pip install -e .[all]
```

## 仪表盘 8 章

1. **项目总览**：指标 + 15 步闭环 + 竞赛维度映射 + 证据边界框
2. **技术路线与 AI 框架**：哈密顿量、AEMTN 架构、训练摘要、AI 框架选型
3. **环境漂移感知与判别**：读出状态可观测性、E0/E1 判别、更新周期经济学
4. **校准决策与安全 shield**：go/no-go 经济门 + 五门交互演示 + 决策边界
5. **真机闭环终测证据**：三类证据层、配对残差、置换直方图、**在线复算按钮**
6. **模型与数据**：数据集契约、特征契约、三 seed 权重、训练管线
7. **边界与限制（诚实声明）**：能说 / 须限定 / 不可说三档 + 技术瓶颈
8. **复现与部署**：评审复现路径 + 3 步发布到 GitHub / Streamlit Cloud

## 目录结构

```
src/       训练、特征、物理、后端、自适应核心（冻结，勿改）
scripts/   从数据到 hybrid final 的原始 CLI（冻结）
tools/     完整性验证、公开结果复算（冻结）
config/    冻结协议与 B4 v4 配置（冻结，不含凭据）
tests/     逐项回归测试（冻结）
models/    三 seed best.pt（冻结）
data/      sim_v3 paper-contract 数据集契约（冻结）
evidence/  终测 pair 表、模拟应急工件、验证报告（冻结）
figures/   12 张独立主图 + 逐图 CSV + 数据簿（冻结）
manifest/  逐文件来源、SHA256、完整包文件表（冻结）
provenance/  来源归属（冻结）
aemtn_b4/  公共 Python API 包装层（本次新增）
app/       Streamlit 仪表盘（本次新增）
packaging/ 发布记录 + 原始 README 备份
```

## 证据边界（不可省略）

- 项目层只写成 `B4_PRESERVED_SIMULATION_ASSISTED`。
- 纯真机注册状态仍是 `INCONCLUSIVE_MISSING_HARDWARE_SESSION1`。
- T* 是点估计，CI 上界撞到观测窗，不能写成稳定生产 SLA。
- H1/H2 是量子测量代理，不是直接温度 / 电磁读数。
- 当前含安全 shield 与规则调度，但**不等于**在线 RL 已在真机部署。

详见 `CLAIM_BOUNDARY.md`、`CODE_INVENTORY.md`、`PIPELINE.md`、
`SECURITY_AND_DATA_BOUNDARY.md`。

## 复现

`packaging/RECORD.md` 记录分层架构与发布清单。`DEPLOY.md` 给发布者的 3 步。
CI（`.github/workflows/ci.yml`）在 Python 3.11 上做冻结校验 + 测试（deselect
3 个依赖作者私有工件的测试）。

## License

Apache-2.0（见 `pyproject.toml`）。核心依赖 `cqlib`、`qutip`、`torch` 均为公开版本。
