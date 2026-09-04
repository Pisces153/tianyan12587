# 发布记录（packaging）

本目录记录 2026-08-31 冻结科研核心之外，为面向评审 / 第三方复用而新增的
**发布层**说明。它不改变科研核心的任何一行。

## 分层架构

| 层 | 目录 | 是否可改 | 说明 |
|----|------|---------|------|
| 冻结科研核心 | `src/` `scripts/` `tools/` `config/` `docs/` `data/` `models/` `figures/` `evidence/` `manifest/` `provenance/` | **否** | 与作者原始包逐字节一致，`manifest/PACKAGE_FILES.csv` 锁定字节 + sha256 |
| 发布包装层 | `aemtn_b4/` `app/` `pyproject.toml` `requirements.txt` | 是 | 新的纯 Python 公共 API + Streamlit 仪表盘（本次新增） |
| 发布配置 | `.github/` `.gitignore` `.gitattributes` `README.md` `DEPLOY.md` | 是 | 让仓库可直接推送到 GitHub 并部署到 Streamlit Cloud |
| 工作区产物 | `.venv/` `__pycache__/` `.pytest_cache/` | **不入库** | `.gitignore` 排除 |

## 本次发布新增 / 变更文件

**新增** `aemtn_b4/`（8 个模块）：`__init__.py` `paths.py` `result.py`
`reproduce.py` `physics.py` `adaptive.py` `verify.py` `cli.py`。全部为**薄封装**，
只 import 冻结核心，不改它。

**新增** `app/`（12 个文件）：多页 Streamlit 仪表盘，面向评审的只读展示 +
两项轻量在线交互（终测复算、安全盾演示）。

**新增** `packaging/`：本记录、原始 README 备份、`_smoke.py` 冒烟脚本。

**新增** `pyproject.toml`：把核心暴露为 `pip install aemtn-b4` 可安装包，
并提供 `aemtn` CLI（`verify` / `reproduce-final` / `final-report` / `dashboard`）。

**新增** `requirements.txt`：Streamlit Cloud 最小依赖（不含 torch / qutip / cqlib）。

**新增** `.github/workflows/ci.yml`：Python 3.11 + 冻结完整性校验 + 测试套件。

**变更** `README.md`：由作者原始说明改写为面向 GitHub 评审 / 复用版本。
原始版本已备份在 `packaging/ORIGINAL_README.md`。

## 冻结核心完整性

- `manifest/PACKAGE_FILES.csv` 包含 333 项：路径、字节数、sha256。
- `aemtn verify`（L0）逐文件比对冻结核心字节，发现任何不一致即失败。
  本次发布**未修改**任何冻结核心文件，因此 L0 通过。
- L1 允许新增的 `aemtn_b4/` `app/` 包装层；L2 忽略 `.venv/` 等工作区产物。
- 注意：`aemtn_b4/verify.py` 是**打包感知**的校验（认识 L1/L2），
  而冻结的 `tools/verify_package.py` 会把这批新目录列为 EXTRA 文件。两者并存无害。

## 证据边界（发布时读者须知）

- 项目层判定：`B4_PRESERVED_SIMULATION_ASSISTED`。
- 纯真机注册状态：`INCONCLUSIVE_MISSING_HARDWARE_SESSION1`。
- T* 仅为点估计（bootstrap 95% CI 跨数量级），不构成可部署 SLA。
- 仓库不含 API 密钥 / `.env`；冻结核心仍含作者机器本地目录名（形如
  `C:\Users\<author>\...`），属溯源元数据且受 sha256 锁定，决定保留、不脱敏。

## 发布前检查清单

1. `pip install -e .` 成功。
2. `aemtn verify` 输出 `OK: 冻结核心完整。`
3. `aemtn reproduce-final` 输出 `PUBLIC FINAL REPRODUCTION PASS`。
4. `pytest` 预期 288 passed / 3 failed（3 个失败依赖作者机器私有工件，
   在 CI 中已 deselect，详见 `DEPLOY.md`）。
5. `aemtn dashboard` 能启动，8 章均可渲染。
