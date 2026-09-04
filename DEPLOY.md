# 发布与部署指南（DEPLOY.md）

面向**接到 GitHub 账号的朋友**：把本仓库发到 GitHub，并让评审通过公网 URL
访问仪表盘。全程约 15 分钟。

---

## 前置

- 一个 GitHub 账号（用来发布；可用你自己的公共账号）。
- 本地有 `git` 与 `git-lfs`（模型权重可选 LFS，见下文）。
- 本仓库目录（已按 `packaging/RECORD.md` 准备好）。

> **仓库可公开也可私有。** Streamlit Community Cloud 支持从私有仓库部署（通过只读
> Deploy Key）。私有仓库部署出的应用默认只对工作区开发者可见，可在 App settings →
> Sharing 里按邮箱加入评审为 viewer（Google 账号或一次性邮件链接登录）。
> 免费层通常只允许 1 个私有应用（第三方资料，未在官方页面核实），公开应用不限。
> 本指南默认公开仓库；选私有则在步骤 3 部署后到 Sharing 里逐个添加评审邮箱。

---

## 步骤 1：推送仓库到 GitHub

```bash
cd <本仓库根目录>

# 首次：初始化并提交
git init
git add .
git commit -m "AEMTN-B4 release 2026-08-31 (frozen core + dashboard)"

# 关联你的远端（把 OWNER 换成你的用户名，REPO 换成仓库名）
git remote add origin git@github.com:OWNER/REPO.git
git push -u origin main
```

**模型权重（约 52 MB，3 个 17.3 MB best.pt）**：
- 方案 A（推荐）：直接提交。`*.pt` 已在 `.gitattributes` 标记为 `binary`，
  单个 17.3 MB < GitHub 100 MB 上限，可正常入库。
- 方案 B：接入 Git LFS（仓库更轻、下载更快）。在仓库里执行：
  ```bash
  git lfs install
  git lfs track "models/**/*.pt"
  git add .gitattributes models/
  git commit -m "track model weights via LFS"
  git push
  ```

> **不要**把 `models/` 从仓库剔除改放 Release 附件：3 个 best.pt 都在冻结
> manifest 里，缺失会直接让 `aemtn verify` 与 Cloud 上的仪表盘校验失败。
> 若用方案 B，注意 LFS 流量额度用尽时 clone 只拿到指针文件，同样会校验失败。

---

## 步骤 2：在 GitHub 上配置 CI（可选但推荐）

仓库推送后，`.github/workflows/ci.yml` 会自动触发。
CI 会做：Python 3.11 环境 → 冻结完整性校验（`aemtn verify`）→ 测试套件
（deselect 3 个依赖作者私有工件的测试）→ 终测复算（`aemtn reproduce-final`）。
在 GitHub 仓库的 **Actions** 页可看到结果，全部绿色即通过。

---

## 步骤 3：部署到 Streamlit Community Cloud

1. 打开 <https://share.streamlit.io>，用 GitHub 登录。
2. **New app** → 选择你刚推送的仓库。
3. 三项配置：
   - **Main file path**：`app/streamlit_app.py`
   - **Python version**：`3.11`
   - **App URL**：自动生成，可改。
4. 点击 **Deploy**。

平台会用根目录的 `requirements.txt` 安装依赖（**不含** torch / qutip / cqlib，
它们是离线训练与真机对接才需要的重量依赖）。部署完成后会给你一个公网 URL。

> **注意：** 依赖里不装 torch/qutip/cqlib，是因为仪表盘的核心展示 + 在线复算
> 只依赖 numpy/scipy/pandas + streamlit/plotly。若你在同一仓库里还想跑训练
> 或有真机需求，请另建一个带完整依赖的部署，或本地运行。

---

## 部署后自查（评审可做）

打开公网 URL，检查：

- 顶部导航 3 个产品区（平台 / 工作台 / 证据）可切换，无报错。
- 「工作台 → 漂移诊断」用内置示例点「执行漂移诊断」，出裁决与结构函数图，可导出 JSON。
- 「工作台 → 决策安全盾」拖动滑块，五门状态条实时变化。
- 「证据 → T176 闭环终测」能点「在线复算」按钮，3 个证据层全部
  `matches_frozen: true`。
- 首页如实展示 `INCONCLUSIVE_MISSING_HARDWARE_SESSION1` 与 T* 的
  `INCONCLUSIVE` 结论，未夸大。

---

## 私有工件说明

仓库含 3 个测试需要作者机器上的私有路径
（`E:\TianYan\...\all_endpoints_timing_grid.json`），该文件**不**随包分发。
因此 CI 已在 `ci.yml` 里 `--deselect` 这 3 个测试；本地跑 `pytest` 时，
没有该文件的机器上这 3 个测试也会失败（其余 288 个通过）。这不是 bug，
是数据边界的一部分，详见 `SECURITY_AND_DATA_BOUNDARY.md`。

---

## 发布前记住

- **可发布**：冻结核心（含 manifest）、包装层 `aemtn_b4/`、仪表盘 `app/`、
  12 张成果图与 source CSV、纯派生 `evidence/` JSON/CSV。
- **已决定保留**：冻结核心（`manifest/ORIGIN.json`、`config/`、`docs/`、`evidence/`
  等 73 个文件）含作者机器的本地目录名（形如 `C:\Users\<author>\...`、`E:\...`）。
  这些是溯源元数据，不含凭据；它们受 sha256 manifest 锁定，改动任一字节都会让
  `aemtn verify` 失败并切断到 `source_git_head` 的溯源链，因此**不脱敏**。
- **凭据**：仓库内无 API 密钥 / `.env`。`SECURITY_AND_DATA_BOUNDARY.md`
  标记平台凭据状态为 `ROTATION_CONFIRMATION_PENDING`，首次个人真机对接前请完成轮换。

若你不确定仓库公开是否会泄露不该公开的信息，先在本地跑一次
`aemtn verify`，并按 `SECURITY_AND_DATA_BOUNDARY.md` 复核后再推。
