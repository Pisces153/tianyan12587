"""路径解析：定位冻结包根目录，不依赖 CWD。

冻结包的 ``src/``、``scripts/`` 内部用 ``Path(__file__).resolve().parents[1]``
推导根目录，因此任何 import 都必须保证包根在 sys.path。这里统一解析一次。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _root(*anchors: str) -> Path:
    best: Path | None = None
    # 优先在包内 anchor：
    for anchor in anchors:
        cand = Path(__file__).resolve().parents[1] / anchor
        if cand.exists():
            best = cand
            return _resolve_root(cand)
    # 退回 CWD 与 PYTHONPATH
    for anchor in anchors:
        for base in [Path.cwd(), *[Path(p) for p in sys.path if p]]:
            cand = (base / anchor).resolve()
            if cand.exists():
                return _resolve_root(cand)
    raise FileNotFoundError(
        "找不到 AEMTN-B4 冻结包根目录。请在仓库根目录运行，"
        "或设置环境变量 AEMTN_PROJECT_ROOT 指向包含 src/ 的目录。"
    )


def _resolve_root(anchor: Path) -> Path:
    # 目标是含 src/、evidence/、config/ 的那一层
    for cand in [anchor, *anchor.parents]:
        if (cand / "src").is_dir() and (cand / "evidence").is_dir():
            return cand
    raise FileNotFoundError(f"锚点 {anchor} 上方未找到含 src/ evidence/ 的包根目录。")


def project_root() -> Path:
    """冻结包根目录（含 src/、evidence/、models/）。"""
    env = __import__("os").environ.get("AEMTN_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return _root("src", "evidence", "config")


PROJECT_ROOT = project_root()

# 冻结核心以顶层包 ``src`` 形式存在、未随 aemtn_b4 安装，因此把包根放进 sys.path，
# 保证 ``aemtn`` 控制台入口在任意 CWD（含 CI、Streamlit Cloud）下都能 import src.*。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

data_dir = lambda: PROJECT_ROOT / "data"  # noqa: E731
evidence_dir = lambda: PROJECT_ROOT / "evidence"  # noqa: E731
manifest_dir = lambda: PROJECT_ROOT / "manifest"  # noqa: E731
models_dir = lambda: PROJECT_ROOT / "models"  # noqa: E731
app_assets_dir = lambda: PROJECT_ROOT / "app" / "assets"  # noqa: E731
