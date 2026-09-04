"""打包完整性校验。

分层策略：
L0 冻结核心（src/scripts/tools/config/docs/data/models/figures/evidence/manifest/provenance）
  必须逐文件满足 manifest::PACKAGE_FILES.csv 记录的 bytes + sha256。
  任何核心文件被改/删/增 -> FAIL。
L1 封装层新增（aemtn_b4/app/packaging/pyproject.toml/requirements*.txt/.github/…）
  允许存在。
L2 工作区产物（.venv/__pycache__/.git 等）全部忽略。

用法:  python -m aemtn_b4.verify
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .paths import PROJECT_ROOT

# 冻结核心必须逐文件比对 manifest 的目录
_CORE_TOP = {
    "src",
    "scripts",
    "tools",
    "config",
    "docs",
    "data",
    "models",
    "figures",
    "evidence",
    "manifest",
    "provenance",
}
# L2 忽略
_IGNORE_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", ".ipynb_checkpoints", "node_modules"}
_IGNORE_SUFFIXES = {".pyc", ".pyo"}

# 发布层可控文件：虽由清单记录，但发布时可重写（如面向 GitHub 的 README）。
# 这些文件不做字节/sha256 校验，只校验"文件仍存在"。
_PUBLISH_EDITABLE = {"README.md"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ignored(path: Path) -> bool:
    return any(part in _IGNORE_PARTS for part in path.parts) or path.suffix in _IGNORE_SUFFIXES


def _is_core(rel: str) -> bool:
    return rel.split("/", 1)[0] in _CORE_TOP


def verify() -> tuple[bool, list[str]]:
    """校验冻结核心完整性。返回 (ok, 问题列表)。"""
    root = PROJECT_ROOT
    manifest_path = root / "manifest" / "PACKAGE_FILES.csv"
    if not manifest_path.is_file():
        return False, [f"缺失核心清单: {manifest_path}"]

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig", newline="")))
    expected = {row["path"]: row for row in rows}
    problems: list[str] = []

    # L0: 每个 manifest 条目都存在、大小与哈希一致
    for relative, row in expected.items():
        path = root / Path(relative)
        if not path.is_file():
            problems.append(f"MISSING {relative}")
            continue
        # 发布层可控文档：只校验存在，不校验字节（允许面向 GitHub 重写）
        if relative in _PUBLISH_EDITABLE:
            continue
        if path.stat().st_size != int(row["bytes"]):
            problems.append(f"SIZE {relative} ({path.stat().st_size} != {row['bytes']})")
        elif _sha256(path) != row["sha256"]:
            problems.append(f"HASH {relative}")

    # 冻结核心目录里出现但不在 manifest 中的文件 -> 被新增/篡改
    _self = "manifest/PACKAGE_FILES.csv"
    for path in root.rglob("*"):
        if not path.is_file() or _ignored(path):
            continue
        rel = path.relative_to(root).as_posix()
        if rel == _self:
            continue  # 清单文件自身不列入 expected，避免自指误报
        if _is_core(rel) and rel not in expected:
            problems.append(f"EXTRA {rel}")

    return (not problems), problems


def main() -> int:
    ok, problems = verify()
    if ok:
        print("OK: 冻结核心完整，所有 manifest 文件大小+sha256 一致。")
        return 0
    print("FAILED:")
    for p in problems:
        print(" -", p)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
