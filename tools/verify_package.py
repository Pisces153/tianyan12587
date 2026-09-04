#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest" / "PACKAGE_FILES.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig", newline="")))
    expected = {row["path"]: row for row in rows}
    failures = []
    for relative, row in expected.items():
        path = ROOT / Path(relative)
        if not path.is_file():
            failures.append(f"MISSING {relative}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"SIZE {relative}")
        if digest(path) != row["sha256"]:
            failures.append(f"HASH {relative}")
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts
    }
    extras = sorted(actual - set(expected))
    failures.extend(f"EXTRA {path}" for path in extras)
    if failures:
        print("PACKAGE VERIFY FAILED")
        print("\n".join(failures))
        return 1
    print(f"PACKAGE VERIFY PASS: {len(rows)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
