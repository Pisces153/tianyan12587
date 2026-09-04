"""AEMTN-B4 统一控制入口（面向评审/复用的薄 CLI）。

与冻结包内 ``b4ctl.py`` 并存但不替代它；本入口把 `aemtn_b4` 公共 API 暴露成
子命令，便于快速验收。

用法示例::

    aemtn verify              # 冻结核心完整性校验
    aemtn reproduce-final     # 在线重算 hybrid 终测
    aemtn final-report        # 打印结构化终测结果
    aemtn dashboard           # 启动 Streamlit 仪表盘（需 app extras）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _cmd_verify() -> int:
    from .verify import verify

    ok, problems = verify()
    if not ok:
        print("FAILED:")
        for p in problems:
            print(" -", p)
        return 1
    print("OK: 冻结核心完整。")
    return 0


def _cmd_reproduce_final() -> int:
    from .reproduce import reproduce_final

    results = reproduce_final()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    all_match = all(r["matches_frozen"] for r in results.values())
    print("PUBLIC FINAL REPRODUCTION " + ("PASS" if all_match else "MISMATCH"))
    return 0 if all_match else 1


def _cmd_final_report() -> int:
    from .result import load_final_report

    report = load_final_report()
    print(f"analysis_label: {report.analysis_label}")
    print(f"project_status: {report.project_status}")
    print(f"registered_hardware_status: {report.registered_hardware_status}")
    print(f"confidence: {report.confidence}")
    print("\n冻结层统计：")
    for layer in report.layers:
        print(f"  {layer.layer}: pairs={layer.pairs} ratio={layer.ratio:.6f} "
              f"reduction={100*layer.relative_reduction:.2f}% p={layer.p_value:.6f} passed={layer.passed}")
    return 0


def _cmd_dashboard() -> int:
    root = Path(__file__).resolve().parents[1]
    app = root / "app" / "streamlit_app.py"
    if not app.is_file():
        print(f"仪表盘不存在: {app}")
        return 1
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless=true"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aemtn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("reproduce-final")
    sub.add_parser("final-report")
    sub.add_parser("dashboard")
    args = parser.parse_args(argv)

    if args.command == "verify":
        return _cmd_verify()
    if args.command == "reproduce-final":
        return _cmd_reproduce_final()
    if args.command == "final-report":
        return _cmd_final_report()
    if args.command == "dashboard":
        return _cmd_dashboard()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
