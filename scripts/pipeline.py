#!/usr/bin/env python3
"""
管道总入口 —— 一条命令完成各阶段工作。

用法：
    python3 scripts/pipeline.py items       # 抓取并生成题库（需联网）
    python3 scripts/pipeline.py test        # 开始答题
    python3 scripts/pipeline.py interpret   # 用最近一次结果生成解读请求
    python3 scripts/pipeline.py gen-prompts # 生成 SFT 训练用的分数组合
    python3 scripts/pipeline.py validate    # 校验生成的训练数据
    python3 scripts/pipeline.py status      # 查看各阶段产出状态
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "scripts"


def run(script, *args):
    cmd = [sys.executable, str(S / script), *args]
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n" + "-" * 60)
    return subprocess.call(cmd)


def status():
    checks = [
        ("题库(全量)", "data/items/ipip_neo_items.json", "extract_ipip_items.py"),
        ("题库(120题)", "data/items/ipip_neo_120.json", "pick_neo120.py"),
        ("分数组合", "data/raw/prompts_bigfive.jsonl", "gen_prompts_bigfive.py"),
        ("生成的解读", "data/raw/generated_bigfive.jsonl", "generate.py"),
        ("通过的样本", "data/curated/bigfive_pass.jsonl", "validate.py"),
        ("训练数据", "data/curated/personality_bigfive.json", "to_llamafactory.py"),
    ]
    print(f"{'阶段':<16s} {'状态':<8s} 文件")
    print("-" * 62)
    for label, rel, script in checks:
        p = ROOT / rel
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8")) if p.suffix == ".jsonl" else ""
            mark = f"✓ {n}条" if n else "✓"
        else:
            mark = "—"
        print(f"{label:<16s} {mark:<8s} {rel}")

    r = ROOT / "data/results"
    n = len(list(r.glob("*.json"))) if r.exists() else 0
    print(f"\n测评结果: {n} 份" + (f" -> {r}" if n else ""))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "items":
        run("extract_ipip_items.py")
        run("pick_neo120.py")
    elif cmd == "test":
        run("take_test.py", *sys.argv[2:])
    elif cmd == "interpret":
        r = ROOT / "data/results"
        files = sorted(r.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            print("没有测评结果，请先运行: python3 scripts/pipeline.py test")
            return
        run("interpret.py", str(files[-1]), *sys.argv[2:])
    elif cmd == "gen-prompts":
        run("gen_prompts_bigfive.py")
    elif cmd == "validate":
        src = sys.argv[2] if len(sys.argv) > 2 else "data/raw/generated_bigfive.jsonl"
        run("validate.py", src)
        run("to_llamafactory.py")
    elif cmd == "status":
        status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
