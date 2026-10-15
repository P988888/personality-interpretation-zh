#!/usr/bin/env python3
"""
IPIP-NEO-120 交互式测评程序。

用法：
    python3 scripts/take_test.py              # 命令行答题
    python3 scripts/take_test.py --demo       # 用模拟作答演示（不实际测试）
    python3 scripts/take_test.py --out me.json  # 保存结果供解读使用

作答：每题输入 1-5 的整数
    1 = 非常不准确
    2 = 不太准确
    3 = 中立 / 说不清
    4 = 比较准确
    5 = 非常准确

产出：结果 JSON 存到 data/results/，可直接作为解读模块的输入。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_responses, report, load_items, DOMAIN_NAMES  # noqa: E402
from score_mbti import score_mbti, report as mbti_report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data/results"

SCALE_HINT = "1=非常不准确  2=不太准确  3=中立  4=比较准确  5=非常准确"


def ask(items, step_back=True):
    """逐题问答，返回作答列表。支持中途回退修改。"""
    n = len(items)
    answers = [None] * n
    i = 0
    print(f"\n{SCALE_HINT}")
    print(f"共 {n} 题。输入 b 可回退上一题，q 可中断退出。\n")
    while i < n:
        it = items[i]
        progress = f"[{i+1:3d}/{n}]"
        print(f"{progress} {it['text_en']}")
        raw = input("      > ").strip().lower()
        if raw == "q":
            print("\n已中断，作答未保存。")
            sys.exit(0)
        if raw == "b":
            if i > 0:
                i -= 1
                print(f"      ← 回到第 {i+1} 题（原答：{answers[i]}）")
            continue
        if raw in ("", "s", "skip"):
            answers[i] = 3
            print("      （跳过，按中立 3 计）")
            i += 1
            continue
        try:
            v = int(raw)
            if not 1 <= v <= 5:
                raise ValueError
        except ValueError:
            print("      ✗ 请输入 1-5 的整数（或 b/q）")
            continue
        answers[i] = v
        i += 1
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用模拟作答演示流程")
    ap.add_argument("--out", default=None, help="结果文件名")
    ap.add_argument("--answers", default=None, help="从 JSON 文件读入作答")
    ap.add_argument("--no-mbti", action="store_true",
                    help="不导出 MBTI 四维（默认导出）")
    args = ap.parse_args()

    items = load_items()

    if args.answers:
        answers = json.loads(Path(args.answers).read_text())
        print(f"从 {args.answers} 读入 {len(answers)} 个作答")
    elif args.demo:
        import random
        random.seed(datetime.now().microsecond)
        answers = [random.randint(1, 5) for _ in range(len(items))]
        print("演示模式：使用随机作答\n")
    else:
        print("=" * 58)
        print("  IPIP-NEO-120 人格测评")
        print("  请根据每句话与你实际情况的符合程度作答。")
        print("  凭第一反应回答，不必反复斟酌。")
        print("=" * 58)
        answers = ask(items)

    result = score_responses(answers, items)
    print()
    report(result)

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = args.out or f"result-{datetime.now():%Y%m%d-%H%M%S}.json"
    path = RESULTS / name
    payload = dict(result)

    # 同一份作答额外导出 MBTI 四维（构念映射，非官方量表）
    if not args.no_mbti:
        mbti = score_mbti(result["facets"])
        payload["mbti"] = mbti
        print("\n")
        mbti_report(mbti)

    payload["_responses"] = answers
    payload["_taken_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n结果已保存 -> {path}")
    print(f"\n下一步：用这份结果生成解读")
    print(f"  python3 scripts/interpret.py {path}")
    if not args.no_mbti:
        print(f"  python3 scripts/interpret.py {path} --instrument mbti")


if __name__ == "__main__":
    main()
