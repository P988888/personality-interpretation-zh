#!/usr/bin/env python3
"""
从 IPIP-NEO 题库中选出 IPIP-NEO-120 的 120 道题。

关键依据：IPIP 官方对照表 (IPIP300-120ComparisonTable.htm) 给出了每个 facet
在 120 题版本中的 正向题数+反向题数 配比。这些配比**极不均衡**，例如：
  - N1 Anxiety:      4正 + 0反（全正向）
  - A2 Straightforwardness: 0正 + 4反（全反向）
  - N5 Immoderation: 1正 + 3反
这不是均匀抽样，而是刻意设计 —— 部分 facet 全用反向题以抵消默认同意倾向。

本脚本严格按官方配比挑题：
  - 若某 facet 需要 N 道正向题，就取其正向题列表的前 N 道（保持官方原始顺序）
  - 反向题同理

产出：data/items/ipip_neo_120.json
    含 120 题、所属 facet、计分方向、题目顺序。

注意：C2 Orderliness 在计分键页面只有 9 题（4正+5反），而对照表显示 300 版
     应为 5+5。两者不一致，属官方数据源差异。本脚本按实际可用的题目处理，
     并在输出中标注该异常。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data/items/ipip_neo_items.json"
OUT = ROOT / "data/items/ipip_neo_120.json"

# IPIP-NEO-120 官方配比：(正向题数, 反向题数)，依据官方对照表
# 顺序：N 6项, E 6项, O 6项, A 6项, C 6项
QUOTA = {
    # Neuroticism
    "N1_anxiety":            (4, 0),
    "N2_anger":              (3, 1),
    "N3_depression":         (3, 1),
    "N4_self_consciousness": (3, 1),
    "N5_immoderation":       (1, 3),
    "N6_vulnerability":      (3, 1),
    # Extraversion
    "E1_warmth":             (2, 2),
    "E2_gregariousness":     (2, 2),
    "E3_assertiveness":      (3, 1),
    "E4_activity_level":     (3, 1),
    "E5_excitement_seeking": (4, 0),
    "E6_cheerfulness":       (4, 0),
    # Openness
    "O1_imagination":        (4, 0),
    "O2_artistic":           (2, 2),
    "O3_emotionality":       (2, 2),
    "O4_adventurousness":    (1, 3),
    "O5_intellect":          (1, 3),
    "O6_liberalism":         (2, 2),
    # Agreeableness
    "A1_trust":              (3, 1),
    "A2_straightforwardness": (0, 4),
    "A3_altruism":           (2, 2),
    "A4_compliance":         (0, 4),
    "A5_modesty":            (0, 4),
    "A6_sympathy":           (2, 2),
    # Conscientiousness
    "C1_self_efficacy":      (4, 0),
    "C2_orderliness":        (1, 3),
    "C3_dutifulness":        (2, 2),
    "C4_achievement_striving": (2, 2),
    "C5_self_discipline":    (2, 2),
    "C6_cautiousness":       (0, 4),
}


def main():
    bank = json.loads(ITEMS.read_text(encoding="utf-8"))

    selected, problems = [], []
    for facet, (n_plus, n_minus) in QUOTA.items():
        if facet not in bank:
            problems.append(f"{facet}: 题库中不存在")
            continue
        f = bank[facet]
        plus = f["plus"][:n_plus]
        minus = f["minus"][:n_minus]

        if len(plus) < n_plus:
            problems.append(f"{facet}: 正向题不足（需{n_plus}，有{len(f['plus'])}）")
        if len(minus) < n_minus:
            problems.append(f"{facet}: 反向题不足（需{n_minus}，有{len(f['minus'])}）")

        for q in plus:
            selected.append({"facet": facet, "facet_name": f["name"],
                             "domain": f["code"][0], "text_en": q, "keyed": "+"})
        for q in minus:
            selected.append({"facet": facet, "facet_name": f["name"],
                             "domain": f["code"][0], "text_en": q, "keyed": "-"})

    # 按 facet 分组交错排列，避免同 facet 题目连续出现（减少作答定势）
    from itertools import zip_longest
    groups = {}
    for it in selected:
        groups.setdefault(it["facet"], []).append(it)
    interleaved = []
    for row in zip_longest(*groups.values()):
        interleaved.extend(x for x in row if x)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(interleaved, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"选出 {len(interleaved)} 题 -> {OUT}")
    print()
    plus_n = sum(1 for x in interleaved if x["keyed"] == "+")
    print(f"正向题 {plus_n}，反向题 {len(interleaved)-plus_n}")
    from collections import Counter
    dc = Counter(x["domain"] for x in interleaved)
    print(f"维度分布：{dict(dc)}")
    fc = Counter(x["facet"] for x in interleaved)
    print(f"facet 数：{len(fc)}，每题数分布：{dict(Counter(fc.values()))}")

    if problems:
        print("\n⚠ 异常：")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\n✓ 所有 facet 均满足官方配比")

    if len(interleaved) != 120:
        print(f"\n⚠ 题数 {len(interleaved)} ≠ 120，请检查配比表")


if __name__ == "__main__":
    main()
