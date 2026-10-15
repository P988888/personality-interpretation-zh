#!/usr/bin/env python3
"""
IPIP-NEO-120 计分器。

流程：
    120 题作答(1-5) → 反向题取反 → facet 原始分 → 维度原始分 → 解读输入

计分规则（依据 IPIP 官方 newScoringInstructions.htm）：
    - 每题 1-5 分
    - 反向题("−" keyed)需先反转：score = 6 - raw
    - facet 原始分 = 该 facet 内所有题目得分之和
    - 维度原始分 = 其 6 个 facet 原始分之和
    - 原始分范围：facet 4-20，维度 24-120

关于百分位的重要说明：
    IPIP 官方**不提供常模**（见 ipip_interp.html：官方明确表示不提供手册或
    解释指南）。因此本脚本输出**原始分与相对高低**，不编造百分位。
    若需百分位，必须自建常模样本，此事在 specs/big-five.md 与
    behavior-rules.md 中都有纪律要求：无可靠常模时不得报百分位。

用法：
    from score import score_responses
    result = score_responses([3,5,2,...])   # 120 个 1-5 的整数
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data/items/ipip_neo_120.json"

DOMAIN_NAMES = {
    "O": "开放性", "C": "尽责性", "E": "外向性",
    "A": "宜人性", "N": "神经质",
}

FACET_NAMES_CN = {
    "O1_imagination": "想象力", "O2_artistic": "审美", "O3_emotionality": "情绪感受",
    "O4_adventurousness": "尝新", "O5_intellect": "思辨", "O6_liberalism": "观念开放",
    "C1_self_efficacy": "自我效能", "C2_orderliness": "秩序", "C3_dutifulness": "尽职",
    "C4_achievement_striving": "成就追求", "C5_self_discipline": "自律", "C6_cautiousness": "审慎",
    "E1_warmth": "热情", "E2_gregariousness": "乐群", "E3_assertiveness": "果断",
    "E4_activity_level": "活力", "E5_excitement_seeking": "寻求刺激", "E6_cheerfulness": "积极情绪",
    "A1_trust": "信任", "A2_straightforwardness": "直率", "A3_altruism": "利他",
    "A4_compliance": "合作", "A5_modesty": "谦逊", "A6_sympathy": "同理心",
    "N1_anxiety": "担忧", "N2_anger": "易怒", "N3_depression": "低落",
    "N4_self_consciousness": "社交不安", "N5_immoderation": "冲动", "N6_vulnerability": "压力敏感",
}


def load_items():
    return json.loads(ITEMS.read_text(encoding="utf-8"))


def score_responses(responses, items=None):
    """把 120 个作答转成分数结构。

    responses: 长度 120 的整数列表，值域 1-5
    返回: dict，含 domains / facets / 原始分 / 标准化到 0-100 的展示分
    """
    items = items or load_items()
    if len(responses) != len(items):
        raise ValueError(f"作答数 {len(responses)} 与题目数 {len(items)} 不符")

    for i, r in enumerate(responses):
        if not isinstance(r, int) or not 1 <= r <= 5:
            raise ValueError(f"第 {i+1} 题作答 {r!r} 非法，应为 1-5 的整数")

    # 逐题计分（反向题反转）
    facet_raw = {}
    for item, r in zip(items, responses):
        v = r if item["keyed"] == "+" else 6 - r
        facet_raw.setdefault(item["facet"], []).append(v)

    facets = {}
    for facet, vals in facet_raw.items():
        raw = sum(vals)
        n = len(vals)
        lo, hi = n * 1, n * 5
        facets[facet] = {
            "name": FACET_NAMES_CN.get(facet, facet),
            "raw": raw,
            "min": lo,
            "max": hi,
            "n_items": n,
            # 展示分：线性映射到 0-100。这不是百分位，只是便于阅读的相对刻度。
            "score_0_100": round((raw - lo) / (hi - lo) * 100, 1),
        }

    domains = {}
    for f, d in facets.items():
        domains.setdefault(f[0], []).append(d["raw"])

    dom_out = {}
    for code, raws in domains.items():
        raw = sum(raws)
        n = sum(facets[f]["n_items"] for f in facets if f[0] == code)
        lo, hi = n * 1, n * 5
        dom_out[code] = {
            "name": DOMAIN_NAMES[code],
            "raw": raw,
            "min": lo,
            "max": hi,
            "n_items": n,
            "score_0_100": round((raw - lo) / (hi - lo) * 100, 1),
        }

    return {
        "instrument": "big-five",
        "version": "IPIP-NEO-120",
        "scale": "raw_and_normalized",
        "norm_source": None,
        "domains": {c: d["score_0_100"] for c, d in dom_out.items()},
        "facets": {f: d["score_0_100"] for f, d in facets.items()},
        "_detail": {"domains": dom_out, "facets": facets},
        "_notice": ("分数为原始分线性映射到 0-100 的相对刻度，非百分位。"
                    "IPIP 官方不提供常模，无可靠常模时不得解释为人群排名。"),
    }


def report(result):
    """人类可读的分数报告。"""
    print(f"IPIP-NEO-120 计分结果\n{'='*54}")
    print(f"\n{'维度':<10s} {'原始分':>12s} {'0-100':>8s}  分布")
    print("-" * 54)
    detail = result["_detail"]["domains"]
    for code in "OCEAN":
        d = detail[code]
        bar = "█" * int(d["score_0_100"] / 5)
        print(f"{d['name']:<10s} {d['raw']:>5d}/{d['max']:<6d} {d['score_0_100']:>7.1f}  {bar}")

    print(f"\n{'facet':<14s} {'原始分':>10s} {'0-100':>8s}")
    print("-" * 54)
    fd = result["_detail"]["facets"]
    for code in "OCEAN":
        print(f"[{DOMAIN_NAMES[code]}]")
        for f in sorted(fd):
            if f[0] != code:
                continue
            x = fd[f]
            print(f"  {x['name']:<12s} {x['raw']:>4d}/{x['max']:<5d} {x['score_0_100']:>7.1f}")

    print(f"\n⚠ {result['_notice']}")


if __name__ == "__main__":
    # 演示：用一份模拟作答跑通全流程
    import random
    random.seed(7)
    demo = [random.randint(1, 5) for _ in range(120)]
    report(score_responses(demo))
