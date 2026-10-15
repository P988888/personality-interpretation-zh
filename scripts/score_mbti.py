#!/usr/bin/env python3
"""
MBTI 四维计分 —— 复用 IPIP-NEO-120 的作答。

设计说明（重要）：
    本模块**不复制 MBTI 官方题目**（版权归 Myers-Briggs 公司），也不自行编造题目
    （那样效度无从谈起）。而是基于 IPIP 公共领域题库中经过验证的构念，
    映射到 MBTI 四个维度。

    映射依据（学术界研究 MBTI 与五因素关系的常见做法）：

        E/I  ←  外向性（E1热情 E2乐群 E3果断 E4活力 E5寻求刺激 E6积极情绪）
        S/N  ←  开放性中的想象与思辨面（O1想象力 O2审美 O5思辨）
        T/F  ←  宜人性中的情感面（A6同理心 A3利他 A1信任）—— 高分偏 F
        J/P  ←  尽责性中的秩序与审慎面（C2秩序 C6审慎）—— 高分偏 J

    这个映射带来一个必须向用户说明的后果：**它把连续分数硬切成二分类型**，
    而 MBTI 的方法论争议恰恰就在这里（见 specs/mbti.md 第五节）。
    因此本模块强制输出 clarity（清晰度），并据此调整解读的确信度。

用法：
    from score_mbti import score_mbti
    result = score_mbti(domains, facets, responses=None)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data/items/ipip_neo_120.json"

# 维度映射：(facet 列表, 极性)
#   极性 +1 表示该 facet 分高 → 偏向前者（E / N / F / J）
#   极性 -1 表示该 facet 分高 → 偏向后者（I / S / T / P）
DIMENSIONS = {
    "EI": {
        "name": "能量方向",
        "left": ("E", "外倾"), "right": ("I", "内倾"),
        "facets": {
            "E1_warmth": +1, "E2_gregariousness": +1, "E3_assertiveness": +1,
            "E4_activity_level": +1, "E5_excitement_seeking": +1, "E6_cheerfulness": +1,
        },
    },
    "SN": {
        "name": "信息获取",
        "left": ("N", "直觉"), "right": ("S", "实感"),
        "facets": {
            "O1_imagination": +1, "O2_artistic": +1, "O5_intellect": +1,
        },
    },
    "TF": {
        "name": "决策方式",
        "left": ("F", "情感"), "right": ("T", "思维"),
        "facets": {
            "A6_sympathy": +1, "A3_altruism": +1, "A1_trust": +1,
        },
    },
    "JP": {
        "name": "生活态度",
        "left": ("J", "判断"), "right": ("P", "知觉"),
        "facets": {
            "C2_orderliness": +1, "C6_cautiousness": +1,
        },
    },
}

# 清晰度分档：依据分数与中点的距离
CLARITY_BANDS = [
    (20, "clear",     "明显"),
    (10, "moderate",  "中等"),
    (0,  "slight",    "轻微"),
    (None, "marginal", "临界"),
]


def _clarity(dist: float) -> tuple:
    """dist 为分数偏离中点的百分点（0-50）。"""
    if dist >= 20:
        return "clear", "明显"
    if dist >= 10:
        return "moderate", "中等"
    if dist >= 5:
        return "slight", "轻微"
    return "marginal", "临界"


def score_mbti(facets_scaled: dict) -> dict:
    """由大五的 facet 分数（0-100 刻度）导出 MBTI 四维。

    facets_scaled: {"E1_warmth": 62.5, ...} —— 即 score.py 输出的 facets
    """
    dims = {}
    letters = []

    for code, spec in DIMENSIONS.items():
        vals = []
        for facet, polarity in spec["facets"].items():
            if facet in facets_scaled:
                vals.append(facets_scaled[facet] * polarity)
        if not vals:
            raise ValueError(f"缺少 {code} 所需 facet")

        pct_left = sum(vals) / len(vals)          # 0-100，越高越偏左（E/N/F/J）
        dist = abs(pct_left - 50)
        clarity, clarity_cn = _clarity(dist)

        left_letter, left_name = spec["left"]
        right_letter, right_name = spec["right"]
        if pct_left >= 50:
            letter, name = left_letter, left_name
            pct = pct_left
        else:
            letter, name = right_letter, right_name
            pct = 100 - pct_left

        letters.append(letter)
        dims[code] = {
            "name": spec["name"],
            "letter": letter,
            "name_cn": name,
            "left": {"letter": left_letter, "name_cn": left_name,
                     "pct": round(pct_left, 1)},
            "right": {"letter": right_letter, "name_cn": right_name,
                      "pct": round(100 - pct_left, 1)},
            "preference_pct": round(pct, 1),
            "clarity": clarity,
            "clarity_cn": clarity_cn,
            "distance": round(dist, 1),
            "source_facets": list(spec["facets"].keys()),
        }

    mbti_type = "".join(letters)
    stack = FUNCTION_STACKS.get(mbti_type)

    marginal = [c for c, d in dims.items() if d["clarity"] == "marginal"]

    return {
        "instrument": "mbti",
        "result": mbti_type,
        "dimensions": {c: d for c, d in dims.items()},
        "preferred_function_stack": stack,
        "clarity": {c: d["clarity"] for c, d in dims.items()},
        "method_note": (
            "本结果由 IPIP-NEO-120 的作答经构念映射导出，非 Myers-Briggs 官方量表。"
            "IPIP 题目的版权状态为公共领域；MBTI 为 Myers-Briggs 公司商标。"
            "四维偏好由连续分数二分化得到，边界临近的维度区分度有限。"
        ),
        "_marginal_dims": marginal,
        "_notice": (
            "MBTI 在主流心理测量学中不被视为严格的分类工具，重测信度偏低。"
            "本结果适合作为自我反思的语言，不宜作为定论。"
        ),
    }


# 16 型功能栈（与 specs/mbti.md 第三节一致）
FUNCTION_STACKS = {
    "INTJ": ["Ni", "Te", "Fi", "Se"], "INTP": ["Ti", "Ne", "Si", "Fe"],
    "ENTJ": ["Te", "Ni", "Se", "Fi"], "ENTP": ["Ne", "Ti", "Fe", "Si"],
    "INFJ": ["Ni", "Fe", "Ti", "Se"], "INFP": ["Fi", "Ne", "Si", "Te"],
    "ENFJ": ["Fe", "Ni", "Se", "Ti"], "ENFP": ["Ne", "Fi", "Te", "Si"],
    "ISTJ": ["Si", "Te", "Fi", "Ne"], "ISFJ": ["Si", "Fe", "Ti", "Ne"],
    "ESTJ": ["Te", "Si", "Ne", "Fi"], "ESFJ": ["Fe", "Si", "Ne", "Ti"],
    "ISTP": ["Ti", "Se", "Ni", "Fe"], "ISFP": ["Fi", "Se", "Ni", "Te"],
    "ESTP": ["Se", "Ti", "Fe", "Ni"], "ESFP": ["Se", "Fi", "Te", "Ni"],
}


def report(result: dict):
    print(f"MBTI 四维倾向（由 IPIP-NEO-120 映射）\n{'='*58}")
    print(f"\n类型：{result['result']}   功能栈："
          f"{' → '.join(result['preferred_function_stack'] or [])}\n")
    print(f"{'维度':<10s} {'倾向':<8s} {'强度':>6s}  {'分布':<28s} 清晰度")
    print("-" * 58)
    for code, d in result["dimensions"].items():
        L, R = d["left"], d["right"]
        bar_len = 20
        pos = int(round(L["pct"] / 100 * bar_len))
        bar = "░" * pos + "▓" * (bar_len - pos)
        print(f"{d['name']:<10s} {d['letter']}({d['name_cn']})"
              f"{'':<2s} {d['preference_pct']:>5.1f}%  {bar:<28s} {d['clarity_cn']}")
        print(f"{'':<10s} {L['letter']} {L['pct']:.0f}% / "
              f"{R['letter']} {R['pct']:.0f}%")

    if result["_marginal_dims"]:
        print(f"\n⚠ 以下维度接近临界，字母区分度低，不宜当作定论：")
        for c in result["_marginal_dims"]:
            d = result["dimensions"][c]
            print(f"   {d['name']}（{d['left']['pct']:.0f}:{d['right']['pct']:.0f}）")

    print(f"\n⚠ {result['_notice']}")
    print(f"   方法：{result['method_note']}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score import score_responses, load_items
    import random
    random.seed(11)
    r = score_responses([random.randint(1, 5) for _ in range(120)], load_items())
    report(score_mbti(r["facets"]))
