#!/usr/bin/env python3
"""
MBTI 分数组合生成器 —— 为 SFT 数据生产提供有控制的输入。

与 gen_prompts_bigfive.py 的差别：
    大五可以直接构造 facet 分数；MBTI 的输入结构不同，需要按
    「四维偏好百分比 + 清晰度 + 功能栈」来构造。

    关键是要覆盖清晰度的各种组合，因为「边界情况」正是 MBTI 最需要
    被正确解读的地方：
      - 四维都明显（clear）→ 类型较可信
      - 有维度临界（marginal）→ 必须提示「不必固定在某一侧」
      - 全部临界 → 应说明这个类型几乎没有区分度

    训练数据若不覆盖这些情况，模型会对所有输入都给出同样确信的语气 ——
    那恰恰是业余与专业的分界。

产出：data/raw/prompts_mbti.jsonl
"""

import json
import random
from pathlib import Path

random.seed(20251015)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/prompts_mbti.jsonl"

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

# 每个维度的字母对（左 = 分数高时取的字母）
PAIRS = {
    "EI": ("E", "I"),
    "SN": ("N", "S"),
    "TF": ("F", "T"),
    "JP": ("J", "P"),
}


def clarity_of(pct: float) -> str:
    """按 score_mbti.py 的同一标准判定清晰度。"""
    dist = abs(pct - 50)
    if dist >= 20:
        return "clear"
    if dist >= 10:
        return "moderate"
    if dist >= 5:
        return "slight"
    return "marginal"


def make_dimensions(profile: dict) -> tuple:
    """profile: {dim: 左字母占比}。返回 (dimensions, letters, clarity)。"""
    dims, letters, clarity = {}, [], {}
    for code, (left, right) in PAIRS.items():
        p = profile[code]
        p = max(5, min(95, p))
        letter = left if p >= 50 else right
        letters.append(letter)
        dims[code] = {left: round(p, 1), right: round(100 - p, 1)}
        clarity[code] = clarity_of(p)
    return dims, "".join(letters), clarity


# 清晰度档位：每档给出「偏离中点的幅度」，配合字母方向构成具体分数。
#
# 设计修正：初版用固定的场景表（四个维度各写死一个占比），结果类型覆盖
# 只有 11/16，且分布严重偏斜（ENTJ 41 条 vs INTP 1 条）—— 因为场景数有限、
# 抖动又不足以改变字母。那样训出来模型会对某些类型过度适应。
# 改为 16 型 × 清晰度档位 的笛卡尔积，保证每个类型、每档清晰度都充分覆盖。
CLARITY_LEVELS = {
    "clear":    (20, 35),   # 偏离 20-35 点
    "moderate": (10, 19),
    "slight":   (5, 9),
    "marginal": (0, 4),
}


def profile_for(type_str: str, level: str) -> dict:
    """按类型字母与清晰度档位，生成四维占比。"""
    lo, hi = CLARITY_LEVELS[level]
    prof = {}
    for code, (left, right) in PAIRS.items():
        pos = {"EI": 0, "SN": 1, "TF": 2, "JP": 3}[code]
        letter = type_str[pos]
        dist = random.uniform(lo, hi)
        prof[code] = 50 + dist if letter == left else 50 - dist
    return prof


def main():
    out = []
    ALL_TYPES = list(FUNCTION_STACKS.keys())      # 16 型
    per_combo = 3                                  # 每 (型 × 清晰度档) 生成 3 条

    for type_str in ALL_TYPES:
        for level in CLARITY_LEVELS:
            for i in range(per_combo):
                prof = profile_for(type_str, level)
                dims, letters, clarity = make_dimensions(prof)
                item = {
                    "instrument": "mbti",
                    "result": letters,
                    "dimensions": dims,
                    "preferred_function_stack": FUNCTION_STACKS[letters],
                    "clarity": clarity,
                    "_meta": {"scenario": level, "variant": i, "type": letters},
                }
                out.append(item)

    random.shuffle(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"生成 {len(out)} 条 MBTI 输入 -> {OUT}")

    from collections import Counter
    tc = Counter(x["_meta"]["type"] for x in out)
    print(f"\n类型覆盖：{len(tc)} / 16")
    for t, n in sorted(tc.items()):
        print(f"  {t}  {n}")

    print(f"\n清晰度场景分布：")
    for tag, n in Counter(x["_meta"]["scenario"] for x in out).most_common():
        print(f"  {tag:10s} {n}")

    # 验证：每条样本声明的清晰度应与分数实际算出的清晰度一致
    bad = 0
    for x in out:
        for code, (left, _) in PAIRS.items():
            p = x["dimensions"][code][left]
            if clarity_of(p) != x["clarity"][code]:
                bad += 1
    print(f"\n清晰度自洽性：{'✓ 全部一致' if bad == 0 else f'✗ {bad} 处不一致'}")
    print(f"理论总量：16 型 × {len(CLARITY_LEVELS)} 档 × {per_combo} = "
          f"{16 * len(CLARITY_LEVELS) * per_combo}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
