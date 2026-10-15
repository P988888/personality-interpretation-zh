#!/usr/bin/env python3
"""
MBTI 解读质量校验器。

与大五校验器的差异（依据 specs/mbti.md）：

1. **必须包含争议说明** —— 这是 MBTI 部分特有的硬性要求（规格第五节），
   缺失即不合格。大五没有这一条。
2. **必须处理清晰度** —— 当输入含 marginal/slight 维度时，解读必须说明
   「不必把字母当定论」。规格第四节规则 2。这是 MBTI 最体现专业度的地方。
3. **必须基于功能栈** —— 解读应讨论认知功能，而非只复述四个字母。
4. **必须说明来源** —— 本项目的结果由构念映射导出，必须声明非官方量表。

用法：
    python3 scripts/validate_mbti.py data/raw/generated_mbti.jsonl
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 继承大五校验器的通用违规检测
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import (CLINICAL_TERMS, check_context_terms,  # noqa: E402
                      check_jargon, FATE_PATTERNS, FLATTERY_PATTERNS)

MIN_LEN, MAX_LEN = 500, 2400

# MBTI 专属：争议说明的关键要素
CONTROVERSY_MARKERS = [
    r"重测", r"信度", r"争议", r"学术", r"心理测量",
    r"不被视为", r"不是科学", r"二分", r"连续",
    r"不宜作为定论", r"反思", r"起点",
]

# 功能栈关键词
FUNCTION_TERMS = [
    "功能", "主导", "辅助", "劣势", "第三",
    "Ni", "Ne", "Si", "Se", "Ti", "Te", "Fi", "Fe",
    "内倾直觉", "外倾直觉", "内倾实感", "外倾实感",
    "内倾思维", "外倾思维", "内倾情感", "外倾情感",
]

# 边界提示语（当有 marginal/slight 维度时必须出现）
BOUNDARY_HINTS = [
    r"不必(把自己)?固定", r"区分度(低|不高|有限)", r"接近(中间|中点|临界)",
    r"边界", r"不一定", r"两面(都|皆)", r"两种(都|皆)", r"临界",
    r"不宜(当作|作为)(定论|结论)", r"可能(两种情况)?都有",
]


def validate_mbti(item: dict) -> tuple:
    out = item.get("output", "")
    reasons = []

    if not out or len(out) < MIN_LEN:
        return "FAIL", [f"输出过短或缺失（{len(out)} 字）"]
    if len(out) > MAX_LEN:
        reasons.append(f"输出过长（{len(out)} 字）")

    # --- 通用违规（复用大五校验器）---
    for t in CLINICAL_TERMS:
        if t in out:
            reasons.append(f"临床词汇：「{t}」")
    reasons.extend(check_context_terms(out))
    reasons.extend(check_jargon(out))
    for pat, label in FATE_PATTERNS:
        m = re.search(pat, out)
        if m:
            reasons.append(f"{label}：「{m.group(0)[:20]}」")
    for pat, label in FLATTERY_PATTERNS:
        m = re.search(pat, out)
        if m:
            reasons.append(f"{label}：「{m.group(0)[:20]}」")

    # --- MBTI 专属检查 ---

    # 1. 争议说明必须在场
    if not any(re.search(p, out) for p in CONTROVERSY_MARKERS):
        reasons.append("缺少必备要素：MBTI 争议说明")

    # 2. 功能栈讨论必须在场
    if not any(re.search(p, out) for p in FUNCTION_TERMS):
        reasons.append("缺少必备要素：认知功能栈分析")

    # 3. 有条件地要求边界提示
    clarity = item.get("clarity", {})
    weak = [c for c, v in clarity.items() if v in ("marginal", "slight")]
    if weak:
        if not any(re.search(p, out) for p in BOUNDARY_HINTS):
            reasons.append(f"缺少必备要素：边界提示（有 {','.join(weak)} 维度接近临界）")

    # 4. 来源声明
    if not re.search(r"(非|不是).{0,12}(官方|Myers|MBTI 官方)|构念|映射|IPIP", out):
        reasons.append("缺少必备要素：结果来源声明")

    return ("PASS" if not reasons else "FAIL"), reasons


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "data/raw/generated_mbti.jsonl"
    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    passed, failed = [], []
    for it in items:
        v, rs = validate_mbti(it)
        (passed if v == "PASS" else failed).append((it, rs))

    op = ROOT / "data/curated/mbti_pass.jsonl"
    of = ROOT / "data/rejected/mbti_fail.jsonl"
    for p in (op, of):
        p.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", encoding="utf-8") as f:
        for it, _ in passed:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with of.open("w", encoding="utf-8") as f:
        for it, rs in failed:
            it["_reject_reasons"] = rs
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"总计 {len(items)} 条")
    print(f"  通过 {len(passed)} -> {op}")
    print(f"  拒绝 {len(failed)} -> {of}")
    if failed:
        from collections import Counter
        print("\n拒绝原因分布：")
        for r, n in Counter(r.split("：")[0] for _, rs in failed for r in rs).most_common():
            print(f"  {r:28s} {n}")


if __name__ == "__main__":
    main()
