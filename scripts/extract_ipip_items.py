#!/usr/bin/env python3
"""
从 IPIP 官方计分键提取题库，生成结构化的 IPIP-NEO 题目表。

数据来源：https://ipip.ori.org/newNEOFacetsKey.htm
IPIP 属公共领域（public domain），可自由使用。

产出：data/items/ipip_neo_items.json
    每个 facet 下的正向题与反向题，含英文原文与中文译法（译法为自建，供中文施测用）。

注意：本页包含的是 IPIP-NEO 完整版的题目（每 facet 约 10 题，共 300 题）。
      IPIP-NEO-120 是它的 120 题子集。子集选择需依据官方对照表确定，
      本脚本先产出全量，子集筛选见 pick_neo120.py。
"""

import json
import re
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/tmp/ipip_facetkey.html")   # 先下载好的页面
OUT = ROOT / "data/items/ipip_neo_items.json"

# IPIP 命名 → 本规格命名 的对照
# 解读规格（specs/big-five.md）用的是 NEO-PI-R 常见命名，而 IPIP 页面用 IPIP 命名。
# 两者指同一构念，必须建立映射，否则分数与解读口径会对不上。
NAME_MAP = {
    "E1: FRIENDLINESS": ("E1_warmth", "Warmth"),
    "A2: MORALITY": ("A2_straightforwardness", "Straightforwardness"),
    "A4: COOPERATION": ("A4_compliance", "Compliance"),
    "O3: EMOTIONALITY": ("O3_emotionality", "Emotionality"),
    "O5: INTELLECT": ("O5_intellect", "Intellect"),
    "N5: IMMODERATION": ("N5_immoderation", "Immoderation"),
    "A5: MODESTY": ("A5_modesty", "Modesty"),
    "O6: LIBERALISM": ("O6_liberalism", "Liberalism"),
    # 与 specs/big-five.md 的 slug 对齐
    "O2: ARTISTIC INTERESTS": ("O2_artistic", "Artistic Interests"),
    "E1: FRIENDLINESS ": ("E1_warmth", "Warmth"),
}


def is_item(line: str) -> bool:
    """判断一行是否是完整题目（以句号结尾的英文句子）。"""
    return bool(re.match(r"^[A-Z][A-Za-z' ,\-]{3,90}\.$", line))


def is_item_fragment(line: str) -> bool:
    """候选题目片段：大写开头、无结尾句号、不含结构标记。"""
    if is_structural(line):
        return False
    return bool(re.match(r"^[A-Z][A-Za-z' ,\-]{3,90}$", line))


def is_structural(line: str) -> bool:
    """结构性行：facet 标题、信度标注、keyed 标记、页面标题。这些绝不能参与拼接。"""
    if re.match(r"^[NEOAC][1-6]:", line):
        return True
    if line.startswith(("(", "+", "–", "-", "=")):
        return True
    if line.lower().startswith(("alpha", "neo facets", "items in", "the ",
                                "measuring ", "scales", "and the ")):
        return True
    return False


def join_wrapped(lines):
    """把被 HTML 硬换行拆开的题目重新拼合。

    实例：'Believe that criminals should receive help rather than'
          + 'punishment.'  应合成一条完整题目。

    关键约束：拼接必须以「续行」为条件 —— 只有下一行以小写字母开头
    （即明确是同一句的延续）才拼接。遇到任何结构性行（facet 标题、keyed
    标记等）立即终止累积并将其原样输出。

    这个约束是必须的：早期版本无条件拼接所有片段，导致页面标题
    'Measuring Constructs ...' 一路吞掉 'Scales' 和整个 'N1: ANXIETY'
    标题及其题目，使 N1 这个 facet 彻底消失。
    """
    out, buf = [], None
    for ln in lines:
        if is_structural(ln):
            if buf:
                out.append(buf)
                buf = None
            out.append(ln)
            continue

        if buf is not None:
            # 只有小写开头才是明确续行；否则说明 buf 本身是残句，先落盘
            if ln[:1].islower():
                buf = buf + " " + ln
                if buf.endswith("."):
                    out.append(buf)
                    buf = None
            else:
                out.append(buf)      # 残句原样保留，交给下游判定
                buf = None
                out.append(ln)
            continue

        if ln.endswith("."):
            out.append(ln)
        elif is_item_fragment(ln):
            buf = ln
        else:
            out.append(ln)

    if buf:
        out.append(buf)
    return out


def canon(code_name: str):
    """把 IPIP 的 'N1: ANXIETY' 转成 ('N1_anxiety', 'Anxiety')。"""
    if code_name in NAME_MAP:
        return NAME_MAP[code_name]
    code, name = code_name.split(":", 1)
    code, name = code.strip(), name.strip()
    slug = f"{code}_{name.lower().replace('-', '_').replace(' ', '_')}"
    return slug, name.title()


def parse():
    raw = SRC.read_text(encoding="latin-1")
    raw = re.sub(r"<script.*?</script>", "", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", "\n", raw)
    text = html.unescape(raw).replace("\xa0", " ")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # 先把被 HTML 硬换行拆断的题目拼合。必须在主循环之前做，
    # 否则半截题目会被当成完整行，或后半截被丢弃 —— 都会污染题库。
    lines = join_wrapped(lines)

    facets = {}
    cur = None
    mode = None   # 'plus' | 'minus'

    for line in lines:
        m = re.match(r"^([NEOAC][1-6]):\s*([A-Z][A-Z \-]+)$", line)
        if m:
            key = f"{m.group(1)}: {m.group(2).strip()}"
            slug, name = canon(key)
            cur = {"code": m.group(1), "ipip_name": m.group(2).strip().title(),
                   "slug": slug, "name": name, "plus": [], "minus": []}
            facets[slug] = cur
            mode = None
            continue

        low = line.lower()
        if "keyed" in low:
            mode = "plus" if "+" in line else "minus"
            continue

        # 题目行：英文句子，含空格，长度适中，以句号结尾
        if cur and mode:
            if is_item(line) and line not in cur[mode]:
                cur[mode].append(line)

    return facets


def main():
    facets = parse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(facets, ensure_ascii=False, indent=2), encoding="utf-8")

    total = 0
    print(f"{'facet':32s} {'正':>4s} {'反':>4s}")
    print("-" * 44)
    for slug, f in facets.items():
        n = len(f["plus"]) + len(f["minus"])
        total += n
        print(f"{slug:32s} {len(f['plus']):4d} {len(f['minus']):4d}")
    print("-" * 44)
    print(f"{'合计':32s} {total:4d} 题")
    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
