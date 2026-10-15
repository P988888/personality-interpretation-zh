#!/usr/bin/env python3
"""
解读质量自动校验器 —— SFT 数据的第一道闸门。

职责：把机器能判定的违规项全部拦掉。机器判不了的（是否套模板、语气是否恰当）
留给人工审阅清单（docs/review-checklist.md）。

判定为 FAIL 的样本不进入训练集。原因是：一条带临床词汇的样本，
会教会模型「这个说法是存在的」，其危害远大于十条平庸样本。

用法：
    python3 scripts/validate.py data/raw/generated_bigfive.jsonl
"""

import json
import re
import sys
from pathlib import Path

# ------------------------------------------------------------ 一、临床/病理词
# 依据 specs/behavior-rules.md 第三节第 1 条：绝不诊断、绝不用病理化语言。
#
# 重要区分（实测修正）：一部分词本身是「边界声明」用的，不是诊断。例如
#   「本量表不测任何临床意义的状态」「这不是一个临床判断」
# 这是**规格要求出现的正确表述**，把它当违规拦掉等于惩罚正确行为。
# 因此对这类词改用「语境判定」：只有出现在肯定/归因语境才算违规。
#
# 纯违规词（任何语境都不该出现）：
CLINICAL_TERMS = [
    "焦虑症", "人格障碍", "自恋型", "边缘型", "回避型人格",
    "强迫症", "躁郁", "双相", "精神分裂", "偏执型", "病态", "心理疾病",
    "心理问题", "确诊", "患病", "需要就医",
]

# 语境敏感词：仅当用于描述受测者时才违规。
# 若前面有否定词/边界声明（不测、不是、不涉及、不代表、没有、不构成、
# 不等同、不轻易、不会有），则放行。
#
# 实测教训：这个词表被误伤过三轮。解读里「不等于情绪不稳定」「不是临床诊断」
# 「不会有障碍」都是**规格要求的正确表述**（主动澄清、声明边界），
# 若按裸词拦截，等于系统性地惩罚正确行为。
CONTEXT_TERMS = [
    "临床", "症状", "治疗", "诊断",
    "情绪不稳定", "情绪有问题", "抑郁", "焦虑",
]
# 说明：「精神」未列入 —— 它在中文里日常用法太常见（有精神、精神焕发、
# 精神头足），而真正需要拦的「精神分裂」已在 CLINICAL_TERMS 纯违规列表中。

# 「障碍」单独处理：它既可能是病理词，也可能是日常用语（遇到障碍、没有障碍）。
# 只有跟在这些词后面才算病理用法。
PATHOLOGY_COLLOCATIONS = ["人格障碍", "心理障碍", "情绪障碍", "精神障碍",
                          "回避型", "边缘型", "自恋型", "行为障碍", "适应障碍"]

# 边界声明的标志：出现在敏感词附近即视为合法用法
BOUNDARY_MARKERS = [
    "不测", "不涉及", "不是", "不代表", "不构成", "没有", "不做",
    "不属于", "并非", "无关", "不应", "不能", "不宜", "不该",
    "意义的状态", "意义上的", "不是诊断",
    "不作", "不含", "不等同", "不衡量", "不评估", "不用于", "不可以",
    "而非", "不是临床", "不等于", "不轻易", "不会有", "无法替代",
    "意义上", "避免", "澄清", "不等于说",
]

# ------------------------------------------------------------ 二、命运预测词
# 依据 behavior-rules 第三节第 2 条：绝不预测命运、职业、关系成败。
FATE_PATTERNS = [
    (r"适合(做|从事|选择)[^。，]{0,12}(工作|职业|行业|管理|销售|艺术|设计)", "职业适配预测"),
    (r"不适合(做|从事|结婚|恋爱|管理)", "否定式命运预测"),
    (r"(你|这类人|这种人)(会|将|注定|必然|一定)(成功|失败|离婚|孤独|孤独终老|幸福|有钱)", "结局预测"),
    (r"(注定|天生就是|命定)", "宿命论表述"),
    (r"你(的)?(智商|智力|能力)(很高|偏低|较高|较差|出众)", "能力判断"),
]

# ------------------------------------------------------------ 三、迎合式表述
# 依据 behavior-rules 第三节第 3 条：绝不迎合求认同式提问。
# 在解读正文中体现为无依据的赞美与优越感制造。
FLATTERY_PATTERNS = [
    (r"你(?:是|非常|很|极其|特别|真的|确实)[^。，\n]{0,6}"
     r"(?:优秀|聪明|厉害|出色|独特|完美|了不起)", "无依据赞美"),
    (r"(?:罕见|少见|稀有)(?:的)?(?:类型|人格|存在)", "优越感制造"),
    (r"(?:万里挑一|凤毛麟角|独一无二的天才|出类拔萃|天赋异禀)", "夸张赞美"),
]

# ------------------------------------------------------------ 四、模板化检测
# 检测「换个人也成立」的通用话术。
TEMPLATE_PHRASES = [
    "你是一个非常有创造力的人",
    "能够平衡好工作与生活",
    "是一个可靠的朋友",
    "你拥有丰富的内心世界",   # 单独出现无意义，需配合具体依据
    "你是一个独特的人",
]

# ------------------------------------------------------------ 四、化验单口吻
# 依据 specs/behavior-rules.md 第三节「以 facet 为单位指的是语义颗粒度，
# 不是报出技术标签」。面向受测者的解读不应出现内部编码或英文术语。
#
# 实测发现：初版规格此处的表述有歧义（「以 facet 为主要解读单位」），
# 导致 44% 的生成样本在正文里写出「N4 社交不安很高」这类化验单式表达。
# 规格已修正，校验器相应加上这条拦截。
JARGON_PATTERNS = [
    (r"\b[NEOAC][1-6]\b", "facet 编码"),
    (r"\b(grecariousness|assertiveness|orderliness|self-discipline|"
     r"straightforwardness|self-consciousness|immoderation|vulnerability|"
     r"extraversion|neuroticism|openness|agreeableness|conscientiousness)\b",
     "英文术语"),
    (r"\b(O1|O2|O3|O4|O5|O6|C1|C2|C3|C4|C5|C6|E1|E2|E3|E4|E5|E6|"
     r"A1|A2|A3|A4|A5|A6|N1|N2|N3|N4|N5|N6)\b", "facet 编码"),
]


def check_jargon(text: str) -> list:
    bad = []
    for pat, label in JARGON_PATTERNS:
        for m in re.finditer(pat, text):
            bad.append(f"{label}：「{m.group(0)}」")
    return bad


# ------------------------------------------------------------ 五、合规要求
# 依据 behavior-rules 第五节：局限性说明必须在场。
REQUIRED_ELEMENTS = {
    "局限性说明": [r"自陈", r"倾向", r"情境", r"不测", r"不涉及能力", r"如何看待自己"],
    "facet或组合分析": [r"但.{0,20}(却|而|却反而)", r"组合", r"张力", r"同时"],
}

MIN_LEN = 400      # 过短必然不合格
MAX_LEN = 2200     # 过长说明未遵守长度控制


def check_context_terms(text: str) -> list:
    """语境敏感的临床词检查：只有肯定语境才算违规。

    判定方式：取敏感词**所在句子**内、其前方的全部文字作为前缀，
    检查是否含边界声明标志。

    两轮实测教训：
      1. 固定 20 字窗口会跨句串扰（前一句的「不等同」会放行后一句的违规）。
         故限定在同一句内。
      2. 句内再设 12 字上限又过紧 ——「本解读不构成任何医疗或心理诊断」
         的前缀有 13 字，被误判为违规。故句内不再设长度上限。
    """
    bad = []
    for term in CONTEXT_TERMS:
        for m in re.finditer(re.escape(term), text):
            start = 0
            for sep in "。！？\n":
                p = text.rfind(sep, 0, m.start())
                start = max(start, p + 1)
            prefix = text[start:m.start()]
            if any(mk in prefix for mk in BOUNDARY_MARKERS):
                continue
            bad.append(f"临床词汇（肯定语境）：「{term}」← …{prefix[-16:]}{term}")
    # 「障碍」类：仅病理搭配算违规
    for coll in PATHOLOGY_COLLOCATIONS:
        if coll in text:
            bad.append(f"临床词汇：「{coll}」")
    return bad


def check_fabricated_numbers(text: str, item: dict) -> list:
    """检查编造的具体数字。

    设计演变（实测修正）：早期版本对任何不在输入中的数字都报违规，
    结果大量误伤 —— 解读里正常会写「接近 90」「超过 80」这类对输入分数的
    复述或近似表述，还有「30 个 facet」「27 个居中」这类结构性计数。

    真正要拦的是**编造的人群排名与百分位**，依据 behavior-rules 第三节第 7 条：
    不得虚构未提供的常模数据、百分位。

    因此改为：只扫描「数字 + 百分位/排名语境」的组合。
    """
    patterns = [
        (r"(前|超过|高于|低于|优于|落后于)\s*\d{1,3}\s*(%|％|个?百分点|的人)", "编造百分位"),
        (r"\d{1,3}\s*(%|％)\s*(的人|分位|百分位)", "编造百分位"),
        (r"(第|排名)\s*\d{1,3}\s*(名|位|百分位)", "编造排名"),
        (r"百分位\s*(是|为|约|大约)?\s*\d{1,3}", "编造百分位"),
        (r"比\s*\d{1,3}\s*(%|％)\s*的人", "编造人群比较"),
        (r"(超过|高于)\s*(全国|全球|同龄人|平均)\s*\d", "编造人群比较"),
    ]
    bad = []
    for pat, label in patterns:
        for m in re.finditer(pat, text):
            bad.append(f"{label}：「{m.group(0)}」")
    return bad


def check_normative_source(item: dict, text: str) -> list:
    """无常模（或非百分位刻度）却宣称人群位置，是更隐蔽的违规。

    依据 specs/big-five.md 第四节：norm_source 缺失时不得输出任何百分位表述。

    实测修正：不能只看 norm_source 字段是否存在。本项目 score.py 输出的是
    原始分线性映射到 0-100 的刻度（scale="raw_and_normalized"），并非百分位。
    即使 norm_source 标了来源，也不支持「高于多数人」这类人群比较。
    因此判定条件是：scale 必须是 percentile 才允许人群比较表述。
    """
    scale = (item.get("scale") or "").lower()
    if scale == "percentile" and item.get("norm_source"):
        return []
    bad = []
    for pat, label in [
        (r"(高于|低于|超过|优于)\s*(大多数|多数|一般人|普通人|常人|同龄人)", "无常模却宣称人群比较"),
        (r"(比|较)\s*(大多数|多数|一般人|普通人)\s*(人)?\s*(更|要)", "无常模却宣称人群比较"),
        (r"(处于|位于)\s*(人群)?\s*(中上|中下|偏上|偏下)\s*(水平|位置)", "无常模却宣称人群位置"),
    ]:
        for m in re.finditer(pat, text):
            bad.append(f"{label}：「{m.group(0)}」")
    return bad


def validate(item: dict) -> tuple:
    """返回 (verdict, reasons)。verdict ∈ {PASS, FAIL}。"""
    output = item.get("output", "")
    reasons = []

    if not output or len(output) < MIN_LEN:
        return "FAIL", [f"输出过短或缺失（{len(output)} 字）"]
    if len(output) > MAX_LEN:
        reasons.append(f"输出过长（{len(output)} 字 > {MAX_LEN}）")

    # 1. 临床词汇 —— 纯违规词全文扫描
    for t in CLINICAL_TERMS:
        if t in output:
            reasons.append(f"临床词汇：「{t}」")

    # 1b. 语境敏感词 —— 仅肯定语境违规（边界声明合法）
    reasons.extend(check_context_terms(output))

    # 2. 命运预测
    for pat, label in FATE_PATTERNS:
        m = re.search(pat, output)
        if m:
            reasons.append(f"{label}：「{m.group(0)[:24]}」")

    # 3. 迎合赞美
    for pat, label in FLATTERY_PATTERNS:
        m = re.search(pat, output)
        if m:
            reasons.append(f"{label}：「{m.group(0)[:24]}」")

    # 4. 模板话术
    for ph in TEMPLATE_PHRASES:
        if ph in output:
            reasons.append(f"模板话术：「{ph}」")

    # 4b. 化验单口吻（内部编码 / 英文术语）
    reasons.extend(check_jargon(output))

    # 5. 编造数字 —— 只拦百分位/排名语境（复述输入分数是合法的）
    reasons.extend(check_fabricated_numbers(output, item))
    reasons.extend(check_normative_source(item, output))

    # 6. 必备要素
    for label, pats in REQUIRED_ELEMENTS.items():
        if not any(re.search(p, output) for p in pats):
            reasons.append(f"缺少必备要素：{label}")

    return ("PASS" if not reasons else "FAIL"), reasons


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/validate.py <jsonl文件>")
        sys.exit(1)

    src = Path(sys.argv[1])
    root = Path(__file__).resolve().parent.parent

    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    passed, failed = [], []
    for it in items:
        verdict, reasons = validate(it)
        (passed if verdict == "PASS" else failed).append((it, reasons))

    out_pass = root / "data/curated/bigfive_pass.jsonl"
    out_fail = root / "data/rejected/bigfive_fail.jsonl"
    for p in (out_pass, out_fail):
        p.parent.mkdir(parents=True, exist_ok=True)

    with out_pass.open("w", encoding="utf-8") as f:
        for it, _ in passed:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with out_fail.open("w", encoding="utf-8") as f:
        for it, reasons in failed:
            it["_reject_reasons"] = reasons
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"总计 {len(items)} 条")
    print(f"  通过 {len(passed)} -> {out_pass}")
    print(f"  拒绝 {len(failed)} -> {out_fail}")

    if failed:
        print("\n拒绝原因分布：")
        from collections import Counter
        c = Counter(r.split("：")[0] for _, rs in failed for r in rs)
        for reason, n in c.most_common():
            print(f"  {reason:24s} {n}")

    # 通过率过低是严重信号
    if items and len(passed) / len(items) < 0.5:
        print(f"\n⚠ 通过率仅 {len(passed)/len(items):.0%}，低于 50%。")
        print("  这通常说明生成用的提示词没有正确注入 specs/ 规则，")
        print("  应回头检查提示词组装，而不是放宽校验标准。")


if __name__ == "__main__":
    main()
