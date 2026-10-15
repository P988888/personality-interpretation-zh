#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 A/B 两份解读做客观的规则合规检查。

词表和判定口径与 scripts/validate.py 保持一致，但这里只做单条输出的抽查，
不替代 validate.py 对训练集的批量校验。

用法：python3 training/eval_ab.py /tmp/ab_test.json
"""
import json
import re
import sys

CLINICAL = ["焦虑症", "抑郁症", "人格障碍", "心理障碍", "情绪障碍", "自恋型",
            "边缘型", "回避型人格", "强迫症", "精神疾病", "病态", "症状", "临床诊断"]
FABRICATED = ["百分位", "百分等级", "排在前", "超过", "常模对比", "PR值",
              "高于同龄", "人群平均", "标准差为"]
# 逐维度平铺的标题式写法，如 "#### 开放性（O）"、"### 神经质（N）"
DOMAIN_HEADING = re.compile(r"^#{2,4}\s*[^\n]{0,6}(开放性|尽责性|外向性|宜人性|神经质)\s*[（(][OCEAN][）)]", re.M)
DOMAIN_LETTER = re.compile(r"[（(][OCEAN][）)]")
FACET_CODE = re.compile(r"\b[OCEAN][1-6](?:_[a-z_]+)?\b")

COST_WORDS = ["代价", "摩擦", "盲区", "风险", "不舒服", "问题在于", "劣势",
              "陷入", "消耗", "懊恼", "负担", "拉扯", "张力", "不足"]
BOUNDARY = ["局限性", "不预测", "不构成", "不涉及", "不是定论", "情境", "会随",
            "仅供参考", "不作为", "不宜", "不能替代"]
SECOND_PERSON = re.compile(r"你")

# 套话检测：这些句子安在任何人身上都成立，是规则 8 明令禁止的模板话术。
# 这一段是基座输出和微调输出差别最直观的地方之一。
BOILERPLATE = ["独一无二", "更好地了解自己", "希望这份报告", "每个人都是",
               "祝你", "相信你一定能", "做最好的自己", "仅供参考之用",
               "你的个性特质正是", "希望你能够"]

# 五个维度是否都被解读到。基座模型漏掉了整个宜人性（A=34），
# 所以"覆盖率"比"违规数"更能暴露问题：漏解读不算违规，但属于实质性缺失。
DOMAIN_KEYWORDS = {
    "开放性": ["开放", "想象", "审美", "艺术", "尝新", "思辨", "好奇", "观念", "新鲜"],
    "尽责性": ["尽责", "条理", "秩序", "自律", "成就", "谨慎", "自我效能", "计划", "纪律"],
    "外向性": ["外向", "热情", "乐群", "果断", "活力", "兴奋", "积极情绪", "社交", "人群"],
    "宜人性": ["宜人", "信任", "坦诚", "利他", "顺从", "谦逊", "同理", "让步"],
    "神经质": ["神经质", "焦虑", "愤怒", "抑郁", "自我意识", "冲动", "脆弱", "敏感", "情绪反应"],
}

# 否定标记。踩过的坑：模型写"本解读不涉及任何百分位"是**正确的边界声明**，
# 早期版本会把「百分位」这个词直接判成"编造常模数字"，精确率很差。
# 口径与 validate.py 一致：回溯到句首，句内出现否定标记就不算违规。
NEGATION = ["不测", "不涉及", "不是", "不代表", "不构成", "没有", "不做", "不作",
            "不含", "不等同", "不衡量", "不评估", "不轻易", "不会有", "无法替代",
            "而非", "不等于", "无需", "不必", "不能", "不宜", "未提供", "不预测"]
SENT_START = re.compile(r"[。！？\n]")


def is_negated(txt, idx):
    """命中词是否处在否定语境里（回溯到句首，看这一段有没有否定标记）。"""
    start = 0
    for m in SENT_START.finditer(txt):
        if m.end() > idx:
            break
        start = m.end()
    return any(n in txt[start:idx + 1] for n in NEGATION)


def scan(txt, words):
    """返回真正违规的命中（排除否定语境里的边界声明）。"""
    bad = []
    for w in words:
        i = txt.find(w)
        while i >= 0:
            if not is_negated(txt, i):
                bad.append(w)
            i = txt.find(w, i + 1)
    return bad


def check(name, txt):
    facet_hits = FACET_CODE.findall(txt)
    covered = [d for d, kws in DOMAIN_KEYWORDS.items() if any(k in txt for k in kws)]
    return {
        "字数": len(txt),
        "facet编码": len(facet_hits),
        "dim编码(如(O))": len(DOMAIN_LETTER.findall(txt)),
        "按维度分节": len(DOMAIN_HEADING.findall(txt)),
        "临床病理词": len(scan(txt, CLINICAL)),
        "编造常模数字": len(scan(txt, FABRICATED)),
        "套话模板句": len([w for w in BOILERPLATE if w in txt]),
        "维度覆盖(满分5)": len(covered),
        "讲代价": len([w for w in COST_WORDS if w in txt]),
        "局限性说明": len([w for w in BOUNDARY if w in txt]),
        "第二人称": len(SECOND_PERSON.findall(txt)),
        "_缺的维度": "、".join(d for d in DOMAIN_KEYWORDS if d not in covered) or "无",
    }


def main():
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab_test.json",
                       encoding="utf-8"))
    a = check("基座", d["base"])
    b = check("微调", d["finetuned"])

    keys = [k for k in a.keys() if not k.startswith("_")]
    w = 18
    print(f"{'指标':<{w}}{'基座':>12}{'微调后':>12}   判定")
    print("-" * 62)
    # 越少越好的指标
    LOWER_BETTER = {"facet编码", "dim编码(如(O))", "按维度分节", "临床病理词",
                    "编造常模数字", "套话模板句"}
    for k in keys:
        va, vb = a[k], b[k]
        note = ""
        if k in LOWER_BETTER:
            if vb < va:
                note = "✓ 微调更符合规范"
            elif vb > va:
                note = "✗ 微调更差"
            elif va == 0:
                note = "= 都合规"
            else:
                note = "= 都不合规"
        else:  # 越多越好
            if vb > va:
                note = "✓ 微调更充分"
            elif vb < va:
                note = "✗ 微调更少"
            else:
                note = "= 持平"
        print(f"{k:<{w}}{va:>12}{vb:>12}   {note}")

    print(f"\n基座漏解读的维度：{a['_缺的维度']}")
    print(f"微调漏解读的维度：{b['_缺的维度']}")


if __name__ == "__main__":
    main()
