#!/usr/bin/env python3
"""
分数组合生成器 —— 为 SFT 数据生产提供有控制的输入样本。

设计要点：
1. 不做均匀随机。均匀随机会产生大量「所有维度都在中点附近」的无聊样本，
   而这类样本恰恰是最不该被解读的（区分度低）。必须按组合类型分层采样。
2. 覆盖规格文件中「高分值组合速查」列的 12 种典型组合。
3. 覆盖边界情况：全高点、全低点、全中点、单项极端。
4. 每条样本带 metadata，说明它属于哪一类组合，便于后续按类型配平与分析。

输出：data/raw/prompts_bigfive.jsonl，每行一个待生成的输入。
"""

import json
import random
from pathlib import Path

random.seed(20251015)  # 固定种子，保证可复现

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- facet 结构
DOMAINS = {
    "O": ["O1_imagination", "O2_artistic", "O3_emotionality",
          "O4_adventurousness", "O5_intellect", "O6_liberalism"],
    "C": ["C1_self_efficacy", "C2_orderliness", "C3_dutifulness",
          "C4_achievement_striving", "C5_self_discipline", "C6_cautiousness"],
    "E": ["E1_warmth", "E2_gregariousness", "E3_assertiveness",
          "E4_activity_level", "E5_excitement_seeking", "E6_cheerfulness"],
    "A": ["A1_trust", "A2_straightforwardness", "A3_altruism",
          "A4_compliance", "A5_modesty", "A6_sympathy"],
    "N": ["N1_anxiety", "N2_anger", "N3_depression",
          "N4_self_consciousness", "N5_immoderation", "N6_vulnerability"],
}
ALL_FACETS = [f for fs in DOMAINS.values() for f in fs]

# 解读纪律：40-60 为「不宜明确解读」的中点区。生成器要控制落在该区的比例，
# 但同时必须保留一部分中点样本 —— 模型要学会「承认这里测不准」。
LOW, HIGH = 15, 92          # 百分位取值范围
MID_LO, MID_HI = 40, 60     # 中点区


def clip(v: int) -> int:
    return max(LOW, min(HIGH, int(round(v))))


def in_mid(v: int) -> bool:
    return MID_LO <= v <= MID_HI


def band(center: int, spread: int = 12) -> int:
    """在 center 附近生成带噪声的分数，模拟真实 facet 间的相关性。

    关键：中心值越靠近中点区(50)，噪声必须越小。否则一个 center=52 的维度
    会被抖出 38~66 的 facet 散布，产生大量「没什么可讲」的中点样本，
    使训练数据被平庸样本稀释。
    """
    # 距离中点越近，噪声越小（center=50 时衰减到 40%，center=80 时保持 100%）
    dist = abs(center - 50) / 50.0
    damping = 0.4 + 0.6 * min(1.0, dist)
    return clip(random.gauss(center, spread * damping))


# ---------------------------------------------------------------- 组合模板
# 每条：(O, C, E, A, N) 维度中心值。这些取自 specs/big-five.md 第三节组合速查表。
#
# 设计纪律：每个模板必须让「承载该组合特征的维度」有强区分度（偏离50达20点以上），
# 其余维度也要给出温和但明确的偏移（偏离50约10点以上）。
# 真实的人不会只在两个维度上有倾向而其余全中性 —— 那样会产出大量无内容的中点
# facet，稀释训练数据。这一点是在第一轮试跑（中点占比 53%）后修正的。
COMBINATIONS = {
    "O_high_C_low":        dict(center=(78, 30, 58, 62, 44),
                                note="想法多启动快收尾难，靠新鲜感驱动"),
    "O_low_C_high":        dict(center=(30, 78, 46, 66, 40),
                                note="稳健可靠路径依赖，执行力强但保守"),
    "E_high_A_high":       dict(center=(62, 66, 80, 78, 36),
                                note="热情好相处爱社交，易过度付出"),
    "E_high_A_low":        dict(center=(66, 64, 78, 28, 46),
                                note="主动强势爱主导，推动力强冲突成本高"),
    "E_low_N_high":        dict(center=(58, 62, 26, 64, 76),
                                note="安静且内心戏多，慢热易内耗"),
    "E2_high_E3_low":      dict(center=(60, 58, 68, 70, 42),
                                note="爱人群但不主导，最常被误读为内向"),
    "C_high_N_high":       dict(center=(58, 80, 56, 62, 78),
                                note="高标准加高焦虑，易倦怠"),
    "A_high_N_high":       dict(center=(58, 60, 54, 82, 76),
                                note="让步多且敏感，边界问题"),
    "A_low_E3_high":       dict(center=(64, 66, 78, 26, 42),
                                note="直接且强势，效率高摩擦成本高"),
    "O3_high_N_low":       dict(center=(78, 62, 62, 70, 22),
                                note="感受丰富但稳，与神经质高区分"),
    "N4_high_E2_high":     dict(center=(62, 60, 70, 68, 74),
                                note="想社交又怕被评价，核心矛盾"),
    "C5_low_O4_high":      dict(center=(78, 28, 64, 64, 46),
                                note="兴趣驱动启动靠热情，非拖延"),

    # ---- 边界与对照组 ----
    "all_high":            dict(center=(84, 84, 84, 84, 18),
                                note="全高（N低），极端样本"),
    "all_low":             dict(center=(20, 20, 20, 20, 20),
                                note="全低，极端样本"),
    "all_mid":             dict(center=(50, 50, 50, 50, 50),
                                note="全中点，训练模型承认测不准", spread=4),
    "spike_O":             dict(center=(90, 44, 60, 64, 42),
                                note="开放性孤高，其余温和偏移"),
    "spike_N":             dict(center=(58, 58, 56, 62, 90),
                                note="神经质孤高，措辞压力测试"),
    "spike_C":             dict(center=(46, 90, 58, 66, 44),
                                note="尽责性孤高，其余温和偏移"),
}


def make_facets(domain_center: dict, spread: int = 12) -> dict:
    """由维度中心值展开到 30 个 facet，加入 facet 内噪声。"""
    facets = {}
    for dom, keys in DOMAINS.items():
        c = domain_center[dom]
        for k in keys:
            facets[k] = band(c, spread)
    return facets


def domain_from_facets(facets: dict) -> dict:
    """由 facet 反算维度分（取均值），保证输入自洽。"""
    return {d: clip(sum(facets[k] for k in ks) / len(ks))
            for d, ks in DOMAINS.items()}


def build_prompt_input(facets: dict, domains: dict, tag: str, note: str) -> dict:
    return {
        "instrument": "big-five",
        "version": "IPIP-NEO-120",
        # 刻度必须与实际计分口径一致。score.py 输出的是 raw_and_normalized
        # （原始分线性映射到 0-100），不是百分位。早期此处误写为 "percentile"
        # 并与 norm_source 一起造成下游校验器放行「高于多数人」这类人群比较表述。
        "scale": "raw_and_normalized",
        "norm_source": None,
        "domains": domains,
        "facets": facets,
        "_meta": {"combination": tag, "note": note},
    }


def main():
    out = []
    per_combo = 100   # 每个组合类型生成 100 份变体（18类 × 100 = 1800 条）

    for tag, spec in COMBINATIONS.items():
        spread = spec.get("spread", 12)
        # 全中点样本用更小的 spread，避免抖动出中点区
        for i in range(per_combo):
            dom_center = dict(zip("OCEAN", spec["center"]))
            facets = make_facets(dom_center, spread)
            domains = domain_from_facets(facets)

            # 统计中点区占比，写入 meta —— 便于后续检查分布
            mid_count = sum(1 for v in facets.values() if in_mid(v))

            item = build_prompt_input(facets, domains, tag, spec["note"])
            item["_meta"]["mid_band_facets"] = mid_count
            item["_meta"]["variant"] = i
            out.append(item)

    random.shuffle(out)

    path = ROOT / "data/raw/prompts_bigfive.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- 分布报告 ----
    print(f"生成 {len(out)} 条输入样本 -> {path}")
    print(f"\n组合分布（共 {len(COMBINATIONS)} 类 × {per_combo}）：")
    from collections import Counter
    for tag, n in Counter(x["_meta"]["combination"] for x in out).most_common():
        print(f"  {tag:22s} {n}")

    mid_total = sum(x["_meta"]["mid_band_facets"] for x in out)
    total_facets = len(out) * len(ALL_FACETS)
    print(f"\n中点区(40-60) facet 占比：{mid_total/total_facets:.1%}"
          f"  （{mid_total}/{total_facets}）")
    print("  说明：真实测评中此比例通常 25-35%，过高意味着样本缺少可解读内容。")


if __name__ == "__main__":
    main()
