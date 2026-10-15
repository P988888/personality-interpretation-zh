#!/usr/bin/env python3
"""
转换为 LLaMA-Factory 训练格式 (sharegpt)。

读入通过校验的 curated 数据，输出与 LLaMA-Factory 的 dataset_info.json 约定一致的 json。

用法：
    python3 scripts/to_llamafactory.py
输出：
    data/curated/personality_bigfive.json
    data/curated/dataset_info.json
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)


def build_train_system(instrument: str = "big-five") -> str:
    """训练时使用的精简 system 提示词。

    设计决策（重要）：训练时**不注入 specs 全文**。

    理由：SFT 的目的是把解读规则内化进权重。若训练时把规格全文放进 prompt，
    模型学到的是「照着面前的说明书回答」；而推理时我们希望它**没有说明书也能答对**。
    因此训练时只给角色定位与输出结构，把「怎么做才对」留给模型从 assistant
    回答中归纳。这样学到的能力才是可迁移的。

    副作用（正面的）：system 从约 1800 token 降到百来 token，总长度回到 4096
    以内，避免样本被截断。

    推理阶段可以（也建议）重新贴上规格全文，作为能力的补强。
    """
    if instrument == "mbti":
        return (
            "你是一个人格测评解读助手，专精 MBTI 的解读。你收到的是已经计算好的"
            "四维倾向分数，你的工作是把它们翻译成一段专业、有分寸的解读。"
            "你不做计分，不猜测缺失信息，不虚构任何数据。\n\n"
            "严格遵守以下准则：\n"
            "1. 解读主干必须基于认知功能栈（主导/辅助/第三/劣势功能），"
            "而不是只复述四个字母。要包含劣势功能在压力下的表现。\n"
            "2. 必须报告各维度的二分清晰度。标为轻微或临界的维度，要明确说明"
            "「你不必把自己固定在这一侧」。\n"
            "3. 必须说明 MBTI 的学术争议：重测信度偏低、二分法在统计上不成立，"
            "适合作为自我反思的语言，不宜作为定论。\n"
            "4. 必须声明结果来源（由构念映射导出，非 Myers-Briggs 官方量表）。\n"
            "5. 绝不使用临床或病理词汇。\n"
            "6. 绝不预测命运、职业适配、关系成败，不把类型当作能力标签。\n"
            "7. 绝不迎合求认同式的期待，绝不套用可安在任何人身上的模板话术。\n"
            "8. 不出现内部编码（如 Ni、Te 可讲其含义，但不堆砌缩写；"
            "尤其不出现 N4、E3 这类 facet 编码）。\n\n"
            "输出结构：整体画像 → 功能栈展开 → 内在张力 → 需要注意的盲区 → "
            "局限性说明（含争议与来源声明）。\n"
            "第二人称，平实有温度，标准长度 600-1000 字。"
        )

    return (
        "你是一个人格测评解读助手。你收到的是已经计算好的分数，"
        "你的工作是把结构化分数翻译成一段专业、有分寸、像一位懂行的咨询师"
        "在与对方交谈那样的解读。你不做计分，不猜测缺失信息，不虚构任何数据。\n\n"
        "严格遵守以下准则：\n"
        "1. 在维度组合的背景下解读，优先讲维度之间的交互，不逐个维度平铺。\n"
        "2. 以 facet（子面向）或认知功能为主要解读单位，大维度只用于开场定调。\n"
        "3. 分数是连续体，用「偏向」「略高」「居中」措辞，不做类型化断言。\n"
        "4. 同时讲优势与代价，不做单向赞美。\n"
        "5. 处于中点区间的分数（约40-60）区分度有限，应明确说明，不强行解读。\n"
        "6. 绝不使用临床或病理词汇（焦虑症、抑郁、人格障碍、情绪不稳定等）。\n"
        "7. 绝不预测命运、职业适配、关系成败。\n"
        "8. 绝不迎合求认同式的期待，绝不套用可安在任何人身上的模板话术。\n"
        "9. 绝不编造分数、常模或百分位等具体数字。\n"
        "10. 结尾必须说明量表测什么、不测什么，以及结果的情境依赖性。\n\n"
        "输出结构：整体画像 → 核心特质展开 → 内在张力 → 需要注意的盲区 → 局限性说明。\n"
        "第二人称，平实有温度，不用文学化表达，标准长度 600-1000 字。"
    )


def build_user_message(item: dict, instrument: str) -> str:
    """构造训练用的 user 消息，需与推理端 interpret.py 保持完全一致。"""
    clean = {k: v for k, v in item.items()
             if not k.startswith("_") and k != "output"}

    if instrument == "mbti":
        note = (
            "注意：\n"
            "1. 本结果由 IPIP-NEO-120 作答经构念映射导出，非 Myers-Briggs 官方量表，"
            "必须在解读的局限性说明中声明这一点。\n"
            "2. clarity 表示各维度的二分清晰度（clear/moderate/slight/marginal）。"
            "标为 slight 或 marginal 的维度，其字母区分度低，必须明确说明"
            "「你不必把自己固定在这一侧」。\n"
            "3. 解读主干必须基于 cognitive function stack（功能栈）来展开，"
            "并包含劣势功能在压力下的表现。\n"
            "4. 必须包含 MBTI 的学术争议说明（见规格第五节）。\n\n"
        )
    else:
        note = ("注意：分数为原始分线性映射到 0-100 的相对刻度，不是百分位，"
                "不可解释为人群排名。\n\n")

    return ("请为以下测评结果生成解读。\n" + note
            + "```json\n" + json.dumps(clean, ensure_ascii=False, indent=2)
            + "\n```\n\n严格按照系统提示词中的输出结构与行为准则执行。")


def convert(instrument: str, src_name: str, out_name: str) -> int:
    src = ROOT / "data/curated" / src_name
    if not src.exists():
        raise SystemExit(f"未找到 {src}，请先运行对应校验器")

    system_prompt = build_train_system(instrument)
    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    records = []
    for it in items:
        records.append({
            "conversations": [
                {"from": "human", "value": build_user_message(it, instrument)},
                {"from": "gpt", "value": it["output"]},
            ],
            "system": system_prompt,
        })

    random.shuffle(records)
    out = ROOT / "data/curated" / out_name
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def main():
    results = {}
    # 两套量表各自独立转换。推理时也分别加载，口径一致。
    for inst, src_name, out_name in [
        ("big-five", "bigfive_pass.jsonl", "personality_bigfive.json"),
        ("mbti", "mbti_pass.jsonl", "personality_mbti.json"),
    ]:
        try:
            n = convert(inst, src_name, out_name)
            results[inst] = (n, out_name)
        except SystemExit as e:
            print(f"跳过 {inst}：{e}")

    print("转换完成：")
    for inst, (n, name) in results.items():
        print(f"  {inst:10s} {n:5d} 条 -> data/curated/{name}")

    # dataset_info 同时登记两套（若都存在）
    info = {}
    if "big-five" in results:
        info["personality_bigfive"] = _dataset_entry("personality_bigfive.json")
    if "mbti" in results:
        info["personality_mbti"] = _dataset_entry("personality_mbti.json")
    (ROOT / "data/curated/dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndataset_info -> data/curated/dataset_info.json（{len(info)} 个数据集）")
    print("\n提示：训练时把 data/curated 加入 LLaMA-Factory 的 dataset_dir。")


def _dataset_entry(fname: str) -> dict:
    return {
        "file_name": fname,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {"role_tag": "from", "content_tag": "value",
                 "user_tag": "human", "assistant_tag": "gpt"},
    }


if __name__ == "__main__":
    main()
