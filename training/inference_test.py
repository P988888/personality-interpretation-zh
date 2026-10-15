#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 对比：基座模型 vs 微调后模型，解读同一份全新测评结果。

为什么要用"训练时的精简 system 提示词"来测：
    训练时刻意没有把 specs 全文放进 prompt（见 to_llamafactory.py 的设计说明），
    目的是让模型把解读规则内化进权重。所以只有用同一个精简 system 来测，
    测到的才是"微调本身带来了什么"，而不是"照着说明书回答"的能力。
    如果给微调模型贴规格全文再比，赢了也不能说明微调有用。

测试用的测评结果是现造的，不在训练集里，而且刻意选得有挑战性：
    - 高开放 + 低尽责 + 低宜人（A=34）+ 中等神经质
    - 7 个 facet 落在 40-60 的中点位区间 —— 考核"不强行解读中点分数"这条规则
    - 低宜人 + 偏高神经质是最容易被模型病理化的组合 —— 考核"不用临床词汇"这条

用法（在训练机上）：python3 inference_test.py [输出json路径]
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = ("/data/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
MERGED = "/data/personality/models/personality-fp16-merged"
TRAIN_DATA = "/data/personality/lf_data/personality_bigfive.json"

# 现造的一份 profile：各 facet 平均值与 domain 分数严格对得上。
# 落在 40-60 中点区间的：C1 42 / C3 45 / C6 46 / A1 41 / N2 58 / N3 51 / N5 55 / N6 52
FRESH = {
    "instrument": "big-five",
    "version": "IPIP-NEO-120",
    "scale": "raw_and_normalized",
    "norm_source": None,
    "domains": {"O": 71, "C": 38, "E": 66, "A": 34, "N": 57},
    "facets": {
        "O1_imagination": 78, "O2_artistic": 74, "O3_emotionality": 66,
        "O4_adventurousness": 72, "O5_intellect": 76, "O6_liberalism": 60,
        "C1_self_efficacy": 42, "C2_orderliness": 31, "C3_dutifulness": 45,
        "C4_achievement_striving": 36, "C5_self_discipline": 28, "C6_cautiousness": 46,
        "E1_warmth": 61, "E2_gregariousness": 58, "E3_assertiveness": 74,
        "E4_activity_level": 69, "E5_excitement_seeking": 72, "E6_cheerfulness": 62,
        "A1_trust": 41, "A2_straightforwardness": 28, "A3_altruism": 39,
        "A4_compliance": 26, "A5_modesty": 33, "A6_sympathy": 37,
        "N1_anxiety": 62, "N2_anger": 58, "N3_depression": 51,
        "N4_self_consciousness": 64, "N5_immoderation": 55, "N6_vulnerability": 52,
    },
}

NOTE = ("注意：分数为原始分线性映射到 0-100 的相对刻度，不是百分位，"
        "不可解释为人群排名。\n\n")
USER = ("请为以下测评结果生成解读。\n" + NOTE + "```json\n"
        + json.dumps(FRESH, ensure_ascii=False, indent=2)
        + "\n```\n\n严格按照系统提示词中的输出结构与行为准则执行。")

GEN_KW = dict(max_new_tokens=1400, do_sample=True, temperature=0.7,
              top_p=0.8, top_k=20, repetition_penalty=1.05)


def generate(path, system, tag):
    print(f"[{time.strftime('%T')}] 加载 {tag}: {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    mdl.eval()

    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": USER}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok([text], return_tensors="pt").to("cuda:0")
    print(f"[{time.strftime('%T')}] 提示词 {ids['input_ids'].shape[1]} token，开始生成", flush=True)

    torch.manual_seed(0)          # 两个模型用同一条随机流，减少采样噪声
    with torch.no_grad():
        out = mdl.generate(**ids, **GEN_KW)
    resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    del mdl
    torch.cuda.empty_cache()
    print(f"[{time.strftime('%T')}] {tag} 生成完毕，{len(resp)} 字", flush=True)
    return resp


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/data/personality/ab_test.json"

    # 直接用训练集的 system 字段，保证和训练时一模一样
    system = json.load(open(TRAIN_DATA, encoding="utf-8"))[0]["system"]
    print(f"system 提示词 {len(system)} 字（训练时用的那份精简版）", flush=True)

    res = {
        "system": system,
        "user": USER,
        "profile": FRESH,
        "base": generate(BASE, system, "基座"),
        "finetuned": generate(MERGED, system, "微调"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
