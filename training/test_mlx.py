#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 MLX 4-bit 量化模型跑一次真实解读，验证转换没坏、质量还在。

为什么在 Mac 上用 MLX 而不是 GGUF：
    这台是 Apple Silicon（M5）。MLX 是苹果原生框架，直接吃 GPU 统一内存，
    比 llama.cpp 的 GGUF 路径更贴合；而且 mlx-lm 一条 pip 就能装，
    GGUF 那条路要 llama.cpp 的转换脚本 + 编译好的 llama-quantize，
    在 github.com 不通的网络下走不通。

用法：python3 training/test_mlx.py [模型目录]
"""
import json
import sys
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models/personality-mlx-4bit")

# 和 API 测试用同一份现造 profile，方便横向对比
FRESH = {
    "instrument": "big-five", "version": "IPIP-NEO-120",
    "scale": "raw_and_normalized", "norm_source": None,
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

# system 直接取训练集里的那份，保证和训练、和服务器 API 逐字一致
system = json.load(open(ROOT / "data/curated/personality_bigfive.json",
                        encoding="utf-8"))[0]["system"]
user = ("请为以下测评结果生成解读。\n" + NOTE + "```json\n"
        + json.dumps(FRESH, ensure_ascii=False, indent=2)
        + "\n```\n\n严格按照系统提示词中的输出结构与行为准则执行。")

print(f"载入 {MODEL} ...", flush=True)
t0 = time.time()
model, tok = load(MODEL)
print(f"  载入耗时 {time.time()-t0:.1f}s", flush=True)

prompt = tok.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True)
print(f"  提示词 {len(prompt)} 字", flush=True)

t0 = time.time()
# 新版 mlx-lm 不再接受 temp=/top_p= 这类 kwarg，采样器要显式构造
sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)
out = generate(model, tok, prompt=prompt, max_tokens=1400, sampler=sampler, verbose=False)
el = time.time() - t0
n = len(tok.encode(out))

print(f"\n生成 {n} token，耗时 {el:.1f}s  =  {n/el:.1f} tok/s")
print(f"输出 {len(out)} 字")
print("=" * 92)
print(out)

(ROOT / "training/mlx_test_output.txt").write_text(out, encoding="utf-8")
json.dump({"output": out, "tokens": n, "seconds": el},
          open(ROOT / "training/mlx_test_output.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已存到 training/mlx_test_output.txt")
