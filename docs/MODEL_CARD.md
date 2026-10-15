---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: apache-2.0
language:
  - zh
tags:
  - lora
  - personality
  - big-five
  - ipip
  - mbti
  - psychology
  - text-generation
pipeline_tag: text-generation
---

# personality —— 人格测评解读模型（Qwen2.5-7B + LoRA）

把**已经计分好的**人格测评结果翻译成一段专业解读。基于 IPIP-NEO-120（大五人格）
与由同一份作答映射出的 MBTI 四维结果。

> **它不是聊天机器人，也不做测评。** 出题和计分由规则代码完成（`scripts/`），
> 模型只负责最后一步：把分数翻译成有意义的语言。详见 Limitations。

## 它做什么

输入一份结构化分数，输出一段 600–1500 字的解读，要求做到：

- 讲**维度之间的交互**和 **facet 组合**，不逐个维度平铺
- 同时讲优势与代价，不做单向赞美，不用"独一无二"这类套话
- 落在中点区间（40–60）的分数**明确说明区分度有限**，不强行解读
- 不使用任何临床/病理词汇（焦虑症、人格障碍、情绪不稳定……）
- 不预测命运、职业适配、关系成败
- 不编造分数、常模或百分位
- 结尾说明量表测什么、不测什么，以及结果的情境依赖性
- MBTI 结果必须报告各维度清晰度，并说明 MBTI 方法论上的学术争议

## 训练数据

1,579 条（大五 1,399 + MBTI 180），由大模型生成后经规则校验器筛除违规样本
（大五通过率 77.7%、MBTI 93.8%）。

**这批数据未经人工逐条审阅。** 应当视为"符合行为规范的示范文"，而不是
专业心理评估的参考答案。

覆盖情况：大五 18 类分数组合（含全中点、极端值、单项孤高）；
MBTI 16 型 × 4 档清晰度均衡分布。token 中位 1814、最长 2246。

## 训练过程

| | |
|---|---|
| 基座 | Qwen2.5-7B-Instruct（frozen） |
| 方法 | bf16 LoRA，**未做量化**（rank 32 / alpha 64 / `lora_target: all`） |
| 可训练参数 | 80.7 M（占 1.05%） |
| 框架 | LLaMA-Factory 0.9.5 |
| 数据 | 1,421 训练 / 158 验证（val_size 0.1） |
| 超参 | batch 4 × grad_accum 4 = 有效 16，lr 1e-4 cosine + warmup 0.1，cutoff_len 4096，3 epoch |
| 硬件 | 单卡 A100-PCIE-40GB，60.7 分钟，267 步 |
| 结果 | train_loss 1.6224（末步 1.3508），**eval_loss 1.4998** |

**注意：eval_loss 在 200 步（共 267 步）就已达 1.4998，最后三分之一轮没有改善。**
下次做类似的同质数据微调，**2 个 epoch 就够了**。

### 训练时的一个刻意设计

训练用的 system 提示词**只有 502 字，不含任何规格全文**。目的是让模型把解读规则
内化进权重，而不是学会"照着面前的说明书回答"。规格全文只在推理时作为补强贴上。

这一点在评估时很关键：**只有用同一个精简 system 提示词对比，测到的才是微调本身的效果。**

## 效果

同一份全新测评结果（高开放 + 低尽责 + 低宜人 A=34 + 中等神经质，含 8 个中点 facet），
在**同一个精简 system 提示词**下对比基座模型：

| 指标 | 基座 Qwen2.5-7B | 本模型 |
|---|---|---|
| 按维度分节平铺 | 3 处 | **0** |
| 输出维度编码（如 `(O)`） | 6 次 | **0** |
| 套话模板句（"独一无二"等） | 5 句 | **0** |
| 五个维度覆盖 | 4/5（**整块漏掉宜人性**） | **5/5** |
| 讲代价/内在张力 | 2 处 | 8 处 |
| 临床病理词 / 编造常模数字 | 0 / 0 | 0 / 0 |

基座输出的典型形态是按维度分节的报告体，并把 A=34 整个漏掉，结尾落在
"记住，每个人都是独一无二的"这类可套用于任何人的模板话术。

本模型能做出 facet 级别的组合推理，例如把 A2 坦诚 28（低）+ N4 自我意识 64（高）
归因成：「你对人的信任起点低，遇到分歧时不让步，事后又容易反复回想自己刚才那句话。
这种组合最容易产生"当时怎么没说"的懊恼。」

## 用法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                            dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, "6guwew/personality-interpretation-zh")
tok = AutoTokenizer.from_pretrained("6guwew/personality-interpretation-zh")

profile = {
    "instrument": "big-five", "version": "IPIP-NEO-120",
    "scale": "raw_and_normalized", "norm_source": None,
    "domains": {"O": 71, "C": 38, "E": 66, "A": 34, "N": 57},
    "facets": { "O1_imagination": 78, "C5_self_discipline": 28, "... ": 0 },
}
user = ("请为以下测评结果生成解读。\n"
        "注意：分数为原始分线性映射到 0-100 的相对刻度，不是百分位，不可解释为人群排名。\n\n"
        "```json\n" + json.dumps(profile, ensure_ascii=False, indent=2)
        + "\n```\n\n严格按照系统提示词中的输出结构与行为准则执行。")

msgs = [{"role": "system", "content": SYSTEM_PROMPT},   # 见仓库 scripts/to_llamafactory.py
        {"role": "user", "content": user}]
ids = tok([tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)],
          return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=1400, do_sample=True, temperature=0.7,
                     top_p=0.8, top_k=20, repetition_penalty=1.05)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

配套的 OpenAI 兼容推理服务在仓库 `training/serve_model.py`（含无分数输入的硬闸门），
实测约 37 tok/s、首字延迟 0.28 s。

## Limitations

**1. 只在收到结构化分数时才工作。** 训练集里全是「已计分 profile → 解读」这一种样本，
**没有"没有数据"的负例**。实测：给「你好」，它会凭空编出一整套分数开始解读；
给一串没定义过的 `A/B` 字母，它会给每个字母编出含义。

推理服务层已用硬闸门堵住（整段对话里没出现结构化分数就不调模型），
但**权重本身仍然有这个倾向**。要在模型层面根治，需要在数据里补"拒绝生成"的负例。

**2. 不报百分位不是保守，是因为没有常模。** IPIP 官方不提供常模，
所以本项目只输出原始分和 0–100 相对刻度。任何声称"高于 X% 的人"的输出都是错的。

**3. MBTI 部分的方法论限制。** 四维结果由 IPIP-NEO-120 的 facet 映射导出，
**不是** Myers-Briggs 官方测评。映射把连续分数硬切成二分类型，而 MBTI 的方法论争议
恰恰在此。S/N 与 T/F 分别只由 3 个 facet 支撑（E/I 有 6 个），稳定性更低。
模型被要求输出 clear/moderate/slight/marginal 清晰度，但**清晰度低时字母本身意义有限**。

**4. 输出偏长。** 规格要求 600–1000 字，实测稳定在 1200–1500 字。已知偏差。

**5. 不是临床工具。** 这是自我反思的语言工具，不是诊断，不替代专业心理评估。

## 商标声明

MBTI is a registered trademark of The Myers-Briggs Company. 本项目与
The Myers-Briggs Company 无任何关联，未获其赞助或背书，不使用其官方题目。
