# 训练流程说明

目标：训练一个专精大五(IPIP)与 MBTI 解读的小模型，本地部署，不依赖云端 API。

基座：Qwen2.5-7B-Instruct　　方式：bf16 LoRA（**不用 QLoRA**）

---

## 零、选显卡：别被显存数字误导

**这是本项目最容易踩的坑，先说清楚。**

按实际数据算（1579 条 × 3 epoch × 1808 token ≈ 8.6M token）：

| 显卡 | 显存 | 架构/算力 | 预计耗时 | 评价 |
|---|---|---|---|---|
| **RTX 4090** | 24G | Ada / 165 TFLOPS | **约 1.1 小时** | ★ **首选**，性价比最高 |
| A100 40G | 40G | Ampere / 312 TFLOPS | 约 40 分钟 | 最快，但贵 2-3 倍 |
| L40S | 48G | Ada | 约 55 分钟 | 可用 |
| RTX 4070Ti Super | 16G | Ada | 约 1.8 小时 | 预算紧时可用 |
| L4 | 24G | Ada / 带宽低 | 约 3.4 小时 | 不推荐 |
| A5000 | 24G | Ampere / 带宽低 | 约 2.6 小时 | 不推荐 |
| **RTX 8000** | 48G | **Turing / 无原生 bf16** | **约 6 小时** | ✗ **避开** |

**关键结论：显存大 ≠ 快。**

- 你的训练**只需约 16GB 显存**（bf16 LoRA：权重 15.2G + LoRA/优化器 0.5G + 激活 0.5G）
- RTX 8000 虽然是 48G，但是 2018 年的 Turing 架构，**不支持原生 bf16**（要退化成 fp16），
  算力约为 4090 的 1/8 —— 它比 4090 慢 5 倍多
- 决定 LoRA 训练速度的是**算力与内存带宽**，不是显存容量

**24G 卡（4090）是这个任务的最佳选择。** 48G 卡只在你想同时跑别的任务、
或要做全参微调时才有意义。

### 三个比换卡更有效的提速手段

1. **关掉 `gradient_checkpointing`** —— 24G 显存有 8G 余量，关掉可提速约 30%
   （配置里已默认关闭）
2. **减少 epoch** —— 数据同质，3 epoch 与 4 epoch 通常无差别，省 25% 时间
3. **加大 batch size** —— 配置里已是 `bs=4 × grad_accum=4 = 有效 batch 16`

---

## 一、完整流程图

```
specs/                      ← 知识源（规则、facet 定义、行为准则）
  │
  ├─ scripts/gen_prompts_bigfive.py   → data/raw/prompts_bigfive.jsonl
  │     分层采样 18 类组合 × 6 变体 = 108 条输入
  │
  ├─ scripts/generate.py              → data/raw/generated_bigfive.jsonl
  │     调 LLM 批量生成解读（并发）
  │
  ├─ scripts/validate.py              → data/curated/bigfive_pass.jsonl
  │     机器校验：临床词/命运预测/迎合/模板/编造数字
  │                                     ↘ data/rejected/bigfive_fail.jsonl
  │
  ├─ 人工审阅（docs/review-checklist.md）
  │     机器查不出的：套模板、无交互、语气、方向性错误
  │
  ├─ scripts/to_llamafactory.py       → data/curated/personality_bigfive.json
  │     转 sharegpt 格式 + 精简 system 提示词
  │
  └─ training/configs/lora_qwen7b.yaml → LLaMA-Factory 训练 → LoRA adapter
```

---

## 二、执行步骤

### 步骤 0：环境（在云端 48G 机器上）

```bash
pip install "llamafactory[torch,metrics]" transformers
# 或按 LLaMA-Factory 官方文档装
```

### 步骤 1：生成分数组合（本地跑，不需要 GPU）

```bash
python3 scripts/gen_prompts_bigfive.py
```

产出 108 条输入，覆盖 18 类组合。当前中点区 facet 占比 34.4%，落在合理区间。

**调数据量**：改脚本里的 `per_combo`（默认 6）。想做 2000 条数据就设 `per_combo=110`。
但注意——**先小批量试跑，确认生成质量后再放量**，别一次生成几千条发现全是废的。

### 步骤 2：批量生成解读

```bash
# 云端 API（推荐，质量高）
export LLM_API_KEY=sk-xxx
python3 scripts/generate.py --backend openai \
    --model gpt-4o --base-url https://api.openai.com/v1 \
    --workers 8 --limit 20          # 先跑 20 条看质量

# 或本地 Ollama
python3 scripts/generate.py --backend ollama --model qwen2.5:32b
```

**先跑 20 条，人工看一眼再放量。** 这一步的产出质量决定了整个项目的上限。

### 步骤 3：机器校验

```bash
python3 scripts/validate.py data/raw/generated_bigfive.jsonl
```

**通过率低于 50% 是严重信号**，说明生成提示词没有正确注入规格规则。
此时应该回头查提示词组装，**不要放宽校验标准**。

### 步骤 4：人工审阅

按 `docs/review-checklist.md` 逐条过。这是不可跳过的一步——机器只能保证下限。

### 步骤 5：转训练格式

```bash
python3 scripts/to_llamafactory.py
```

产出 `data/curated/personality_bigfive.json` 和 `dataset_info.json`。

### 步骤 6：训练

```bash
# 把数据放进 LLaMA-Factory 能找到的位置
cp data/curated/personality_bigfive.json data/curated/dataset_info.json \
   /path/to/LLaMA-Factory/data/

llamafactory-cli train training/configs/lora_qwen7b.yaml
```

48G 卡上，2000 条 × 4 epoch 大约 1–2 小时。

### 步骤 7：合并与部署

```bash
llamafactory-cli export training/configs/merge_qwen7b.yaml
# 然后用 vLLM 起服务，或 ollama 导入 gguf
```

---

## 三、踩过的坑（实测记录）

### 坑 0：校验器会被「正确行为」误伤（连续四轮修正）

这是本项目耗时最多的一个问题，值得完整记录，因为它揭示了自动校验的固有陷阱。

**现象**：第一轮生成 108 条，校验器拒绝了 19 条，但人工一看，**大部分是误伤**——
被拦的恰恰是规格要求的正确表述：

| 被拦内容 | 为什么其实是对的 |
|---|---|
| 「本量表**不测**任何**临床**意义的状态」 | 这是在**声明边界**，规格要求必须说 |
| 「这**不等同于**情绪不稳定」 | 这是在**否定病理化解读**，正确做法 |
| 「遇到**障碍**时会先停下来想」 | 「障碍」在此是日常用法（阻碍），非病理词 |
| 「你的开放性约在 **74** 左右」 | 这是**复述输入分数**，不是编造 |
| 「其余 **27** 个 facet 都在中点」 | 这是对 30 个 facet 的**结构性计数**，是事实 |

**根因**：裸词匹配 + 数字白名单两种做法都无法区分「使用」与「澄清」。

**四次修正的过程**：
1. 把「临床/症状/诊断」等改成语境敏感词 —— 若前面有否定词则放行
2. 补充边界标记词表（不作/不含/不等同/不衡量/不能替代…）
3. 把「编造数字」从「不在输入中的数字」改为「**百分位/排名语境**下的数字」
4. 修正边界检测的窗口：先用固定 20 字（跨句串扰），改句内 12 字（又过紧），
   最终定为**句内不限长度**

**最终精度：14/14**（4 类合法表述放行，10 类违规拦截）。

**教训**：自动校验器拦下的东西，**每一条都要人工看上下文**。否则你会系统性地
惩罚正确行为，训出来的模型反而不敢声明边界、不敢澄清——那是更糟的结果。

### 坑 0b：规格表述歧义导致 44% 的样本「化验单化」

**现象**：第一轮生成中，**48/108 条（44%）**在正文里写「你的 N4 社交不安很高」
「C1 到 C4 四项同时到位」这类内部编码。

**根因**：`behavior-rules.md` 原文写的是「**必须以 facet 为主要解读单位**」。
这句话有歧义 —— 模型理解成「要报出 facet 的名称」。

**修正**：改成明确区分「语义颗粒度」与「技术标签」：

> 以 facet 为单位指的是**语义颗粒度**，不是让你报出 facet 的技术标签。
> - ✗ 你的 **N4 社交不安**是 87
> - ✓ **你很在意别人怎么看你，却又喜欢和人待在一起**

同时在校验器加入 facet 编码与英文术语检测。修正后违规从 48 条降至 6 条。

**教训**：规格里每一句「做什么」都要追问一句「模型会不会理解成别的意思」。
模糊的指令会被模型按最技术化的方式执行。

### 坑 0c：输入标注与实际计分口径不一致

`gen_prompts_bigfive.py` 早期把输入标为 `scale: "percentile"` 并填了 `norm_source`，
但 `score.py` 实际输出的是 `scale: "raw_and_normalized"`、`norm_source: None`。

这个不一致让下游校验器**放行**了本应拦截的人群比较表述（「高于多数普通人」）。
已统一为 `raw_and_normalized` / `None`。

**教训**：跨模块的元数据字段必须一致，否则校验逻辑会基于错误前提放行。

### 坑 1：token 数不能靠字数乘系数估算

**我犯的错**：按「中文 1.5 token/字」粗估，算出 3509 字 ≈ 5263 token，
判定 `cutoff_len=4096` 不够，差点去改一个本来正确的配置。

**实测结果**：用 Qwen2.5 的真实 tokenizer 量，实际只有 **2090 token**。
Qwen 对中文的压缩率约 **0.65 token/字**，不是 1.5。

| 段落 | 字数 | 实测 token |
|---|---|---|
| system | 502 | 346 |
| user（含 30 facet JSON） | 1088 | 491 |
| assistant 输出 | 1919 | 1253 |
| **合计** | **3509** | **2090** |

**结论**：`cutoff_len=4096` 足够，余量约一倍。

**教训**：调参前用真实 tokenizer 量：
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print(len(tok.apply_chat_template(msgs, tokenize=True)))
```
不同模型中文压缩率差别很大，拍脑袋的系数会让你改错配置。

### 坑 2：组合模板不能把非重点维度设在中点

**我犯的错**：第一版 18 个组合模板里，为了突出某个组合特征，
把其余维度都设成 50 附近的中性值。结果产出的大量 facet 落在 40–60 中点区，
**中点区占比 53%**，意味着超过一半的分数「没什么可讲」，样本被严重稀释。

**修正**：让每个模板的非重点维度也给出温和但明确的偏移（偏离 50 约 10 点以上）。
真实的人不会只在两个维度上有倾向而其余全中性。

**结果**：中点区占比从 53% → **34.4%**，落进合理区间。

### 坑 3：训练时不该注入规格全文

**最初的直觉**：把 `specs/big-five.md` 全文放进 system 字段，让模型「看着说明书答」。

**为什么是错的**：SFT 的目的是把规则**内化进权重**。如果训练时说明书就在眼前，
模型学到的是「照着面前的说明书回答」；而推理时你希望它**没有说明书也能答对**。
训练时不给全文，模型才会被迫从 assistant 回答中归纳出规则——这样学到的能力才可迁移。

**修正**：`to_llamafactory.py` 里改用精简的 `build_train_system()`，
只给角色定位 + 10 条准则 + 输出结构（502 字）。
推理阶段再贴上规格全文作为补强。

### 坑 4：校验器会误伤「焦虑」「抑郁」这两个词

NEO 的 facet 学名里就有 anxiety 和 depression。但校验器对解读正文做了全文禁止扫描，
**这是有意的**——面向受测者的解读里绝不能出现这两个词，即使是在叙述分数。

如果生成时模型习惯性写出 facet 学名，会被大量拦截。这不是校验器的问题，
是提示词里应该明确要求「使用中文学名时必须换成描述性措辞」。
规格文件的「措辞纪律」一节已经写了这条，生成时务必确认模型遵守。

---

## 四、常见问题

**Q：为什么不用 QLoRA？**
48G 显存跑 7B 完全够 bf16。QLoRA 是为 24G 以下显存做的妥协，会引入量化误差。
你的卡不需要这个妥协。

**Q：为什么 rank 选 32？**
解读风格需要一定容量。16 可能偏紧（学不完全部规则），64 在 2000 条数据上有过拟合风险。
32 + alpha 64 是稳妥起点。如果欠拟合改 64，过拟合改 16。

**Q：要不要做全参微调？**
先不要。LoRA 训坏了删 adapter 就回到原模型，全参训坏了要重下权重。
而且数据有问题时，全参只会**更彻底地学到错误**。
等 LoRA 满意了、确实感觉能力被 adapter 限制，再考虑全参（48G 勉强可行）。

**Q：数据要多少条？**
1500–3000 条精选。瓶颈是你的审阅时间，不是生成能力。
策略：生成 2 倍量，机器筛一批，分层抽样审阅，保留约 2000 条。

**Q：训完发现效果不好怎么办？**
按这个顺序排查：
1. 数据里有没有违规样本？（跑 validate，再看 curated 文件）
2. 是不是套模板太严重？（人工审阅，看组合之间的差异度）
3. 基座能力是否不足？（7B 换 14B 试试）
4. 是不是过拟合？（看 eval_loss 曲线，减少 epoch 或加数据）

**先怀疑数据，再怀疑参数。** 这个场景下 90% 的问题出在数据。
