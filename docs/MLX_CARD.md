---
base_model: 6guwew/personality-interpretation-zh
library_name: mlx
license: apache-2.0
language:
  - zh
tags:
  - mlx
  - lora
  - personality
  - big-five
  - ipip
  - mbti
  - psychology
  - text-generation
  - apple-silicon
pipeline_tag: text-generation
---

# personality —— 人格测评解读模型（MLX 4-bit）

这是 [`6guwew/personality-interpretation-zh`](https://huggingface.co/6guwew/personality-interpretation-zh)
的 **MLX 4-bit 量化版**，专为 Apple Silicon 本地推理准备。把**已经计分好的**人格测评结果
翻译成一段专业解读，基于 IPIP-NEO-120（大五人格）与由同一份作答映射出的 MBTI 四维结果。

量化前的 LoRA 适配器已合并进 Qwen2.5-7B-Instruct 基座，再用 MLX 转为 4-bit。

> **它不是聊天机器人，也不做测评。** 出题和计分由规则代码完成，模型只负责最后一步：
> 把分数翻译成有意义的语言。**给它一句"你好"，它会凭空编出一整套分数开始解读** ——
> 数据里没有"没有数据"的负例。调用方必须在服务层做硬闸门。

## 量化规格

| | |
|---|---|
| 基座 | Qwen2.5-7B-Instruct（Apache-2.0） |
| 适配器 | LoRA rank 32 / alpha 64 / `lora_target: all`，已合并 |
| 量化 | MLX 4-bit，group size 64 |
| 平均位宽 | **4.501 bits per weight** |
| 体积 | 约 4.0 GB |
| 实测（M5 / 16GB） | 载入 3.7 秒，生成 7–10 tok/s，一份完整解读约 100 秒 |

质量未退化：规则合规指标与 fp16 版本一致（按维度平铺 0、维度编码 0、套话 0、
临床词 0、编造常模 0、五维度覆盖 5/5）。

## 用法

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tok = load("6guwew/personality-interpretation-zh-mlx-4bit")

sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)
out = generate(model, tok, prompt=prompt, max_tokens=1400,
               sampler=sampler, verbose=False)
```

`prompt` 需要用 chat template 把 system（解读行为规范）与 user（结构化分数 JSON）拼好。
完整的 system 提示词、计分脚本与推理服务见 GitHub 仓库：
**https://github.com/P988888/personality-interpretation-zh**

## 输入格式

分数必须是**已计分**的结构化结果，不是原始作答：

```json
{
  "instrument": "big-five", "version": "IPIP-NEO-120",
  "scale": "raw_and_normalized", "norm_source": null,
  "domains": {"O": 71, "C": 38, "E": 66, "A": 34, "N": 57},
  "facets": {"O1_imagination": 78, "C5_self_discipline": 28}
}
```

`scale` 与 `norm_source` 必须如实标注。IPIP 官方不提供常模，本项目只输出原始分和
0–100 相对刻度，**任何"高于 X% 的人"的说法都是错的**。

## Limitations

1. **只在收到结构化分数时才工作**，否则会编造分数（权重层面的已知偏差）。
2. **不报百分位是因为没有常模**，不是保守。
3. **MBTI 部分**由 IPIP facet 映射导出，**不是** Myers-Briggs 官方测评；S/N 与 T/F
   分别只有 3 个 facet 支撑，稳定性低于 E/I（6 个）。
4. **输出偏长**：规格要求 600–1000 字，实测 1200–1500 字。
5. **不是临床工具**，不替代专业心理评估。

训练数据未经人工逐条审阅，应视为"符合行为规范的示范文"，而非专业心理评估的参考答案。

## 商标声明

MBTI is a registered trademark of The Myers-Briggs Company. 本项目与
The Myers-Briggs Company 无任何关联，未获其赞助或背书，不使用其官方题目。

## 许可

Apache-2.0，继承自 Qwen2.5-7B-Instruct。详见 GitHub 仓库的 `NOTICE`。
