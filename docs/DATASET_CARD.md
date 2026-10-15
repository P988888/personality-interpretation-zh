---
language:
  - zh
license: apache-2.0
task_categories:
  - text-generation
tags:
  - personality
  - big-five
  - ipip
  - mbti
  - psychology
  - synthetic
  - sft
pretty_name: 人格测评解读训练集（IPIP-NEO-120 + MBTI）
---

# 人格测评解读训练集

用于微调 [`6guwew/personality-interpretation-zh`](https://huggingface.co/6guwew/personality-interpretation-zh)
的监督微调（SFT）数据。每条样本是「**已计分**的结构化测评结果 → 一段专业中文解读」。

> **这些数据由大模型生成，未经人工逐条审阅。** 应视为"符合行为规范的示范文"，
> 而不是专业心理评估的参考答案。解读规则来自项目人工撰写的规格文件（`specs/`），
> 生成后经规则校验器筛除违规样本。

## 构成

| 文件 | 条数 | 通过率 | 内容 |
|---|---|---|---|
| `personality_bigfive.json` | 1,399 | 77.7%（1,399 / 1,800） | 大五人格解读，LLaMA-Factory sharegpt 格式 |
| `personality_mbti.json` | 180 | 93.8%（180 / 192） | MBTI 四维解读，同上 |
| `bigfive_pass.jsonl` | 1,399 | — | 与上表同内容，JSONL 便于流式处理 |
| `mbti_pass.jsonl` | 180 | — | 同上 |
| `dataset_info.json` | — | — | LLaMA-Factory 数据集注册文件 |

合计 **1,579 条**。token 中位 1,814、最长 2,246（训练时 `cutoff_len=4096`，余量充足）。

**覆盖情况**：大五覆盖 18 类分数组合（含全中点、极端值、单项孤高）；
MBTI 覆盖 16 型 × 4 档清晰度，均衡分布。

## 训练/验证划分

训练时用 `val_size=0.1` 随机划分，即 **1,421 训练 / 158 验证**。

## 被拒样本

生成阶段共拒掉 22.3% 的大五样本和 6.2% 的 MBTI 样本，主要失败模式：
临床/病理词汇、输出 facet 技术编码、按维度平铺分节、编造百分位、
对中点分数强行解读。被拒样本与其原因留在项目 GitHub 仓库的 `data/rejected/`，
供分析失败模式使用：
**https://github.com/P988888/personality-interpretation-zh**

## 计分口径（重要）

输入里的分数是**原始分线性映射到 0–100 的相对刻度，不是百分位**。
IPIP 官方不提供常模，所以**任何"高于 X% 的人"的解读都是错的**。
样本中的 `scale` 字段标注为 `raw_and_normalized`、`norm_source` 为 `null`。

## 题目来源

- 英文题目：IPIP 官方公开题库 `https://ipip.ori.org/newNEOFacetsKey.htm`（公共领域）
- 中文题目：本项目自建翻译
- **未使用** Myers-Briggs 官方题目（版权归 The Myers-Briggs Company）
- MBTI 四维结果由 IPIP-NEO-120 的 facet 映射导出，**不是**官方 MBTI 测评

## 许可与商标

Apache-2.0。MBTI is a registered trademark of The Myers-Briggs Company.
本项目与其无任何关联，未获赞助或背书。

AI 生成数据的版权归属在法律上仍不确定，详见项目仓库的 `docs/OPEN-SOURCE-NOTES.md`。
