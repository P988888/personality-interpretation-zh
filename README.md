# 人格测评模型

一个专精**大五人格（IPIP-NEO-120）**与 **MBTI** 解读的项目。目标：本地部署、不依赖云端 API、
解读达到专业咨询师水准。

---

## 这个项目能做什么

1. **做测评**：120 道题，答完自动计分，得到 5 个维度 + 30 个 facet 的分数
2. **出解读**：根据分数生成专业解读（讲维度交互、facet 组合、内在张力，不套模板）
3. **训模型**（可选）：用生成的解读数据做 LoRA 微调，得到一个本地跑的专用小模型

---

## 快速开始

### 想测试自己 → 三步

```bash
# 1. 准备题库（首次运行需要，之后跳过）
python3 scripts/pipeline.py items

# 2. 答题（120 题，约 15 分钟）
#    答完同时得到大五分数 + MBTI 四维倾向
python3 scripts/pipeline.py test

# 3. 生成解读请求
python3 scripts/pipeline.py interpret              # 大五解读
python3 scripts/interpret.py <结果.json> --instrument mbti   # MBTI 解读
```

第 3 步会输出一份提示词，把它贴到任意大模型（ChatGPT / Claude / 本地模型）就能得到解读。
如果你配了 API key，可以直接一步到位：

```bash
export LLM_API_KEY=sk-xxx
python3 scripts/interpret.py data/results/你的结果.json --mode api --model gpt-4o
```

### 想在自己电脑上跑 → 本地 MLX 服务

模型已量化成 **MLX 4-bit（4.0 GB）**，Apple Silicon 原生跑，不需要联网：

```bash
python3 training/serve_mlx.py --port 8010
```

然后浏览器打开 http://127.0.0.1:8010 ，或在任意客户端里配：
Base URL `http://127.0.0.1:8010/v1`，API Key 随便填，模型名 `personality`。

**为什么是 MLX 而不是 GGUF**：在 Apple Silicon 上 MLX 是原生框架，直接吃 GPU 统一内存；
而且 `pip install mlx-lm` 一条就够，GGUF 那条路要 llama.cpp 的转换脚本加编译好的
`llama-quantize`，在 `github.com` 不通的网络里走不下去。

实测（M5 / 16GB）：载入 3.7 秒，生成约 10 tok/s，一份完整解读约 100 秒。
比 A100 上的 37 tok/s 慢，但本地、免费、不依赖任何外部服务。

⚠️ **服务端有硬闸门**：整段对话里没有结构化分数就不会调用模型。
这不是洁癖 —— 实测给模型「你好」，它会凭空编出一整套分数开始解读（详见「已知问题」）。

### 想训练模型 → 四步

```bash
python3 scripts/pipeline.py gen-prompts              # 1. 生成分数组合
export LLM_API_KEY=sk-xxx
python3 scripts/generate.py --backend openai --model gpt-4o --workers 4   # 2. 批量生成解读
python3 scripts/pipeline.py validate                 # 3. 校验 + 转训练格式
# 4. 按 docs/training-guide.md 训练
```

---

## 目录结构

```
README.md / QUICKSTART.md      项目全貌 / 30 秒上手
LICENSE / NOTICE               Apache-2.0 全文 / 基座归属与修改声明

specs/                         知识源 —— 项目的核心资产
  behavior-rules.md              解读行为准则（红线清单、违例示例）
  big-five.md                    大五规格（30 个 facet 详解、组合速查表）
  mbti.md                        MBTI 规格（认知功能栈、争议处理）

scripts/                       管线脚本
  extract_ipip_items.py          抓取 IPIP 官方题库
  pick_neo120.py                 按官方配比选出 120 题
  score.py / score_mbti.py       大五计分 / MBTI 四维映射
  take_test.py                   交互式答题（一次作答，两套结果）
  interpret.py                   生成解读请求 / 调用 API 出解读
  gen_prompts_*.py               生成训练用的分数组合
  generate.py                    批量生成解读样本（SFT 数据）
  validate*.py                   质量校验（自动拦截违规样本）
  to_llamafactory.py             转训练格式
  pipeline.py                    总入口

data/
  items/                       题库（全量 + 120 题版）
  curated/                     通过校验的训练数据（1579 条）
  rejected/                    被拒样本 + 失败模式说明
  raw/                         生成过程中间产物（不进 Git，可重建）

training/
  configs/                     LoRA 训练 / 合并 / 冒烟测试配置
  logs/                        完整训练日志与环境搭建记录（18 份）
  serve_mlx.py                 本地推理服务（Apple Silicon 原生，带安全闸门）
  serve_model.py               通用推理服务（transformers，OpenAI 兼容）
  run_train.sh                 训练启动脚本
  inference_test.py            基座 vs 微调 A/B 生成
  eval_ab.py                   合规指标对比
  test_mlx.py                  MLX 量化模型验证

dashboard/                     训练监控大屏（暗色科技风 + 3D 球体）
docs/                          模型卡、API 文档、训练流程、人工审阅清单、开源义务
models/                        模型权重，20G，不进 Git —— 详见 QUICKSTART
```

---

## 关键设计决定

**为什么计分不交给模型？**
计分是纯规则（反向计分、求和、映射），代码做是 100% 准确的，模型做一定会出错。
模型只负责它擅长的部分：把分数翻译成有意义的语言。这条界线贯穿整个项目。

**为什么不报百分位？**
IPIP 官方明确不提供常模（见 ipip_interp.html）。本项目因此只报原始分与相对刻度，
**不编造人群排名**。这不是保守，是诚实——没有常模支撑的百分位是假的。

**为什么解读规格这么重要？**
因为模型不会因为看了样本就「懂」人格测评，它只学会模仿样本的样子。
样本里的专业判断来自 `specs/`，这是整个项目真正的知识资产。
规格写错，训一万条数据只会得到一万个错误答案的一致性。

**为什么训练时不注入规格全文？**
SFT 的目的是把规则内化进权重。训练时把规格放在 prompt 里，模型学的是
「照着面前的说明书回答」；不给全文，它才会从回答中归纳规则。
推理时再把规格贴上做补强。

---

## 当前状态

| 模块 | 状态 |
|---|---|
| 题库（30 facet / 299 题） | ✓ 完成 |
| IPIP-NEO-120（120 题，官方配比） | ✓ 完成 |
| 大五计分器 | ✓ 完成 |
| MBTI 四维计分（构念映射） | ✓ 完成 |
| 交互式答题（一次作答两套结果） | ✓ 完成 |
| 解读规格（大五 + MBTI） | ✓ 完成 |
| 解读请求组装（两套量表） | ✓ 完成 |
| 训练数据集 | ✓ 1579 条（大五 1399 + MBTI 180） |
| 质量校验器（两套，精度 14/14） | ✓ 完成 |
| **LoRA 微调** | **✓ 完成**（267 步 / 3 epoch / 60.7 分钟 / eval_loss 1.4998） |
| **效果验证** | **✓ A/B 对比基座模型，合规项全面优于基座** |
| 推理服务（OpenAI 兼容） | ✓ 完成，约 37 tok/s |
| 训练监控大屏 | ✓ 完成（本地 8000 端口） |

### 数据集现状

| | 生成 | 通过校验 | 通过率 |
|---|---|---|---|
| 大五 | 1800 | **1399** | 77.7% |
| MBTI | 192 | **180** | 93.8% |

- 大五覆盖 18 类分数组合（含全中点、极端值、单项孤高）
- MBTI 覆盖 16 型 × 4 档清晰度，均衡分布
- token 中位 1814、最长 2246，`cutoff_len=4096` 余量充足
- 被拒样本留在 `data/rejected/` 供分析失败模式

**下一步**：模型已训完并验证，可以直接发布。发布前先读
[`docs/OPEN-SOURCE-NOTES.md`](docs/OPEN-SOURCE-NOTES.md) —— 里面列了 Apache-2.0 的义务、
MBTI 商标声明、以及权重为什么不能进 Git 仓库。模型卡在 `docs/MODEL_CARD.md`，
可直接发到 Hugging Face。

---

## MBTI 部分的重要说明

**本项目不使用 Myers-Briggs 官方题目**（版权归 Myers-Briggs 公司），也不自行编造题目
（那样效度无从谈起）。

采用的是学术界研究 MBTI 与五因素关系时常用的做法：**基于 IPIP 公共领域题库中经过
验证的构念，映射到 MBTI 四个维度**。

| MBTI 维度 | 取自 IPIP-NEO-120 的 facet |
|---|---|
| E / I | 外向性全部 6 个 facet |
| S / N | O1 想象力、O2 审美、O5 思辨 |
| T / F | A6 同理心、A3 利他、A1 信任 |
| J / P | C2 秩序、C6 审慎 |

**好处**：一次作答（120 题）同时得到大五和 MBTI 两套结果，不必做两遍。
**必须说明的代价**：这个映射把连续分数硬切成二分类型，而 MBTI 的方法论争议
恰恰就在这里。因此系统**强制输出清晰度（clarity）**：

- `clear`（偏离中点 ≥20 点）：可以较有把握地描述该偏好
- `moderate`（≥10）：中等把握
- `slight`（≥5）：区分度低，须提示「不必把自己固定在这一侧」
- `marginal`（<5）：完全无法区分，仅名义上有个字母

实测中，随机作答往往得到四个维度全为 `slight` 或 `marginal` 的类型——这不是 bug，
而是 MBTI 二分法问题的活体展示。解读时这一判断必须传达给用户。

---

## 已知问题

- **模型没有"没有数据"这个行为**（最重要的一条）。训练集里全是「已计分 profile → 解读」
  这一种样本，没有负例。实测给「你好」，它会凭空编出一整套分数开始解读；给一串从没定义过的
  `A/B` 字母，它会给每个字母编出含义。这是窄域 SFT 的典型盲区，**靠 system 提示词约束
  不可靠**（提示词里写着"不虚构任何数据"，它照样无视）。
  推理服务层已用硬闸门堵住（见 `docs/API.md`），但**权重本身仍有这个倾向**；
  要在模型层面根治，需在数据里补"拒绝生成"的负例。
- **输出偏长**：规格要求 600–1000 字，实测稳定在 1200–1500 字。
- **eval_loss 在 200/267 步就已收敛**（1.4998），最后三分之一轮无改善。同类同质数据
  微调 **2 个 epoch 就够**。
- **C2 秩序 facet 在 IPIP 官方两个页面数据不一致**：计分键页面有 9 题（5正+4反），
  对照表显示应为 10 题（5+5）。本项目按实际可用题目处理（120 题版取 1正+3反，
  满足配比）。如需完全对齐官方 300 题版，需补一道反向题。
- **MBTI 的 S/N、T/F 维度覆盖面偏窄**：分别只由 3 个 facet 构成，相比 E/I 的
  6 个 facet，稳定性较低。这是构念映射的固有限制，不是实现缺陷。
- **训练数据未经人工逐条审阅**：1579 条由模型生成后经规则校验器筛除违规样本，
  应视为"符合行为规范的示范文"，不是专业心理评估的参考答案。
