# 快速上手

这是个什么东西：一个只做**人格测评解读**的模型。你把已经算好的测评分数给它，
它写出一段像咨询师那样有分寸的解读。

**它不出题、不聊天、不猜数据。** 出题和计分由规则代码做（那样才 100% 准确），
模型只负责把分数翻译成有意义的话。

---

## 一、最快：30 秒看到效果（Apple Silicon Mac）

```bash
pip install mlx-lm
python3 training/serve_mlx.py --port 8010
```

权重不在 Git 仓库里，脚本会自动从 Hugging Face 拉取
[`6guwew/personality-interpretation-zh-mlx-4bit`](https://huggingface.co/6guwew/personality-interpretation-zh-mlx-4bit)（约 4GB，只需一次）。
如果你手上已经有 `models/personality-mlx-4bit/`，它会直接用本地的，不联网。

浏览器打开 **http://127.0.0.1:8010** —— 输入框里预填了一份完整测评结果，
点「生成解读」就行。约 100 秒出一份 1500 字左右的解读。

想接进自己的客户端（Cherry Studio / ChatBox 等）就填：

| 项 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:8010/v1` |
| API Key | 随便填 |
| 模型名 | `personality` |

只用 CPU 的机器或非 Mac 请走 `training/serve_model.py`（transformers 版，同样接口）。

---

## 二、做一次真实的测评

模型需要**分数**才能工作。分数这样来：

```bash
python3 scripts/pipeline.py items      # 首次：准备 120 道题
python3 scripts/pipeline.py test        # 答题，约 15 分钟
```

这道题是 **IPIP-NEO-120**，一次作答同时得到两套结果：

- 大五人格：5 个维度 + 30 个 facet 的分数
- MBTI：由同一份作答的 facet 映射出的四维倾向，**带清晰度分级**

结果存在 `data/results/`，把那份 JSON 原样贴进上面那个网页，就能拿到解读。

> 为什么 MBTI 不用官方题目：官方题目有版权。这里用的是学术界研究 MBTI 与五因素关系时
> 常用的做法——用 IPIP 公共领域的构念映射到四个维度。代价是它把连续分数切成了二分类型，
> 而 MBTI 的方法论争议恰恰在这里，所以系统强制输出清晰度，低清晰度的字母必须被说明为
> 「不必把自己固定在这一侧」。

---

## 三、这个文件夹里有什么

```
README.md            项目全貌与关键设计决定
LICENSE              Apache-2.0
NOTICE               基座模型归属与修改声明（Apache-2.0 要求）
docs/
  OPEN-SOURCE-NOTES.md  开源义务：许可证、商标、AI 生成数据的版权判断
  MODEL_CARD.md         适配器模型卡（已发布到 Hugging Face）
  MLX_CARD.md           MLX 4-bit 量化版模型卡（已发布）
  API.md                两个推理服务的说明与实测性能
  training-guide.md     完整训练流程
  review-checklist.md   人工审阅清单
specs/               知识源 —— 整个项目最核心的资产
  behavior-rules.md     解读行为准则（红线清单、违例示例）
  big-five.md           大五规格：30 个 facet 详解 + 组合速查表
  mbti.md               MBTI 规格：认知功能栈、争议处理
scripts/             管线：抓题库 → 选题 → 计分 → 组装解读请求 → 校验
dashboard/           训练监控大屏（本地 8000 端口，暗色科技风 + 3D 球体）
data/
  items/              题库
  curated/            训练数据集（1579 条，大五 1399 + MBTI 180）
  rejected/           被规则校验器拒掉的样本（供分析失败模式）
training/
  configs/            LoRA 训练与合并配置
  logs/               完整训练日志与环境搭建记录
  serve_mlx.py        本地推理服务（MLX，带安全闸门）
  serve_model.py      通用推理服务（transformers）
  run_train.sh        训练启动脚本
  eval_ab.py          微调效果 A/B 评估
models/               模型权重（20G，不进 Git —— 见第四节，已发布到 Hugging Face）
```

## 四、模型在哪

**已在 Hugging Face 发布**，不用等别人打包给你：

| 模型 | 体积 | 用途 |
|---|---|---|
| [personality-interpretation-zh](https://huggingface.co/6guwew/personality-interpretation-zh) | 323 MB | LoRA 适配器，开源发布的主体 |
| [personality-interpretation-zh-mlx-4bit](https://huggingface.co/6guwew/personality-interpretation-zh-mlx-4bit) | 4.0 GB | **Apple Silicon 本地跑，推荐** |

如果发送方连权重一起打包了，`models/` 目录里是这三份：

| 目录 | 体积 | 用途 |
|---|---|---|
| `personality-mlx-4bit/` | 4.0 G | 同上面的 MLX 版 |
| `personality-lora-v1/` | 323 M | 同上面的适配器 |
| `personality-fp16-merged/` | 14 G | 合并后的完整 fp16 权重（**未发布**，网上已有基座，传上去只是重复） |

**只有适配器也能复现全部效果。** 323MB 的适配器配上基座 `Qwen/Qwen2.5-7B-Instruct`
即可合并出完整模型：

```bash
python3 -m llamafactory-cli export training/configs/merge_qwen7b.yaml
```

`models/` 没有进 Git 是因为 GitHub 单文件限 100MB，所以权重走 Hugging Face
（原因和处理方式见 `docs/OPEN-SOURCE-NOTES.md`）。

---

## 五、一个必须知道的边界

**没有分数时它不会给解读。** 推理服务里有一道硬闸门——整段对话中没有出现过结构化分数，
就直接返回一段说明，不调用模型。

这不是洁癖。实测给它「你好」，它会凭空编出一整套分数开始分析；给它一串从没定义过的
`A/B` 字母，它会给每个字母编出含义。原因是训练集里只有「已计分结果 → 解读」这一种样本，
**没有"没有数据"这个负例**。要在模型层面根治，需要在数据里补"拒绝生成"的样本。

服务层已经堵住了，但**权重本身仍有这个倾向**，别拿它当聊天机器人用。
