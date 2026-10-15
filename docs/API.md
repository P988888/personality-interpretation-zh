# 模型 API —— 直接试效果

微调后的模型（`models/personality-fp16-merged`）以 OpenAI 兼容协议提供服务。

## 连接信息

| 项 | 值 |
|---|---|
| Base URL | `http://127.0.0.1:8001/v1` |
| API Key | 随便填（服务端不校验），例如 `sk-local` |
| 模型名 | `personality` |
| 网页试玩 | 浏览器打开 http://127.0.0.1:8001 |

Cherry Studio / ChatBox / NextChat / LobeChat 这类客户端，选「自定义 OpenAI 兼容」，
把上面三项填进去即可。已开启 CORS，浏览器直连也不会被拦。

## 这个本地地址依赖 SSH 隧道

模型跑在租的训练机上，`127.0.0.1:8001` 是本机隧道转过去的。
**隧道断了或电脑重启后，跑这一行就能恢复**（不会弹密码，密钥已配好）：

```bash
ssh -f -N -L 8001:127.0.0.1:8001 personality-gpu
```

想换本地端口就改 `-L` 左边那个数字。

## 一个必须知道的边界：没给分数就不会生成

服务端有一道**硬闸门**：整段对话里没有出现过结构化分数（`facets` / `domains` / `dimensions`），
就完全不调用模型，直接返回一段固定说明，告诉你该先去做题。

为什么要做在服务端而不是只写进 system 提示词 —— 因为实测**模型会无视提示词**：

| 输入 | 修复前的行为 |
|---|---|
| `你好` | 「从这份分数看，你身上最突出的一条线是……」凭空开始解读 |
| `我想测一下我的MBTI，怎么开始？` | 先声明"本次输入的是大五分数"（输入里根本没有分数），然后编出一整套画像 |
| `1、A 2、A 3、B 4、A 5、B 6、A` | 给 A/B 编出了从没定义过的含义 |

根因是窄域 SFT 的典型盲区：训练集里只有「已计分 profile → 解读」这一种样本，
**没有"没有数据"这个行为**，所以任何输入都会触发它去写解读，缺的分数它就编。
这类盲区靠提示词约束不可靠，只有在服务端拦死。

这也意味着：**别把这个接口当聊天机器人用**，它只做解读这一步。

## 完整流程

```
scripts/take_test.py        120 道 IPIP-NEO-120，1-5 评分（同时导出 MBTI 四维）
        ↓
data/results/result-*.json  原始分 + 0-100 相对刻度 + MBTI 维度与清晰度
        ↓
本接口                       贴进去，出解读
```


**1. 贴裸 profile，自动包装（最省事）**

服务端识别到消息体是一个含 `instrument`/`domains`/`facets` 的 JSON 时，
会自动套上训练时用的那套 user 消息模板，并按 `instrument` 自动选 big-five 或 MBTI 的
system 提示词。所以直接把测评结果丢进去就行：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"personality","messages":[{"role":"user","content":"{\"instrument\":\"big-five\",\"domains\":{\"O\":45,\"C\":78,\"E\":41,\"A\":72,\"N\":63},\"facets\":{...}}"}],"stream":true}'
```

**2. 标准对话**

自己给 system 和 user，行为和普通 OpenAI 接口一致；不给 system 时服务端会补上训练用的默认提示词。

**3. `/v1/interpret`**

请求体直接就是 profile JSON，不用套 `messages` 外壳。

## 实测性能

单用户、非并发：

| | |
|---|---|
| 生成速度 | ~37 tok/s |
| 首字延迟（流式） | 0.28 s |
| 一份完整解读（约 800 token） | 约 21 s |

## 为什么不装 vLLM

这台机器的 transformers 刚经历过 4.56.2 / 5.6.0 混装的坑（见 `training/fix_env.sh`），
vLLM 对 transformers 版本卡得很死，装它极可能再把环境搅坏，连 LLaMA-Factory 一起废掉。
单用户试效果用 transformers 直接推理完全够。**真要上生产、要并发**，另开一个干净 venv
装 vLLM（或直接用 TF/sglang），别动这个环境。

另外注意：单卡串行，服务端有锁，同时发两个请求会排队。生成期间显存占用约 15–16GB。

## 运维

```bash
# 重启（脚本里有 pkill，必须放在脚本里跑，否则 pkill -f 会匹配到自己那条命令行）
ssh personality-gpu 'bash /data/personality/training/start_remote_api.sh 8001'

# 看日志
ssh personality-gpu 'tail -f /tmp/model_api.log'

# 停掉
ssh personality-gpu 'pkill -f serve_model.py'
```
