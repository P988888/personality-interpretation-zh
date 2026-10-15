#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把微调后的模型起成 OpenAI 兼容接口，方便用现成客户端直接试效果。

为什么不用 vLLM：
    这台机器的 transformers 刚经历过一次 4.56.2 / 5.6.0 混装的坑（见 training/fix_env.sh）。
    vLLM 对 transformers 版本卡得很死，装它极可能把环境再搅坏一次，
    连 LLaMA-Factory 一起废掉。试效果是单用户场景，transformers 直接推理够用。
    真要上生产，另开一个干净 venv 装 vLLM 部署。

接口：
    GET  /v1/models             列出模型
    POST /v1/chat/completions   标准 OpenAI 协议，支持 stream
    GET  /                      最简网页聊天框（先在浏览器里试）
    POST /v1/interpret          直接丢一份 profile JSON，省掉拼消息

用法（在训练机上）：
    setsid python3 serve_model.py > /tmp/model_api.log 2>&1 < /dev/null &
"""
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 注意这是训练机上的路径（本仓库的 models/ 只是本机下载后的存放位置，两者不同）
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/personality/training/output/qwen7b-merged")
PORT = int(os.environ.get("PORT", "8001"))
MODEL_ID = os.environ.get("MODEL_ID", "personality")
LF_DATA = "/data/personality/lf_data"

# 默认采样参数，和 A/B 测试保持一致
GEN_DEFAULTS = dict(temperature=0.7, top_p=0.8, top_k=20,
                    repetition_penalty=1.05, max_tokens=1400)

_lock = threading.Lock()          # 单卡，一次只跑一个请求
_tok = None
_mdl = None
_systems = {}


def load_systems():
    """直接取训练集里的 system 字段 —— 和训练时用的提示词逐字一致，不会走样。"""
    for inst, fname in (("big-five", "personality_bigfive.json"),
                        ("mbti", "personality_mbti.json")):
        try:
            with open(os.path.join(LF_DATA, fname), encoding="utf-8") as f:
                _systems[inst] = json.load(f)[0]["system"]
        except Exception as e:
            print(f"[warn] 读取 {fname} 的 system 失败: {e}", flush=True)
    if not _systems:
        _systems["big-five"] = (
            "你是一个人格测评解读助手。你收到的是已经计算好的分数，你的工作是把结构化分数"
            "翻译成一段专业、有分寸、像一位懂行的咨询师在与对方交谈那样的解读。"
            "你不做计分，不猜测缺失信息，不虚构任何数据。第二人称，标准长度 600-1000 字。"
        )
    print(f"[info] 已载入 {len(_systems)} 套 system 提示词: {list(_systems)}", flush=True)


def load_model():
    global _tok, _mdl
    print(f"[{time.strftime('%T')}] 载入 tokenizer…", flush=True)
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[{time.strftime('%T')}] 载入模型 {MODEL_PATH}（约 15G，要等一会）…", flush=True)
    _mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    _mdl.eval()
    print(f"[{time.strftime('%T')}] 模型就绪，监听 0.0.0.0:{PORT}", flush=True)


# ---------------------------------------------------------------- 消息构造

PROFILE_NOTE = {
    "big-five": "注意：分数为原始分线性映射到 0-100 的相对刻度，不是百分位，不可解释为人群排名。\n\n",
    "mbti": ("注意：\n"
             "1. 本结果由 IPIP-NEO-120 作答经构念映射导出，非 Myers-Briggs 官方量表，"
             "必须在解读的局限性说明中声明这一点。\n"
             "2. clarity 表示各维度的二分清晰度（clear/moderate/slight/marginal）。"
             "标为 slight 或 marginal 的维度，其字母区分度低，必须明确说明"
             "「你不必把自己固定在这一侧」。\n"
             "3. 解读主干必须基于 cognitive function stack（功能栈）来展开，"
             "并包含劣势功能在压力下的表现。\n"
             "4. 必须包含 MBTI 的学术争议说明（见规格第五节）。\n\n"),
}


def detect_instrument(text):
    if re.search(r'"instrument"\s*:\s*"mbti"', text) or '"dimensions"' in text:
        return "mbti"
    return "big-five"


def wrap_profile(profile, inst):
    """把裸的 profile JSON 包成训练时那种完整 user 消息。"""
    clean = {k: v for k, v in profile.items()
             if not k.startswith("_") and k != "output"}
    return ("请为以下测评结果生成解读。\n" + PROFILE_NOTE[inst]
            + "```json\n" + json.dumps(clean, ensure_ascii=False, indent=2)
            + "\n```\n\n严格按照系统提示词中的输出结构与行为准则执行。")


def looks_like_profile(text):
    """用户是不是直接贴了一份 profile JSON（而不是在聊天）。"""
    t = text.strip()
    if not (t.startswith("{") and t.rstrip().endswith("}")):
        return False
    try:
        obj = json.loads(t)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    return any(k in obj for k in ("facets", "dimensions", "domains", "instrument"))


# 在整段对话里找"有没有给过分"。用这个做兜底闸门而不是靠提示词约束，
# 因为实测模型会无视 system 里的"不虚构任何数据"：
#   输入"你好"          -> 它编了一整套分数开始解读
#   输入"1、A 2、A ..." -> 它给 A/B 编出了从没定义过的含义
# 窄域 SFT 只有"有分数->写解读"这一种样本，没有"没数据"这个行为。
# 提示词约束在这个盲区里不可靠，必须在服务端拦死。
PROFILE_SHAPE = re.compile(r'"(facets|domains|dimensions)"\s*:\s*[' + r'{' + r'\[]')


def message_text(m):
    c = m.get("content") or ""
    if isinstance(c, list):          # 有些客户端把内容发成分段数组
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def has_profile(messages):
    return any(PROFILE_SHAPE.search(message_text(m)) for m in messages)


NO_PROFILE_MSG = """我这边只做一件事：**把已经算好的测评结果翻译成一段解读**。出题和计分不归我管，所以在收到结构化分数之前，我不会给任何解读——凭空生成分数正是这个模型最需要避免的错。

完整流程是三步，前两步在本地脚本里跑：

**1. 做题**（IPIP-NEO-120，120 道，每题 1-5 分）
```bash
python3 scripts/take_test.py --out me.json
```

**2. 计分**（反向题已处理好；同时从同一份作答导出 MBTI 四维和清晰度）
```bash
# take_test.py 已经把结果写到 data/results/result-*.json，不用再单独跑
```

**3. 解读**：把那份结果 JSON 原样贴给我就行，本接口会自动识别并套上对应的提示词。

如果你手上已经有一份结果，直接贴过来，格式长这样：

```json
{
  "instrument": "big-five",
  "scale": "raw_and_normalized",
  "domains": { "O": 71, "C": 38, "E": 66, "A": 34, "N": 57 },
  "facets": { "O1_imagination": 78, "C5_self_discipline": 28, "...": 0 }
}
```

（`facets` 要给全 30 项。MBTI 用 `"instrument": "mbti"` 和 `"dimensions"` 字段。）"""


def build_messages(body):
    msgs = list(body.get("messages") or [])
    inst = detect_instrument(json.dumps(msgs, ensure_ascii=False))

    # 把裸 profile JSON 自动包成完整消息，省得用户手拼
    if msgs and msgs[-1].get("role") == "user" and looks_like_profile(message_text(msgs[-1])):
        profile = json.loads(message_text(msgs[-1]).strip())
        inst = profile.get("instrument") or detect_instrument(
            json.dumps(profile, ensure_ascii=False))
        if inst not in _systems:
            inst = "big-five"
        msgs = [{"role": "user", "content": wrap_profile(profile, inst)}]

    # 没给 system 就补上训练时那套
    if not any(m.get("role") == "system" for m in msgs):
        if inst not in _systems:
            inst = "big-five"
        msgs = [{"role": "system", "content": _systems[inst]}] + msgs
    return msgs


# ---------------------------------------------------------------- 生成

def generate(messages, **kw):
    """产出 (text_iter_or_none, full_text)。stream=True 时返回迭代器。"""
    params = {**GEN_DEFAULTS}
    for k in GEN_DEFAULTS:
        if kw.get(k) is not None:
            params[k] = kw[k]
    stream = bool(kw.get("stream"))

    text = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = _tok([text], return_tensors="pt").to("cuda:0")
    prompt_len = ids["input_ids"].shape[1]

    gen_kw = dict(max_new_tokens=int(params["max_tokens"]),
                  do_sample=True,
                  temperature=float(params["temperature"]),
                  top_p=float(params["top_p"]),
                  top_k=int(params["top_k"]),
                  repetition_penalty=float(params["repetition_penalty"]))

    if not stream:
        with torch.no_grad():
            out = _mdl.generate(**ids, **gen_kw)
        resp = _tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return None, resp, prompt_len, out.shape[1] - prompt_len

    streamer = TextIteratorStreamer(_tok, skip_prompt=True, skip_special_tokens=True)
    box = {}

    def worker():
        with torch.no_grad():
            box["out"] = _mdl.generate(**ids, streamer=streamer, **gen_kw)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return streamer, None, prompt_len, None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _sse_headers(self):
        # SSE 没有 Content-Length，在 HTTP/1.1 下客户端无法判断响应何时结束，
        # 会一直挂着等。用 Connection: close 让读完 [DONE] 后连接自然收尾。
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": int(time.time()),
                 "owned_by": "local"}]})
        elif path in ("/", "/index.html"):
            body = PLAYGROUND.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/health":
            self._json(200, {"ok": _mdl is not None, "model": MODEL_ID})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        path = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json(400, {"error": {"message": f"请求体不是合法 JSON: {e}"}})
            return

        if path == "/v1/interpret":
            body = {"messages": [{"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
                    "stream": False}
            path = "/v1/chat/completions"

        if path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return

        try:
            messages = build_messages(body)
        except Exception as e:
            self._json(400, {"error": {"message": f"消息构造失败: {e}"}})
            return

        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        want_stream = bool(body.get("stream"))

        # 兜底闸门：整段对话里从没出现过结构化分数，就完全不调模型，直接回一段固定说明。
        # 放这一层而不是只写进 system 提示词，是因为实测模型会无视提示词里的
        # "不虚构任何数据" —— 给"你好"它都能编出一整套分数。闸门放在这儿它就没有机会编。
        if not has_profile(body.get("messages") or []):
            if want_stream:
                self._sse_headers()
                self._emit(cid, created, {"role": "assistant", "content": NO_PROFILE_MSG}, None)
                self._emit(cid, created, {}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._json(200, {
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": NO_PROFILE_MSG}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })
            return

        try:
            with _lock:
                streamer, full, plen, clen = generate(
                    messages,
                    temperature=body.get("temperature"),
                    top_p=body.get("top_p"),
                    max_tokens=body.get("max_tokens"),
                    stream=want_stream,
                )
                if want_stream:
                    self._sse_headers()
                    self._emit(cid, created, {"role": "assistant", "content": ""}, None)
                    for piece in streamer:
                        if piece:
                            self._emit(cid, created, {"content": piece}, None)
                    self._emit(cid, created, {}, "stop")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
        except Exception as e:
            self._json(500, {"error": {"message": f"生成失败: {e}"}})
            return

        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": full}}],
            "usage": {"prompt_tokens": plen, "completion_tokens": clen,
                      "total_tokens": plen + (clen or 0)},
        })

    def _emit(self, cid, created, delta, finish):
        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": MODEL_ID,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def log_message(self, fmt, *a):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%T"), fmt % a))


PLAYGROUND = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>personality · 试效果</title><style>
body{margin:0;background:#050b14;color:#cfe9f5;font:14px/1.7 "PingFang SC",system-ui,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:22px}
h1{font-size:16px;letter-spacing:2px;color:#22d3ee;margin:0 0 4px}
p.hint{color:#6b8ba4;font-size:12px;margin:0 0 16px}
textarea{width:100%;height:230px;background:#08131f;color:#cfe9f5;border:1px solid #21455f;
  border-radius:4px;padding:12px;font:12.5px/1.6 ui-monospace,Menlo,monospace;resize:vertical}
button{margin-top:10px;padding:9px 22px;background:linear-gradient(135deg,#0e7490,#22d3ee);
  color:#02141c;border:0;border-radius:4px;font-weight:700;cursor:pointer;letter-spacing:1px}
button:disabled{opacity:.5;cursor:default}
#out{margin-top:20px;white-space:pre-wrap;border-left:2px solid #22d3ee;padding:4px 0 4px 16px;
  min-height:60px}
</style></head><body><div class="wrap">
<h1>personality · 试效果</h1>
<p class="hint">直接贴一份 profile JSON（含 instrument/domains/facets），服务端会自动包成完整消息。也可以随便聊。</p>
<textarea id="in">{
  "instrument": "big-five",
  "version": "IPIP-NEO-120",
  "scale": "raw_and_normalized",
  "norm_source": null,
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
    "N4_self_consciousness": 64, "N5_immoderation": 55, "N6_vulnerability": 52
  }
}</textarea>
<button id="go">生成解读</button>
<div id="out"></div></div>
<script>
const out=document.getElementById('out'), go=document.getElementById('go');
go.onclick=async()=>{
  out.textContent=''; go.disabled=true; go.textContent='生成中…';
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({stream:true,messages:[{role:'user',content:document.getElementById('in').value}]})});
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    for(;;){
      const {done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\\n\\n'); buf=parts.pop();
      for(const p of parts){
        const line=p.replace(/^data: /,'').trim();
        if(!line||line==='[DONE]')continue;
        try{const j=JSON.parse(line); const d=j.choices?.[0]?.delta?.content; if(d)out.textContent+=d;}catch(e){}
      }
    }
  }catch(e){ out.textContent='出错了：'+e; }
  go.disabled=false; go.textContent='生成解读';
};
</script></body></html>"""


def main():
    load_systems()
    load_model()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止", flush=True)


if __name__ == "__main__":
    main()
