#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 MLX 推理服务（OpenAI 兼容），带"没有分数就不生成"的硬闸门。

为什么不用现成的 mlx_lm.server：
    它没有闸门。实测这个模型在收到任何输入时都会去写解读，缺的分数就编 ——
    给它「你好」，它会凭空编出一整套分数开始分析。窄域 SFT 只有
    「已计分 profile → 解读」这一种样本，没有"没有数据"这个行为，
    所以必须在服务层拦死，光靠 system 提示词不管用。

用法：
    python3 training/serve_mlx.py                    # 默认 127.0.0.1:8010
    python3 training/serve_mlx.py --port 8010 --model models/personality-mlx-4bit

客户端配置：Base URL http://127.0.0.1:8010/v1 ，API Key 随便填，模型名 personality
网页试玩：  浏览器打开 http://127.0.0.1:8010
"""
import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "personality"
GEN = dict(max_tokens=1400, temp=0.7, top_p=0.8, top_k=20)

_model = _tok = None
_lock = threading.Lock()
_systems = {}


def load_systems():
    """system 提示词直接取训练集里的那份，和训练时逐字一致。"""
    for inst, fn in (("big-five", "personality_bigfive.json"),
                     ("mbti", "personality_mbti.json")):
        try:
            with open(ROOT / "data/curated" / fn, encoding="utf-8") as f:
                _systems[inst] = json.load(f)[0]["system"]
        except Exception as e:
            print(f"[warn] 读 {fn} 失败: {e}")
    print(f"[info] system 提示词: {list(_systems)}")


PROFILE_NOTE = {
    "big-five": "注意：分数为原始分线性映射到 0-100 的相对刻度，不是百分位，不可解释为人群排名。\n\n",
    "mbti": ("注意：\n1. 本结果由 IPIP-NEO-120 作答经构念映射导出，非 Myers-Briggs 官方量表，"
             "必须在解读的局限性说明中声明这一点。\n"
             "2. clarity 表示各维度的二分清晰度（clear/moderate/slight/marginal）。"
             "标为 slight 或 marginal 的维度，其字母区分度低，必须明确说明"
             "「你不必把自己固定在这一侧」。\n"
             "3. 解读主干必须基于 cognitive function stack（功能栈）来展开，"
             "并包含劣势功能在压力下的表现。\n"
             "4. 必须包含 MBTI 的学术争议说明（见规格第五节）。\n\n"),
}

PROFILE_SHAPE = re.compile(r'"(facets|domains|dimensions)"\s*:\s*[' + '{' + r'\[]')


def msg_text(m):
    c = m.get("content") or ""
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)


def has_profile(msgs):
    return any(PROFILE_SHAPE.search(msg_text(m)) for m in msgs)


def detect_inst(text):
    return "mbti" if ('"instrument": "mbti"' in text or '"instrument":"mbti"' in text
                      or '"dimensions"' in text) else "big-five"


def wrap_profile(profile, inst):
    clean = {k: v for k, v in profile.items() if not k.startswith("_") and k != "output"}
    return ("请为以下测评结果生成解读。\n" + PROFILE_NOTE[inst]
            + "```json\n" + json.dumps(clean, ensure_ascii=False, indent=2)
            + "\n```\n\n严格按照系统提示词中的输出结构与行为准则执行。")


def build_messages(body):
    msgs = list(body.get("messages") or [])
    inst = detect_inst(json.dumps(msgs, ensure_ascii=False))
    if msgs and msgs[-1].get("role") == "user":
        txt = msg_text(msgs[-1]).strip()
        if txt.startswith("{") and txt.endswith("}"):
            try:
                prof = json.loads(txt)
                if isinstance(prof, dict) and any(
                        k in prof for k in ("facets", "domains", "dimensions", "instrument")):
                    inst = prof.get("instrument") or inst
                    if inst not in _systems:
                        inst = "big-five"
                    msgs = [{"role": "user", "content": wrap_profile(prof, inst)}]
            except Exception:
                pass
    if not any(m.get("role") == "system" for m in msgs):
        if inst not in _systems:
            inst = "big-five"
        msgs = [{"role": "system", "content": _systems[inst]}] + msgs
    return msgs


NO_PROFILE_MSG = """我这边只做一件事：**把已经算好的测评结果翻译成一段解读**。出题和计分不归我管，所以在收到结构化分数之前，我不会给任何解读——凭空生成分数正是这个模型最需要避免的错。

完整流程两步，都在本地脚本里跑：

**1. 做题 + 计分**（IPIP-NEO-120，120 道，每题 1-5 分）
```bash
python3 scripts/take_test.py --out me.json
```

**2. 把结果贴给我**
`take_test.py` 会把结果写到 `data/results/`，把那份 JSON 原样贴过来即可，
本服务会自动识别并套上对应的提示词（大五 / MBTI 自动判断）。

如果你手上已经有一份结果，格式长这样（`facets` 要给全 30 项）：

```json
{
  "instrument": "big-five",
  "scale": "raw_and_normalized",
  "domains": { "O": 71, "C": 38, "E": 66, "A": 34, "N": 57 },
  "facets": { "O1_imagination": 78, "C5_self_discipline": 28, "...": 0 }
}
```"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": int(time.time()),
                 "owned_by": "local"}]})
        elif p in ("/", "/index.html"):
            b = PLAYGROUND.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif p == "/health":
            self._json(200, {"ok": _model is not None, "model": MODEL_ID})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json(400, {"error": {"message": f"请求体不是合法 JSON: {e}"}})
            return
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return

        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        stream = bool(body.get("stream"))

        # 硬闸门：整段对话里没有结构化分数就不调模型
        if not has_profile(body.get("messages") or []):
            if stream:
                self._sse_open()
                self._emit(cid, created, {"role": "assistant", "content": NO_PROFILE_MSG}, None)
                self._emit(cid, created, {}, "stop")
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            else:
                self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                 "model": MODEL_ID,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant",
                                                          "content": NO_PROFILE_MSG}}],
                                 "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                           "total_tokens": 0}})
            return

        try:
            msgs = build_messages(body)
        except Exception as e:
            self._json(400, {"error": {"message": f"消息构造失败: {e}"}})
            return

        prompt = _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(
            temp=float(body.get("temperature") or GEN["temp"]),
            top_p=float(body.get("top_p") or GEN["top_p"]),
            top_k=GEN["top_k"])
        max_tokens = int(body.get("max_tokens") or GEN["max_tokens"])
        plen = len(_tok.encode(prompt))

        try:
            with _lock:
                if stream:
                    self._sse_open()
                    self._emit(cid, created, {"role": "assistant", "content": ""}, None)
                    nout = 0
                    for r in stream_generate(_model, _tok, prompt, max_tokens=max_tokens,
                                             sampler=sampler):
                        if r.text:
                            nout += 1
                            self._emit(cid, created, {"content": r.text}, None)
                    self._emit(cid, created, {}, "stop")
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                    return
                parts = []
                for r in stream_generate(_model, _tok, prompt, max_tokens=max_tokens,
                                         sampler=sampler):
                    parts.append(r.text)
        except Exception as e:
            self._json(500, {"error": {"message": f"生成失败: {e}"}})
            return

        full = "".join(parts)
        clen = len(_tok.encode(full))
        self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                         "model": MODEL_ID,
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": full}}],
                         "usage": {"prompt_tokens": plen, "completion_tokens": clen,
                                   "total_tokens": plen + clen}})

    def _sse_open(self):
        self.close_connection = True      # SSE 无 Content-Length，靠关连接收尾
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _emit(self, cid, created, delta, finish):
        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": MODEL_ID,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
        self.wfile.flush()

    def log_message(self, fmt, *a):
        print("[%s] %s" % (time.strftime("%T"), fmt % a), flush=True)


PLAYGROUND = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>personality · 本地试效果</title><style>
body{margin:0;background:#050b14;color:#cfe9f5;font:14px/1.7 "PingFang SC",system-ui,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px}
h1{font-size:16px;letter-spacing:2px;color:#22d3ee;margin:0 0 4px}
p.hint{color:#6b8ba4;font-size:12px;margin:0 0 16px}
textarea{width:100%;height:240px;background:#08131f;color:#cfe9f5;border:1px solid #21455f;
  border-radius:4px;padding:12px;font:12.5px/1.6 ui-monospace,Menlo,monospace;resize:vertical}
button{margin-top:10px;padding:9px 22px;background:linear-gradient(135deg,#0e7490,#22d3ee);
  color:#02141c;border:0;border-radius:4px;font-weight:700;cursor:pointer;letter-spacing:1px}
button:disabled{opacity:.5;cursor:default}
#out{margin-top:20px;white-space:pre-wrap;border-left:2px solid #22d3ee;padding:4px 0 4px 16px;min-height:60px}
</style></head><body><div class="wrap">
<h1>personality · 本地试效果</h1>
<p class="hint">贴一份 profile JSON（含 instrument/domains/facets），会自动套上正确提示词。没有分数的输入不会生成解读。</p>
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
<button id="go">生成解读</button><div id="out"></div></div>
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/personality-mlx-4bit"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    load_systems()
    print(f"[{time.strftime('%T')}] 载入 {a.model} ...", flush=True)
    global _model, _tok
    _model, _tok = load(a.model)
    print(f"[{time.strftime('%T')}] 就绪 → http://{a.host}:{a.port}   (Ctrl-C 退出)", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
