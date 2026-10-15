#!/usr/bin/env python3
"""
SFT 数据生成器 —— 读入分数组合，调用 LLM 生成解读，自动校验过滤。

流程：
    prompts_bigfive.jsonl  →  [LLM 生成]  →  [validate.py 校验]  →  curated

支持两种后端：
    --backend openai    走 OpenAI 兼容接口（也适用于 DeepSeek / 通义 / 智谱等）
    --backend ollama    走本地 Ollama

用法：
    # 云端 API
    export LLM_API_KEY=sk-xxx
    python3 scripts/generate.py --backend openai \
        --model gpt-4o --base-url https://api.openai.com/v1 \
        --input data/raw/prompts_bigfive.jsonl --workers 4

    # 本地
    python3 scripts/generate.py --backend ollama --model qwen2.5:32b

注意：生成质量决定训练上限。用能力不足的模型生成，再好的规格也救不回来。
      建议用 GPT-4o / Claude Sonnet 级别；本地 Ollama 至少 32B。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "specs"


def load_system_prompt(instrument: str = "big-five") -> str:
    """组装系统提示词：行为准则 + 量表规格。"""
    base = (ROOT / "prompts/system-prompt.md").read_text(encoding="utf-8")
    if instrument == "big-five":
        scale = (SPECS / "big-five.md").read_text(encoding="utf-8")
    else:
        scale = (SPECS / "mbti.md").read_text(encoding="utf-8")

    # 把 {{SCALE_SPEC}} 替换为量表规格全文
    return base.replace("{{SCALE_SPEC}}", scale)


def build_user_message(item: dict, instrument: str = "big-five") -> str:
    """把结构化分数转成给模型的用户消息。不含 _meta 等内部字段。"""
    clean = {k: v for k, v in item.items() if not k.startswith("_")}

    if instrument == "mbti":
        note = (
            "注意：\n"
            "1. 本结果由 IPIP-NEO-120 作答经构念映射导出，非 Myers-Briggs 官方量表，"
            "必须在解读的局限性说明中声明这一点。\n"
            "2. clarity 表示各维度的二分清晰度（clear/moderate/slight/marginal）。"
            "标为 slight 或 marginal 的维度，其字母区分度低，必须明确说明"
            "「你不必把自己固定在这一侧」。\n"
            "3. 解读主干必须基于 cognitive function stack（功能栈）来展开，"
            "并包含劣势功能在压力下的表现。\n"
            "4. 必须包含 MBTI 的学术争议说明（见规格第五节）。\n\n"
        )
    else:
        note = (
            "注意：分数为原始分线性映射到 0-100 的相对刻度，不是百分位，"
            "不可解释为人群排名。\n\n"
        )

    return (
        "请为以下测评结果生成解读。\n" + note
        + "```json\n"
        + json.dumps(clean, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        "严格按照系统提示词中的输出结构与行为准则执行。"
    )


# --------------------------------------------------------------- 后端实现
def call_openai(system: str, user: str, model: str, base_url: str,
                temperature: float, retries: int = 3) -> str:
    import urllib.request

    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("未设置 LLM_API_KEY 环境变量")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 2000,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API 调用失败: {last_err}")


def call_ollama(system: str, user: str, model: str,
                temperature: float, retries: int = 3) -> str:
    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 2000},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama 调用失败: {last_err}")


# --------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--input", default=None,
                    help="输入池；不指定则按 --instrument 自动选择")
    ap.add_argument("--output", default=None,
                    help="输出文件；不指定则按 --instrument 自动选择")
    ap.add_argument("--instrument", default="big-five")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="0 表示全部")
    ap.add_argument("--resume", action="store_true", help="跳过已生成项")
    args = ap.parse_args()

    # 按 instrument 推导默认路径。
    #
    # 实测教训：早期 --input 硬编码为 prompts_bigfive.jsonl，导致
    # 「--instrument mbti」时读到大五的输入池，变成用大五分数配 MBTI 提示词
    # 的灾难性错配（模型在回复里自己都指出「这并不是 MBTI」）。
    # 改为自动匹配，从根上杜绝这种组合错误。
    suffix = "mbti" if args.instrument == "mbti" else "bigfive"
    if args.input is None:
        args.input = f"data/raw/prompts_{suffix}.jsonl"
    if args.output is None:
        args.output = f"data/raw/generated_{suffix}.jsonl"

    # 一致性校验：输入文件的 instrument 必须与参数一致
    _probe = (ROOT / args.input)
    if _probe.exists():
        _first = _probe.read_text(encoding="utf-8").splitlines()
        if _first:
            _inst = json.loads(_first[0]).get("instrument")
            if _inst and _inst != args.instrument:
                raise SystemExit(
                    f"✗ 输入文件 {args.input} 的 instrument='{_inst}' "
                    f"与 --instrument '{args.instrument}' 不一致，"
                    f"这会导致分数与提示词错配。请检查路径。")

    system = load_system_prompt(args.instrument)
    src = ROOT / args.input
    dst = ROOT / args.output
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"输入 {args.input}\n输出 {args.output}")

    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]

    done = set()
    if args.resume and dst.exists():
        for l in dst.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    d = json.loads(l)
                    done.add(json.dumps(d.get("_meta", {}), sort_keys=True))
                except Exception:
                    pass
        print(f"断点续跑：已完成 {len(done)} 条")

    call = call_openai if args.backend == "openai" else call_ollama
    print(f"后端 {args.backend} / 模型 {args.model}")
    print(f"系统提示词 {len(system)} 字，待生成 {len(items)} 条，并发 {args.workers}\n")

    def work(item):
        key = json.dumps(item.get("_meta", {}), sort_keys=True)
        if key in done:
            return None
        user = build_user_message(item, args.instrument)
        if args.backend == "openai":
            out = call(system, user, args.model, args.base_url, args.temperature)
        else:
            out = call(system, user, args.model, args.temperature)
        rec = dict(item)
        rec["output"] = out.strip()
        return rec

    ok, err = 0, 0
    with dst.open("a" if args.resume else "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(work, it): it for it in items}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rec = fut.result()
                    if rec is None:
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    ok += 1
                    print(f"[{i}/{len(items)}] ✓ {len(rec['output'])} 字  "
                          f"{rec['_meta'].get('combination','')}")
                except Exception as e:
                    err += 1
                    print(f"[{i}/{len(items)}] ✗ {e}")

    print(f"\n完成：成功 {ok}，失败 {err}")
    print(f"输出 -> {dst}")
    print(f"\n下一步：python3 scripts/validate.py {args.output}")


if __name__ == "__main__":
    main()
