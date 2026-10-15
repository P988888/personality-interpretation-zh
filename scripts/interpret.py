#!/usr/bin/env python3
"""
解读模块 —— 把计分结果转成解读。

两种模式：
    --mode prompt    只输出组装好的提示词（供你贴到任意 LLM 使用）  ← 默认
    --mode api       直接调用 LLM 生成解读（需配置 LLM_API_KEY）

用法：
    python3 scripts/interpret.py data/results/result-xxx.json
    python3 scripts/interpret.py data/results/result-xxx.json --mode api \
        --model gpt-4o --base-url https://api.openai.com/v1

设计说明：
    解读质量取决于 specs/ 里的规格文件，不取决于本脚本。
    本脚本只负责：组装提示词 → （可选）调模型 → 输出。
    训练微调模型时，训练数据也应由同一套提示词生成，保证训练与推理口径一致。
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "specs"


def build_prompt(result: dict, instrument: str = "big-five") -> tuple:
    """返回 (system_prompt, user_message)。"""
    base = (ROOT / "prompts/system-prompt.md").read_text(encoding="utf-8")
    if instrument == "big-five":
        scale = (SPECS / "big-five.md").read_text(encoding="utf-8")
    else:
        scale = (SPECS / "mbti.md").read_text(encoding="utf-8")
    system = base.replace("{{SCALE_SPEC}}", scale)

    if instrument == "mbti":
        if "mbti" not in result:
            raise ValueError(
                "该结果文件中没有 MBTI 数据。请用 take_test.py 重新测评"
                "（不要加 --no-mbti），或先运行 score_mbti.py 导出。")
        m = result["mbti"]
        clean = {
            "instrument": "mbti",
            "result": m["result"],
            "dimensions": {
                "EI": {"E": m["dimensions"]["EI"]["left"]["pct"],
                       "I": m["dimensions"]["EI"]["right"]["pct"]},
                "SN": {"S": m["dimensions"]["SN"]["right"]["pct"],
                       "N": m["dimensions"]["SN"]["left"]["pct"]},
                "TF": {"T": m["dimensions"]["TF"]["right"]["pct"],
                       "F": m["dimensions"]["TF"]["left"]["pct"]},
                "JP": {"J": m["dimensions"]["JP"]["left"]["pct"],
                       "P": m["dimensions"]["JP"]["right"]["pct"]},
            },
            "preferred_function_stack": m["preferred_function_stack"],
            "clarity": m["clarity"],
        }
        note = (
            "注意：\n"
            "1. 本结果由 IPIP-NEO-120 作答经构念映射导出，非 Myers-Briggs 官方量表，"
            "必须在解读的局限性说明中声明这一点。\n"
            "2. clarity 表示各维度的二分清晰度（clear/moderate/slight/marginal）。"
            "标为 slight 或 marginal 的维度，其字母区分度低，必须明确说明"
            "「你不必把自己固定在这一侧」。\n"
            "3. 解读主干必须基于 cognitive function stack（功能栈）来展开，"
            "并包含劣势功能在压力下的表现。\n"
            "4. 必须包含 MBTI 的学术争议说明（见规格第五节）。\n"
        )
    else:
        clean = {
            "instrument": result["instrument"],
            "version": result["version"],
            "scale": result["scale"],
            "norm_source": result.get("norm_source"),
            "domains": result["domains"],
            "facets": result["facets"],
        }
        note = (
            "注意：分数为原始分线性映射到 0-100 的相对刻度，不是百分位，"
            "不可解释为人群排名。\n"
        )

    user = (
        "请为以下测评结果生成解读。\n" + note + "\n"
        "```json\n" + json.dumps(clean, ensure_ascii=False, indent=2) + "\n```\n\n"
        "严格按照系统提示词中的输出结构与行为准则执行。"
    )
    return system, user


def call_api(system, user, model, base_url, temperature=0.7):
    import urllib.request
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("未设置 LLM_API_KEY")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": 2000,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="计分结果 JSON 路径")
    ap.add_argument("--mode", choices=["prompt", "api"], default="prompt")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--save", action="store_true", help="把解读存到结果同目录")
    ap.add_argument("--instrument", choices=["big-five", "mbti"], default="big-five",
                    help="解读哪套量表（mbti 需要结果文件含 mbti 字段）")
    args = ap.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    system, user = build_prompt(result, args.instrument)

    sys_tok = len(system)
    print(f"提示词组装完成：system {sys_tok} 字，user {len(user)} 字", file=sys.stderr)

    if args.mode == "prompt":
        out = ROOT / "prompts/last-request.json"
        out.write_text(json.dumps(
            {"system": system, "user": user}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n提示词已保存 -> {out}", file=sys.stderr)
        print("可直接用于任意 LLM（把 system 与 user 两段分别贴入）。", file=sys.stderr)
        print("\n" + "=" * 60)
        print("USER MESSAGE:")
        print("=" * 60)
        print(user)
        return

    text = call_api(system, user, args.model, args.base_url)
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

    if args.save:
        p = Path(args.result)
        outp = p.with_name(p.stem + "-interpretation.md")
        outp.write_text(text, encoding="utf-8")
        print(f"\n解读已保存 -> {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
