#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练服务器侧的指标采集器。

职责：把 /data/personality/fulltrain.log 解析成一个小 JSON 打到 stdout。
被本地大屏通过 ssh 调用，每次只回传 2-5 KB，而不是把几 MB 的日志搬来搬去。

放在服务器上而不是本地解析的原因：日志是追加写的，还在增长；
每次抓全量会越来越大，而这里只传「解析后的结论」。
"""
import glob
import json
import os
import re
import subprocess
import time

ROOT = "/data/personality"
LOG = os.path.join(ROOT, "fulltrain.log")
OUTDIR = os.path.join(ROOT, "models/personality-lora-v1")
CFG = os.path.join(ROOT, "training/configs/full.yaml")


def sh(cmd, timeout=8):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "").strip()
    except Exception:
        return ""


def to_sec(s):
    """tqdm 的 1:02:03 / 25:06 转秒。"""
    try:
        parts = [int(x) for x in s.split(":")]
    except Exception:
        return None
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec


# tqdm 进度： " 41%|████      | 110/267 [25:06<36:14, 13.85s/it]"
# 不要带前导的 \|，否则会先匹到左边的 "| 41%" 那一根。
RE_PRG = re.compile(r"(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)\s*(s/it|it/s)\]")
RE_SUM = re.compile(
    r"\{'loss':\s*([-\d.eE+]+),\s*'grad_norm':\s*([-\d.eEnNaiIfF]+),"
    r"\s*'learning_rate':\s*([-\d.eE+]+),\s*'epoch':\s*([-\d.eE+]+)\}"
)
RE_EVAL = re.compile(r"\{'eval_loss':\s*([-\d.eE+]+),.*?'epoch':\s*([-\d.eE+]+)\}")
# 训练收尾时 LLaMA-Factory 换一种格式再打一遍：
#   ***** eval metrics *****
#     epoch     = 3.0
#     eval_loss = 1.4998
# 这套是最终值，只认字典格式会漏掉，面板上就会一直显示倒数第二次的 eval_loss。
RE_EVAL_BLOCK = re.compile(r"\*{5}\s*eval metrics\s*\*{5}")
RE_EVAL_KV = re.compile(r"^\s*(\w+)\s*=\s*(\S+)\s*$")
# tqdm 写日志时会夹带 ESC 光标控制码，ssh 之后常留下 "[A" 之类残渣
RE_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
RE_STRAY = re.compile(r"^\[[0-9;?]*[A-K]")


def clean_line(ln):
    return RE_STRAY.sub("", RE_ANSI.sub("", ln)).strip()


def read_epochs():
    try:
        with open(CFG, "r", encoding="utf-8", errors="replace") as f:
            m = re.search(r"^num_train_epochs:\s*([\d.]+)", f.read(), re.M)
        return float(m.group(1)) if m else 3.0
    except Exception:
        return 3.0


def parse_log():
    try:
        with open(LOG, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
    except Exception as e:
        return {"log_error": str(e), "train": [], "evals": [], "progress": None}

    # 进度条用 \r 刷新，同一行会被覆盖无数次；拆成行后逐个扫描。
    lines = data.replace("\r", "\n").split("\n")

    # 日志里不止一条进度条：前缀有数据预处理（如 4/4），期间还有 eval（40/40）。
    # 不能"取第一个"——那会锁到预处理上。主训练条刷新次数最多，
    # 所以按出现次数投票，同票取更大的总数。
    votes = {}
    for ln in lines:
        m = RE_PRG.search(ln)
        if m:
            votes[int(m.group(2))] = votes.get(int(m.group(2)), 0) + 1
    total = None
    if votes:
        total = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]

    epochs = read_epochs()
    cur = 0
    elapsed = None
    sec_it = None
    train, evals = [], []

    for ln in lines:
        m = RE_PRG.search(ln)
        if m:
            step, tot, el, _eta, rate, unit = m.groups()
            if int(tot) == total:
                cur = int(step)
                elapsed = to_sec(el)
                if unit == "s/it":
                    sec_it = float(rate)
                elif float(rate) > 0:
                    sec_it = 1.0 / float(rate)
            continue

        m = RE_SUM.search(ln)
        if m:
            g = m.groups()
            ep = float(g[3])
            # 用 epoch 反推 step 比用进度条更准：
            # 摘要行往往打在进度条刷新之前，读到的会是上一步。
            step = round(ep * total / epochs) if (total and epochs) else cur
            train.append({
                "step": step,
                "loss": float(g[0]),
                "grad_norm": g[1],
                "lr": float(g[2]),
                "epoch": ep,
            })
            continue

        m = RE_EVAL.search(ln)
        if m:
            ep = float(m.group(2))
            step = round(ep * total / epochs) if (total and epochs) else cur
            evals.append({"step": step, "eval_loss": float(m.group(1)), "epoch": ep})
            continue

    # 收尾的 "***** eval metrics *****" 块：覆盖成最终值
    final = None
    for i, ln in enumerate(lines):
        if RE_EVAL_BLOCK.search(ln):
            kv = {}
            for nxt in lines[i + 1: i + 14]:
                mm = RE_EVAL_KV.match(nxt)
                if not mm:
                    break
                kv[mm.group(1)] = mm.group(2)
            if "eval_loss" in kv:
                try:
                    final = {"eval_loss": float(kv["eval_loss"]),
                             "epoch": float(kv.get("epoch", 0) or 0)}
                except ValueError:
                    pass
    if final:
        step = round(final["epoch"] * total / epochs) if (total and epochs) else cur
        # 同一个 step 已有记录就替换，否则追加
        for e in evals:
            if e["step"] == step:
                e["eval_loss"] = final["eval_loss"]
                e["epoch"] = final["epoch"]
                break
        else:
            evals.append({"step": step, "eval_loss": final["eval_loss"],
                          "epoch": final["epoch"]})

    # 日志里的进度条是覆盖式的，末尾可能停在比 train 曲线更新的位置，
    # 所以以「最后一次进度条」为准判断当前步。
    if total:
        epoch_now = round(cur / total * epochs, 3) if total else 0
    else:
        epoch_now = 0

    return {
        "train": train,
        "evals": evals,
        "total_steps": total,
        "cur_step": cur,
        "elapsed_sec": elapsed,
        "sec_per_it": sec_it,
        "epochs": epochs,
        "epoch_now": epoch_now,
    }


def gpu():
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    raw = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits")
    if not raw:
        return None
    try:
        util, mu, mt, temp, pw = [x.strip() for x in raw.splitlines()[0].split(",")]
        return {
            "util": float(util),
            "mem_used": float(mu),
            "mem_total": float(mt),
            "temp": float(temp),
            "power": float(pw),
        }
    except Exception:
        return None


def artifacts():
    got = {}
    for path in glob.glob(os.path.join(OUTDIR, "*")):
        if os.path.isfile(path):
            got[os.path.basename(path)] = os.path.getsize(path)
    return got


def main():
    p = parse_log()
    total = p.get("total_steps")
    cur = p.get("cur_step") or 0
    elapsed = p.get("elapsed_sec")
    sec_it = p.get("sec_per_it")

    eta = None
    pct = None
    if total and sec_it:
        pct = round(cur / total * 100, 2)
        eta = round((total - cur) * sec_it)
    elif total and elapsed and cur:
        sec_it = elapsed / cur
        pct = round(cur / total * 100, 2)
        eta = round((total - cur) * sec_it)

    # 日志尾部：不能直接 tail 原始字节——进度条用 \r 刷新且会被从中间截断，
    # 前端会读到半截的 "48% |" 碎片。这里只挑完整且有信息量的行，
    # 并且把连续刷新的进度条压成一条。
    log_lines = []
    try:
        raw = sh(f"tail -c 20000 {LOG}").replace("\r", "\n")
        entries = []          # (是不是进度条, 文本)
        for ln in raw.split("\n"):
            ln = clean_line(ln)
            if not ln:
                continue
            if RE_PRG.search(ln):
                entries.append((True, ln))
            elif any(k in ln for k in ("'loss'", "'eval_loss'", "'train_runtime'",
                                       "Traceback", "Error", "error")):
                entries.append((False, ln))

        last_bar = max((i for i, (b, _) in enumerate(entries) if b), default=None)
        log_lines = [t for i, (b, t) in enumerate(entries)
                     if not b or i == last_bar][-9:]
    except Exception:
        pass

    out = {
        "ts": time.time(),
        "host": sh("hostname"),
        # 注意 [l] 的写法：subprocess 起的是 sh -c "pgrep -f 'llamafactory-cli train'"，
        # 那个 shell 自己的命令行里就含这串字符，不加方括号 pgrep 会匹配到自己，
        # alive 永远是 True。（和 pkill 那个自杀坑是同一类问题。）
        "alive": bool(sh("pgrep -f '[l]lamafactory-cli train'")),
        "finished": os.path.exists(os.path.join(OUTDIR, "adapter_model.safetensors")),
        "progress": {
            "cur": cur,
            "total": total,
            "pct": pct,
            "epoch_now": p.get("epoch_now"),
            "epochs": p.get("epochs"),
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "sec_per_it": round(sec_it, 2) if sec_it else None,
        },
        "loss_curve": p.get("train") or [],
        "eval_curve": p.get("evals") or [],
        "gpu": gpu(),
        "artifacts": artifacts(),
        "log_lines": log_lines,
        "log_error": p.get("log_error"),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
