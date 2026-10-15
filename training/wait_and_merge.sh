#!/bin/bash
# 等训练跑完 -> 自动合并 LoRA adapter -> 打印产物清单
#
# 用 setsid 起，和 ssh 会话解耦；日志写在 /data/personality/merge.log，
# 随时 tail 就能看进度。重复运行是安全的（已在跑就直接退出）。
LOG=/data/personality/merge.log
cd /data/personality || exit 1

# 用 flock 防重复启动。不用 pgrep 自查——脚本自己的命令行里就含脚本名，
# 那种写法很容易把自己判成"已在运行"或者反过来把自己杀掉。
exec 9>/tmp/wait_and_merge.lock
if ! flock -n 9; then
  echo "已有实例在跑，退出"
  exit 0
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

{
  log "守护启动，等待训练结束…"
  while pgrep -f "[l]lamafactory-cli train" >/dev/null; do sleep 20; done
  log "训练进程已退出"

  ADAPTER=models/personality-lora-v1/adapter_model.safetensors
  if [ ! -f "$ADAPTER" ]; then
    log "!! 没找到 $ADAPTER —— 训练可能中途失败，放弃合并"
    tail -60 fulltrain.log
    exit 1
  fi

  log "训练产物："
  ls -la models/personality-lora-v1/

  # 模型只在 torch 这个 conda 环境里，系统 python3 没有 llamafactory-cli
  source /data/miniconda/etc/profile.d/conda.sh
  conda activate torch
  export HF_HOME=/data/hf_cache
  export HF_HUB_OFFLINE=1              # 全程离线，防止 HF 去联网卡住
  unset HF_HUB_ENABLE_HF_TRANSFER

  log "开始合并（CPU，约几分钟）…"
  llamafactory-cli export training/configs/merge_qwen7b.yaml
  log "合并命令返回 rc=$?"

  log "合并产物："
  ls -la models/personality-fp16-merged/ 2>/dev/null | head -20
  du -sh models/personality-fp16-merged/ 2>/dev/null
  log "全部完成"
} >>"$LOG" 2>&1
