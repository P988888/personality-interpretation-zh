#!/bin/bash
# 在训练机上重启模型 API。
# 和 start_remote.sh 一样，必须放在脚本里跑：直接把 pkill 和启动命令写在同一条
# ssh 命令行里，pkill -f 会匹配到后半句里的脚本名，把自己杀掉（而且静默无输出）。
cd "$(dirname "$0")" || exit 1

pkill -f "serve_model.py" >/dev/null 2>&1
sleep 1

source /data/miniconda/etc/profile.d/conda.sh
conda activate torch
export HF_HOME=/data/hf_cache
export HF_HUB_OFFLINE=1
# 服务器上的目录仍是 training/output/，不要把本仓库的 models/ 布局套过来
export MODEL_PATH=/data/personality/training/output/qwen7b-merged
export PORT="${1:-8001}"

setsid python3 /data/personality/training/serve_model.py >/tmp/model_api.log 2>&1 </dev/null &
sleep 3
if pgrep -f "serve_model.py" >/dev/null; then
  echo "模型 API 已启动 pid=$(pgrep -f 'serve_model.py' | head -1)，端口=${PORT}"
  echo "（模型 15G，首次载入约 1-3 分钟，进度看 /tmp/model_api.log）"
else
  echo "启动失败："; cat /tmp/model_api.log
fi
