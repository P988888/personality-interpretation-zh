#!/bin/bash
# 从训练机把全部产物拉到本机。
#
# 关于速度：实测本机到服务器的链路封顶约 7 MB/s（单流 5.4，aes128-gcm 6.1，
# 四流并行 7.0 —— 并发只涨 29%，说明瓶颈在服务器出口带宽，不在单连接）。
# 所以别指望靠调传输参数提速，唯一有效的办法是少传字节：
#   只传 adapter 不传 merged，体积从 17G 降到 350M。
# 本脚本默认就只传"开源需要的东西"，merged 那 15G 单独用 --with-merged 才拉。
set -u

HOST="${TRAIN_SSH_HOST:-personality-gpu}"
DEST="$(cd "$(dirname "$0")/.." && pwd)"
SSH_OPT=(-e "ssh -T -c aes128-gcm@openssh.com -o ServerAliveInterval=30 -o ServerAliveCountMax=6")
R="rsync -a --partial --info=none ${SSH_OPT[*]}"

cd "$DEST" || exit 1
mkdir -p training/output training/logs/build data

echo "==> 1/4 LoRA 适配器（323MB，开源发布的主体）"
$R "$HOST:/data/personality/training/output/qwen7b-lora-v1/" models/personality-lora-v1/

echo "==> 2/4 数据集"
$R "$HOST:/data/personality/data/curated/" data/curated/

echo "==> 3/4 日志与构建记录"
$R "$HOST:/data/personality/fulltrain.log" training/logs/
$R "$HOST:/data/personality/merge.log"   training/logs/merge.log
$R "$HOST:/data/personality/merge2.log"  training/logs/merge2.log
$R "$HOST:/data/personality/ab_test.json" training/logs/build/

echo "==> 4/4 代码（dashboard / 训练脚本）"
$R "$HOST:/data/personality/dashboard/" dashboard/
rsync -a "${SSH_OPT[@]}" --exclude 'output' --exclude '__pycache__' \
     "$HOST:/data/personality/training/" training/

if [ "${1:-}" = "--with-merged" ]; then
  echo "==> 额外：合并后的完整权重（15G，约 40 分钟）"
  echo "    注意这里 98% 是 Qwen2.5-7B-Instruct 的公开权重，"
  echo "    只是为了拿一个免合并、可直接加载的 fp16 模型。"
  $R "$HOST:/data/personality/training/output/qwen7b-merged/" models/personality-fp16-merged/
else
  echo
  echo "跳过 merged（15G）。真要一份能直接跑的本地模型，更划算的是在服务器上"
  echo "转成 GGUF Q4_K_M（约 4.5G）再拉 —— 体积只有三分之一。"
  echo "需要完整的 fp16 权重就加参数：bash training/download_all.sh --with-merged"
fi

echo
echo "==> 完成，落地体积："
du -sh models/personality-lora-v1 data/curated training/logs dashboard 2>/dev/null
