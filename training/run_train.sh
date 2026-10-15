#!/usr/bin/env bash
# 一键训练脚本 —— 在租用的 GPU 机器上执行
#
# 前提：
#   1. 已把本项目上传到机器（含 data/curated/ 与 training/configs/）
#   2. 机器已装好 CUDA 驱动
#
# 用法：
#   bash training/run_train.sh              # 完整流程
#   bash training/run_train.sh --smoke      # 只跑 20 步验证管线
#
# 脚本做四件事：装依赖 → 拉数据 → 训练 → 合并 adapter
# 每步失败即中止（set -e），避免训练到一半才发现数据没挂上。

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SMOKE=0
[ "$1" = "--smoke" ] && SMOKE=1

echo "=============================================="
echo " 人格测评模型 LoRA 训练"
echo " 项目目录: $ROOT"
echo " 模式: $([ $SMOKE = 1 ] && echo '冒烟测试（20步）' || echo '完整训练')"
echo "=============================================="

# ---------------------------------------------------------------- 0. 环境检查
echo -e "\n[0/4] 检查环境..."
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader || { echo "✗ 未检测到 GPU"; exit 1; }
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" \
    2>/dev/null || echo "  提示: PyTorch 未安装，即将安装"

command -v llamafactory-cli >/dev/null 2>&1 || {
    echo "  未发现 llamafactory-cli，开始安装依赖..."
    pip install --quiet "llamafactory[torch,metrics]" || {
        echo "✗ LLaMA-Factory 安装失败，请手动安装后重试"; exit 1; }
}

# ---------------------------------------------------------------- 1. 数据检查
echo -e "\n[1/4] 检查训练数据..."
for f in data/curated/personality_bigfive.json \
         data/curated/personality_mbti.json \
         data/curated/dataset_info.json; do
    if [ ! -f "$f" ]; then
        echo "✗ 缺少 $f"
        echo "  请先在本机运行: python3 scripts/pipeline.py validate"
        exit 1
    fi
    echo "  ✓ $f ($(wc -c < "$f" | tr -d ' ') 字节)"
done

# 挂到 LLaMA-Factory 的 data 目录
LF_DATA="$(python3 -c "
import importlib.util, os
spec = importlib.util.find_spec('llamafactory')
print(os.path.join(os.path.dirname(spec.origin), '..', 'data')) if spec else print('')
" 2>/dev/null)"
if [ -n "$LF_DATA" ] && [ -d "$LF_DATA" ]; then
    cp data/curated/personality_*.json data/curated/dataset_info.json "$LF_DATA/"
    echo "  ✓ 已复制数据到 LLaMA-Factory: $LF_DATA"
else
    echo "  ⚠ 未能定位 LLaMA-Factory data 目录，请手动复制 data/curated/ 下的文件"
fi

# ---------------------------------------------------------------- 2. 训练
echo -e "\n[2/4] 开始训练..."
CFG="training/configs/lora_qwen7b.yaml"

if [ $SMOKE = 1 ]; then
    # 冒烟测试：只跑少量步数，验证数据能读、模型能载、显存够用
    llamafactory-cli train "$CFG" \
        --max_steps 20 \
        --save_steps 20 \
        --eval_steps 20 \
        --output_dir models/smoke
    echo -e "\n✓ 冒烟测试通过。确认无误后运行完整训练："
    echo "  bash training/run_train.sh"
    exit 0
fi

llamafactory-cli train "$CFG"

# ---------------------------------------------------------------- 3. 合并
echo -e "\n[3/4] 合并 LoRA adapter..."
llamafactory-cli export training/configs/merge_qwen7b.yaml

# ---------------------------------------------------------------- 4. 完成
echo -e "\n[4/4] 训练完成"
echo "=============================================="
echo " LoRA adapter : models/personality-lora-v1"
echo " 合并后模型   : models/personality-fp16-merged"
echo ""
echo " 部署方式（二选一）："
echo "   vLLM:"
echo "     python -m vllm.entrypoints.openai.api_server \\"
echo "         --model models/personality-fp16-merged \\"
echo "         --served-model-name personality --port 8000"
echo "   或转 GGUF 本地跑（Ollama / llama.cpp）"
echo "=============================================="
