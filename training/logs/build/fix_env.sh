#!/bin/bash
# 修复 transformers 混合安装，然后重新合并 LoRA adapter。
#
# 症状：import transformers 直接抛
#   ValueError: Backend should be defined in the BACKENDS_MAPPING. Offending backend: tensorflow_text
# 原因：site-packages 里 4.56.2 和 5.6.0 两套文件混在一起（两个 dist-info 并存）。
#   5.x 已经删掉 TensorFlow 支持，BACKENDS_MAPPING 里没有 tf/tensorflow_text，
#   而 4.56 的 __init__.py 仍在 import_structure 里引用它们 —— 于是惰性模块初始化就炸了。
#   pip 的 --force-reinstall 不会删掉新版本里已被移除的旧文件，所以必须整个目录 rm 掉重装。
set -u

SP=/data/miniconda/envs/torch/lib/python3.12/site-packages
[ -d "$SP/torch" ] || { echo "site-packages 路径不对，中止"; exit 1; }

cd /data/personality || exit 1
source /data/miniconda/etc/profile.d/conda.sh
conda activate torch
export HF_HOME=/data/hf_cache
export HF_HUB_OFFLINE=1

echo "=== 修复前的版本线索（应当不一致）==="
grep -m1 -oE '^__version__ = "[^"]+"' "$SP/transformers/__init__.py" 2>/dev/null
grep -m1 -oE '^__version__ = "[^"]+"' "$SP/transformers/utils/import_utils.py" 2>/dev/null
ls -d "$SP"/transformers*.dist-info 2>/dev/null

echo
echo "=== 清除混合目录与残留 dist-info ==="
rm -rf "$SP"/transformers "$SP"/transformers-*.dist-info
find "$SP" -maxdepth 1 -name "transformers*" -print

echo
echo "=== 重装 transformers==4.56.2 ==="
echo "（约束：llamafactory 0.9.5 要 >=4.55.0,<=5.6.0，且 !=4.57.0；trl 0.24 要 >=4.56.1）"
pip install --no-cache-dir -q \
    -i https://repo.huaweicloud.com/repository/pypi/simple \
    "transformers==4.56.2" 2>&1 | tail -5

echo
echo "=== 验证 ==="
python3 -c "import transformers, peft, trl; print('transformers', transformers.__version__); print('peft', peft.__version__); print('trl', trl.__version__)" 2>&1 | tail -6
echo "--- pip check ---"
pip check 2>&1 | head -8
