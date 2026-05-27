#!/bin/bash
# ============================================================
# 三合一兼容测试: 剪枝 + 量化 + ReST-KV
# 加载已有的剪枝+量化模型，启用 ReST-KV，做简单生成测试
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 剪枝后的基座模型
PRUNED_MODEL="${PRUNED_MODEL:-$PROJECT_ROOT/outputs/pruned_models/Llama-2-7b-pruned-20pct}"
# 量化参数路径 (旧脚本因 $MODEL_NAME 未定义, 路径以 - 开头)
ABQ_PARAMS="${ABQ_PARAMS:-$PROJECT_ROOT/outputs/pruned_quantized_models/-pruned20-w2a8/log/abq_parameters.pth}"
# 如果旧路径不存在，尝试新路径
if [ ! -f "$ABQ_PARAMS" ]; then
    MODEL_SHORT=$(basename "${PRUNED_MODEL}" | sed 's/-pruned.*//;s/-chat.*//;s/-hf$//')
    ABQ_PARAMS="$PROJECT_ROOT/outputs/pruned_quantized_models/${MODEL_SHORT}-pruned20-w${WBITS}a${ABITS}-ep40-bs4/log/abq_parameters.pth"
fi
WBITS="${WBITS:-2}"
ABITS="${ABITS:-8}"

echo "============================================"
echo "  三合一兼容测试: 剪枝+量化+ReST-KV"
echo "  剪枝模型: ${PRUNED_MODEL}"
echo "  量化参数: ${ABQ_PARAMS}"
echo "============================================"

cd "$PROJECT_ROOT"

python3 pipelines/test_three_in_one.py \
    --pruned_model "$PRUNED_MODEL" \
    --abq_params "$ABQ_PARAMS" \
    --wbits "$WBITS" \
    --abits "$ABITS"
