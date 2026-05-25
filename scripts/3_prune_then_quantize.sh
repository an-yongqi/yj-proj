#!/bin/bash
# ============================================================
# Script 3: 剪枝 + 量化
# FANG 20% 剪枝 → ABQ-LLM W2A8 量化
# 评估: PiQA, ARC-e, ARC-c, BoolQ, HellaSwag, Winogrande, OBQA
# ============================================================

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ===== 配置 =====
MODEL_PATH="${MODEL_PATH:-/PATH/TO/Llama-2-7b}"
PRUNING_RATIO="${PRUNING_RATIO:-0.2}"
WBITS="${WBITS:-2}"
ABITS="${ABITS:-8}"
EPOCHS="${EPOCHS:-40}"
NSAMPLES="${NSAMPLES:-128}"

# ===== 获取项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  yj-proj: 剪枝 + 量化 Pipeline"
echo "  模型: ${MODEL_PATH}"
echo "  剪枝: ${PRUNING_RATIO} → 量化: W${WBITS}A${ABITS}"
echo "============================================"

cd "$PROJECT_ROOT"

python pipelines/run_prune_quantize.py \
    --model "$MODEL_PATH" \
    --pruning_ratio "$PRUNING_RATIO" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --epochs "$EPOCHS" \
    --nsamples "$NSAMPLES" \
    --use_mask

echo "Done!"
