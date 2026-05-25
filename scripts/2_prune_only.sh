#!/bin/bash
# ============================================================
# Script 2: FANG 结构化剪枝 (20%)
# 五阶段: 聚类 → 重要性评分 → 神经元分组 → 稀疏度分配 → 剪枝
# 评估: PiQA, ARC-e, ARC-c, BoolQ, HellaSwag, Winogrande, OBQA
# ============================================================

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ===== 配置 =====
MODEL_PATH="${MODEL_PATH:-/PATH/TO/Llama-2-7b}"
PRUNING_RATIO="${PRUNING_RATIO:-0.2}"
NSAMPLES="${NSAMPLES:-128}"

# ===== 获取项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  yj-proj: FANG 结构化剪枝 Pipeline"
echo "  模型: ${MODEL_PATH}"
echo "  剪枝比例: ${PRUNING_RATIO}"
echo "============================================"

cd "$PROJECT_ROOT"

python pipelines/run_prune.py \
    --model "$MODEL_PATH" \
    --pruning_ratio "$PRUNING_RATIO" \
    --nsamples "$NSAMPLES" \
    --use_mask

echo "Done!"
