#!/bin/bash
# ============================================================
# FANG 20% Mask 剪枝 + PPL 评估
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Llama-2-7b-chat-hf}"
PRUNING_RATIO="${PRUNING_RATIO:-0.2}"
NSAMPLES="${NSAMPLES:-128}"

# 从模型名推断输出目录
# 短名: Llama-2-7b-chat-hf → Llama-2-7b
MODEL_SHORT=$(basename "$MODEL_PATH" | sed 's/-chat.*//;s/-hf$//')
PR_INT=$(python3 -c "print(int(${PRUNING_RATIO}*100))")
PRUNED_DIR="$PROJECT_ROOT/outputs/pruned_models/${MODEL_SHORT}-pruned-${PR_INT}pct"

echo "============================================"
echo "  FANG ${PR_INT}% Mask 剪枝 + PPL 评估"
echo "  模型: ${MODEL_PATH}"
echo "  输出: ${PRUNED_DIR}"
echo "============================================"

cd "$PROJECT_ROOT"

# Step 1: 剪枝
python3 pipelines/run_prune.py \
    --model "$MODEL_PATH" \
    --pruning_ratio "$PRUNING_RATIO" \
    --nsamples "$NSAMPLES" \
    --save_model "$PRUNED_DIR" \
    --use_mask

# Step 2: PPL 评估
python3 pipelines/run_eval_only.py \
    --model "$PRUNED_DIR" \
    --name "pruned-${PR_INT}pct" \
    --skip_zeroshot

echo "Done!"
