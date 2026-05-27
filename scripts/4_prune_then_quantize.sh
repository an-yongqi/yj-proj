#!/bin/bash
# ============================================================
# 对已剪枝(mask)模型进行 W2A8 量化 + PPL 评估
# 前置条件: 已跑完 2_prune_and_eval.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Llama-2-7b-chat-hf}"
PRUNING_RATIO="${PRUNING_RATIO:-0.2}"
WBITS="${WBITS:-2}"
ABITS="${ABITS:-8}"
EPOCHS="${EPOCHS:-40}"
NSAMPLES="${NSAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"

MODEL_SHORT=$(basename "$MODEL_PATH" | sed 's/-chat.*//;s/-hf$//')
PR_INT=$(python3 -c "print(int(${PRUNING_RATIO}*100))")
PRUNED_DIR="$PROJECT_ROOT/outputs/pruned_models/${MODEL_SHORT}-pruned-${PR_INT}pct"
SAVE_DIR="$PROJECT_ROOT/outputs/pruned_quantized_models/${MODEL_SHORT}-pruned${PR_INT}-w${WBITS}a${ABITS}-ep${EPOCHS}-bs${BATCH_SIZE}"
LOG_DIR="$SAVE_DIR/log"

echo "============================================"
echo "  剪枝模型 W${WBITS}A${ABITS} 量化 + PPL 评估"
echo "  剪枝模型: ${PRUNED_DIR}"
echo "  输出: ${SAVE_DIR}"
echo "============================================"

# 检查剪枝模型是否存在
if [ ! -d "$PRUNED_DIR" ]; then
    echo "错误: 剪枝模型不存在: $PRUNED_DIR"
    echo "请先运行 2_prune_and_eval.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

# Step 1: 量化剪枝后的模型
python3 pipelines/run_quantize.py \
    --model "$PRUNED_DIR" \
    --save_dir "$SAVE_DIR" \
    --output_dir "$LOG_DIR" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --epochs "$EPOCHS" \
    --nsamples "$NSAMPLES" \
    --batch_size "$BATCH_SIZE"

# Step 2: PPL 评估
python3 pipelines/run_eval_only.py \
    --model "$PRUNED_DIR" \
    --abq_params "$LOG_DIR/abq_parameters.pth" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --name "pruned${PR_INT}-w${WBITS}a${ABITS}" \
    --skip_zeroshot

echo "Done!"
