#!/bin/bash
# ============================================================
# ABQ-LLM W2A8 量化 + PPL 评估
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Llama-2-7b-chat-hf}"
WBITS="${WBITS:-2}"
ABITS="${ABITS:-8}"
EPOCHS="${EPOCHS:-40}"
NSAMPLES="${NSAMPLES:-128}"

MODEL_NAME="$(basename "$MODEL_PATH")"
SAVE_DIR="$PROJECT_ROOT/outputs/quantized_models/${MODEL_NAME}-w${WBITS}a${ABITS}"
LOG_DIR="$PROJECT_ROOT/outputs/quantized_models/log"

echo "============================================"
echo "  ABQ-LLM W${WBITS}A${ABITS} 量化 + PPL 评估"
echo "  模型: ${MODEL_PATH}"
echo "  输出: ${SAVE_DIR}"
echo "============================================"

cd "$PROJECT_ROOT"

# Step 1: 量化
python3 pipelines/run_quantize.py \
    --model "$MODEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --epochs "$EPOCHS" \
    --nsamples "$NSAMPLES"

# Step 2: PPL 评估
python3 pipelines/run_eval_only.py \
    --model "$MODEL_PATH" \
    --abq_params "$LOG_DIR/abq_parameters.pth" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --name "w${WBITS}a${ABITS}" \
    --skip_zeroshot

echo "Done!"
