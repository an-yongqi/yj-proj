#!/bin/bash
# ============================================================
# 基线模型 PPL 评估
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Llama-2-7b-chat-hf}"

echo "============================================"
echo "  基线模型 PPL 评估"
echo "  模型: ${MODEL_PATH}"
echo "============================================"

cd "$PROJECT_ROOT"

python pipelines/run_eval_only.py \
    --model "$MODEL_PATH" \
    --name baseline \
    --skip_zeroshot

echo "Done!"
