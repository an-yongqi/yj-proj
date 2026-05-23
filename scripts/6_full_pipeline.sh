#!/bin/bash
# ============================================================
# Script 6: 全局集成流水线
# Prune (20%) → Quantize (W2A8) → LoRA → KV Cache → Evaluate
# ============================================================

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ===== 配置 =====
MODEL_PATH="${MODEL_PATH:-/PATH/TO/Llama-2-7b}"
PRUNING_RATIO="${PRUNING_RATIO:-0.2}"
WBITS="${WBITS:-2}"
ABITS="${ABITS:-8}"
LORA_R="${LORA_R:-16}"
KV_METHOD="${KV_METHOD:-restkv}"

# ===== 获取项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  yj-proj: 全局集成 Pipeline"
echo "  模型: ${MODEL_PATH}"
echo "  流程: Prune ${PRUNING_RATIO} → W${WBITS}A${ABITS} → LoRA r${LORA_R} → ${KV_METHOD}"
echo "============================================"

cd "$PROJECT_ROOT"

python pipelines/run_full_pipeline.py \
    --model "$MODEL_PATH" \
    --pruning_ratio "$PRUNING_RATIO" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --lora_r "$LORA_R" \
    --kv_method "$KV_METHOD"

echo "Done!"
