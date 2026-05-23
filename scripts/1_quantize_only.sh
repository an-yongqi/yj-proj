#!/bin/bash
# ============================================================
# Script 1: W2A8 量化
# 使用 ABQ-LLM 对 LLaMA-2-7B 进行 W2A8 量化
# 评估: PiQA, ARC-e, ARC-c, BoolQ, HellaSwag, Winogrande, OBQA
# ============================================================

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ===== 配置 =====
MODEL_PATH="${MODEL_PATH:-/PATH/TO/Llama-2-7b}"
WBITS="${WBITS:-2}"
ABITS="${ABITS:-8}"
EPOCHS="${EPOCHS:-40}"
NSAMPLES="${NSAMPLES:-128}"

# ===== 获取项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  yj-proj: W${WBITS}A${ABITS} 量化 Pipeline"
echo "  模型: ${MODEL_PATH}"
echo "============================================"

cd "$PROJECT_ROOT"

python pipelines/run_quantize.py \
    --model "$MODEL_PATH" \
    --wbits "$WBITS" \
    --abits "$ABITS" \
    --epochs "$EPOCHS" \
    --nsamples "$NSAMPLES"

echo "Done!"
