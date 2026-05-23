#!/bin/bash
# ============================================================
# Script 5: LoRA 训练 + 组合 + 评估
# 1. 在 7 个任务上训练 LLaMA-2-7B 的 LoRA adapter
# 2. 用 Nevergrad 无梯度优化组合多个 LoRA 权重
# 3. merge 后评估 7 个下游任务
# ============================================================

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ===== 配置 =====
MODEL_PATH="${MODEL_PATH:-/PATH/TO/Llama-2-7b}"
LORA_R="${LORA_R:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_INFERENCE_STEP="${MAX_INFERENCE_STEP:-40}"

# ===== 获取项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  yj-proj: LoRA Pipeline"
echo "  模型: ${MODEL_PATH}"
echo "  LoRA rank: ${LORA_R}"
echo "============================================"

cd "$PROJECT_ROOT"

# 全流程: train → compose → evaluate
python pipelines/run_lora.py all \
    --base_model "$MODEL_PATH" \
    --lora_r "$LORA_R" \
    --num_epochs "$NUM_EPOCHS" \
    --max_inference_step "$MAX_INFERENCE_STEP"

echo "Done!"
