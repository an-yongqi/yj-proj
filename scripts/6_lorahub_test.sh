#!/bin/bash
# ============================================================
# LoraHub 快速测试
# 1. 创建多个随机 LoRA adapter
# 2. 对不同输入，用 Nevergrad 无梯度优化组合权重
# 3. 对比不同输入下的权重分布
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Llama-2-7b-chat-hf}"
NUM_LORAS="${NUM_LORAS:-5}"
LORA_R="${LORA_R:-16}"
ADAPTER_DIR="${ADAPTER_DIR:-$PROJECT_ROOT/outputs/lora_adapters}"
MAX_STEP="${MAX_STEP:-20}"
GPU="${GPU:-0}"

echo "============================================"
echo "  LoraHub 自动 LoRA 组合测试"
echo "  模型: ${MODEL_PATH}"
echo "  LoRA 数量: ${NUM_LORAS}"
echo "  LoRA rank: ${LORA_R}"
echo "  优化步数: ${MAX_STEP}"
echo "  GPU: ${GPU}"
echo "============================================"

cd "$PROJECT_ROOT"

# ── Step 1: 创建随机 LoRA adapter ──
echo ""
echo ">>> [1/2] 创建 ${NUM_LORAS} 个随机 LoRA adapter ..."
python3 pipelines/create_dummy_loras.py \
    --base_model "$MODEL_PATH" \
    --num_loras "$NUM_LORAS" \
    --lora_r "$LORA_R" \
    --output_dir "$ADAPTER_DIR"

# ── Step 2: 测试 LoraHub 组合 ──
echo ""
echo ">>> [2/2] 测试 LoraHub 自动组合 ..."
python3 pipelines/run_lorahub_test.py \
    --base_model "$MODEL_PATH" \
    --adapter_dir "$ADAPTER_DIR" \
    --max_inference_step "$MAX_STEP"

echo ""
echo "============================================"
echo "  完成! 结果: outputs/lorahub_test_results.json"
echo "============================================"
