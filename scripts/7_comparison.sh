#!/bin/bash
# ============================================================
# Script 7: Baseline vs 优化模型 对比评测 + Demo 生成
# 1. PPL + 7 个 Zero-shot 任务对比表格
# 2. 10 个场景的生成质量 + 速度对比
# ============================================================

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ===== 配置 =====
MODEL_PATH="${MODEL_PATH:-/PATH/TO/Llama-2-7b}"
OPTIMIZED_PATH="${OPTIMIZED_PATH:-outputs/quantized_models/Llama-2-7b-pruned-w2a8}"
OPTIMIZED_NAME="${OPTIMIZED_NAME:-Prune20%+W2A8}"

# ===== 获取项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  yj-proj: 对比评测 + Demo"
echo "  Baseline: ${MODEL_PATH}"
echo "  Optimized: ${OPTIMIZED_PATH}"
echo "============================================"

cd "$PROJECT_ROOT"

# Step 1: 对比评测
echo ""
echo ">>> Step 1: PPL + Zero-shot 对比评测"
python pipelines/run_comparison.py \
    --baseline "$MODEL_PATH" \
    --optimized "$OPTIMIZED_PATH" \
    --optimized_name "$OPTIMIZED_NAME"

# Step 2: Demo 生成对比
echo ""
echo ">>> Step 2: Demo 生成对比"
python pipelines/run_demo.py \
    --baseline "$MODEL_PATH" \
    --optimized "$OPTIMIZED_PATH" \
    --optimized_name "$OPTIMIZED_NAME"

echo ""
echo "Done! 结果保存在:"
echo "  outputs/eval_results/comparison.json"
echo "  outputs/eval_results/comparison.md"
echo "  outputs/demo/demo_results.json"
echo "  outputs/demo/demo_results.md"
