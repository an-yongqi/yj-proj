#!/bin/bash
# ============================================================
# Script 4: ReST-KV KV Cache 压缩
# 使用 ReST-KV 对 LLaMA-2-7B 进行 KV cache 驱逐
# 评估: PiQA, ARC-e, ARC-c, BoolQ, HellaSwag, Winogrande, OBQA
# ============================================================

set -e

# ===== 配置 =====
MODEL_PATH="${MODEL_PATH:-/PATH/TO/Llama-2-7b}"
METHOD="${METHOD:-restkv}"
MAX_CAPACITY="${MAX_CAPACITY:-128}"

# ===== 获取项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  yj-proj: ReST-KV KV Cache 压缩 Pipeline"
echo "  模型: ${MODEL_PATH}"
echo "  方法: ${METHOD}"
echo "  容量: ${MAX_CAPACITY}"
echo "============================================"

cd "$PROJECT_ROOT"

python pipelines/run_kv_cache.py \
    --model_path "$MODEL_PATH" \
    --method "$METHOD" \
    --max_capacity_prompts "$MAX_CAPACITY" \
    --use_wo \
    --use_ema \
    --alpha 0.3 \
    --metric_mode after \
    --pooling adaptive \
    --scale 2000

echo "Done!"
