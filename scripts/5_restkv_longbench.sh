#!/bin/bash
# ============================================================
# ReST-KV LongBench 评估
# 对比 FullKV (无压缩) vs ReST-KV (KV cache 压缩)
# 前置条件: third_party/rest-kv/data/LongBench/ 下有 JSONL 数据
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESTKV_DIR="$PROJECT_ROOT/third_party/rest-kv"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Llama-2-7b-chat-hf}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/outputs/restkv_longbench}"
# sdpa 兼容性最好; 若安装了 flash-attn 可改为 flash_attention_2
ATTN="${ATTN:-sdpa}"
# 4 个短上下文任务 (适合 4K 上下文的 Llama-2)
DATASETS="${DATASETS:-hotpotqa,trec,triviaqa,samsum}"
# 每个数据集最多测试样本数 (快速验证用, 设 0 或不设表示全量)
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"
GPU="${GPU:-0}"

echo "============================================"
echo "  ReST-KV LongBench 评估"
echo "  模型: ${MODEL_PATH}"
echo "  输出: ${SAVE_DIR}"
echo "  任务: ${DATASETS}"
echo "  每任务最多: ${MAX_EXAMPLES} 条"
echo "  Attention: ${ATTN}"
echo "  GPU: ${GPU}"
echo "============================================"

# 检查数据目录
DATA_DIR="$RESTKV_DIR/data/LongBench"
if [ ! -d "$DATA_DIR" ]; then
    echo ""
    echo "错误: LongBench 数据不存在: $DATA_DIR"
    echo ""
    echo "下载方法 (任选其一):"
    echo ""
    echo "  方法1 - huggingface_hub:"
    echo "    pip3 install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple"
    echo "    python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('THUDM/LongBench', repo_type='dataset', local_dir='$DATA_DIR', allow_patterns='data/*.jsonl')\""
    echo ""
    echo "  方法2 - hf-mirror.com 手动下载:"
    echo "    mkdir -p $DATA_DIR"
    echo "    cd $DATA_DIR"
    echo "    for f in hotpotqa trec triviaqa samsum; do"
    echo "      wget https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data/\${f}.jsonl"
    echo "    done"
    echo ""
    exit 1
fi

cd "$RESTKV_DIR"

MAX_EXAMPLES_FLAG=""
if [ "$MAX_EXAMPLES" -gt 0 ] 2>/dev/null; then
    MAX_EXAMPLES_FLAG="--max_num_examples $MAX_EXAMPLES"
fi

# ── Step 1: FullKV Baseline ──
echo ""
echo ">>> [1/2] FullKV Baseline 推理..."
LONGBENCH_DATASETS="$DATASETS" \
LONGBENCH_DATA_DIR="$DATA_DIR" \
python3 run_longbench.py \
    --method FullKV \
    --model_path "$MODEL_PATH" \
    --max_capacity_prompts 128 \
    --window_sizes 32 \
    --attn_implementation "$ATTN" \
    --save_dir "$SAVE_DIR/fullkv" \
    --add_file_name "fullkv" \
    $MAX_EXAMPLES_FLAG

# ── Step 2: ReST-KV ──
echo ""
echo ">>> [2/2] ReST-KV 推理 (capacity=128)..."
LONGBENCH_DATASETS="$DATASETS" \
LONGBENCH_DATA_DIR="$DATA_DIR" \
python3 run_longbench.py \
    --method restkv \
    --model_path "$MODEL_PATH" \
    --max_capacity_prompts 128 \
    --window_sizes 32 \
    --kernel_sizes 21 \
    --attn_implementation "$ATTN" \
    --save_dir "$SAVE_DIR/restkv" \
    --add_file_name "restkv" \
    --use_wo \
    --use_ema \
    --alpha 0.3 \
    --metric_mode after \
    --pooling adaptive \
    --scale 2000 \
    $MAX_EXAMPLES_FLAG

# ── Step 3: 评估 ──
echo ""
echo ">>> 计算评估指标..."
MODEL_NAME_LOWER=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

for label in fullkv restkv; do
    RESULTS_PATH="$SAVE_DIR/$label/${MODEL_NAME_LOWER}_128"
    if [ -d "$RESULTS_PATH" ]; then
        echo "  评估 $label ..."
        python3 eval.py --results_dir "$RESULTS_PATH" 2>&1 || echo "  (部分评估跳过)"
    else
        echo "  警告: 结果目录不存在 $RESULTS_PATH"
    fi
done

echo ""
echo "============================================"
echo "  完成! 结果: $SAVE_DIR/"
echo "============================================"
