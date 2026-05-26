#!/bin/bash
# ============================================================
# 下载 LongBench 数据到 third_party/rest-kv/data/LongBench/
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/third_party/rest-kv/data/LongBench"

# 默认只下载 4 个短任务; 传参 "all" 下载全部 16 个
if [ "$1" = "all" ]; then
    TASKS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
else
    TASKS="${1:-hotpotqa trec triviaqa samsum}"
fi

echo "下载 LongBench 数据到: $DATA_DIR"
echo "任务: $TASKS"
echo ""

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 尝试方法1: huggingface_hub
if python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "使用 huggingface_hub 下载..."
    for task in $TASKS; do
        echo "  下载 ${task}.jsonl ..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('THUDM/LongBench', filename='data/${task}.jsonl',
    repo_type='dataset', local_dir='$DATA_DIR/..')
" 2>/dev/null && echo "    OK" || echo "    失败, 尝试 wget..."
    done
    # huggingface_hub 可能下载到 data/ 子目录
    if [ -d "$DATA_DIR/../data" ] && [ ! -f "$DATA_DIR/hotpotqa.jsonl" ]; then
        mv "$DATA_DIR/../data/"*.jsonl "$DATA_DIR/" 2>/dev/null || true
    fi
else
    echo "huggingface_hub 未安装, 使用 wget 下载..."
    echo "(如需安装: pip3 install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple)"
    echo ""
    for task in $TASKS; do
        if [ -f "${task}.jsonl" ]; then
            echo "  ${task}.jsonl 已存在, 跳过"
            continue
        fi
        echo "  下载 ${task}.jsonl ..."
        # 尝试多个 URL 格式
        wget -q "https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data/${task}.jsonl" -O "${task}.jsonl" 2>/dev/null \
        || wget -q "https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/${task}.jsonl" -O "${task}.jsonl" 2>/dev/null \
        || curl -sL "https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data/${task}.jsonl" -o "${task}.jsonl" 2>/dev/null \
        || { echo "    下载失败: ${task}"; rm -f "${task}.jsonl"; }
        [ -f "${task}.jsonl" ] && echo "    OK"
    done
fi

echo ""
echo "下载完成! 文件列表:"
ls -lh "$DATA_DIR/"*.jsonl 2>/dev/null || echo "  (无文件)"
