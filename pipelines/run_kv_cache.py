"""
Pipeline 4: ReST-KV KV Cache 压缩
对 LLaMA-2-7B 应用 ReST-KV monkeypatch，评估 7 个下游任务

ReST-KV 通过运行时替换注意力前向函数实现 KV cache 驱逐，
不修改模型权重，因此可与任何标准 HuggingFace 模型兼容。

用法:
    python pipelines/run_kv_cache.py \
        --model_path /path/to/Llama-2-7b \
        --method restkv \
        --max_capacity_prompts 128
"""

import os
import sys
import argparse
import torch

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "rest-kv"))

from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_restkv_config(model, args):
    """
    将 ReST-KV 参数写入每层注意力的 config
    参考 rest-kv/run_longbench.py:251-272
    """
    layers = len(model.model.layers)

    window_sizes = [args.window_size] * layers
    max_capacity_prompts = [args.max_capacity_prompts] * layers
    kernel_sizes = [args.kernel_size] * layers
    ratio = [0.0] * layers
    recent_size = [args.recent_size] * layers

    for i in range(layers):
        layer_config = model.model.layers[i].self_attn.config
        layer_config.window_size = window_sizes[i]
        layer_config.max_capacity_prompt = max_capacity_prompts[i]
        layer_config.kernel_size = kernel_sizes[i]
        layer_config.pooling = args.pooling
        layer_config.merge = args.merge
        layer_config.floor = args.floor
        layer_config.ratio = ratio[i]
        layer_config.recent_size = recent_size[i]

        if args.method == "restkv":
            layer_config.use_wo = args.use_wo
            layer_config.use_norm = args.use_norm
            layer_config.use_ema = args.use_ema
            layer_config.use_pyramid = args.use_pyramid
            layer_config.alpha = args.alpha
            layer_config.metric_mode = args.metric_mode
            layer_config.tau = args.tau
            layer_config.scale = args.scale


def main():
    parser = argparse.ArgumentParser(description="ReST-KV KV Cache 压缩 Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--method", type=str, default="restkv",
                        choices=["restkv", "snapkv", "h2o", "streamingllm", "pyramidkv",
                                 "cam", "l2norm", "fullkv"])
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])

    # ReST-KV 参数
    parser.add_argument("--max_capacity_prompts", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--pooling", type=str, default="adaptive")
    parser.add_argument("--merge", type=str, default="None")
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--recent_size", type=int, default=0)

    # ReST-KV 专有参数
    parser.add_argument("--use_wo", action="store_true", default=True)
    parser.add_argument("--use_norm", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--use_pyramid", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--metric_mode", type=str, default="after")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=2000)

    # 评估参数
    parser.add_argument("--tasks", type=str,
                        default="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,openbookqa")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    # Step 1: Monkeypatch 注意力（必须在加载模型之前）
    if args.method != "fullkv":
        print(f"应用 {args.method} monkeypatch...")
        from restkv.monkeypatch import replace_llama
        replace_llama(args.method)

    # Step 2: 加载模型
    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # Step 3: 配置 ReST-KV 参数到每层
    if args.method != "fullkv":
        print("配置 KV cache 参数...")
        setup_restkv_config(model, args)

    # Step 4: 用统一评估模块评估
    print("开始评估下游任务...")
    task_list = args.tasks.split(",")

    from unified.eval_harness import evaluate_zero_shot, format_results

    results = evaluate_zero_shot(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_path,
        task_list=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )

    # 格式化输出
    save_dir = args.save_dir or os.path.join(PROJECT_ROOT, "outputs", "eval_results")
    technique_name = f"kv_cache_{args.method}"
    format_results(results, technique_name, save_dir=save_dir)


if __name__ == "__main__":
    main()
