"""
Pipeline 6: 全局集成流水线
完整流程: Prune → Quantize → LoRA → KV Cache → Evaluate

LLaMA-2-7B (原始)
    ↓ FANG 20% 剪枝
剪枝后模型 (save_pretrained)
    ↓ ABQ-LLM W2A8 量化
量化后模型 (save_pretrained, fake-quant FP16 权重)
    ↓ LoRA 训练 + Nevergrad 组合
LoRA-merged 模型
    ↓ ReST-KV monkeypatch
最终模型 → evaluate_zero_shot()

用法:
    python pipelines/run_full_pipeline.py \
        --model /path/to/Llama-2-7b \
        --pruning_ratio 0.2 --wbits 2 --abits 8

    # 跳过已完成的步骤
    python pipelines/run_full_pipeline.py \
        --model /path/to/Llama-2-7b \
        --skip_prune --skip_quantize \
        --pruned_quantized_model outputs/pruned_quantized_models/...
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TASKS = ["piqa", "arc_easy", "arc_challenge", "boolq", "hellaswag", "winogrande", "openbookqa"]


def run_command(cmd):
    """执行命令"""
    print(f"\n>>> {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"命令执行失败，返回码: {proc.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="全局集成 Pipeline")
    parser.add_argument("--model", type=str, required=True, help="LLaMA-2-7B 模型路径")

    # 剪枝参数
    parser.add_argument("--pruning_ratio", type=float, default=0.2)
    parser.add_argument("--skip_prune", action="store_true")

    # 量化参数
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--quant_epochs", type=int, default=40)
    parser.add_argument("--skip_quantize", action="store_true")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_epochs", type=int, default=3)
    parser.add_argument("--max_inference_step", type=int, default=40)
    parser.add_argument("--skip_lora", action="store_true")

    # KV Cache 参数
    parser.add_argument("--kv_method", type=str, default="restkv")
    parser.add_argument("--max_capacity_prompts", type=int, default=128)
    parser.add_argument("--skip_kv_cache", action="store_true")

    # 已有模型路径（配合 skip 使用）
    parser.add_argument("--pruned_model", type=str, default=None)
    parser.add_argument("--pruned_quantized_model", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)

    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Mask 模式剪枝: 保持维度不变，直接兼容量化")

    args = parser.parse_args()

    pr_int = int(args.pruning_ratio * 100)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 路径规划
    pruned_dir = args.pruned_model or os.path.join(
        PROJECT_ROOT, "outputs", "pruned_models", f"Llama-2-7b-pruned-{pr_int}pct")
    pq_dir = args.pruned_quantized_model or os.path.join(
        PROJECT_ROOT, "outputs", "pruned_quantized_models",
        f"Llama-2-7b-pruned{pr_int}-w{args.wbits}a{args.abits}")
    lora_adapter_dir = os.path.join(PROJECT_ROOT, "outputs", "lora_adapters")
    lora_compose_dir = os.path.join(PROJECT_ROOT, "outputs", "lora_composed")
    lora_model_dir = args.lora_model or os.path.join(lora_compose_dir, "model")

    print("=" * 60)
    print("  yj-proj 全局集成 Pipeline")
    print("=" * 60)
    print(f"  基础模型: {args.model}")
    print(f"  剪枝: {args.pruning_ratio} {'(跳过)' if args.skip_prune else ''}")
    print(f"  量化: W{args.wbits}A{args.abits} {'(跳过)' if args.skip_quantize else ''}")
    print(f"  LoRA: r={args.lora_r} {'(跳过)' if args.skip_lora else ''}")
    print(f"  KV Cache: {args.kv_method} {'(跳过)' if args.skip_kv_cache else ''}")
    print("=" * 60)

    # ==================== Step 1: 剪枝 ====================
    if not args.skip_prune:
        print(f"\n{'='*60}")
        print(f"  Step 1/4: FANG {pr_int}% 结构化剪枝")
        print(f"{'='*60}")
        prune_cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, "pipelines", "run_prune.py"),
            "--model", args.model,
            "--pruning_ratio", str(args.pruning_ratio),
            "--nsamples", str(args.nsamples),
            "--save_model", pruned_dir,
        ]
        if args.use_mask:
            prune_cmd.append("--use_mask")
        run_command(prune_cmd)
    else:
        print(f"\n[跳过] Step 1: 使用已有剪枝模型 {pruned_dir}")

    # ==================== Step 2: 量化 ====================
    if not args.skip_quantize:
        print(f"\n{'='*60}")
        print(f"  Step 2/4: ABQ-LLM W{args.wbits}A{args.abits} 量化")
        print(f"{'='*60}")
        from unified.model_utils import detect_net_name
        net_name = detect_net_name(args.model)
        run_command([
            sys.executable, os.path.join(PROJECT_ROOT, "pipelines", "run_quantize.py"),
            "--model", pruned_dir,
            "--save_dir", pq_dir,
            "--wbits", str(args.wbits),
            "--abits", str(args.abits),
            "--epochs", str(args.quant_epochs),
            "--net", net_name,
        ])
    else:
        print(f"\n[跳过] Step 2: 使用已有量化模型 {pq_dir}")

    # ==================== Step 3: LoRA ====================
    if not args.skip_lora:
        print(f"\n{'='*60}")
        print(f"  Step 3/4: LoRA 训练 + 组合")
        print(f"{'='*60}")
        # 在量化后的模型上训练 LoRA
        run_command([
            sys.executable, os.path.join(PROJECT_ROOT, "pipelines", "run_lora.py"),
            "all",
            "--base_model", pq_dir,
            "--lora_r", str(args.lora_r),
            "--num_epochs", str(args.lora_epochs),
            "--max_inference_step", str(args.max_inference_step),
            "--adapter_dir", lora_adapter_dir,
            "--compose_dir", lora_compose_dir,
        ])
    else:
        print(f"\n[跳过] Step 3: 使用已有 LoRA 模型 {lora_model_dir}")

    # ==================== Step 4: KV Cache + 最终评估 ====================
    if not args.skip_kv_cache:
        print(f"\n{'='*60}")
        print(f"  Step 4/4: ReST-KV + 最终评估")
        print(f"{'='*60}")
        final_model = lora_model_dir
        run_command([
            sys.executable, os.path.join(PROJECT_ROOT, "pipelines", "run_kv_cache.py"),
            "--model_path", final_model,
            "--method", args.kv_method,
            "--max_capacity_prompts", str(args.max_capacity_prompts),
            "--use_wo", "--use_ema",
            "--save_dir", os.path.join(PROJECT_ROOT, "outputs", "eval_results"),
        ])
    else:
        # 仅评估（无 KV cache）
        print(f"\n{'='*60}")
        print(f"  Step 4/4: 最终评估（无 KV cache）")
        print(f"{'='*60}")
        final_model = lora_model_dir

        from unified.model_utils import load_model_and_tokenizer
        from unified.eval_harness import evaluate_zero_shot, format_results

        model, tokenizer = load_model_and_tokenizer(final_model)
        results = evaluate_zero_shot(model, tokenizer, model_name=final_model)
        save_dir = os.path.join(PROJECT_ROOT, "outputs", "eval_results")
        format_results(results, "full_pipeline", save_dir=save_dir)

    print(f"\n{'='*60}")
    print(f"  全局集成 Pipeline 完成！")
    print(f"  结果保存在: outputs/eval_results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
