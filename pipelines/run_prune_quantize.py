"""
Pipeline 3: 剪枝 + 量化
先执行 FANG 20% 剪枝，再对剪枝后模型执行 ABQ-LLM W2A8 量化

用法:
    python pipelines/run_prune_quantize.py \
        --model /path/to/Llama-2-7b \
        --pruning_ratio 0.2 --wbits 2 --abits 8
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd):
    """执行命令"""
    print(f"\n>>> {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"命令执行失败，返回码: {proc.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="剪枝 + 量化 Pipeline")
    parser.add_argument("--model", type=str, required=True, help="LLaMA-2-7B 模型路径")
    parser.add_argument("--pruning_ratio", type=float, default=0.2)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--skip_prune", action="store_true",
                        help="跳过剪枝步骤（使用已有剪枝模型）")
    parser.add_argument("--pruned_model_path", type=str, default=None,
                        help="已有剪枝模型路径（配合 --skip_prune 使用）")
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Mask 模式剪枝: 保持维度不变，直接兼容量化")
    args = parser.parse_args()

    pr_int = int(args.pruning_ratio * 100)
    pruned_dir = os.path.join(PROJECT_ROOT, "outputs", "pruned_models",
                              f"Llama-2-7b-pruned-{pr_int}pct")
    final_dir = os.path.join(PROJECT_ROOT, "outputs", "pruned_quantized_models",
                             f"Llama-2-7b-pruned{pr_int}-w{args.wbits}a{args.abits}")

    # Step 1: 剪枝
    if not args.skip_prune:
        print("=" * 60)
        print("  Step 1/2: FANG 结构化剪枝")
        print("=" * 60)
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
        if args.pruned_model_path:
            pruned_dir = args.pruned_model_path
        print(f"跳过剪枝，使用已有模型: {pruned_dir}")

    # Step 2: 量化剪枝后的模型
    print("=" * 60)
    print(f"  Step 2/2: ABQ-LLM W{args.wbits}A{args.abits} 量化")
    print("=" * 60)
    run_command([
        sys.executable, os.path.join(PROJECT_ROOT, "pipelines", "run_quantize.py"),
        "--model", pruned_dir,
        "--save_dir", final_dir,
        "--wbits", str(args.wbits),
        "--abits", str(args.abits),
        "--epochs", str(args.epochs),
        "--net", "Llama-2-7b",
    ])

    print(f"\n剪枝+量化完成！最终模型保存至: {final_dir}")


if __name__ == "__main__":
    main()
