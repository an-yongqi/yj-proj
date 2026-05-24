"""
Pipeline 1: W2A8 量化
使用 ABQ-LLM 对 LLaMA-2-7B 进行 W2A8 量化并评估 7 个下游任务

用法:
    python pipelines/run_quantize.py \
        --model /path/to/Llama-2-7b \
        --save_dir outputs/quantized_models/Llama-2-7b-w2a8
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABQ_DIR = os.path.join(PROJECT_ROOT, "third_party", "ABQ-LLM", "algorithm")

TASKS = "piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,openbookqa"


def run_command(cmd, cwd=None):
    """执行命令并实时输出"""
    print(f"\n>>> {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        print(f"命令执行失败，返回码: {proc.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="W2A8 量化 Pipeline")
    parser.add_argument("--model", type=str, required=True, help="LLaMA-2-7B 模型路径")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "quantized_models", "Llama-2-7b-w2a8"))
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--net", type=str, default=None, help="显式指定网络名称")
    parser.add_argument("--tasks", type=str, default=TASKS)
    parser.add_argument("--skip_act_scales", action="store_true", help="跳过激活统计生成")
    args = parser.parse_args()

    # 确定 net 名称（两个步骤必须一致）
    net_name = args.net
    if net_name is None:
        # 从模型路径推断，与 main.py 的模糊匹配逻辑保持一致
        net_name = args.model.rstrip('/').split('/')[-1]

    # Step 1: 生成激活统计
    if not args.skip_act_scales:
        print("=" * 60)
        print("  Step 1: 生成激活统计 (act_scales / act_shifts)")
        print("=" * 60)
        cmd = [
            sys.executable, "generate_act_scale_shift.py",
            "--model", args.model,
            "--num-samples", str(args.nsamples),
            "--net", net_name,
        ]
        run_command(cmd, cwd=ABQ_DIR)

    # Step 2: W2A8 量化 + 评估
    print("=" * 60)
    print(f"  Step 2: W{args.wbits}A{args.abits} 量化 + 评估")
    print("=" * 60)
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "quantized_models", "log")
    cmd = [
        sys.executable, "main.py",
        "--model", args.model,
        "--eval_ppl",
        "--epochs", str(args.epochs),
        "--output_dir", output_dir,
        "--wbits", str(args.wbits),
        "--abits", str(args.abits),
        "--lwc", "--let",
        "--save_dir", args.save_dir,
        "--tasks", args.tasks,
        "--net", net_name,
    ]
    run_command(cmd, cwd=ABQ_DIR)

    print("\n量化完成！模型保存至:", args.save_dir)


if __name__ == "__main__":
    main()
