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
    parser.add_argument("--batch_size", type=int, default=4, help="量化训练 batch size (默认4, 增大可加速)")
    parser.add_argument("--net", type=str, default=None, help="显式指定网络名称")
    parser.add_argument("--tasks", type=str, default=TASKS)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="量化参数输出目录 (默认: outputs/quantized_models/log)")
    parser.add_argument("--skip_act_scales", action="store_true", help="跳过激活统计生成")
    args = parser.parse_args()

    # 确定 act_scales 文件名（从模型路径推断）
    act_scales_name = args.model.rstrip('/').split('/')[-1]
    # 确定 net 名称（ABQ-LLM 要求从固定列表选，用于模型架构识别）
    if args.net:
        net_name = args.net
    else:
        sys.path.insert(0, PROJECT_ROOT)
        from unified.model_utils import detect_net_name
        net_name = detect_net_name(args.model)
        print(f"自动检测 net 名称: {net_name}")
    # act_scales/act_shifts 文件路径
    act_scales_path = os.path.join(ABQ_DIR, "act_scales", f"{act_scales_name}.pt")
    act_shifts_path = os.path.join(ABQ_DIR, "act_shifts", f"{act_scales_name}.pt")

    # Step 1: 生成激活统计
    if not args.skip_act_scales:
        print("=" * 60)
        print("  Step 1: 生成激活统计 (act_scales / act_shifts)")
        print("=" * 60)
        cmd = [
            sys.executable, "generate_act_scale_shift.py",
            "--model", args.model,
            "--num-samples", str(args.nsamples),
            "--net", act_scales_name,
        ]
        run_command(cmd, cwd=ABQ_DIR)

    # Step 2: W2A8 量化 + 评估
    print("=" * 60)
    print(f"  Step 2: W{args.wbits}A{args.abits} 量化 + 评估")
    print("=" * 60)
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "quantized_models", "log")
    cmd = [
        sys.executable, "main.py",
        "--model", args.model,
        "--eval_ppl",
        "--epochs", str(args.epochs),
        "--output_dir", output_dir,
        "--wbits", str(args.wbits),
        "--abits", str(args.abits),
        "--batch_size", str(args.batch_size),
        "--lwc", "--let",
        "--save_dir", args.save_dir,
        "--tasks", args.tasks,
        "--net", net_name,
        "--act-scales", act_scales_path,
        "--act-shifts", act_shifts_path,
    ]
    run_command(cmd, cwd=ABQ_DIR)

    print("\n量化完成！模型保存至:", args.save_dir)


if __name__ == "__main__":
    main()
