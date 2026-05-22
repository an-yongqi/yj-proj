"""
Pipeline 5: LoRA 训练 + 组合 + 评估
三步走:
1. train: 在 7 个任务上分别训练 LLaMA-2-7B 的 LoRA adapter
2. compose: 用 Nevergrad 无梯度优化组合多个 LoRA 权重
3. evaluate: 评估组合后的模型

用法:
    # 训练所有任务的 LoRA
    python pipelines/run_lora.py train --base_model /path/to/Llama-2-7b

    # 组合 LoRA 模块
    python pipelines/run_lora.py compose --base_model /path/to/Llama-2-7b

    # 评估组合模型
    python pipelines/run_lora.py evaluate --model outputs/lora_composed/model

    # 全流程
    python pipelines/run_lora.py all --base_model /path/to/Llama-2-7b
"""

import os
import sys
import argparse
import subprocess

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


def cmd_train(args):
    """Step 1: 训练 per-task LoRA adapters"""
    tasks = args.tasks.split(",") if args.tasks else TASKS
    adapter_dir = args.adapter_dir or os.path.join(PROJECT_ROOT, "outputs", "lora_adapters")

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  训练 LoRA adapter: {task}")
        print(f"{'='*60}")
        output = os.path.join(adapter_dir, task)

        if os.path.exists(os.path.join(output, "adapter_config.json")) and not args.force:
            print(f"  已存在，跳过 (使用 --force 强制重新训练)")
            continue

        run_command([
            sys.executable, "-m", "unified.lora_train",
            "--base_model", args.base_model,
            "--task", task,
            "--output_dir", output,
            "--lora_r", str(args.lora_r),
            "--num_epochs", str(args.num_epochs),
            "--batch_size", str(args.batch_size),
            "--max_samples", str(args.max_samples),
        ])

    print(f"\n所有 LoRA adapter 已保存至: {adapter_dir}")


def cmd_compose(args):
    """Step 2: Nevergrad 组合 LoRA 模块"""
    adapter_dir = args.adapter_dir or os.path.join(PROJECT_ROOT, "outputs", "lora_adapters")
    compose_dir = args.compose_dir or os.path.join(PROJECT_ROOT, "outputs", "lora_composed")
    os.makedirs(compose_dir, exist_ok=True)

    # 收集可用的 LoRA 模块
    tasks = args.tasks.split(",") if args.tasks else TASKS
    lora_modules = []
    for task in tasks:
        adapter_path = os.path.join(adapter_dir, task)
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            lora_modules.append(adapter_path)
        else:
            print(f"  警告: {task} 的 LoRA adapter 不存在，跳过")

    if len(lora_modules) < 2:
        print("错误: 至少需要 2 个 LoRA 模块才能进行组合")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  组合 {len(lora_modules)} 个 LoRA 模块")
    print(f"{'='*60}")

    # 准备少量样本用于优化
    from unified.lora_causal import lorahub_learning
    import json

    # 使用 BoolQ 的少量样本作为优化目标
    from datasets import load_dataset
    eval_dataset = load_dataset("boolq", split="validation")
    example_inputs = []
    example_outputs = []
    for i in range(min(5, len(eval_dataset))):
        ex = eval_dataset[i]
        example_inputs.append(f"Passage: {ex['passage'][:300]}\nQuestion: {ex['question']}\nAnswer:")
        example_outputs.append("Yes" if ex["answer"] else "No")

    # 运行 Nevergrad 优化
    weights, model, tokenizer = lorahub_learning(
        lora_module_list=lora_modules,
        example_inputs=example_inputs,
        example_outputs=example_outputs,
        max_inference_step=args.max_inference_step,
        model_name_or_path=args.base_model,
        batch_size=1,
        seed=args.seed,
    )

    if model is not None:
        # 保存组合后的模型
        model_save_path = os.path.join(compose_dir, "model")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        # 保存权重
        import json
        weights_info = {
            "modules": lora_modules,
            "weights": weights.tolist() if hasattr(weights, 'tolist') else list(weights),
        }
        with open(os.path.join(compose_dir, "composition_weights.json"), "w") as f:
            json.dump(weights_info, f, indent=2)

        print(f"\n组合模型已保存至: {model_save_path}")
        print(f"组合权重: {weights}")


def cmd_evaluate(args):
    """Step 3: 评估组合后的模型"""
    compose_dir = args.compose_dir or os.path.join(PROJECT_ROOT, "outputs", "lora_composed")
    model_path = args.model or os.path.join(compose_dir, "model")

    print(f"\n{'='*60}")
    print(f"  评估组合模型: {model_path}")
    print(f"{'='*60}")

    from unified.model_utils import load_model_and_tokenizer
    from unified.eval_harness import evaluate_zero_shot, format_results

    model, tokenizer = load_model_and_tokenizer(model_path)
    tasks = args.tasks.split(",") if args.tasks else TASKS

    results = evaluate_zero_shot(
        model=model,
        tokenizer=tokenizer,
        model_name=model_path,
        task_list=tasks,
    )

    save_dir = os.path.join(PROJECT_ROOT, "outputs", "eval_results")
    format_results(results, "lora_composed", save_dir=save_dir)


def cmd_all(args):
    """全流程: train → compose → evaluate"""
    cmd_train(args)
    cmd_compose(args)
    cmd_evaluate(args)


def main():
    parser = argparse.ArgumentParser(description="LoRA Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 共有参数
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base_model", type=str, default=None, help="基础模型路径")
    common.add_argument("--tasks", type=str, default=None, help="任务列表（逗号分隔）")
    common.add_argument("--adapter_dir", type=str, default=None)
    common.add_argument("--compose_dir", type=str, default=None)
    common.add_argument("--seed", type=int, default=42)

    # train 子命令
    train_parser = subparsers.add_parser("train", parents=[common], help="训练 per-task LoRA")
    train_parser.add_argument("--lora_r", type=int, default=16)
    train_parser.add_argument("--num_epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--max_samples", type=int, default=5000)
    train_parser.add_argument("--force", action="store_true", help="强制重新训练")

    # compose 子命令
    compose_parser = subparsers.add_parser("compose", parents=[common], help="组合 LoRA 模块")
    compose_parser.add_argument("--max_inference_step", type=int, default=40)

    # evaluate 子命令
    eval_parser = subparsers.add_parser("evaluate", parents=[common], help="评估组合模型")
    eval_parser.add_argument("--model", type=str, default=None, help="模型路径")

    # all 子命令
    all_parser = subparsers.add_parser("all", parents=[common], help="全流程")
    all_parser.add_argument("--lora_r", type=int, default=16)
    all_parser.add_argument("--num_epochs", type=int, default=3)
    all_parser.add_argument("--batch_size", type=int, default=4)
    all_parser.add_argument("--max_samples", type=int, default=5000)
    all_parser.add_argument("--max_inference_step", type=int, default=40)
    all_parser.add_argument("--force", action="store_true")
    all_parser.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        if not args.base_model:
            print("错误: --base_model 是必需参数")
            sys.exit(1)
        cmd_train(args)
    elif args.command == "compose":
        if not args.base_model:
            print("错误: --base_model 是必需参数")
            sys.exit(1)
        cmd_compose(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "all":
        if not args.base_model:
            print("错误: --base_model 是必需参数")
            sys.exit(1)
        cmd_all(args)


if __name__ == "__main__":
    main()
