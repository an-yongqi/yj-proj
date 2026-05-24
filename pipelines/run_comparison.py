"""
Baseline vs 优化后模型 对比评测

评测内容:
1. WikiText-2 Perplexity
2. 7 个 Zero-shot 下游任务

输出:
- 终端打印对比表格
- 保存 JSON 结果到 outputs/eval_results/comparison.json
- 保存 Markdown 表格到 outputs/eval_results/comparison.md

用法:
    python pipelines/run_comparison.py \
        --baseline /path/to/Llama-2-7b \
        --optimized /path/to/pruned-quantized-model \
        --optimized_name "Prune20%+W2A8"
"""

import os
import sys
import json
import argparse
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from unified.model_utils import load_model_and_tokenizer
from unified.pruned_model_loader import load_pruned_model
from unified.eval_harness import evaluate_ppl

TASK_LIST = ["piqa", "arc_easy", "arc_challenge", "boolq", "hellaswag", "winogrande", "openbookqa"]

# 任务显示名映射
TASK_DISPLAY = {
    "piqa": "PiQA",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "winogrande": "Winogrande",
    "openbookqa": "OBQA",
}


def eval_zero_shot_abq(model, tokenizer, model_name, batch_size=4):
    """使用 ABQ-LLM 内置 lm_eval 进行 zero-shot 评估"""
    abq_path = os.path.join(PROJECT_ROOT, "third_party", "ABQ-LLM", "algorithm")
    # 清理已加载的 lm_eval
    for key in list(sys.modules.keys()):
        if key.startswith("lm_eval"):
            del sys.modules[key]
    sys.path.insert(0, abq_path)

    from lm_eval import evaluator
    from lm_eval.models.huggingface import AutoCausalLM
    from lm_eval.base import CacheHook

    lm = AutoCausalLM.__new__(AutoCausalLM)
    lm.cache_hook = CacheHook(None)
    lm.model = model
    lm.tokenizer = tokenizer
    lm.tokenizer.padding_side = "left"
    lm._batch_size = batch_size
    lm._max_length = 2048
    lm._max_gen_toks = 256
    lm._add_special_tokens = False
    lm._config = model.config
    lm._device = next(model.parameters()).device
    lm.vocab_size = model.config.vocab_size

    results = evaluator.simple_evaluate(
        lm=lm,
        tasks=",".join(TASK_LIST),
        num_fewshot=0,
        limit=None,
    )

    sys.path.remove(abq_path)

    # 提取 acc
    task_scores = {}
    for task in TASK_LIST:
        if task in results["results"]:
            metrics = results["results"][task]
            if "acc_norm" in metrics:
                task_scores[task] = round(metrics["acc_norm"] * 100, 2)
            elif "acc" in metrics:
                task_scores[task] = round(metrics["acc"] * 100, 2)
    return task_scores


def evaluate_model(model_path, label, batch_size=4):
    """加载并评测一个模型，返回 ppl + zero-shot 结果"""
    print(f"\n{'='*60}")
    print(f"  正在评测: {label}")
    print(f"  模型路径: {model_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load_pruned_model(model_path)

    # PPL
    print(">>> 评测 WikiText-2 Perplexity...")
    ppl = evaluate_ppl(model, tokenizer, dataset="wikitext2")
    print(f"    PPL = {ppl:.2f}")

    # Zero-shot
    print(">>> 评测 7 个 Zero-shot 任务...")
    task_scores = eval_zero_shot_abq(model, tokenizer, model_path, batch_size)
    for task, score in task_scores.items():
        print(f"    {TASK_DISPLAY.get(task, task):15s}: {score:.2f}%")

    avg = round(sum(task_scores.values()) / len(task_scores), 2) if task_scores else 0
    print(f"    {'Average':15s}: {avg:.2f}%")

    # 释放显存
    del model
    torch.cuda.empty_cache()

    return {"ppl": round(ppl, 2), "tasks": task_scores, "average": avg}


def print_comparison_table(baseline_results, optimized_results, baseline_name, optimized_name):
    """打印对比表格"""
    print("\n")
    print("=" * 75)
    print("  Baseline vs Optimized 对比表格")
    print("=" * 75)

    header = f"{'Metric':<18} | {baseline_name:>15} | {optimized_name:>15} | {'Δ':>10}"
    print(header)
    print("-" * 75)

    # PPL (越低越好)
    b_ppl = baseline_results["ppl"]
    o_ppl = optimized_results["ppl"]
    delta_ppl = o_ppl - b_ppl
    sign = "+" if delta_ppl > 0 else ""
    print(f"{'PPL ↓':<18} | {b_ppl:>15.2f} | {o_ppl:>15.2f} | {sign}{delta_ppl:>9.2f}")
    print("-" * 75)

    # Zero-shot tasks (越高越好)
    for task in TASK_LIST:
        display = TASK_DISPLAY.get(task, task)
        b_score = baseline_results["tasks"].get(task, 0)
        o_score = optimized_results["tasks"].get(task, 0)
        delta = o_score - b_score
        sign = "+" if delta > 0 else ""
        print(f"{display:<18} | {b_score:>14.2f}% | {o_score:>14.2f}% | {sign}{delta:>8.2f}%")

    print("-" * 75)
    b_avg = baseline_results["average"]
    o_avg = optimized_results["average"]
    delta_avg = o_avg - b_avg
    sign = "+" if delta_avg > 0 else ""
    print(f"{'Average':<18} | {b_avg:>14.2f}% | {o_avg:>14.2f}% | {sign}{delta_avg:>8.2f}%")
    print("=" * 75)


def save_results(baseline_results, optimized_results, baseline_name, optimized_name, save_dir):
    """保存 JSON 和 Markdown"""
    os.makedirs(save_dir, exist_ok=True)

    # JSON
    data = {
        "baseline": {"name": baseline_name, **baseline_results},
        "optimized": {"name": optimized_name, **optimized_results},
    }
    json_path = os.path.join(save_dir, "comparison.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON 结果已保存: {json_path}")

    # Markdown
    md_lines = [
        f"# {baseline_name} vs {optimized_name} 评测对比\n",
        f"| Metric | {baseline_name} | {optimized_name} | Δ |",
        "|--------|------:|------:|------:|",
    ]

    b_ppl = baseline_results["ppl"]
    o_ppl = optimized_results["ppl"]
    delta_ppl = o_ppl - b_ppl
    sign = "+" if delta_ppl > 0 else ""
    md_lines.append(f"| PPL ↓ | {b_ppl:.2f} | {o_ppl:.2f} | {sign}{delta_ppl:.2f} |")

    for task in TASK_LIST:
        display = TASK_DISPLAY.get(task, task)
        b_s = baseline_results["tasks"].get(task, 0)
        o_s = optimized_results["tasks"].get(task, 0)
        d = o_s - b_s
        sign = "+" if d > 0 else ""
        md_lines.append(f"| {display} | {b_s:.2f}% | {o_s:.2f}% | {sign}{d:.2f}% |")

    b_avg = baseline_results["average"]
    o_avg = optimized_results["average"]
    d_avg = o_avg - b_avg
    sign = "+" if d_avg > 0 else ""
    md_lines.append(f"| **Average** | **{b_avg:.2f}%** | **{o_avg:.2f}%** | **{sign}{d_avg:.2f}%** |")

    md_path = os.path.join(save_dir, "comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Markdown 表格已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Optimized 模型对比评测")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline 模型路径")
    parser.add_argument("--optimized", type=str, required=True, help="优化后模型路径")
    parser.add_argument("--baseline_name", type=str, default="LLaMA-2-7B")
    parser.add_argument("--optimized_name", type=str, default="Prune20%+W2A8")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "eval_results"))
    parser.add_argument("--skip_baseline", action="store_true", help="跳过 baseline 评测（从 JSON 加载）")
    parser.add_argument("--skip_optimized", action="store_true", help="跳过 optimized 评测（从 JSON 加载）")
    args = parser.parse_args()

    json_path = os.path.join(args.save_dir, "comparison.json")
    cached = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            cached = json.load(f)

    if args.skip_baseline and "baseline" in cached:
        print(">>> 从缓存加载 Baseline 结果")
        baseline_results = {k: v for k, v in cached["baseline"].items() if k != "name"}
    else:
        baseline_results = evaluate_model(args.baseline, args.baseline_name, args.batch_size)

    if args.skip_optimized and "optimized" in cached:
        print(">>> 从缓存加载 Optimized 结果")
        optimized_results = {k: v for k, v in cached["optimized"].items() if k != "name"}
    else:
        optimized_results = evaluate_model(args.optimized, args.optimized_name, args.batch_size)

    print_comparison_table(baseline_results, optimized_results, args.baseline_name, args.optimized_name)
    save_results(baseline_results, optimized_results, args.baseline_name, args.optimized_name, args.save_dir)


if __name__ == "__main__":
    main()
