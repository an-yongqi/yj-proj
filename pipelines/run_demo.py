"""
Demo 生成对比: Baseline vs 优化后模型

在多个场景下对比两个模型的生成结果，用于直观展示优化前后的质量差异。

包含 5 类 Demo 场景:
1. 知识问答 (Knowledge QA)
2. 常识推理 (Commonsense Reasoning)
3. 文本续写 (Text Completion)
4. 指令跟随 (Instruction Following)
5. 数学推理 (Math Reasoning)

输出:
- 终端打印对比结果
- 保存 JSON 到 outputs/demo/demo_results.json
- 保存 Markdown 到 outputs/demo/demo_results.md

用法:
    python pipelines/run_demo.py \
        --baseline /path/to/Llama-2-7b \
        --optimized /path/to/pruned-quantized-model \
        --optimized_name "Prune20%+W2A8"
"""

import os
import sys
import json
import time
import argparse
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from unified.pruned_model_loader import load_pruned_model

# Demo 场景定义
DEMO_PROMPTS = [
    {
        "category": "知识问答",
        "id": "qa_1",
        "prompt": "Q: What is the capital of France?\nA:",
        "max_new_tokens": 30,
        "description": "简单事实性问答",
    },
    {
        "category": "知识问答",
        "id": "qa_2",
        "prompt": "Q: Explain what photosynthesis is in one sentence.\nA:",
        "max_new_tokens": 60,
        "description": "科学概念解释",
    },
    {
        "category": "常识推理",
        "id": "reason_1",
        "prompt": "If it is raining outside, you should bring",
        "max_new_tokens": 30,
        "description": "日常常识补全",
    },
    {
        "category": "常识推理",
        "id": "reason_2",
        "prompt": "The trophy doesn't fit in the suitcase because the",
        "max_new_tokens": 30,
        "description": "Winograd 式代词消解",
    },
    {
        "category": "文本续写",
        "id": "complete_1",
        "prompt": "In a distant future, humanity has colonized Mars. The first generation born on the red planet",
        "max_new_tokens": 100,
        "description": "科幻故事续写",
    },
    {
        "category": "文本续写",
        "id": "complete_2",
        "prompt": "The following is a recipe for chocolate cake:\n1.",
        "max_new_tokens": 120,
        "description": "菜谱续写",
    },
    {
        "category": "指令跟随",
        "id": "instruct_1",
        "prompt": "List three benefits of regular exercise:\n1.",
        "max_new_tokens": 80,
        "description": "列举式指令",
    },
    {
        "category": "数学推理",
        "id": "math_1",
        "prompt": "Q: If a train travels at 60 miles per hour for 2.5 hours, how far does it travel?\nA: Let me calculate step by step.",
        "max_new_tokens": 60,
        "description": "简单数学计算",
    },
    {
        "category": "数学推理",
        "id": "math_2",
        "prompt": "Q: A store sells apples for $2 each and oranges for $3 each. If you buy 4 apples and 3 oranges, how much do you pay?\nA:",
        "max_new_tokens": 60,
        "description": "应用题",
    },
    {
        "category": "知识问答",
        "id": "qa_3",
        "prompt": "Q: Who wrote the novel '1984'?\nA:",
        "max_new_tokens": 30,
        "description": "文学知识",
    },
]


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0):
    """生成文本并测量耗时"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start_time = time.time()
    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.time() - start_time

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_tokens = len(generated_ids)
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

    return {
        "text": generated_text.strip(),
        "num_tokens": num_tokens,
        "time_sec": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


def run_demo_on_model(model_path, label):
    """在一个模型上运行所有 demo 场景"""
    print(f"\n{'='*60}")
    print(f"  加载模型: {label}")
    print(f"  路径: {model_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load_pruned_model(model_path)

    results = []
    for i, demo in enumerate(DEMO_PROMPTS):
        print(f"  [{i+1}/{len(DEMO_PROMPTS)}] {demo['category']} - {demo['description']}")
        output = generate_text(model, tokenizer, demo["prompt"], demo["max_new_tokens"])
        print(f"    Prompt: {demo['prompt'][:60]}...")
        print(f"    Output: {output['text'][:80]}...")
        print(f"    ({output['num_tokens']} tokens, {output['time_sec']}s, {output['tokens_per_sec']} tok/s)")
        print()

        results.append({
            "id": demo["id"],
            "category": demo["category"],
            "description": demo["description"],
            "prompt": demo["prompt"],
            "output": output,
        })

    del model
    torch.cuda.empty_cache()

    return results


def print_side_by_side(baseline_results, optimized_results, baseline_name, optimized_name):
    """终端打印对比"""
    print("\n" + "=" * 80)
    print("  Demo 生成对比")
    print("=" * 80)

    for b, o in zip(baseline_results, optimized_results):
        print(f"\n{'─'*80}")
        print(f"  [{b['category']}] {b['description']}")
        print(f"  Prompt: {b['prompt']}")
        print(f"{'─'*80}")
        print(f"  {baseline_name}:")
        print(f"    {b['output']['text']}")
        print(f"    ⏱ {b['output']['time_sec']}s | {b['output']['tokens_per_sec']} tok/s")
        print()
        print(f"  {optimized_name}:")
        print(f"    {o['output']['text']}")
        print(f"    ⏱ {o['output']['time_sec']}s | {o['output']['tokens_per_sec']} tok/s")

    # 速度汇总
    print(f"\n{'='*80}")
    print("  生成速度汇总")
    print(f"{'='*80}")
    b_avg_speed = sum(r["output"]["tokens_per_sec"] for r in baseline_results) / len(baseline_results)
    o_avg_speed = sum(r["output"]["tokens_per_sec"] for r in optimized_results) / len(optimized_results)
    speedup = o_avg_speed / b_avg_speed if b_avg_speed > 0 else 0
    print(f"  {baseline_name} 平均: {b_avg_speed:.1f} tok/s")
    print(f"  {optimized_name} 平均: {o_avg_speed:.1f} tok/s")
    print(f"  加速比: {speedup:.2f}x")
    print(f"{'='*80}\n")


def save_demo_results(baseline_results, optimized_results, baseline_name, optimized_name, save_dir):
    """保存 JSON 和 Markdown"""
    os.makedirs(save_dir, exist_ok=True)

    # JSON
    data = {
        "timestamp": datetime.now().isoformat(),
        "baseline_name": baseline_name,
        "optimized_name": optimized_name,
        "demos": [],
    }
    for b, o in zip(baseline_results, optimized_results):
        data["demos"].append({
            "id": b["id"],
            "category": b["category"],
            "description": b["description"],
            "prompt": b["prompt"],
            "baseline_output": b["output"],
            "optimized_output": o["output"],
        })

    json_path = os.path.join(save_dir, "demo_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存: {json_path}")

    # Markdown
    md_lines = [
        f"# Demo 生成对比: {baseline_name} vs {optimized_name}\n",
        f"生成时间: {data['timestamp']}\n",
    ]

    # 速度汇总表
    b_avg_speed = sum(r["output"]["tokens_per_sec"] for r in baseline_results) / len(baseline_results)
    o_avg_speed = sum(r["output"]["tokens_per_sec"] for r in optimized_results) / len(optimized_results)
    speedup = o_avg_speed / b_avg_speed if b_avg_speed > 0 else 0

    md_lines.append("## 生成速度汇总\n")
    md_lines.append(f"| Model | Avg Speed (tok/s) | Speedup |")
    md_lines.append(f"|-------|------------------:|--------:|")
    md_lines.append(f"| {baseline_name} | {b_avg_speed:.1f} | 1.00x |")
    md_lines.append(f"| {optimized_name} | {o_avg_speed:.1f} | {speedup:.2f}x |")
    md_lines.append("")

    # 逐条对比
    md_lines.append("## 逐条对比\n")

    current_category = None
    for b, o in zip(baseline_results, optimized_results):
        if b["category"] != current_category:
            current_category = b["category"]
            md_lines.append(f"### {current_category}\n")

        md_lines.append(f"**{b['description']}**\n")
        md_lines.append(f"**Prompt:**")
        md_lines.append(f"```")
        md_lines.append(b["prompt"])
        md_lines.append(f"```\n")

        md_lines.append(f"| | {baseline_name} | {optimized_name} |")
        md_lines.append(f"|---|---|---|")

        # 截断过长输出用于表格
        b_text = b["output"]["text"].replace("\n", " ").replace("|", "\\|")
        o_text = o["output"]["text"].replace("\n", " ").replace("|", "\\|")
        md_lines.append(f"| Output | {b_text} | {o_text} |")
        md_lines.append(f"| Speed | {b['output']['tokens_per_sec']} tok/s | {o['output']['tokens_per_sec']} tok/s |")
        md_lines.append(f"| Tokens | {b['output']['num_tokens']} | {o['output']['num_tokens']} |")
        md_lines.append("")

    md_path = os.path.join(save_dir, "demo_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Markdown 已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo 生成对比")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline 模型路径")
    parser.add_argument("--optimized", type=str, required=True, help="优化后模型路径")
    parser.add_argument("--baseline_name", type=str, default="LLaMA-2-7B")
    parser.add_argument("--optimized_name", type=str, default="Prune20%+W2A8")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "demo"))
    args = parser.parse_args()

    baseline_results = run_demo_on_model(args.baseline, args.baseline_name)
    optimized_results = run_demo_on_model(args.optimized, args.optimized_name)

    print_side_by_side(baseline_results, optimized_results, args.baseline_name, args.optimized_name)
    save_demo_results(baseline_results, optimized_results, args.baseline_name, args.optimized_name, args.save_dir)


if __name__ == "__main__":
    main()
