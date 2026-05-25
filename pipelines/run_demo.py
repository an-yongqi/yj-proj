"""
Demo 生成对比: Baseline vs W2A8 量化模型

在多个中文场景下对比两个模型的生成结果，用于直观展示量化前后的质量差异。

用法:
    python pipelines/run_demo.py \
        --baseline /path/to/Llama-2-7b-hf \
        --quantized /path/to/Llama-2-7b-w2a8
"""

import os
import sys
import json
import time
import argparse
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Demo 场景定义（中文）
DEMO_PROMPTS = [
    {
        "category": "知识问答",
        "id": "qa_1",
        "prompt": "问：中国的首都是哪里？\n答：",
        "max_new_tokens": 30,
        "description": "简单事实问答",
    },
    {
        "category": "知识问答",
        "id": "qa_2",
        "prompt": "问：请用一句话解释什么是光合作用。\n答：",
        "max_new_tokens": 80,
        "description": "科学概念解释",
    },
    {
        "category": "知识问答",
        "id": "qa_3",
        "prompt": "问：《红楼梦》的作者是谁？\n答：",
        "max_new_tokens": 30,
        "description": "文学知识",
    },
    {
        "category": "常识推理",
        "id": "reason_1",
        "prompt": "如果外面正在下雨，出门时你应该带上",
        "max_new_tokens": 30,
        "description": "日常常识补全",
    },
    {
        "category": "常识推理",
        "id": "reason_2",
        "prompt": "问：冰箱里的食物放了一个月还能吃吗？为什么？\n答：",
        "max_new_tokens": 80,
        "description": "生活常识推理",
    },
    {
        "category": "文本续写",
        "id": "complete_1",
        "prompt": "在遥远的未来，人类已经在火星上建立了第一个城市。这座城市",
        "max_new_tokens": 100,
        "description": "科幻故事续写",
    },
    {
        "category": "文本续写",
        "id": "complete_2",
        "prompt": "春天来了，万物复苏。田野里",
        "max_new_tokens": 80,
        "description": "散文续写",
    },
    {
        "category": "指令跟随",
        "id": "instruct_1",
        "prompt": "请列举坚持运动的三个好处：\n1.",
        "max_new_tokens": 100,
        "description": "列举式指令",
    },
    {
        "category": "数学推理",
        "id": "math_1",
        "prompt": "问：一列火车以每小时120公里的速度行驶，行驶了2.5小时，总共走了多远？\n答：让我一步步计算。",
        "max_new_tokens": 80,
        "description": "简单数学计算",
    },
    {
        "category": "数学推理",
        "id": "math_2",
        "prompt": "问：商店里苹果每个3元，橘子每个5元。小明买了4个苹果和3个橘子，一共需要多少钱？\n答：",
        "max_new_tokens": 80,
        "description": "应用题",
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


def load_standard_model(model_path):
    """标准 HF 模型加载"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def run_demo_on_model(model_path, label):
    """在一个模型上运行所有 demo 场景"""
    print(f"\n{'='*60}")
    print(f"  加载模型: {label}")
    print(f"  路径: {model_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load_standard_model(model_path)

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


def print_side_by_side(baseline_results, quant_results, baseline_name, quant_name):
    """终端打印对比"""
    print("\n" + "=" * 80)
    print("  Demo 生成对比")
    print("=" * 80)

    for b, q in zip(baseline_results, quant_results):
        print(f"\n{'─'*80}")
        print(f"  [{b['category']}] {b['description']}")
        print(f"  Prompt: {b['prompt']}")
        print(f"{'─'*80}")
        print(f"  {baseline_name}:")
        print(f"    {b['output']['text']}")
        print(f"    {b['output']['time_sec']}s | {b['output']['tokens_per_sec']} tok/s")
        print()
        print(f"  {quant_name}:")
        print(f"    {q['output']['text']}")
        print(f"    {q['output']['time_sec']}s | {q['output']['tokens_per_sec']} tok/s")

    # 速度汇总
    print(f"\n{'='*80}")
    print("  生成速度汇总")
    print(f"{'='*80}")
    b_avg_speed = sum(r["output"]["tokens_per_sec"] for r in baseline_results) / len(baseline_results)
    q_avg_speed = sum(r["output"]["tokens_per_sec"] for r in quant_results) / len(quant_results)
    speedup = q_avg_speed / b_avg_speed if b_avg_speed > 0 else 0
    print(f"  {baseline_name} 平均: {b_avg_speed:.1f} tok/s")
    print(f"  {quant_name} 平均: {q_avg_speed:.1f} tok/s")
    print(f"  加速比: {speedup:.2f}x")
    print(f"{'='*80}\n")


def save_demo_results(baseline_results, quant_results, baseline_name, quant_name, save_dir):
    """保存 JSON 和 Markdown"""
    os.makedirs(save_dir, exist_ok=True)

    # JSON
    data = {
        "timestamp": datetime.now().isoformat(),
        "baseline_name": baseline_name,
        "quantized_name": quant_name,
        "demos": [],
    }
    for b, q in zip(baseline_results, quant_results):
        data["demos"].append({
            "id": b["id"],
            "category": b["category"],
            "description": b["description"],
            "prompt": b["prompt"],
            "baseline_output": b["output"],
            "quantized_output": q["output"],
        })

    json_path = os.path.join(save_dir, "demo_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存: {json_path}")

    # Markdown
    md_lines = [
        f"# Demo 生成对比: {baseline_name} vs {quant_name}\n",
        f"生成时间: {data['timestamp']}\n",
    ]

    # 速度汇总表
    b_avg_speed = sum(r["output"]["tokens_per_sec"] for r in baseline_results) / len(baseline_results)
    q_avg_speed = sum(r["output"]["tokens_per_sec"] for r in quant_results) / len(quant_results)
    speedup = q_avg_speed / b_avg_speed if b_avg_speed > 0 else 0

    md_lines.append("## 生成速度汇总\n")
    md_lines.append(f"| 模型 | 平均速度 (tok/s) | 加速比 |")
    md_lines.append(f"|------|------------------:|--------:|")
    md_lines.append(f"| {baseline_name} | {b_avg_speed:.1f} | 1.00x |")
    md_lines.append(f"| {quant_name} | {q_avg_speed:.1f} | {speedup:.2f}x |")
    md_lines.append("")

    # 逐条对比
    md_lines.append("## 逐条对比\n")

    current_category = None
    for b, q in zip(baseline_results, quant_results):
        if b["category"] != current_category:
            current_category = b["category"]
            md_lines.append(f"### {current_category}\n")

        md_lines.append(f"**{b['description']}**\n")
        md_lines.append(f"**Prompt:**")
        md_lines.append(f"```")
        md_lines.append(b["prompt"])
        md_lines.append(f"```\n")

        # 用独立段落展示，避免表格里换行问题
        md_lines.append(f"**{baseline_name}:**")
        md_lines.append(f"> {b['output']['text']}")
        md_lines.append(f"> *({b['output']['tokens_per_sec']} tok/s, {b['output']['num_tokens']} tokens)*\n")

        md_lines.append(f"**{quant_name}:**")
        md_lines.append(f"> {q['output']['text']}")
        md_lines.append(f"> *({q['output']['tokens_per_sec']} tok/s, {q['output']['num_tokens']} tokens)*\n")
        md_lines.append("---\n")

    md_path = os.path.join(save_dir, "demo_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Markdown 已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo 生成对比: Baseline vs 量化模型")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline 模型路径")
    parser.add_argument("--quantized", type=str, required=True, help="量化后模型路径")
    parser.add_argument("--baseline_name", type=str, default="LLaMA-2-7B (Baseline)")
    parser.add_argument("--quantized_name", type=str, default="LLaMA-2-7B (W2A8)")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "demo"))
    args = parser.parse_args()

    baseline_results = run_demo_on_model(args.baseline, args.baseline_name)
    quant_results = run_demo_on_model(args.quantized, args.quantized_name)

    print_side_by_side(baseline_results, quant_results, args.baseline_name, args.quantized_name)
    save_demo_results(baseline_results, quant_results, args.baseline_name, args.quantized_name, args.save_dir)


if __name__ == "__main__":
    main()
