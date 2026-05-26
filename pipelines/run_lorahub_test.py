"""
LoraHub 快速测试: 验证多个 LoRA 权重能否根据输入自动选择最优组合

测试流程:
1. 加载基础模型 + 多个 LoRA adapter
2. 对不同类型的输入，用 Nevergrad 优化组合权重
3. 比较不同输入下的权重分布是否不同

用法:
    python pipelines/run_lorahub_test.py \
        --base_model models/Llama-2-7b-chat-hf \
        --adapter_dir outputs/lora_adapters \
        --max_inference_step 20
"""

import os
import sys
import argparse
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# 两组不同类型的测试样本，期望产生不同的 LoRA 组合权重
TEST_SCENARIOS = {
    "math": {
        "inputs": [
            "What is 15 + 27? Answer:",
            "Calculate 8 * 12. Answer:",
            "What is 100 - 37? Answer:",
            "If x + 5 = 12, what is x? Answer:",
            "What is 144 / 12? Answer:",
        ],
        "outputs": ["42", "96", "63", "7", "12"],
    },
    "qa": {
        "inputs": [
            "What is the capital of France? Answer:",
            "Who wrote Romeo and Juliet? Answer:",
            "What is the largest planet in our solar system? Answer:",
            "In what year did World War II end? Answer:",
            "What element has the chemical symbol O? Answer:",
        ],
        "outputs": ["Paris", "William Shakespeare", "Jupiter", "1945", "Oxygen"],
    },
}


def main():
    parser = argparse.ArgumentParser(description="LoraHub 组合测试")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--max_inference_step", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adapter_dir = args.adapter_dir or os.path.join(PROJECT_ROOT, "outputs", "lora_adapters")

    # 收集可用的 LoRA 模块
    lora_modules = []
    for name in sorted(os.listdir(adapter_dir)):
        adapter_path = os.path.join(adapter_dir, name)
        if os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
            lora_modules.append(adapter_path)

    if len(lora_modules) < 2:
        print(f"错误: {adapter_dir} 下至少需要 2 个 LoRA adapter")
        print(f"  请先运行: python pipelines/create_dummy_loras.py --base_model {args.base_model}")
        sys.exit(1)

    module_names = [os.path.basename(m) for m in lora_modules]
    print(f"找到 {len(lora_modules)} 个 LoRA 模块: {module_names}")

    from unified.lora_causal import lorahub_learning

    results = {}
    for scenario_name, scenario in TEST_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  场景: {scenario_name}")
        print(f"  样本: {scenario['inputs'][0][:50]}...")
        print(f"{'='*60}")

        weights, model, tokenizer = lorahub_learning(
            lora_module_list=lora_modules,
            example_inputs=scenario["inputs"],
            example_outputs=scenario["outputs"],
            max_inference_step=args.max_inference_step,
            model_name_or_path=args.base_model,
            batch_size=len(scenario["inputs"]),
            seed=args.seed,
        )

        if weights is not None:
            weight_dict = {name: float(w) for name, w in zip(module_names, weights)}
            results[scenario_name] = weight_dict
            print(f"\n  组合权重:")
            for name, w in weight_dict.items():
                bar = "#" * int(abs(w) * 20)
                sign = "+" if w >= 0 else "-"
                print(f"    {name:>12s}: {w:+.4f}  {sign}{bar}")

        # 释放显存
        del model, tokenizer
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 对比
    print(f"\n{'='*60}")
    print("  权重对比 (不同输入 → 不同组合)")
    print(f"{'='*60}")
    if len(results) >= 2:
        scenarios = list(results.keys())
        print(f"\n  {'Module':>12s}  {'':>2s}  ", end="")
        for s in scenarios:
            print(f"{s:>10s}  ", end="")
        print()
        print("  " + "-" * (14 + 12 * len(scenarios)))
        for name in module_names:
            print(f"  {name:>12s}  {'':>2s}  ", end="")
            for s in scenarios:
                w = results[s].get(name, 0)
                print(f"{w:+10.4f}  ", end="")
            print()

        # 计算权重差异
        import numpy as np
        w1 = np.array([results[scenarios[0]][n] for n in module_names])
        w2 = np.array([results[scenarios[1]][n] for n in module_names])
        diff = np.linalg.norm(w1 - w2)
        cosine = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8)
        print(f"\n  权重 L2 差异: {diff:.4f}")
        print(f"  权重余弦相似度: {cosine:.4f}")
        print(f"  结论: {'不同输入产生了不同的 LoRA 组合' if diff > 0.01 else '权重差异较小'}")

    # 保存结果
    save_path = os.path.join(PROJECT_ROOT, "outputs", "lorahub_test_results.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {save_path}")


if __name__ == "__main__":
    main()
