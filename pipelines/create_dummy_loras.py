"""
创建多个随机 LoRA adapter 用于测试 LoraHub 组合算法
不需要训练数据，直接用随机初始化的 LoRA 权重

用法:
    python pipelines/create_dummy_loras.py \
        --base_model models/Llama-2-7b-chat-hf \
        --num_loras 5 \
        --output_dir outputs/lora_adapters
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_dummy_lora(base_model_path, output_path, lora_r=16, seed=0):
    """创建一个随机初始化的 LoRA adapter"""
    torch.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # 用不同随机种子产生不同的 LoRA 权重
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(param, a=5**0.5)
            elif "lora_B" in name:
                # 不同 seed 给不同的非零初始化，使各 adapter 有区别
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    peft_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"  已创建: {output_path}")

    # 释放显存
    del peft_model, model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    parser = argparse.ArgumentParser(description="创建随机 LoRA adapter")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--num_loras", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "outputs", "lora_adapters")
    task_names = ["math", "reasoning", "qa", "summarize", "translate",
                  "code", "classify", "sentiment"][:args.num_loras]

    print(f"创建 {args.num_loras} 个随机 LoRA adapter ...")
    print(f"基础模型: {args.base_model}")
    print(f"输出目录: {output_dir}")

    for i, task in enumerate(task_names):
        adapter_path = os.path.join(output_dir, task)
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"  已存在，跳过: {adapter_path}")
            continue
        create_dummy_lora(args.base_model, adapter_path, lora_r=args.lora_r, seed=i * 42 + 7)

    print(f"\n完成! 共 {args.num_loras} 个 adapter 在 {output_dir}")


if __name__ == "__main__":
    main()
