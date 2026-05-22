"""
LoRA per-task 训练脚本
在 7 个下游任务上为 LLaMA-2-7B 训练独立的 LoRA adapter

每个 adapter 保存到 outputs/lora_adapters/{task_name}/
后续由 lora_causal.py 中的 Nevergrad 算法进行组合

用法:
    python -m unified.lora_train \
        --base_model /path/to/Llama-2-7b \
        --task piqa \
        --output_dir outputs/lora_adapters/piqa \
        --lora_r 16 --num_epochs 3
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


# 任务到 HuggingFace datasets 的映射
TASK_DATASET_MAP = {
    "piqa": {"path": "piqa", "split": "train"},
    "arc_easy": {"path": "ai2_arc", "name": "ARC-Easy", "split": "train"},
    "arc_challenge": {"path": "ai2_arc", "name": "ARC-Challenge", "split": "train"},
    "boolq": {"path": "boolq", "split": "train"},
    "hellaswag": {"path": "Rowan/hellaswag", "split": "train"},
    "winogrande": {"path": "winogrande", "name": "winogrande_xl", "split": "train"},
    "openbookqa": {"path": "openbookqa", "name": "main", "split": "train"},
}


def format_piqa(example):
    """PiQA: 给定 goal，选择 sol1 或 sol2"""
    goal = example["goal"]
    sol1 = example["sol1"]
    sol2 = example["sol2"]
    label = example["label"]
    answer = sol1 if label == 0 else sol2
    return {"text": f"Question: {goal}\nAnswer: {answer}"}


def format_arc(example):
    """ARC: 多选题"""
    question = example["question"]
    choices = example["choices"]
    answer_key = example["answerKey"]

    choice_texts = []
    answer_text = ""
    for label, text in zip(choices["label"], choices["text"]):
        choice_texts.append(f"({label}) {text}")
        if label == answer_key:
            answer_text = text

    choices_str = " ".join(choice_texts)
    return {"text": f"Question: {question}\nChoices: {choices_str}\nAnswer: {answer_text}"}


def format_boolq(example):
    """BoolQ: 是非题"""
    passage = example["passage"][:500]  # 截断避免过长
    question = example["question"]
    answer = "Yes" if example["answer"] else "No"
    return {"text": f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}"}


def format_hellaswag(example):
    """HellaSwag: 句子补全"""
    ctx = example["ctx"]
    endings = example["endings"]
    label = int(example["label"]) if example["label"] != "" else 0
    answer = endings[label] if label < len(endings) else endings[0]
    return {"text": f"Context: {ctx}\nCompletion: {answer}"}


def format_winogrande(example):
    """Winogrande: 代词消歧"""
    sentence = example["sentence"]
    option1 = example["option1"]
    option2 = example["option2"]
    answer = option1 if example["answer"] == "1" else option2
    filled = sentence.replace("_", answer)
    return {"text": f"Sentence: {filled}"}


def format_openbookqa(example):
    """OpenBookQA: 常识推理"""
    question = example["question_stem"]
    choices = example["choices"]
    answer_key = example["answerKey"]

    choice_texts = []
    answer_text = ""
    for label, text in zip(choices["label"], choices["text"]):
        choice_texts.append(f"({label}) {text}")
        if label == answer_key:
            answer_text = text

    choices_str = " ".join(choice_texts)
    return {"text": f"Question: {question}\nChoices: {choices_str}\nAnswer: {answer_text}"}


TASK_FORMAT_MAP = {
    "piqa": format_piqa,
    "arc_easy": format_arc,
    "arc_challenge": format_arc,
    "boolq": format_boolq,
    "hellaswag": format_hellaswag,
    "winogrande": format_winogrande,
    "openbookqa": format_openbookqa,
}


def load_and_format_dataset(task_name, max_samples=5000):
    """加载并格式化任务数据集"""
    ds_info = TASK_DATASET_MAP[task_name]
    kwargs = {"path": ds_info["path"], "split": ds_info["split"]}
    if "name" in ds_info:
        kwargs["name"] = ds_info["name"]

    dataset = load_dataset(**kwargs)

    # 限制样本数
    if len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    # 格式化
    format_fn = TASK_FORMAT_MAP[task_name]
    dataset = dataset.map(format_fn, remove_columns=dataset.column_names)

    return dataset


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """tokenize 数据集"""
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # 对于 CausalLM，labels = input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="LoRA per-task 训练")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_DATASET_MAP.keys()))
    parser.add_argument("--output_dir", type=str, required=True, help="LoRA adapter 保存路径")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"任务: {args.task}")
    print(f"基础模型: {args.base_model}")
    print(f"LoRA rank: {args.lora_r}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    print(f"加载 {args.task} 数据集...")
    dataset = load_and_format_dataset(args.task, max_samples=args.max_samples)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=args.max_length)

    # 加载模型
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 配置 LoRA
    target_modules = args.target_modules.split(",")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print(f"开始训练 {args.task} LoRA adapter...")
    trainer.train()

    # 保存 LoRA adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter 已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
