"""
LoraHub → CausalLM 移植
将 LoraHub 的无梯度 LoRA 组合算法从 Seq2Seq (Flan-T5) 移植到 LLaMA-2-7B (decoder-only)

核心思路不变:
1. 加载基础模型 + 多个预训练 LoRA 模块
2. 用 Nevergrad 无梯度优化器学习各 LoRA 模块的组合权重
3. 加权求和得到最终 LoRA → merge 到基础模型

原始代码参考: third_party/lorahub/lorahub/algorithm.py
"""

import os
import copy
import random
from typing import List, Optional, Union
from functools import partial

import torch
import numpy
import nevergrad as ng
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from peft import PeftModel, PeftConfig
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict


def load_base_model_and_lora_modules(
    lora_module_list: List[str],
    model_name_or_path: Optional[str] = None,
    torch_dtype=torch.float16,
):
    """
    加载基础模型和多个 LoRA 模块

    与原版区别: AutoModelForSeq2SeqLM → AutoModelForCausalLM

    Args:
        lora_module_list: LoRA 模块路径列表（本地路径或 HuggingFace ID）
        model_name_or_path: 基础模型路径，None 则从第一个 LoRA 的 config 中推断
        torch_dtype: 数据类型

    Returns:
        (peft_model, tokenizer, cache) 元组
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    default_peft_model_id = lora_module_list[0]

    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path

    # 关键变化: 使用 CausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except Exception as e:
        raise Exception(f"{default_peft_model_id} 无法加载到模型 {model_name_or_path}: {e}")

    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> 开始加载 LoRA 模块")
    cache = {}
    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print(f"> 加载 {peft_model_id} ...")
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        try:
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except AssertionError:
            raise Exception(f"LoRA 模块 {peft_model_id} 架构不兼容（rank 不同）")

    return peft_model, tokenizer, cache


def preprocess_function_causal(examples, tokenizer, max_length=2048):
    """
    CausalLM 的预处理函数

    与 Seq2Seq 的区别:
    - Seq2Seq: 输入和输出分别编码到 encoder/decoder
    - CausalLM: 输入+输出拼接，prompt 部分的 label 设为 -100

    Args:
        examples: {"input": [...], "output": [...]}
        tokenizer: tokenizer 实例
        max_length: 最大长度
    """
    inputs = examples["input"]
    targets = examples["output"]

    # 拼接 input + output
    prompts_with_targets = [f"{inp} {out}" for inp, out in zip(inputs, targets)]

    model_inputs = tokenizer(
        prompts_with_targets,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # 创建 labels: 将 prompt 部分 mask 为 -100
    labels = model_inputs["input_ids"].clone()
    for i, inp in enumerate(inputs):
        # 编码 prompt 部分
        prompt_ids = tokenizer.encode(inp, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        # prompt 部分不计算 loss
        labels[i, :prompt_len] = -100

    # padding token 也不计算 loss
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    return model_inputs


def load_dataset_causal(example_inputs, example_outputs, tokenizer):
    """加载数据集并预处理"""
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)

    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))

    preprocess_func = partial(preprocess_function_causal, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_loss(example_dataset, model, batch_size):
    """
    计算模型在样例数据集上的 loss
    与原版相同，outputs.loss 对 CausalLM 同样有效
    """
    data_batch_size = len(example_dataset) if batch_size is None else min(len(example_dataset), batch_size)
    train_dataloader = DataLoader(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    return float(loss) / len(example_dataset["input"])


def default_l1_regularization(weights):
    """L1 正则化"""
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_score(weights, model, cache, example_dataset, batch_size, get_loss, get_regular):
    """计算组合权重的评分（loss + 正则化）"""
    final_state_dict = {}
    lora_module_list = list(cache.keys())
    keys = cache[lora_module_list[0]].keys()

    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )

    set_peft_model_state_dict(model, final_state_dict)
    loss = get_loss(example_dataset, model, batch_size)
    metric_val = loss + get_regular(weights)
    return metric_val


def get_final_weights(weights, lora_module_list, cache):
    """根据权重合成最终 LoRA state dict"""
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict


def lorahub_learning(
    lora_module_list: List[str],
    example_inputs: List[str],
    example_outputs: List[str],
    max_inference_step: int = 40,
    model_name_or_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    get_loss=default_get_loss,
    get_regular=default_l1_regularization,
    seed: int = 42,
):
    """
    LoraHub 核心学习算法: Nevergrad 无梯度优化 LoRA 组合权重

    与原版完全相同的优化逻辑，仅基础模型改为 CausalLM。

    Args:
        lora_module_list: LoRA 模块路径列表
        example_inputs: 少量样本的输入
        example_outputs: 少量样本的输出
        max_inference_step: Nevergrad 优化步数
        model_name_or_path: 基础模型路径
        batch_size: batch size
        get_loss: 自定义 loss 函数
        get_regular: 自定义正则化函数
        seed: 随机种子

    Returns:
        (weights, model, tokenizer) 元组
    """
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> 未提供 LoRA 模块")
        return None, None, None

    # 加载模型和 LoRA 模块
    model, tokenizer, cache = load_base_model_and_lora_modules(
        lora_module_list, model_name_or_path
    )

    # 预处理数据集
    dataset = load_dataset_causal(example_inputs, example_outputs, tokenizer)

    get_score_partial = partial(
        get_score,
        model=model,
        cache=cache,
        example_dataset=dataset,
        batch_size=batch_size,
        get_loss=get_loss,
        get_regular=get_regular,
    )

    # Nevergrad 优化
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)

    print("> 开始无梯度优化 LoRA 组合权重 ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)

    # 应用最终权重并 merge
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()

    return recommendation.value, model, tokenizer


def lorahub_inference(
    example_inputs: List[str],
    model_or_name_path: Union[AutoModelForCausalLM, str],
    tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
    batch_size: int,
    max_new_tokens: int = 256,
    example_outputs: Optional[List[str]] = None,
):
    """
    使用组合后的模型进行推理

    Args:
        example_inputs: 输入列表
        model_or_name_path: 模型或路径
        tokenizer_or_tokenizer_path: tokenizer 或路径
        batch_size: 推理 batch size
        max_new_tokens: 最大生成 token 数
        example_outputs: 可选的 ground truth（用于计算准确率）

    Returns:
        (predictions, accuracy) 元组
    """
    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                correct += 1
            total += 1
        return correct / total * 100

    if isinstance(model_or_name_path, str):
        model = AutoModelForCausalLM.from_pretrained(
            model_or_name_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = model_or_name_path

    if isinstance(tokenizer_or_tokenizer_path, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_path, use_fast=False)
    else:
        tokenizer = tokenizer_or_tokenizer_path

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    example_predictions = []
    for i in range(0, len(example_inputs), batch_size):
        batch_inputs = example_inputs[i:i + batch_size]
        inputs = tokenizer(
            batch_inputs,
            max_length=2048,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 只取新生成的部分
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            example_predictions.append(generated)

    if example_outputs is not None:
        task_perf = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf = None

    return example_predictions, task_perf
