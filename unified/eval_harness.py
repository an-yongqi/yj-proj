"""
统一评估模块
为所有优化技术提供一致的下游任务评估接口

支持两种评估后端：
1. pip 安装的 lm_eval 0.3.0 (用于标准 HuggingFace 模型)
2. ABQ-LLM 内置的 lm_eval (用于量化模型的 LMClass)
"""

import os
import sys
import json
import torch
import torch.nn as nn
from datetime import datetime

# 默认 7 个下游任务
TASK_LIST = [
    "piqa", "arc_easy", "arc_challenge", "boolq",
    "hellaswag", "winogrande", "openbookqa"
]


def evaluate_zero_shot(
    model,
    tokenizer,
    model_name="llama-2-7b",
    task_list=None,
    num_fewshot=0,
    batch_size=32,
    limit=None,
):
    """
    用 pip 安装的 lm_eval 评估标准 HuggingFace 模型
    复用 FANG/lib/eval.py:103-135 的 eval_zero_shot() 模式

    Args:
        model: 已加载的 HuggingFace CausalLM 模型
        tokenizer: 对应的 tokenizer
        model_name: 模型名称（用于 lm_eval 的 model_args）
        task_list: 评估任务列表
        num_fewshot: few-shot 数量（默认 0 = zero-shot）
        batch_size: 评估 batch size
        limit: 限制每个任务的样本数（用于调试）

    Returns:
        dict: 包含每个任务准确率的结果字典
    """
    if task_list is None:
        task_list = TASK_LIST

    import fnmatch
    from lm_eval import tasks, evaluator

    task_names = []
    for pattern in task_list:
        for matching in fnmatch.filter(tasks.ALL_TASKS, pattern):
            task_names.append(matching)

    model_args = f"pretrained={model_name}"

    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=False,
    )

    return results


def evaluate_abq_quantized(lm_class, task_list=None, num_fewshot=0, limit=None):
    """
    用 ABQ-LLM 内置 lm_eval 评估量化模型

    ABQ 的 LMClass 实现了 BaseLM 接口（_model_call, tok_encode 等），
    必须用 ABQ 内置的 evaluator 来驱动。

    Args:
        lm_class: ABQ-LLM 的 LMClass 实例
        task_list: 评估任务列表（逗号分隔的字符串或列表）
        num_fewshot: few-shot 数量
        limit: 限制样本数

    Returns:
        dict: 评估结果
    """
    if task_list is None:
        task_list = TASK_LIST

    # 构建 tasks 字符串
    if isinstance(task_list, list):
        tasks_str = ",".join(task_list)
    else:
        tasks_str = task_list

    # 导入 ABQ-LLM 内置的 evaluator
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abq_algo_path = os.path.join(project_root, "third_party", "ABQ-LLM", "algorithm")
    sys.path.insert(0, abq_algo_path)

    from lm_eval import evaluator as abq_evaluator

    results = abq_evaluator.simple_evaluate(
        lm_class,
        tasks=tasks_str,
        num_fewshot=num_fewshot,
        limit=None if limit == -1 else limit,
    )

    # 恢复 sys.path
    sys.path.remove(abq_algo_path)

    return results


def evaluate_ppl(model, tokenizer, dataset="wikitext2", device=None, seqlen=2048):
    """
    Perplexity 评估（WikiText-2 / PTB / C4）

    Args:
        model: HuggingFace CausalLM 模型
        tokenizer: 对应的 tokenizer
        dataset: 数据集名称
        device: 评估设备
        seqlen: 序列长度

    Returns:
        float: perplexity 值
    """
    if device is None:
        device = next(model.parameters()).device

    # 加载测试数据
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fang_path = os.path.join(project_root, "third_party", "FANG")
    sys.path.insert(0, fang_path)
    from lib.data import get_loaders
    sys.path.remove(fang_path)

    _, testloader = get_loaders(dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer)
    testenc = testloader.input_ids

    nsamples = testenc.numel() // seqlen
    nlls = []

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            outputs = model(batch)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(device)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return ppl.item()


def format_results(results, technique_name, save_dir=None):
    """
    格式化并打印/保存评估结果

    Args:
        results: lm_eval 返回的结果字典
        technique_name: 技术名称（如 "W2A8", "Prune20%"）
        save_dir: 保存路径（可选）

    Returns:
        dict: 格式化后的结果
    """
    formatted = {"technique": technique_name, "timestamp": datetime.now().isoformat()}
    task_results = {}

    if "results" in results:
        for task, metrics in results["results"].items():
            if "acc_norm" in metrics:
                task_results[task] = round(metrics["acc_norm"] * 100, 2)
            elif "acc" in metrics:
                task_results[task] = round(metrics["acc"] * 100, 2)

    formatted["tasks"] = task_results

    if task_results:
        formatted["average"] = round(
            sum(task_results.values()) / len(task_results), 2
        )

    # 打印结果
    print("\n" + "=" * 60)
    print(f"  评估结果: {technique_name}")
    print("=" * 60)
    for task, acc in sorted(task_results.items()):
        print(f"  {task:20s}: {acc:.2f}%")
    if "average" in formatted:
        print(f"  {'AVERAGE':20s}: {formatted['average']:.2f}%")
    print("=" * 60 + "\n")

    # 保存到文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{technique_name}.json")
        with open(save_path, "w") as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)
        print(f"结果已保存至: {save_path}")

    return formatted
