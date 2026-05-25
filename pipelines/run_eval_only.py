"""
独立评估脚本：对已保存的标准 HF 模型做 PPL + 7 个 Zero-shot 评估

适用于：
- 量化后的 fake-quant FP16 模型（ABQ-LLM save_pretrained 输出）
- 原始 baseline 模型
- 任何标准 HuggingFace CausalLM 模型

用法:
    python pipelines/run_eval_only.py --model /path/to/model --name "W2A8"
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_LIST = ["piqa", "arc_easy", "arc_challenge", "boolq", "hellaswag", "winogrande", "openbookqa"]


def eval_ppl_wikitext2(model, tokenizer, seqlen=2048):
    """WikiText-2 PPL 评估"""
    from datasets import load_dataset
    import torch.nn as nn

    device = next(model.parameters()).device
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    testenc = testenc.input_ids

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
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    model.config.use_cache = use_cache
    return ppl.item()


def eval_zero_shot(model, tokenizer, batch_size=4):
    """用 ABQ-LLM 内置 lm_eval 做 zero-shot 评估"""
    abq_path = os.path.join(PROJECT_ROOT, "third_party", "ABQ-LLM", "algorithm")
    # 确保用 ABQ 内置 lm_eval
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

    task_scores = {}
    for task in TASK_LIST:
        if task in results["results"]:
            metrics = results["results"][task]
            if "acc_norm" in metrics:
                task_scores[task] = round(metrics["acc_norm"] * 100, 2)
            elif "acc" in metrics:
                task_scores[task] = round(metrics["acc"] * 100, 2)
    return task_scores


def main():
    parser = argparse.ArgumentParser(description="独立模型评估 (PPL + Zero-shot)")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--name", type=str, default=None, help="模型名称（用于输出）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--skip_ppl", action="store_true", help="跳过 PPL 评估")
    parser.add_argument("--skip_zeroshot", action="store_true", help="跳过 zero-shot 评估")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "eval_results"))
    args = parser.parse_args()

    if args.name is None:
        args.name = args.model.rstrip("/").split("/")[-1]

    print(f"\n{'='*60}")
    print(f"  评估模型: {args.name}")
    print(f"  路径: {args.model}")
    print(f"{'='*60}\n")

    # 加载模型
    print(">>> 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    results = {"name": args.name, "model_path": args.model, "timestamp": datetime.now().isoformat()}

    # PPL
    if not args.skip_ppl:
        print(">>> 评估 WikiText-2 Perplexity...")
        ppl = eval_ppl_wikitext2(model, tokenizer)
        results["ppl"] = round(ppl, 2)
        print(f"    PPL = {ppl:.2f}")

    # Zero-shot
    if not args.skip_zeroshot:
        print("\n>>> 评估 7 个 Zero-shot 任务...")
        task_scores = eval_zero_shot(model, tokenizer, args.batch_size)
        results["tasks"] = task_scores
        avg = round(sum(task_scores.values()) / len(task_scores), 2) if task_scores else 0
        results["average"] = avg

        print(f"\n{'─'*40}")
        for task in TASK_LIST:
            score = task_scores.get(task, 0)
            print(f"  {task:20s}: {score:.2f}%")
        print(f"  {'Average':20s}: {avg:.2f}%")
        print(f"{'─'*40}")

    # 保存
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    main()
