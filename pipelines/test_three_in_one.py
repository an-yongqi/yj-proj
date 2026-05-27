"""
三合一兼容测试: 剪枝(FANG) + 量化(ABQ-LLM) + KV cache 压缩(ReST-KV)

测试流程:
1. 加载剪枝+量化模型 (已有的 abq_parameters.pth)
2. 启用 ReST-KV KV cache 压缩
3. 对比 有/无 ReST-KV 的生成结果和 PPL

用法:
    python pipelines/test_three_in_one.py \
        --pruned_model outputs/pruned_models/Llama-2-7b-pruned-20pct \
        --abq_params outputs/pruned_quantized_models/-pruned20-w2a8/log/abq_parameters.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def eval_ppl_wikitext2(model, tokenizer, seqlen=2048, max_samples=20):
    """简化版 WikiText-2 PPL 评估 (限制样本数加速)"""
    from datasets import load_dataset

    device = next(model.parameters()).device
    try:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        local_path = os.path.join(PROJECT_ROOT, "data", "wikitext-2-raw", "test.parquet")
        testdata = load_dataset("parquet", data_files=local_path, split="train")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    testenc = testenc.input_ids

    nsamples = min(testenc.numel() // seqlen, max_samples)
    nlls = []

    model.eval()
    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            outputs = model(batch, use_cache=False)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(device)
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


def test_generate(model, tokenizer, prompt, max_new_tokens=50):
    """简单生成测试"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def main():
    parser = argparse.ArgumentParser(description="三合一兼容测试")
    parser.add_argument("--pruned_model", type=str, required=True, help="剪枝后的模型路径")
    parser.add_argument("--abq_params", type=str, required=True, help="量化参数路径")
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--max_capacity_prompts", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=32)
    args = parser.parse_args()

    # ── Step 1: 加载剪枝+量化模型 ──
    print("\n" + "=" * 60)
    print("  Step 1: 加载剪枝+量化模型")
    print("=" * 60)
    from unified.abq_model_loader import load_abq_quantized_model
    model, tokenizer = load_abq_quantized_model(
        base_model_path=args.pruned_model,
        abq_params_path=args.abq_params,
        wbits=args.wbits,
        abits=args.abits,
    )

    # ── Step 2: 无 ReST-KV 的 PPL ──
    print("\n" + "=" * 60)
    print("  Step 2: 剪枝+量化 PPL (无 ReST-KV)")
    print("=" * 60)
    ppl_no_restkv = eval_ppl_wikitext2(model, tokenizer)
    print(f"  PPL (剪枝+量化, 无 KV 压缩) = {ppl_no_restkv:.2f}")

    # ── Step 3: 无 ReST-KV 的生成测试 ──
    print("\n" + "=" * 60)
    print("  Step 3: 生成测试 (无 ReST-KV)")
    print("=" * 60)
    prompt = "The meaning of life is"
    # 生成需要 use_cache=True，先临时启用
    model.config.use_cache = True
    gen_no_restkv = test_generate(model, tokenizer, prompt)
    model.config.use_cache = False
    print(f"  Prompt: {prompt}")
    print(f"  Output: {gen_no_restkv}")

    # ── Step 4: 启用 ReST-KV ──
    print("\n" + "=" * 60)
    print("  Step 4: 启用 ReST-KV KV cache 压缩")
    print("=" * 60)
    from unified.restkv_compat import enable_restkv_on_quant_model
    enable_restkv_on_quant_model(
        model,
        max_capacity_prompts=args.max_capacity_prompts,
        window_size=args.window_size,
    )

    # ── Step 5: 有 ReST-KV 的 PPL ──
    # 注意: PPL 评估用 use_cache=False，ReST-KV 仅在 prefix 阶段
    # (past_key_value=None) 触发。use_cache=False 时 past_key_value
    # 始终为 None，所以每次 forward 都是 prefix 阶段，会触发压缩。
    # 这模拟了 ReST-KV 在长序列推理时的效果。
    print("\n" + "=" * 60)
    print("  Step 5: 剪枝+量化+ReST-KV PPL")
    print("=" * 60)
    ppl_with_restkv = eval_ppl_wikitext2(model, tokenizer)
    print(f"  PPL (剪枝+量化+ReST-KV) = {ppl_with_restkv:.2f}")

    # ── Step 6: 有 ReST-KV 的生成测试 ──
    print("\n" + "=" * 60)
    print("  Step 6: 生成测试 (有 ReST-KV)")
    print("=" * 60)
    model.config.use_cache = True
    gen_with_restkv = test_generate(model, tokenizer, prompt)
    model.config.use_cache = False
    print(f"  Prompt: {prompt}")
    print(f"  Output: {gen_with_restkv}")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("  结果汇总: 剪枝 + 量化 + ReST-KV")
    print("=" * 60)
    print(f"  PPL 无 ReST-KV:  {ppl_no_restkv:.2f}")
    print(f"  PPL 有 ReST-KV:  {ppl_with_restkv:.2f}")
    print(f"  PPL 差异:        {ppl_with_restkv - ppl_no_restkv:+.2f}")
    print()
    print(f"  生成 (无 ReST-KV): {gen_no_restkv[:80]}...")
    print(f"  生成 (有 ReST-KV): {gen_with_restkv[:80]}...")
    print()
    if ppl_with_restkv < ppl_no_restkv * 1.5:
        print("  结论: 三合一兼容成功! PPL 在合理范围内。")
    else:
        print("  警告: PPL 偏差较大，需要检查兼容性。")
    print("=" * 60)


if __name__ == "__main__":
    main()
