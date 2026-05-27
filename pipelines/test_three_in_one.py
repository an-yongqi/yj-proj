"""
三合一兼容测试: 剪枝(FANG) + 量化(ABQ-LLM) + KV cache 压缩(ReST-KV)

测试流程:
1. 加载剪枝+量化模型
2. 短 prompt 生成 (无 ReST-KV vs 有 ReST-KV)
3. 长 prompt 生成 (>max_capacity_prompts, 触发 KV 压缩)
4. 逐 token loss 对比: 用长文本喂入，对比有/无压缩的 next-token 预测 loss
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def eval_ppl_wikitext2(model, tokenizer, seqlen=2048, max_samples=20):
    """WikiText-2 PPL 评估"""
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


def generate_with_cache(model, tokenizer, prompt, max_new_tokens=50):
    """用 KV cache 生成 (会触发 ReST-KV 压缩)"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=1.0,  # 避免 warning
            top_p=1.0,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return full_text, generated_text


def compute_generate_loss(model, tokenizer, prompt, continuation, device):
    """
    计算给定 continuation 在 prompt 之后的 cross-entropy loss
    用 use_cache=True 模拟实际 generate 场景下 ReST-KV 的效果
    """
    full_text = prompt + continuation
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    model.eval()
    with torch.no_grad():
        # prefix 阶段: 用 use_cache=True 触发 ReST-KV
        prefix_out = model(prompt_ids, use_cache=True)
        past_kv = prefix_out.past_key_values

        # decode 阶段: 逐 token 计算 loss
        cont_ids = inputs["input_ids"][:, prompt_len:]
        total_loss = 0.0
        n_tokens = cont_ids.shape[1]
        prev_token = inputs["input_ids"][:, prompt_len - 1:prompt_len]

        for t in range(n_tokens):
            out = model(prev_token, past_key_values=past_kv, use_cache=True)
            logits = out.logits[:, -1, :]
            target = cont_ids[:, t]
            loss = nn.CrossEntropyLoss()(logits, target)
            total_loss += loss.item()
            past_kv = out.past_key_values
            prev_token = cont_ids[:, t:t+1]

    return total_loss / n_tokens


def build_long_prompt(tokenizer, min_tokens=200):
    """构造超过 max_capacity_prompts 的长 prompt"""
    # 用重复文本构造足够长的 prompt
    text = (
        "The history of artificial intelligence began in antiquity, with myths and stories "
        "of artificial beings endowed with intelligence. The seeds of modern AI were planted "
        "by philosophers who attempted to describe the process of human thinking as the "
        "mechanical manipulation of symbols. This work culminated in the invention of the "
        "programmable digital computer in the 1940s. This machine inspired a handful of "
        "scientists to begin seriously discussing the possibility of building an electronic "
        "brain. The field of AI research was founded at a workshop held on the campus of "
        "Dartmouth College during the summer of 1956. The attendees became the leaders of "
        "AI research for decades. Many of them predicted that a machine as intelligent as "
        "a human being would exist in no more than a generation, and they were given millions "
        "of dollars to make this vision come true. Eventually it became obvious that commercial "
        "developers and investors of AI had been far too optimistic about the ability of the "
        "first AI systems. The question of what it means to be intelligent remains unanswered. "
    )
    # 重复直到超过 min_tokens
    while len(tokenizer.encode(text)) < min_tokens:
        text = text + text
    # 截断到合理长度
    ids = tokenizer.encode(text)[:min_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text


def disable_restkv(model):
    """临时禁用 ReST-KV"""
    clusters = {}
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, 'self_attn', None)
        if attn and hasattr(attn, 'kv_cluster'):
            clusters[i] = attn.kv_cluster
            delattr(attn, 'kv_cluster')
    return clusters


def restore_restkv(model, clusters):
    """恢复 ReST-KV"""
    for i, cluster in clusters.items():
        model.model.layers[i].self_attn.kv_cluster = cluster


def main():
    parser = argparse.ArgumentParser(description="三合一兼容测试")
    parser.add_argument("--pruned_model", type=str, required=True)
    parser.add_argument("--abq_params", type=str, required=True)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--abits", type=int, default=8)
    parser.add_argument("--max_capacity_prompts", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ── Step 2: Baseline PPL (无 ReST-KV) ──
    print("\n" + "=" * 60)
    print("  Step 2: Baseline PPL (剪枝+量化, 无 ReST-KV)")
    print("=" * 60)
    ppl_baseline = eval_ppl_wikitext2(model, tokenizer)
    print(f"  PPL = {ppl_baseline:.2f}")

    # ── Step 3: 短 prompt 生成 (无 ReST-KV) ──
    print("\n" + "=" * 60)
    print("  Step 3: 短 prompt 生成 (无 ReST-KV)")
    print("=" * 60)
    short_prompt = "The meaning of life is"
    model.config.use_cache = True
    _, gen_short_no_restkv = generate_with_cache(model, tokenizer, short_prompt)
    model.config.use_cache = False
    print(f"  Prompt ({len(tokenizer.encode(short_prompt))} tokens): {short_prompt}")
    print(f"  Output: {gen_short_no_restkv}")

    # ── Step 4: 启用 ReST-KV ──
    print("\n" + "=" * 60)
    print("  Step 4: 启用 ReST-KV")
    print("=" * 60)
    from unified.restkv_compat import enable_restkv_on_quant_model
    enable_restkv_on_quant_model(
        model,
        max_capacity_prompts=args.max_capacity_prompts,
        window_size=args.window_size,
    )

    # ── Step 5: 短 prompt 生成 (有 ReST-KV, 但不触发压缩) ──
    print("\n" + "=" * 60)
    print(f"  Step 5: 短 prompt 生成 (有 ReST-KV, <{args.max_capacity_prompts} tokens, 不触发压缩)")
    print("=" * 60)
    model.config.use_cache = True
    _, gen_short_restkv = generate_with_cache(model, tokenizer, short_prompt)
    model.config.use_cache = False
    print(f"  Output: {gen_short_restkv}")
    match_short = gen_short_no_restkv == gen_short_restkv
    print(f"  与无 ReST-KV 一致: {match_short} (预期: True, 因为未触发压缩)")

    # ── Step 6: 长 prompt 生成对比 ──
    print("\n" + "=" * 60)
    print(f"  Step 6: 长 prompt 生成 (>{args.max_capacity_prompts} tokens, 触发压缩)")
    print("=" * 60)
    long_prompt = build_long_prompt(tokenizer, min_tokens=args.max_capacity_prompts + 50)
    prompt_tokens = len(tokenizer.encode(long_prompt))
    print(f"  Prompt 长度: {prompt_tokens} tokens (阈值: {args.max_capacity_prompts})")

    # 有 ReST-KV
    model.config.use_cache = True
    _, gen_long_restkv = generate_with_cache(model, tokenizer, long_prompt, max_new_tokens=30)
    model.config.use_cache = False
    print(f"  有 ReST-KV: {gen_long_restkv[:100]}...")

    # 无 ReST-KV
    saved_clusters = disable_restkv(model)
    model.config.use_cache = True
    _, gen_long_no_restkv = generate_with_cache(model, tokenizer, long_prompt, max_new_tokens=30)
    model.config.use_cache = False
    restore_restkv(model, saved_clusters)
    print(f"  无 ReST-KV: {gen_long_no_restkv[:100]}...")

    match_long = gen_long_restkv == gen_long_no_restkv
    print(f"  输出一致: {match_long} (预期: False, 因为 KV 被压缩)")

    # ── Step 7: 长 prompt decode loss 对比 ──
    print("\n" + "=" * 60)
    print("  Step 7: 长 prompt decode 阶段 loss 对比")
    print("=" * 60)
    # 用一段固定文本作为 continuation
    continuation = " In conclusion, the development of artificial intelligence represents one of the most"

    # 有 ReST-KV
    model.config.use_cache = True
    loss_restkv = compute_generate_loss(model, tokenizer, long_prompt, continuation, device)
    model.config.use_cache = False

    # 无 ReST-KV
    saved_clusters = disable_restkv(model)
    model.config.use_cache = True
    loss_no_restkv = compute_generate_loss(model, tokenizer, long_prompt, continuation, device)
    model.config.use_cache = False
    restore_restkv(model, saved_clusters)

    print(f"  Decode loss 无 ReST-KV: {loss_no_restkv:.4f}")
    print(f"  Decode loss 有 ReST-KV: {loss_restkv:.4f}")
    print(f"  Loss 差异:              {loss_restkv - loss_no_restkv:+.4f} ({(loss_restkv/loss_no_restkv - 1)*100:+.1f}%)")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("  结果汇总: 剪枝 + 量化 + ReST-KV 三合一")
    print("=" * 60)
    print(f"  Baseline PPL (无 ReST-KV):     {ppl_baseline:.2f}")
    print(f"  短 prompt 生成一致性:           {'通过' if match_short else '不一致'}")
    print(f"  长 prompt 触发压缩:             {'是' if not match_long else '否 (异常)'}")
    print(f"  Decode loss 无 ReST-KV:        {loss_no_restkv:.4f}")
    print(f"  Decode loss 有 ReST-KV:        {loss_restkv:.4f}")
    print(f"  Loss 退化:                     {(loss_restkv/loss_no_restkv - 1)*100:+.1f}%")
    print()
    if loss_restkv < loss_no_restkv * 1.3:
        print("  结论: 三合一兼容成功! KV 压缩损失在合理范围内 (<30%)")
    elif loss_restkv < loss_no_restkv * 1.5:
        print("  结论: 三合一基本兼容, KV 压缩有一定损失但可接受")
    else:
        print("  警告: KV 压缩损失偏大, 建议调整 max_capacity_prompts")
    print("=" * 60)


if __name__ == "__main__":
    main()
