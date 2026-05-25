"""
加载 FANG 结构化剪枝后的模型

支持两种模式:
1. Mask 模式 (推荐): 剪枝后维度不变，权重置零，直接用 from_pretrained 加载
2. 物理剪枝模式 (旧): 每层维度不同，需要扫描 state_dict 并替换 Linear 层
"""

import os
import glob
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_pruned_model_mask(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
):
    """
    加载 mask 模式剪枝的模型（推荐）

    Mask 模式下模型维度不变，直接用 from_pretrained 加载即可。

    Args:
        model_dir: 剪枝后模型目录
        torch_dtype: 数据类型
        device_map: 设备映射

    Returns:
        (model, tokenizer) 元组
    """
    print(f"[load_pruned_model_mask] 加载 mask 剪枝模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map=device_map,
    )
    model.eval()

    if not hasattr(model, "seqlen"):
        model.seqlen = 2048

    # 打印剪枝统计
    mask_path = os.path.join(model_dir, "pruning_masks.pt")
    if os.path.exists(mask_path):
        masks = torch.load(mask_path, map_location="cpu")
        total_pruned = 0
        total_neurons = 0
        for idx in sorted(masks.keys()):
            mlp_mask = masks[idx]['mlp']
            attn_mask = masks[idx]['attn']
            total_pruned += (~mlp_mask).sum().item() + (~attn_mask).sum().item()
            total_neurons += mlp_mask.numel() + attn_mask.numel()
        print(f"[load_pruned_model_mask] 剪枝率: {total_pruned}/{total_neurons} = {total_pruned/total_neurons:.1%}")

    print(f"[load_pruned_model_mask] 加载完成")
    return model, tokenizer


def _load_state_dict_from_dir(model_dir):
    """从目录加载完整 state_dict（支持 safetensors 和 bin 格式）"""
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if safetensor_files:
        from safetensors.torch import load_file
        state_dict = {}
        for f in safetensor_files:
            state_dict.update(load_file(f, device="cpu"))
        return state_dict

    bin_files = sorted(glob.glob(os.path.join(model_dir, "*.bin")))
    if bin_files:
        state_dict = {}
        for f in bin_files:
            state_dict.update(torch.load(f, map_location="cpu"))
        return state_dict

    raise FileNotFoundError(f"No model weights found in {model_dir}")


def _set_module(model, path, new_module):
    """通过点分路径替换模块"""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def scan_dimensions(model_dir):
    """扫描剪枝后模型的每层实际维度"""
    state_dict = _load_state_dict_from_dir(model_dir)
    layer_dims = {}
    proj_names = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]

    for key, tensor in state_dict.items():
        for proj in proj_names:
            if f".{proj}.weight" in key:
                parts = key.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        if layer_idx not in layer_dims:
                            layer_dims[layer_idx] = {}
                        layer_dims[layer_idx][proj] = tuple(tensor.shape)
                        break

    return layer_dims


def print_dimension_report(model_dir):
    """打印剪枝后每层的维度变化"""
    config = AutoConfig.from_pretrained(model_dir)
    layer_dims = scan_dimensions(model_dir)

    orig_hidden = config.hidden_size
    orig_intermediate = config.intermediate_size
    orig_heads = config.num_attention_heads
    head_dim = orig_hidden // orig_heads

    print(f"\n{'='*80}")
    print(f"  剪枝模型维度报告: {model_dir}")
    print(f"  原始: hidden={orig_hidden}, intermediate={orig_intermediate}, heads={orig_heads}")
    print(f"{'='*80}")
    print(f"{'Layer':>5} | {'MLP(gate/up)':>12} | {'MLP(down_in)':>12} | {'QKV_out':>10} | {'O_proj_in':>10} | {'Heads':>5}")
    print(f"{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}")

    for layer_idx in sorted(layer_dims.keys()):
        dims = layer_dims[layer_idx]
        gate_out = dims.get("gate_proj", (0, 0))[0]
        down_in = dims.get("down_proj", (0, 0))[1]
        q_out = dims.get("q_proj", (0, 0))[0]
        o_in = dims.get("o_proj", (0, 0))[1]
        heads = q_out // head_dim if head_dim > 0 else "?"
        print(f"{layer_idx:>5} | {gate_out:>12} | {down_in:>12} | {q_out:>10} | {o_in:>10} | {heads:>5}")

    print(f"{'='*80}\n")


def load_pruned_model(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
):
    """
    加载 FANG 结构化剪枝后的模型

    处理流程:
    1. 加载 state_dict，扫描每层的实际维度
    2. 创建标准模型骨架
    3. 替换维度不匹配的 Linear 层
    4. 用 load_state_dict 加载实际权重
    5. 修正每层的 attention 参数

    Args:
        model_dir: 剪枝后模型目录
        torch_dtype: 数据类型
        device_map: 设备映射

    Returns:
        (model, tokenizer) 元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    config = AutoConfig.from_pretrained(model_dir)
    head_dim = config.hidden_size // config.num_attention_heads

    # Step 1: 加载 state_dict 并扫描实际维度
    print(f"[load_pruned_model] 加载 state_dict...")
    state_dict = _load_state_dict_from_dir(model_dir)

    # Step 2: 创建模型骨架（用原始 config，不初始化权重以加速）
    print(f"[load_pruned_model] 创建模型骨架...")
    # 临时 monkeypatch 跳过随机初始化
    original_init = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda self: None
    try:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    finally:
        torch.nn.Linear.reset_parameters = original_init

    # Step 3: 扫描 state_dict，替换维度不匹配的 Linear 层
    print(f"[load_pruned_model] 替换剪枝后的 Linear 层...")
    proj_names = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    replaced = 0

    for key, tensor in state_dict.items():
        # 只处理 Linear 权重
        if not key.endswith(".weight"):
            continue
        is_proj = False
        for proj in proj_names:
            if f".{proj}.weight" in key:
                is_proj = True
                break
        if not is_proj:
            continue

        # 获取对应模块路径 (去掉 .weight)
        module_path = key[:-len(".weight")]
        try:
            module = model
            for p in module_path.split("."):
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)
        except (AttributeError, IndexError):
            continue

        if not isinstance(module, nn.Linear):
            continue

        # 检查维度是否匹配
        ckpt_out, ckpt_in = tensor.shape
        if module.out_features != ckpt_out or module.in_features != ckpt_in:
            new_linear = nn.Linear(ckpt_in, ckpt_out, bias=module.bias is not None, dtype=torch_dtype)
            _set_module(model, module_path, new_linear)
            replaced += 1

    print(f"[load_pruned_model] 替换了 {replaced} 个 Linear 层")

    # Step 4: 加载权重
    print(f"[load_pruned_model] 加载权重...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # 过滤掉 rotary_emb 的 inv_freq（正常缺失）
        real_missing = [k for k in missing if "inv_freq" not in k]
        if real_missing:
            print(f"[load_pruned_model] 警告: {len(real_missing)} 个权重缺失")

    # Step 5: 修正每层 Attention 参数
    print(f"[load_pruned_model] 修正每层 Attention 参数...")
    for layer in model.model.layers:
        attn = layer.self_attn
        actual_q_out = attn.q_proj.out_features
        actual_k_out = attn.k_proj.out_features
        actual_num_heads = actual_q_out // head_dim
        actual_num_kv_heads = actual_k_out // head_dim
        if actual_num_heads != attn.num_heads:
            attn.num_heads = actual_num_heads
            attn.hidden_size = actual_num_heads * head_dim
            attn.num_key_value_heads = actual_num_kv_heads
            attn.num_key_value_groups = actual_num_heads // actual_num_kv_heads

    # Step 6: 移动到 GPU
    if device_map == "auto":
        print(f"[load_pruned_model] 移动模型到 GPU...")
        model = model.cuda()
    model.eval()

    if not hasattr(model, "seqlen"):
        model.seqlen = 2048

    return model, tokenizer
