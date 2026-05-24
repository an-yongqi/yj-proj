"""
加载 FANG 结构化剪枝后的模型

FANG 的结构化剪枝会改变每层的 MLP/Attention 维度，且不同层剪枝比例不同，
导致标准 from_pretrained 无法加载。

本模块通过以下方式解决:
1. 先用原始 config 创建模型骨架
2. 扫描 state_dict 确定每层的实际维度
3. 替换维度不匹配的 Linear 层
4. 加载实际权重
"""

import os
import glob
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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


def _get_module(model, path):
    """通过点分路径获取模块"""
    parts = path.split(".")
    module = model
    for p in parts:
        if p.isdigit():
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module


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
    """
    扫描剪枝后模型的每层实际维度

    Returns:
        dict: 每层的维度信息
        {
            0: {"gate_proj": (out, in), "up_proj": (out, in), "down_proj": (out, in),
                "q_proj": (out, in), "k_proj": (out, in), "v_proj": (out, in), "o_proj": (out, in)},
            1: {...},
            ...
        }
    """
    state_dict = _load_state_dict_from_dir(model_dir)
    layer_dims = {}
    proj_names = ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]

    for key, tensor in state_dict.items():
        for proj in proj_names:
            if f".{proj}.weight" in key:
                # 解析层号: model.layers.X.mlp.gate_proj.weight
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
    1. 尝试标准 from_pretrained（如果 config 已正确更新）
    2. 失败则用自定义加载: 创建骨架 → 替换 Linear 层 → 加载权重

    Args:
        model_dir: 剪枝后模型目录
        torch_dtype: 数据类型
        device_map: 设备映射

    Returns:
        (model, tokenizer) 元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    # 先尝试标准加载
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype, device_map=device_map,
            low_cpu_mem_usage=True,
        )
        model.eval()
        if not hasattr(model, "seqlen"):
            model.seqlen = 2048
        print(f"[load_pruned_model] 标准加载成功")
        return model, tokenizer
    except (ValueError, RuntimeError) as e:
        print(f"[load_pruned_model] 标准加载失败: {e}")
        print(f"[load_pruned_model] 使用自定义加载器...")

    # 自定义加载
    print(f"[load_pruned_model] 正在读取权重文件...")
    state_dict = _load_state_dict_from_dir(model_dir)
    print(f"[load_pruned_model] 读取完成，共 {len(state_dict)} 个参数")

    # 创建模型骨架（跳过权重初始化，快速且省内存）
    print(f"[load_pruned_model] 创建模型骨架...")
    config = AutoConfig.from_pretrained(model_dir)
    # 跳过 reset_parameters 避免慢速随机初始化
    _orig_linear_reset = nn.Linear.reset_parameters
    _orig_embed_reset = nn.Embedding.reset_parameters
    nn.Linear.reset_parameters = lambda self: None
    nn.Embedding.reset_parameters = lambda self: None
    try:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    finally:
        nn.Linear.reset_parameters = _orig_linear_reset
        nn.Embedding.reset_parameters = _orig_embed_reset

    # 遍历 state_dict，找出维度不匹配的 Linear 层并替换
    print(f"[load_pruned_model] 替换非均匀维度的 Linear 层...")
    replaced = 0
    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            continue

        module_path = key[:-7]  # remove ".weight"
        try:
            module = _get_module(model, module_path)
        except (AttributeError, IndexError):
            continue

        if isinstance(module, nn.Linear) and module.weight.shape != tensor.shape:
            out_features, in_features = tensor.shape
            has_bias = module.bias is not None
            new_linear = nn.Linear(in_features, out_features, bias=has_bias, dtype=torch_dtype)
            _set_module(model, module_path, new_linear)
            replaced += 1

    print(f"[load_pruned_model] 替换了 {replaced} 个 Linear 层")

    # 加载权重
    print(f"[load_pruned_model] 加载权重到模型...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # 过滤掉正常缺失的（如 inv_freq）
        real_missing = [k for k in missing if "inv_freq" not in k]
        if real_missing:
            print(f"[load_pruned_model] 警告: 缺失 {len(real_missing)} 个权重: {real_missing[:5]}...")
    if unexpected:
        print(f"[load_pruned_model] 警告: 多余 {len(unexpected)} 个权重: {unexpected[:5]}...")

    # Patch 每层的 Attention 参数（num_heads 等）以匹配实际权重
    print(f"[load_pruned_model] 修正每层 Attention 参数...")
    head_dim = config.hidden_size // config.num_attention_heads
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

    # 释放 state_dict 节省内存
    del state_dict

    # 移动到 GPU
    if device_map == "auto":
        model = model.cuda()
    model.eval()

    if not hasattr(model, "seqlen"):
        model.seqlen = 2048

    return model, tokenizer
