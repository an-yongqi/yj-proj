"""
加载 ABQ-LLM 量化后的模型用于推理

ABQ-LLM 的 save_pretrained 输出包含 QuantLinear 特有的参数
(weight_quantizer.scales, weight_quantizer.zeros, bound_factor 等),
标准 AutoModelForCausalLM.from_pretrained 会丢弃这些参数，导致输出乱码。

本模块通过以下方式正确加载:
1. 加载原始基座模型
2. 用 QuantLlamaDecoderLayer 包装每层
3. 注册 LET smooth 参数
4. 从 abq_parameters.pth 加载训练好的参数
5. 执行 smooth_and_quant_inplace 应用量化
6. 注册 scales 和 zeros
"""

import os
import sys
import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABQ_DIR = os.path.join(PROJECT_ROOT, "third_party", "ABQ-LLM", "algorithm")


def _setup_abq_imports():
    """确保 ABQ-LLM 的模块可以导入"""
    if ABQ_DIR not in sys.path:
        sys.path.insert(0, ABQ_DIR)


def _make_quant_args(wbits=2, abits=8, lwc=True, let=True):
    """创建量化所需的 args 对象"""
    args = SimpleNamespace()
    args.wbits = wbits
    args.abits = abits
    args.lwc = lwc
    args.let = let
    args.symmetric = False
    args.disable_zero_point = False
    args.w_dynamic_method = "per_channel"
    args.a_dynamic_method = "per_token"
    args.group_size = None
    args.alpha = 0.5

    args.weight_quant_params = {
        "n_bits": wbits,
        "per_channel_axes": [0],
        "symmetric": False,
        "dynamic_method": "per_channel",
        "group_size": None,
        "lwc": lwc,
        "disable_zero_point": False,
    }
    args.act_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.q_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.k_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.v_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    args.p_quant_params = {
        "n_bits": abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": "per_token",
    }
    return args


def load_abq_quantized_model(
    base_model_path,
    abq_params_path,
    wbits=2,
    abits=8,
    lwc=True,
    let=True,
    torch_dtype=torch.float16,
):
    """
    加载 ABQ-LLM 量化后的模型用于推理/生成

    Args:
        base_model_path: 原始基座模型路径 (如 Llama-2-7b-hf)
        abq_params_path: ABQ-LLM 训练产出的 abq_parameters.pth 路径
        wbits: 权重量化位数
        abits: 激活量化位数
        lwc: 是否使用 learnable weight clipping
        let: 是否使用 LET (learnable equivalent transformation)
        torch_dtype: 数据类型

    Returns:
        (model, tokenizer) 元组
    """
    _setup_abq_imports()

    from models.int_llama_layer import QuantLlamaDecoderLayer
    from quantize.int_linear import QuantLinear
    from quantize.utils import smooth_and_quant_inplace, register_scales_and_zeros, set_quant_state

    args = _make_quant_args(wbits, abits, lwc, let)

    # Step 1: 加载原始基座模型
    print(f"[load_abq_quantized] 加载基座模型: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    config = AutoConfig.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, config=config, torch_dtype=torch_dtype, device_map=None,
    )
    model.eval()

    # Step 2: 加载 abq_parameters
    print(f"[load_abq_quantized] 加载量化参数: {abq_params_path}")
    abq_parameters = torch.load(abq_params_path, map_location="cpu")

    layers = model.model.layers
    num_layers = len(layers)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = {
        "q_proj": "qkv",
        "o_proj": "out",
        "up_proj": "fc1",
        "down_proj": "fc2",
    }

    print(f"[load_abq_quantized] 量化 {num_layers} 层...")
    for i in range(num_layers):
        layer = layers[i].to(dev)

        # Step 3: 用 QuantLlamaDecoderLayer 包装
        qlayer = QuantLlamaDecoderLayer(config, layer, args)
        qlayer = qlayer.to(dev)

        # Step 4: 注册 LET smooth 参数 (初始值会被 load_state_dict 覆盖)
        if let:
            k_out = layer.self_attn.k_proj.out_features
            qlayer.register_parameter(
                "qkt_smooth_scale",
                nn.Parameter(torch.ones(k_out, device=dev, dtype=torch_dtype)),
            )
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs:
                        if key in name:
                            in_features = module.in_features
                            scale = torch.ones(in_features, device=dev, dtype=torch_dtype)
                            shift = torch.zeros(in_features, device=dev, dtype=torch_dtype)

                            # o_proj GQA 调整
                            if "self_attn.o_proj" in name and qlayer.self_attn.num_key_value_groups > 1:
                                head_dim = config.hidden_size // config.num_attention_heads
                                num_kv_heads = qlayer.self_attn.num_key_value_heads
                                scale = torch.ones(num_kv_heads * head_dim, device=dev, dtype=torch_dtype)
                                shift = torch.zeros_like(scale)

                            qlayer.register_parameter(
                                f"{pairs[key]}_smooth_shift",
                                nn.Parameter(shift),
                            )
                            qlayer.register_parameter(
                                f"{pairs[key]}_smooth_scale",
                                nn.Parameter(scale),
                            )

                    # 补偿向量 (前3层和后4层)
                    if ("down_proj" in name) and (i <= 2 or i >= num_layers - 4):
                        name_tmp = name.replace(".", "_")
                        comp_left = torch.zeros(module.out_features, 1, device=dev)
                        comp_right = torch.ones(1, module.in_features, device=dev)
                        qlayer.register_parameter(
                            f"{name_tmp}_compensation_left",
                            nn.Parameter(comp_left),
                        )
                        qlayer.register_parameter(
                            f"{name_tmp}_compensation_right",
                            nn.Parameter(comp_right),
                        )

        # Step 5: 加载训练好的参数
        if i in abq_parameters:
            qlayer.load_state_dict(abq_parameters[i], strict=False)

        # Step 6: 应用量化
        smooth_and_quant_inplace(qlayer, args, True)

        # Step 7: 注册 scales 和 zeros
        qlayer.half()
        register_scales_and_zeros(qlayer)

        # 替换原始层
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

        if (i + 1) % 8 == 0 or i == num_layers - 1:
            print(f"  [{i+1}/{num_layers}] 完成")

    # 移动模型到 GPU
    print(f"[load_abq_quantized] 移动模型到 GPU...")
    model = model.cuda()
    model.eval()
    model.config.use_cache = False  # 避免 DynamicCache 兼容问题

    print(f"[load_abq_quantized] 加载完成")
    return model, tokenizer
