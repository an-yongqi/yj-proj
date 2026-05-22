"""
统一模型加载工具
支持加载原始、剪枝后、量化后的 LLaMA-2 模型
"""

import os
import sys
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_config(config_path="configs/paths.yaml"):
    """加载项目配置"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, config_path)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model_and_tokenizer(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
):
    """
    统一的模型加载接口

    Args:
        model_path: HuggingFace 模型路径或本地路径
        torch_dtype: 数据类型
        device_map: 设备映射策略
        attn_implementation: 注意力实现方式 (eager/sdpa/flash_attention_2)

    Returns:
        (model, tokenizer) 元组
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()

    # 设置 seqlen（FANG 和 ABQ-LLM 都需要）
    if not hasattr(model, "seqlen"):
        model.seqlen = 2048

    return model, tokenizer


def detect_net_name(model_path):
    """
    从模型路径或 config 中检测网络架构名称
    用于 ABQ-LLM 的 --net 参数

    Args:
        model_path: 模型路径

    Returns:
        str: 匹配的网络名称，如 "Llama-2-7b"
    """
    net_choices = [
        "opt-125m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b",
        "opt-30b", "opt-66b",
        "llama-7b", "llama-13b", "llama-30b", "llama-65b",
        "Llama-2-7b", "Llama-2-13b", "Llama-2-70b",
        "Llama-2-7b-chat", "Llama-2-13b-chat",
        "falcon-180b", "falcon-7b",
        "mixtral-8x7b",
    ]

    path_name = model_path.rstrip("/").split("/")[-1]

    # 精确匹配
    if path_name in net_choices:
        return path_name

    # 模糊匹配
    path_lower = path_name.lower()
    for net in net_choices:
        if net.lower() in path_lower:
            return net

    # 通过 config 检测
    try:
        config = AutoConfig.from_pretrained(model_path)
        arch = config.architectures[0].lower() if config.architectures else ""
        if "llama" in arch:
            # 根据 hidden_size 判断模型大小
            hidden = config.hidden_size
            if hidden <= 4096:
                return "Llama-2-7b"
            elif hidden <= 5120:
                return "Llama-2-13b"
            else:
                return "Llama-2-70b"
        elif "opt" in arch:
            return "opt-6.7b"  # 默认
        elif "falcon" in arch:
            return "falcon-7b"
    except Exception:
        pass

    # 默认
    return "Llama-2-7b"
