"""
ReST-KV 兼容层: 在 ABQ-LLM 量化模型上启用 KV cache 压缩

用法:
    from unified.restkv_compat import enable_restkv_on_quant_model

    model = load_abq_quantized(...)  # 加载量化模型
    enable_restkv_on_quant_model(model, max_capacity_prompts=128, window_size=32, ...)

原理:
    ReST-KV 原版通过 monkeypatch 替换 LlamaAttention.forward，
    但 ABQ-LLM 使用自定义的 QuantLlamaAttention，monkeypatch 不生效。
    本模块直接在 QuantLlamaAttention 实例上挂载 RestKVCluster，
    QuantLlamaAttention.forward 中已内置兼容逻辑 (hasattr(self, 'kv_cluster'))。
"""

import sys
import os


def enable_restkv_on_quant_model(
    model,
    max_capacity_prompts=128,
    window_size=32,
    kernel_size=21,
    pooling="adaptive",
    use_wo=True,
    use_ema=True,
    use_norm=False,
    use_pyramid=False,
    alpha=0.3,
    metric_mode="after",
    tau=1,
    scale=2000,
    merge=None,
):
    """
    在量化模型的每个 QuantLlamaAttention 上挂载 RestKVCluster

    Args:
        model: ABQ-LLM 量化后的模型 (含 QuantLlamaDecoderLayer)
        其余参数: ReST-KV 配置，默认值与 scripts/5_restkv_longbench.sh 一致
    """
    # 确保 rest-kv 代码可以被 import
    restkv_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "rest-kv"
    )
    if restkv_dir not in sys.path:
        sys.path.insert(0, restkv_dir)

    from restkv.restkv_utils import RestKVCluster

    # 找到所有 decoder layer
    try:
        layers = model.model.layers
    except AttributeError:
        raise ValueError("模型结构不兼容: 未找到 model.model.layers")

    num_hidden_layers = len(layers)
    count = 0

    for i, layer in enumerate(layers):
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            continue

        # 确保 layer_idx 存在
        if attn.layer_idx is None:
            attn.layer_idx = i

        attn.kv_cluster = RestKVCluster(
            num_hidden_layers=num_hidden_layers,
            layer_idx=i,
            window_size=window_size,
            max_capacity_prompt=max_capacity_prompts,
            kernel_size=kernel_size,
            pooling=pooling,
            merge=merge,
            use_wo=use_wo,
            use_norm=use_norm,
            use_ema=use_ema,
            use_pyramid=use_pyramid,
            alpha=alpha,
            metric_mode=metric_mode,
            tau=tau,
            scale=scale,
        )
        count += 1

    print(f"[ReST-KV] 已在 {count} 个 QuantLlamaAttention 层启用 KV cache 压缩")
    print(f"  max_capacity_prompts={max_capacity_prompts}, window_size={window_size}, "
          f"pooling={pooling}, metric_mode={metric_mode}")
