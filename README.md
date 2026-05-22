# yj-proj: 统一 LLM 优化框架

面向 LLaMA-2-7B 的统一优化框架，整合 **量化**、**剪枝**、**KV Cache 压缩**、**LoRA 组合** 四种技术。

## 技术组成

| 模块 | 来源 | 技术 | 论文 |
|------|------|------|------|
| **ABQ-LLM** | ByteDance | 任意比特量化 (W2A8) | — |
| **FANG** | AAAI 2026 | 功能感知神经元分组剪枝 (20%) | Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping |
| **ReST-KV** | ICLR 2026 | KV Cache 驱逐 | ReST-KV: Residual-based Streaming KV Cache Compression |
| **LoraHub** | COLM 2024 | LoRA 模块动态组合（移植至 CausalLM） | LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition |

## 评估任务

7 个 zero-shot 下游任务：PiQA, ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, WinoGrande, OpenBookQA

## 硬件要求

- 单卡 NVIDIA A100/A800 (80GB VRAM)
- ~100GB 磁盘（模型 checkpoint + 中间产物）

## 环境配置

```bash
# 方式 1: conda (推荐)
conda env create -f environment.yml
conda activate yj-proj

# 方式 2: pip
pip install -r requirements.txt

# 可选: Flash Attention (需要 CUDA)
pip install flash-attn==2.5.8 --no-build-isolation

# 安装本地包
pip install -e third_party/rest-kv
pip install -e third_party/lorahub
```

### 关键版本

| 依赖 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.1.2 |
| Transformers | 4.42.4 |
| PEFT | 0.10.0 |
| lm-eval | 0.3.0 |
| Nevergrad | >=0.11.0 |

## 配置

修改 `configs/paths.yaml` 中的模型路径：

```yaml
model_path: "/your/path/to/Llama-2-7b"
```

或通过环境变量：

```bash
export MODEL_PATH=/your/path/to/Llama-2-7b
```

## 运行

### 6 个独立 Pipeline

```bash
# 1. 量化 (W2A8)
MODEL_PATH=/path/to/Llama-2-7b bash scripts/1_quantize_only.sh

# 2. 剪枝 (20%)
MODEL_PATH=/path/to/Llama-2-7b bash scripts/2_prune_only.sh

# 3. 剪枝 + 量化
MODEL_PATH=/path/to/Llama-2-7b bash scripts/3_prune_then_quantize.sh

# 4. KV Cache 压缩
MODEL_PATH=/path/to/Llama-2-7b bash scripts/4_kv_cache_compress.sh

# 5. LoRA 训练 + 组合
MODEL_PATH=/path/to/Llama-2-7b bash scripts/5_lora_pipeline.sh

# 6. 全局集成 (Prune → Quantize → LoRA → KV Cache)
MODEL_PATH=/path/to/Llama-2-7b bash scripts/6_full_pipeline.sh
```

### Python 接口

```bash
# 量化
python pipelines/run_quantize.py --model /path/to/Llama-2-7b --wbits 2 --abits 8

# 剪枝
python pipelines/run_prune.py --model /path/to/Llama-2-7b --pruning_ratio 0.2

# 剪枝+量化
python pipelines/run_prune_quantize.py --model /path/to/Llama-2-7b

# KV Cache
python pipelines/run_kv_cache.py --model_path /path/to/Llama-2-7b --method restkv

# LoRA (训练→组合→评估)
python pipelines/run_lora.py all --base_model /path/to/Llama-2-7b

# 全局集成 (支持 --skip_prune 等跳过已完成步骤)
python pipelines/run_full_pipeline.py --model /path/to/Llama-2-7b
```

## 项目结构

```
yj-proj/
├── README.md
├── environment.yml                     # conda 环境
├── requirements.txt                    # pip 依赖
├── configs/
│   └── paths.yaml                      # 路径和超参配置
├── unified/                            # 统一封装层
│   ├── eval_harness.py                 # 统一评估 (7 任务)
│   ├── model_utils.py                  # 模型加载工具
│   ├── lora_causal.py                  # LoraHub → CausalLM 移植
│   └── lora_train.py                   # per-task LoRA 训练
├── pipelines/                          # Python 入口
│   ├── run_quantize.py                 # Pipeline 1: 量化
│   ├── run_prune.py                    # Pipeline 2: 剪枝
│   ├── run_prune_quantize.py           # Pipeline 3: 剪枝+量化
│   ├── run_kv_cache.py                 # Pipeline 4: KV Cache
│   ├── run_lora.py                     # Pipeline 5: LoRA
│   └── run_full_pipeline.py            # Pipeline 6: 全局集成
├── scripts/                            # Shell 脚本
│   ├── 1_quantize_only.sh
│   ├── 2_prune_only.sh
│   ├── 3_prune_then_quantize.sh
│   ├── 4_kv_cache_compress.sh
│   ├── 5_lora_pipeline.sh
│   └── 6_full_pipeline.sh
├── third_party/                        # 第三方代码库
│   ├── ABQ-LLM/                        # 量化
│   ├── FANG/                           # 剪枝
│   ├── rest-kv/                        # KV Cache
│   └── lorahub/                        # LoRA (原始参考)
├── outputs/                            # 运行产物
│   ├── pruned_models/
│   ├── quantized_models/
│   ├── pruned_quantized_models/
│   ├── lora_adapters/
│   ├── lora_composed/
│   └── eval_results/
└── data/                               # 数据缓存
```

## 全局集成流水线

```
LLaMA-2-7B (原始)
    ↓ FANG 20% 结构化剪枝
剪枝后模型 (save_pretrained)
    ↓ ABQ-LLM W2A8 量化
量化后模型 (fake-quant FP16 权重)
    ↓ LoRA 训练 + Nevergrad 组合
LoRA-merged 模型
    ↓ ReST-KV monkeypatch (推理时)
最终模型 → evaluate_zero_shot (7 任务)
```

## 结果表格

| 技术 | PiQA | ARC-e | ARC-c | BoolQ | HellaSwag | WinoGrande | OBQA | 平均 | PPL(wiki) |
|------|------|-------|-------|-------|-----------|------------|------|------|-----------|
| Baseline (FP16) | — | — | — | — | — | — | — | — | — |
| W2A8 量化 | — | — | — | — | — | — | — | — | — |
| 剪枝 20% | — | — | — | — | — | — | — | — | — |
| 剪枝+量化 | — | — | — | — | — | — | — | — | — |
| KV Cache (ReST-KV) | — | — | — | — | — | — | — | — | — |
| LoRA 组合 | — | — | — | — | — | — | — | — | — |
| 全局集成 | — | — | — | — | — | — | — | — | — |

*运行实验后填入*

## 第三方代码修改说明

### ABQ-LLM (2 处修改)
- `algorithm/main.py:268-273`: 添加 net 名称模糊匹配，支持剪枝后模型路径
- `algorithm/generate_act_scale_shift.py:133,149`: 添加 `--net` 参数，支持显式指定网络名

### FANG、ReST-KV、LoraHub
- 无修改。LoRA 的 CausalLM 移植代码在 `unified/lora_causal.py`（新建文件）

## 引用

```
ABQ-LLM: Arbitrary-Bit Quantization Inference Acceleration for Large Language Models
FANG: Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping (AAAI 2026)
ReST-KV: Residual-based Streaming KV Cache Compression (ICLR 2026)
LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition (COLM 2024)
```

## License

各子项目遵循各自的开源协议 (Apache 2.0 / MIT)。
