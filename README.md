# yj-proj: LLaMA-2-7B 统一优化框架

整合 **量化 (ABQ-LLM)** + **剪枝 (FANG)** + **KV Cache 压缩 (ReST-KV)** + **LoRA 组合 (LoraHub)** 四种 LLM 优化技术，面向 LLaMA-2-7B-chat。

## 技术组成

| 模块 | 技术 | 来源 | 说明 |
|------|------|------|------|
| **ABQ-LLM** | 任意比特量化 (W2A8) | ByteDance | 权重 2-bit、激活 8-bit，逐层训练 |
| **FANG** | 结构化剪枝 (20%) | AAAI 2026 | 功能感知神经元分组，mask 方式置零 |
| **ReST-KV** | KV Cache 压缩 | ICLR 2026 | 残差流重要性评分，推理时 monkeypatch |
| **LoraHub** | LoRA 动态组合 | COLM 2024 | Nevergrad 无梯度优化，已移植至 CausalLM |

## 四者兼容性

```
LLaMA-2-7B
   ├─ 1. FANG 剪枝 (mask, 不改维度)
   ├─ 2. LoRA 训练 + merge (可选)
   ├─ 3. ABQ-LLM W2A8 量化
   └─ 4. ReST-KV 推理时 KV 压缩
```

| 组合 | 兼容 | 说明 |
|------|------|------|
| 剪枝 + 量化 | 直接兼容 | 剪枝后模型结构不变，正常量化 |
| 剪枝 + LoRA | 直接兼容 | LoRA 挂载在标准 Linear 上 |
| LoRA + 量化 | 先 merge 再量化 | 量化前 LoRA 必须 merge 进模型 |
| 量化 + ReST-KV | 已适配 | QuantLlamaAttention 内置 kv_cluster 支持 |
| 四合一 | 剪枝 → LoRA merge → 量化 → ReST-KV 推理 |

## 环境配置

```bash
# conda (推荐)
conda env create -f environment.yml
conda activate yj-proj

# 或 pip
pip install -r requirements.txt
```

### 关键版本

| 依赖 | 版本 |
|------|------|
| Python | 3.9+ |
| PyTorch | 2.1.0+ |
| Transformers | 4.42.4 |
| PEFT | 0.10.0 |
| Nevergrad | >=0.11.0 |

> Flash Attention 可选，未安装时自动使用 eager attention。

## 快速开始

### 前提

将 LLaMA-2-7B-chat-hf 放到 `models/` 下，或通过 `MODEL_PATH` 环境变量指定。

### 运行脚本

所有脚本通过 `CUDA_VISIBLE_DEVICES` 控制 GPU：

```bash
# 1. Baseline 评估 (PPL)
CUDA_VISIBLE_DEVICES=0 bash scripts/1_eval_baseline.sh

# 2. 剪枝 + 评估
CUDA_VISIBLE_DEVICES=0 bash scripts/2_prune_and_eval.sh

# 3. 量化 + PPL 评估
CUDA_VISIBLE_DEVICES=1 bash scripts/3_quantize_and_eval.sh

# 4. 剪枝后量化 (需先跑 script 2)
CUDA_VISIBLE_DEVICES=2 bash scripts/4_prune_then_quantize.sh

# 5. ReST-KV LongBench 评估 (FullKV vs ReST-KV)
CUDA_VISIBLE_DEVICES=3 bash scripts/5_restkv_longbench.sh

# 6. LoraHub 组合测试
CUDA_VISIBLE_DEVICES=4 bash scripts/6_lorahub_test.sh

# 7. 三合一兼容测试 (剪枝+量化+ReST-KV)
CUDA_VISIBLE_DEVICES=5 bash scripts/7_test_prune_quant_restkv.sh
```

### 常用参数

```bash
# 量化: 调整 epochs 和 batch_size
EPOCHS=20 BATCH_SIZE=8 CUDA_VISIBLE_DEVICES=1 bash scripts/3_quantize_and_eval.sh

# ReST-KV: 指定数据集和样本数
DATASETS=hotpotqa MAX_EXAMPLES=20 ATTN=eager bash scripts/5_restkv_longbench.sh
```

## 项目结构

```
yj-proj/
├── scripts/                            # Shell 入口脚本
│   ├── 1_eval_baseline.sh              # Baseline PPL 评估
│   ├── 2_prune_and_eval.sh             # FANG 20% 剪枝 + 评估
│   ├── 3_quantize_and_eval.sh          # ABQ-LLM W2A8 量化 + PPL
│   ├── 4_prune_then_quantize.sh        # 剪枝模型再量化
│   ├── 5_restkv_longbench.sh           # ReST-KV LongBench 评估
│   ├── 6_lorahub_test.sh              # LoraHub 组合测试
│   ├── 7_test_prune_quant_restkv.sh    # 三合一兼容测试
│   └── download_longbench.sh           # 下载 LongBench 数据
│
├── pipelines/                          # Python Pipeline 入口
│   ├── run_quantize.py                 # 量化
│   ├── run_prune.py                    # 剪枝 (5 阶段)
│   ├── run_prune_quantize.py           # 剪枝 → 量化
│   ├── run_kv_cache.py                 # ReST-KV
│   ├── run_lora.py                     # LoRA (train/compose/evaluate)
│   ├── run_eval_only.py                # 独立评估 (PPL + zero-shot)
│   ├── run_full_pipeline.py            # 全流程集成
│   ├── run_comparison.py               # Baseline vs 优化对比
│   ├── run_demo.py                     # 生成 Demo
│   ├── create_dummy_loras.py           # 创建随机 LoRA (测试用)
│   ├── run_lorahub_test.py             # LoraHub 权重选择测试
│   └── test_three_in_one.py            # 剪枝+量化+ReST-KV 兼容测试
│
├── unified/                            # 统一封装层
│   ├── model_utils.py                  # 模型加载 (baseline/pruned/quantized)
│   ├── eval_harness.py                 # 统一评估 (7 个 zero-shot 任务)
│   ├── abq_model_loader.py             # ABQ-LLM 量化模型加载器
│   ├── pruned_model_loader.py          # FANG 剪枝模型加载器
│   ├── lora_causal.py                  # LoraHub CausalLM 移植
│   ├── lora_train.py                   # per-task LoRA 训练
│   └── restkv_compat.py                # ReST-KV + 量化模型兼容层
│
├── third_party/                        # 第三方代码 (含适配修改)
│   ├── ABQ-LLM/                        # 量化
│   ├── FANG/                           # 剪枝
│   ├── rest-kv/                        # KV Cache 压缩
│   └── lorahub/                        # LoRA 组合 (原始参考)
│
├── configs/paths.yaml                  # 路径和超参配置
├── outputs/                            # 运行产物
│   ├── pruned_models/                  # 剪枝后模型
│   ├── quantized_models/               # 量化后模型
│   ├── pruned_quantized_models/        # 剪枝+量化模型
│   ├── lora_adapters/                  # LoRA adapter
│   └── eval_results/                   # 评估结果 JSON
└── data/                               # 数据缓存
```

## 输出路径命名规则

```
outputs/
├── pruned_models/Llama-2-7b-pruned-20pct/
├── quantized_models/Llama-2-7b-w2a8-ep40-bs4/
│   └── log/abq_parameters.pth
├── pruned_quantized_models/Llama-2-7b-pruned20-w2a8-ep40-bs4/
│   └── log/abq_parameters.pth
└── lora_adapters/{task_name}/
```

短名规则: `Llama-2-7b-chat-hf` → `Llama-2-7b`

## 评估

### PPL (WikiText-2)

```bash
python pipelines/run_eval_only.py \
    --model outputs/pruned_models/Llama-2-7b-pruned-20pct \
    --name "Pruned-20%" --skip_zeroshot
```

### 量化模型评估

```bash
python pipelines/run_eval_only.py \
    --model models/Llama-2-7b-chat-hf \
    --abq_params outputs/quantized_models/Llama-2-7b-w2a8-ep40-bs4/log/abq_parameters.pth \
    --name "W2A8" --skip_zeroshot
```

### 7 个 Zero-shot 任务

PiQA, ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, WinoGrande, OpenBookQA

## 第三方代码修改说明

### ABQ-LLM

| 文件 | 修改 |
|------|------|
| `models/int_llama_layer.py` | QuantLlamaAttention 兼容 DynamicCache (transformers 4.42)；支持非均匀剪枝维度推断；内置 ReST-KV kv_cluster 可选挂载点 |
| `models/LMClass.py` | `_model_call` 添加 `use_cache=False` 避免 DynamicCache 报错 |
| `algorithm/main.py` | net 名称模糊匹配，支持剪枝后模型路径 |

### FANG

| 文件 | 修改 |
|------|------|
| `main.py` | 模型保存移到评估之前 (防止 OOM 导致未保存) |
| `lib/eval.py` | zero-shot eval batch_size 改为 1 (防止 OOM) |

### ReST-KV

| 文件 | 修改 |
|------|------|
| `restkv/llama_model.py` | flash_attn import 改为可选 (try/except)；eager/sdpa forward 补充 Wo 参数 |
| `restkv/llama_model_think.py` | 同上 flash_attn 兼容 |
| `run_longbench.py` | 支持 `LONGBENCH_DATASETS` 和 `LONGBENCH_DATA_DIR` 环境变量 |

### LoraHub

无直接修改。CausalLM 移植在 `unified/lora_causal.py`。

## Python API 示例

### 加载量化模型 + 启用 ReST-KV

```python
from unified.abq_model_loader import load_abq_quantized_model
from unified.restkv_compat import enable_restkv_on_quant_model

# 加载剪枝+量化模型
model, tokenizer = load_abq_quantized_model(
    base_model_path="outputs/pruned_models/Llama-2-7b-pruned-20pct",
    abq_params_path="outputs/pruned_quantized_models/.../log/abq_parameters.pth",
)

# 启用 ReST-KV KV Cache 压缩
enable_restkv_on_quant_model(model, max_capacity_prompts=128, window_size=32)

# 生成
model.config.use_cache = True
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
```

### LoraHub 组合

```python
from unified.lora_causal import lorahub_learning

weights, model, tokenizer = lorahub_learning(
    lora_module_list=["outputs/lora_adapters/math", "outputs/lora_adapters/qa"],
    example_inputs=["What is 2+2?", ...],
    example_outputs=["4", ...],
    max_inference_step=40,
    model_name_or_path="models/Llama-2-7b-chat-hf",
)
```

## 已知限制

- 量化训练逐层串行，无法多卡并行 (层间依赖)
- LoraHub 是 task-level 组合，非 sample-aware
- 服务器无法直接访问 HuggingFace Hub，数据需手动下载
- flash-attn 未安装时使用 eager attention，性能较慢

## 引用

```
ABQ-LLM: Arbitrary-Bit Quantization Inference Acceleration for Large Language Models
FANG: Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping (AAAI 2026)
ReST-KV: Residual-based Streaming KV Cache Compression (ICLR 2026)
LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition (COLM 2024)
```

## License

各子项目遵循各自的开源协议 (Apache 2.0 / MIT)。
