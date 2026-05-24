"""修复剪枝后模型的 config.json，使 intermediate_size 匹配实际权重维度"""
import json
import os
import sys
from safetensors import safe_open
import glob

model_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/pruned_models/Llama-2-7b-pruned-20pct"

# 读取实际权重维度
safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
if not safetensor_files:
    # 尝试 bin 格式
    import torch
    bin_files = glob.glob(os.path.join(model_dir, "*.bin"))
    state = torch.load(bin_files[0], map_location="cpu")
    for key in state:
        if "gate_proj.weight" in key:
            actual_intermediate = state[key].shape[0]
            break
else:
    with safe_open(safetensor_files[0], framework="pt") as f:
        for key in f.keys():
            if "gate_proj.weight" in key:
                actual_intermediate = f.get_tensor(key).shape[0]
                break

# 更新 config
config_path = os.path.join(model_dir, "config.json")
with open(config_path) as f:
    config = json.load(f)

old_val = config["intermediate_size"]
if old_val != actual_intermediate:
    config["intermediate_size"] = actual_intermediate
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"已修复 intermediate_size: {old_val} → {actual_intermediate}")
else:
    print(f"intermediate_size 已正确: {actual_intermediate}")
