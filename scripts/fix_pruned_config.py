"""
扫描并报告 FANG 剪枝后模型的每层维度变化

用法:
    python scripts/fix_pruned_config.py /path/to/pruned_model
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from unified.pruned_model_loader import print_dimension_report

model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "outputs/pruned_models/Llama-2-7b-pruned-20pct")
print_dimension_report(model_dir)
