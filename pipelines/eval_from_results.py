"""
从 JSONL 逐样本数据加载并计算评估指标

读取评估产出的 JSONL 文件 (每条含各选项 log-likelihood 和 token 长度)，
按任务类型用 acc (argmax logprob) 或 acc_norm (argmax logprob/length)
判定每条样本是否正确，聚合后输出指标表格。

用法:
    python pipelines/eval_from_results.py [--data_dir outputs/eval_results]
    python pipelines/eval_from_results.py --tsv   # 额外输出可粘贴到 Excel 的 TSV
"""

import os
import json
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_DISPLAY = {
    "piqa": "PiQA",
    "arc_easy": "ARC-e",
    "arc_challenge": "ARC-c",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "winogrande": "Winogrande",
    "openbookqa": "OBQA",
}

TASK_METRIC = {
    "piqa":          "acc",
    "arc_easy":      "acc_norm",
    "arc_challenge": "acc_norm",
    "boolq":         "acc",
    "hellaswag":     "acc_norm",
    "winogrande":    "acc",
    "openbookqa":    "acc_norm",
}

TASK_ORDER = ["piqa", "arc_easy", "arc_challenge", "boolq",
              "hellaswag", "winogrande", "openbookqa"]


def judge_sample(sample):
    """根据任务指标类型判定该样本是否正确"""
    task = sample["task"]
    gold = sample["gold"]
    logprobs = sample["choices_logprobs"]
    lengths = sample["choices_lengths"]
    metric = TASK_METRIC[task]

    if metric == "acc":
        prediction = max(range(len(logprobs)), key=lambda i: logprobs[i])
    else:  # acc_norm
        prediction = max(range(len(logprobs)),
                         key=lambda i: logprobs[i] / lengths[i])

    return 1 if prediction == gold else 0


def load_experiment(data_dir, exp_name):
    meta_path = os.path.join(data_dir, f"{exp_name}_meta.json")
    samples_path = os.path.join(data_dir, f"{exp_name}_samples.jsonl")

    with open(meta_path) as f:
        meta = json.load(f)

    task_correct = defaultdict(list)
    with open(samples_path) as f:
        for line in f:
            sample = json.loads(line)
            correct = judge_sample(sample)
            task_correct[sample["task"]].append(correct)

    return meta, task_correct


def compute_task_accuracies(task_correct):
    accs = {}
    for task in TASK_ORDER:
        vals = task_correct[task]
        accs[task] = sum(vals) / len(vals) * 100
    return accs


def print_detail(meta, accs):
    avg = sum(accs.values()) / len(accs)
    print(f"[{meta['display_name']}]  模型体积: {meta['model_size_gb']}GB")
    for task in TASK_ORDER:
        metric_tag = TASK_METRIC[task]
        print(f"  {TASK_DISPLAY[task]:>12s} ({metric_tag:>8s}): {accs[task]:.1f}%")
    print(f"  {'平均准确率':>20s}: {avg:.1f}%")
    print()


def print_table(rows):
    baseline = rows[0]
    baseline_avg = sum(baseline["accs"].values()) / len(baseline["accs"])
    baseline_size = baseline["size"]

    hdr_tasks = "".join(f"{TASK_DISPLAY[t]:>12s}" for t in TASK_ORDER)
    hdr = f"{'实验设置':<18s}{'模型体积':>10s}{'压缩率':>10s}{'保留率':>10s}{'平均准确率':>12s}{hdr_tasks}"
    sep = "─" * len(hdr)

    print(sep)
    print(hdr)
    print(sep)

    for r in rows:
        accs = r["accs"]
        size = r["size"]
        avg = sum(accs.values()) / len(accs)

        if r is baseline:
            comp_str = "0%"
            ret_str = "100.0%"
        else:
            comp_str = f"{(1 - size / baseline_size) * 100:.0f}%"
            ret_str = f"{avg / baseline_avg * 100:.1f}%"

        cols = f"{r['name']:<18s}{size:>8.2f}GB{comp_str:>10s}{ret_str:>10s}{avg:>11.1f}%"
        for t in TASK_ORDER:
            cols += f"{accs[t]:>11.1f}%"
        print(cols)

    print(sep)


def print_tsv(rows):
    baseline = rows[0]
    baseline_avg = sum(baseline["accs"].values()) / len(baseline["accs"])
    baseline_size = baseline["size"]

    headers = ["实验设置", "模型体积", "体积压缩率", "能力保留率", "平均准确率"]
    headers += [TASK_DISPLAY[t] for t in TASK_ORDER]
    print("\t".join(headers))

    for r in rows:
        accs = r["accs"]
        size = r["size"]
        avg = sum(accs.values()) / len(accs)

        if r is baseline:
            comp_str = "0%"
            ret_str = "100.0%"
        else:
            comp_str = f"{(1 - size / baseline_size) * 100:.0f}%"
            ret_str = f"{avg / baseline_avg * 100:.1f}%"

        vals = [r["name"], f"{size:.2f}GB", comp_str, ret_str, f"{avg:.1f}%"]
        vals += [f"{accs[t]:.1f}%" for t in TASK_ORDER]
        print("\t".join(vals))


def main():
    parser = argparse.ArgumentParser(description="从评估数据计算指标")
    parser.add_argument("--data_dir", default=os.path.join(
        PROJECT_ROOT, "outputs", "eval_results"))
    parser.add_argument("--tsv", action="store_true",
                        help="额外输出 TSV 格式 (可直接粘贴到 Excel)")
    args = parser.parse_args()

    exp_names = ["baseline", "w2a16_prune20"]
    table_rows = []

    print(f"数据目录: {args.data_dir}\n")

    for exp_name in exp_names:
        meta, task_correct = load_experiment(args.data_dir, exp_name)
        accs = compute_task_accuracies(task_correct)
        print_detail(meta, accs)
        table_rows.append({
            "name": meta["display_name"],
            "size": meta["model_size_gb"],
            "accs": accs,
        })

    print("=" * 40)
    print("  评估结果汇总")
    print("=" * 40 + "\n")
    print_table(table_rows)

    if args.tsv:
        print("\n--- TSV (可粘贴到表格软件) ---\n")
        print_tsv(table_rows)


if __name__ == "__main__":
    main()
