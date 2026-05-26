"""
Pipeline 2: FANG 结构化剪枝 (20%)
五阶段 pipeline: 聚类 → 重要性评分 → 神经元分组 → 稀疏度分配 → 剪枝+评估

用法:
    python pipelines/run_prune.py \
        --model /path/to/Llama-2-7b \
        --pruning_ratio 0.2 \
        --save_model outputs/pruned_models/Llama-2-7b-pruned-20pct
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FANG_DIR = os.path.join(PROJECT_ROOT, "third_party", "FANG")


def run_command(cmd, cwd=None):
    """执行命令并实时输出"""
    print(f"\n>>> {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        print(f"命令执行失败，返回码: {proc.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FANG 结构化剪枝 Pipeline")
    parser.add_argument("--model", type=str, required=True, help="LLaMA-2-7B 模型路径")
    parser.add_argument("--pruning_ratio", type=float, default=0.2)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--num_experts_mlp", type=int, default=7)
    parser.add_argument("--pca_components", type=int, default=64)
    parser.add_argument("--Lambda", type=float, default=0.1,
                        help="稀疏度分配的 Lambda 参数 (建议 0.5 * pruning_ratio)")
    parser.add_argument("--tau", type=float, default=9, help="softmax 温度")
    parser.add_argument("--prune_method", type=str, default="noiter_group_weight",
                        choices=["ziplm", "noiter", "noiter_group", "noiter_group_weight"])
    parser.add_argument("--save_model", type=str,
                        default=None)
    parser.add_argument("--work_dir", type=str, default=None,
                        help="中间文件保存目录（默认 outputs/pruned_models/work）")
    parser.add_argument("--skip_stages", type=str, default="",
                        help="跳过的阶段，逗号分隔 (1,2,3,4)")
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Mask 模式: 置零而非物理移除，保持维度不变，兼容后续量化")
    args = parser.parse_args()

    pr_int = int(args.pruning_ratio * 100)
    if args.save_model is None:
        model_name = os.path.basename(args.model.rstrip("/"))
        args.save_model = os.path.join(PROJECT_ROOT, "outputs", "pruned_models", f"{model_name}-pruned-{pr_int}pct")

    if args.work_dir is None:
        args.work_dir = os.path.join(PROJECT_ROOT, "outputs", "pruned_models", "work")
    os.makedirs(args.work_dir, exist_ok=True)

    skip = set(args.skip_stages.split(",")) if args.skip_stages else set()
    lam_int = int(args.Lambda * 100)

    # Stage 1: K-means 聚类
    if "1" not in skip:
        print("=" * 60)
        print("  Stage 1/5: K-means 聚类")
        print("=" * 60)
        cluster_dir = os.path.join(args.work_dir, "clustering")
        run_command([
            sys.executable, "hidden_feature_clustering_wikitext.py",
            "--model", args.model,
            "--nsamples", str(args.nsamples),
            "--save_path", cluster_dir,
            "--mlp_cluster",
            "--num_experts_mlp", str(args.num_experts_mlp),
            "--pca_components", str(args.pca_components),
            "--n_jobs", "16",
            "--max_iter", "25",
        ], cwd=FANG_DIR)

    # Stage 2: 梯度重要性评分
    if "2" not in skip:
        print("=" * 60)
        print("  Stage 2/5: 梯度重要性评分")
        print("=" * 60)
        grads_dir = os.path.join(args.work_dir, "grads")
        cluster_dir = os.path.join(args.work_dir, "clustering")
        run_command([
            sys.executable, "split_gradient_get_grads_wikitext.py",
            "--model", args.model,
            "--nsamples", str(args.nsamples),
            "--save_path", grads_dir,
            "--mlp_cluster",
            "--pca_components", str(args.pca_components),
            "--gate_weights_file", os.path.join(cluster_dir, "gate_weights.pt"),
        ], cwd=FANG_DIR)

    # Stage 3: 神经元分组
    if "3" not in skip:
        print("=" * 60)
        print("  Stage 3/5: 神经元分组")
        print("=" * 60)
        neuron_dir = os.path.join(args.work_dir, "neuron_groups")
        grads_dir = os.path.join(args.work_dir, "grads")
        # 查找 importance scores 文件
        score_file = os.path.join(grads_dir, "importance_scores_mlp_up.pt")
        if not os.path.exists(score_file):
            # 尝试其他可能的文件名
            for candidate in ["importance_scores.pt", "importance_scores_mlp.pt"]:
                alt = os.path.join(grads_dir, candidate)
                if os.path.exists(alt):
                    score_file = alt
                    break

        run_command([
            sys.executable, "split_gradient_residual.py",
            "--model_path", args.model,
            "--save_path", neuron_dir,
            "--score_file", score_file,
            "--mlp_cluster",
            "--num_experts_mlp", str(args.num_experts_mlp),
            "--num_experts_residual_mlp", "1",
        ], cwd=FANG_DIR)

    # Stage 4: 稀疏度分配
    if "4" not in skip:
        print("=" * 60)
        print("  Stage 4/5: 稀疏度分配")
        print("=" * 60)
        sparsity_dir = os.path.join(args.work_dir, "sparsity")
        run_command([
            sys.executable, "FC_sparsity_allocation.py",
            "--model", args.model,
            "--nsamples", str(args.nsamples),
            "--pruning_ratio", str(args.pruning_ratio),
            "--Lambda", str(args.Lambda),
            "--save_dir", sparsity_dir,
        ], cwd=FANG_DIR)

    # Stage 5: 剪枝 + 评估
    print("=" * 60)
    print("  Stage 5/5: 剪枝 + 评估")
    print("=" * 60)
    neuron_dir = os.path.join(args.work_dir, "neuron_groups")
    cluster_dir = os.path.join(args.work_dir, "clustering")
    sparsity_dir = os.path.join(args.work_dir, "sparsity")
    sparsity_file = os.path.join(sparsity_dir, f"FC_sp{pr_int}_lambda{lam_int}.pt")

    cmd = [
        sys.executable, "main.py",
        "--model", args.model,
        "--nsamples", str(args.nsamples),
        "--pruning_ratio", str(args.pruning_ratio),
        "--tau", str(args.tau),
        "--prune_method", args.prune_method,
        "--group_method", "group_res",
        "--neuron_indices_file", os.path.join(neuron_dir, "neuron_residual_indices.pt"),
        "--cluster_center_file", os.path.join(cluster_dir, "gate_weights.pt"),
        "--sparsity_allocation_file", sparsity_file,
        "--save_model", args.save_model,
    ]
    if args.use_mask:
        cmd.append("--use_mask")
    run_command(cmd, cwd=FANG_DIR)

    print(f"\n剪枝完成！模型保存至: {args.save_model}")


if __name__ == "__main__":
    main()
