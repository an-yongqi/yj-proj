import argparse
import os 
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from importlib.metadata import version

from lib.prune import ziplm_prune, check_sparsity, ziplm_noiter, ziplm_noiter_group_prune, ziplm_noiter_group_weight_prune
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.seqlen = 2048
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0.2, help='Pruning ratio.')
    parser.add_argument('--tau', type=float, default=None, help='temperature for softmax')
    parser.add_argument("--prune_method", type=str, default="ziplm",
                         choices=["ziplm", "noiter", "noiter_group", "noiter_group_weight"])
    parser.add_argument("--group_method", type=str, default="group_res",
                         choices=["group_res", "group_nores", "onlyres"])
    parser.add_argument("--attn_group", action="store_true", help="use group method for attn head")
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument('--unstr', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument("--eval_dataset", type=str, default='wikitext2', choices=['c4', 'ptb', 'wikitext2'])
    parser.add_argument("--prune_calibration", type=str, default='wikitext2', choices=['c4', 'ptb', 'wikitext2'])
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--neuron_indices_file", default=None, type=str, help="file containing neuron indices for group")
    parser.add_argument("--cluster_center_file", default=None, type=str, help="file containing cluster centers")
    parser.add_argument("--sparsity_allocation_file", default=None, type=str, help="file containing sparsity ratios")
    args = parser.parse_args()
    
    # Setting seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model or "70b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    print("pruning starts")
    if args.prune_method == "ziplm":
        ziplm_prune(args, model, tokenizer, device)
    elif args.prune_method == "noiter":
        ziplm_noiter(args, model, tokenizer, device)
    elif args.prune_method == "noiter_group":
        ziplm_noiter_group_prune(args, model, tokenizer, device)
    elif args.prune_method == "noiter_group_weight":
        ziplm_noiter_group_weight_prune(args, model, tokenizer, device)
    else:
        raise ValueError("Invalid prune method")
    # Check the sparsity of the model
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")
    print("*"*30)

    # Evaluate the model
    model_name = args.model.split("/")[-1]
    if args.eval:
        ppl = eval_ppl(model, tokenizer, device, 'wikitext2')
        print(f"ppl on wikitext {ppl:.2f}")
        # PTB 数据集需要联网下载，离线环境跳过
        # ppl = eval_ppl(model, tokenizer, device, 'ptb')
        # print(f"ppl on ptb {ppl:.2f}")
    if args.eval_zero_shot:

        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        for task in task_list:
            result = results['results'][task]
            if 'acc_norm' in result:
                acc = result['acc_norm'] * 100
                print(f"{task}: {acc:.2f}")
            else:
                acc = result['acc'] * 100
                print(f"{task}: {acc:.2f}")
    # Save the model
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        # 更新 config 以反映剪枝后的实际维度
        # MLP: gate_proj/up_proj 的 out_features = intermediate_size
        first_mlp = model.model.layers[0].mlp
        actual_intermediate = first_mlp.gate_proj.out_features
        if actual_intermediate != model.config.intermediate_size:
            print(f"更新 config.intermediate_size: {model.config.intermediate_size} → {actual_intermediate}")
            model.config.intermediate_size = actual_intermediate
        # Attention: 检查 num_heads 是否变化
        first_attn = model.model.layers[0].self_attn
        actual_hidden = first_attn.o_proj.out_features
        if actual_hidden != model.config.hidden_size:
            print(f"更新 config.hidden_size: {model.config.hidden_size} → {actual_hidden}")
            model.config.hidden_size = actual_hidden
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    

if __name__ == '__main__':
    main()