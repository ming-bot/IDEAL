import argparse
import os
import numpy as np
import torch
import gc
from utils import load_model
from data_prepare import LoadDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from cal_KFAC import cal_ihvp, cal_grad, cal_influence
import warnings
import datetime
import pandas as pd
# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# 初始化分布式训练
def setup_distributed():
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=180000))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main(args):
    # 分布式环境初始化
    local_rank = setup_distributed()
    
    # load model
    model, tokenizer = load_model(args.model_path)
    # 将模型移动到 GPU 并包裹为 DDP
    # model = model.cuda(local_rank)
    # model = DDP(model, device_ids=[local_rank])

    train_sample_rate = 0.5
    val_sample_rate = 1.0
    # calculate KFAC;主要目的是计算海塞矩阵
    train_dataset = LoadDataset(all_file_paths=args.full_train,
                                tokenizer=tokenizer,
                                max_seq_length=1024,
                                sample_percentage=train_sample_rate)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8, drop_last=True)
    cal_ihvp(train_dataloader, model, args.save_path, args, local_rank)
    del train_dataloader
    del train_sampler
    del train_dataset
    torch.cuda.empty_cache()
    gc.collect()

    per_val_list = []
    # calculate validation gradient;主要目的是计算综合的验证集（定向/平均）的梯度
    validation_dataset = LoadDataset(all_file_paths=args.validation_path,
                                    tokenizer=tokenizer,
                                    max_seq_length=1024,
                                    sample_percentage=val_sample_rate)
    validation_sampler = DistributedSampler(validation_dataset, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=8)
    validation_grad_path = cal_grad(validation_dataloader, model, args.save_path + f"/val_avg_grad", args, local_rank)
    per_val_list.append(validation_grad_path)
    del validation_dataloader
    del validation_sampler
    del validation_dataset
    gc.collect()

    # 计算验证集子集的梯度
    for subset in list_subdirectories(args.validation_path):
        subset_path = os.path.join(args.validation_path, subset)
        if os.path.exists(subset_path):
            dataset = LoadDataset(all_file_paths=subset_path,
                                  tokenizer=tokenizer,
                                  max_seq_length=1024,
                                  sample_percentage=val_sample_rate)
            sampler = DistributedSampler(dataset, shuffle=True)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
            subset_grad_path = cal_grad(dataloader, model, args.save_path + f"/val_{subset}_grad", args, local_rank)
            per_val_list.append(subset_grad_path)
            # release memory
            del dataloader
            del sampler
            del dataset
            torch.cuda.empty_cache()
            gc.collect()
    
    if local_rank == 0:
        print(per_val_list)
    
    
    final_result = np.zeros((len(per_val_list), len(args.sub_train)))
    per_train_list = []
    for i, subset in enumerate(args.sub_train):
        subset_path = os.path.join(args.full_train, subset)
        if os.path.exists(subset_path):
            dataset = LoadDataset(all_file_paths=subset_path,
                                tokenizer=tokenizer,
                                max_seq_length=1024,
                                sample_percentage=train_sample_rate)
            sampler = DistributedSampler(dataset, shuffle=True)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
            avg_grad_path = cal_grad(dataloader, model, args.save_path + f"/train_{subset}_grad", args, local_rank)
            per_train_list.append(avg_grad_path)
            # release memory
            del dataloader
            del sampler
            del dataset
            torch.cuda.empty_cache()
            gc.collect()
    
    if local_rank == 0:
        print(per_train_list)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    
    for i in range(len(per_train_list)):
        for j in range(len(per_val_list)):
            influence_list = cal_influence(hessian_path = args.save_path,
                        train_grad_path = per_train_list[i],
                        validation_grad_path = per_val_list[j], 
                        local_rank = local_rank)
            total_sum = sum(influence_list)
            final_result[j, i] = total_sum.item()

            del influence_list
            torch.cuda.empty_cache()
            gc.collect()

    if local_rank == 0:
        print(final_result)
        df = pd.DataFrame(final_result, index=per_val_list, columns=per_train_list)
        df.to_csv(args.save_path + f"/influence.csv", index=True, header=True)

def list_subdirectories(directory):
    subdirectories = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            subdirectories.append(dir_name)
    return subdirectories

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=f"/mnt/petrelfs/mingchenlin/LLaMA-Factory/saves/neurips/llama3.1-8b/sft-IDEAL-M2-2_442")
    parser.add_argument("--full-train", type=str, default=f"/mnt/petrelfs/mingchenlin/DataEvolution/train_dataset_1_M1_M2-2")
    parser.add_argument("--validation-path", type=str, default=f"/mnt/petrelfs/mingchenlin/DataEvolution/validation/")
    parser.add_argument("--sub-train", type=list, default=[f"Mathematics","Coding","bbh","Instruction","TrustAI"])
    parser.add_argument("--save-path", type=str, default=f"/mnt/petrelfs/mingchenlin/DataEvolution/proportion_M2-2")
    parser.add_argument("--use-full-layer", type=bool, default=True)
    parser.add_argument("--target-layers", type=list, default=["model.layers.1.mlp.gate_proj", "model.layers.5.mlp.gate_proj", "model.layers.10.mlp.gate_proj", "model.layers.15.mlp.gate_proj"
    "model.layers.20.mlp.gate_proj", "model.layers.24.mlp.gate_proj", "model.layers.25.mlp.gate_proj", "model.layers.26.mlp.gate_proj", "model.layers.27.mlp.gate_proj", "model.layers.28.mlp.gate_proj"])
    parser.add_argument("--without-output", type=bool, default=True)
    parser.add_argument("--without-attention", type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Directory '{args.save_path}' created.")

    main(args)
    torch.distributed.destroy_process_group()