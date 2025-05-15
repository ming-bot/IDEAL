import torch
import sys
from tqdm import tqdm
import os
import torch.nn as nn
import gc

# def prepare_batch(batch, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     for key in batch:
#         batch[key] = batch[key].to(device)
def prepare_batch(batch, rank):
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch

def filter_layers(name, module, args):
    if not args.use_full_layer:
        if name not in args.target_layers:
            return False
        return True
    
    if not isinstance(module, nn.Linear):
        return False
    if not module.weight.requires_grad:
        return False
    if args.without_attention:
        if "self_attn" in name:
            return False
    if args.without_output:
        if "lm_head" in name:
            return False
    return True

def cal_grad(dataloader, model, output_dir, args, local_rank):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = next(model.parameters()).dtype

    # print("Starting calculating gradients!")
    # sys.stdout.flush()

    # Start computing ihvp for each sample
    layer_info = {}
    mlp_blocks = []
    for name, module in model.named_modules():
        if filter_layers(name, module, args):
            layer_info[name] = {}
            mlp_blocks.append([name, module])
    
    avg_dataset_grads = [[] for _ in range(len(mlp_blocks))]
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        model.zero_grad()
        prepare_batch(batch, local_rank)

        loss = model(**batch).loss
        loss.backward()
        # 记录bp值
        for name_p, p in model.named_parameters():
            if p.grad is not None:
                for name_m, module in model.named_modules():
                    if filter_layers(name_m, module, args) and module.weight is p:
                        layer_info[name_m]["weight_grad"] = p.grad.detach().clone()
    
        for i, [name, module] in enumerate(mlp_blocks):
            if len(avg_dataset_grads[i]) == 0:
                avg_dataset_grads[i] = layer_info[name]["weight_grad"].cpu() / len(dataloader)
            else:
                avg_dataset_grads[i] += layer_info[name]["weight_grad"].cpu() / len(dataloader)
            
        for name in layer_info.keys():
            del layer_info[name]["weight_grad"]
        torch.cuda.empty_cache()
        gc.collect()
    
    # print(len(avg_dataset_grads))
    # print("Every dimension of avg_dataset_grads shape is:")
    # for i in range(len(avg_dataset_grads)):
    #     print(avg_dataset_grads[i].shape)
    del mlp_blocks, layer_info
    torch.cuda.empty_cache()
    
    file_path = output_dir + ".pt"
    if local_rank == 0:
        torch.save(avg_dataset_grads, file_path)
    torch.distributed.barrier()
    
    del avg_dataset_grads
    torch.cuda.empty_cache()
    gc.collect()

    # print(f"Average dataset gradients have saved to {file_path}.")

    return file_path