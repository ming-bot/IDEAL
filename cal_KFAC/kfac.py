'''
Developed by: Ming
'''
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time
import einops
import warnings
import gc
import logging
# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

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

# Create forward hook to record forward input
def forward_hook_fn(layer_info, name):
    def hook(module, input, output):
        layer_info[name]["input"] = input[0].detach().clone()
    return hook

# Create backward hook to record output gradient
def backward_hook_fn(layer_info, name):
    def hook(module, grad_input, grad_output):
        if name in layer_info:
            layer_info[name]["grad_output"] = grad_output[0].detach().clone()
    return hook

def get_Q_list(kfac_input_covs, kfac_grad_covs, mlp_blocks, output_dir, dtype, local_rank):
    if local_rank == 0:
        print("Entering Getting Q Lists")
        sys.stdout.flush()
    # kfac_input_covs = torch.load(os.path.join(output_dir, "kfac_input_covs.pt"))
    q_a_list = []
    # q_a_t_list = []
    for i in tqdm(range(len(mlp_blocks)-1, -1, -1)):
        # # print(f"Getting Layer {i}'s Q_list")
        # sys.stdout.flush()
        input_covs_gpu = kfac_input_covs[i].cuda()
        _, q_a = torch.linalg.eigh(input_covs_gpu)
        q_a_cpu = q_a.to(dtype).cpu()
        q_a_list.append(q_a_cpu)
        # q_a_t_cpu = q_a_t.cpu()
        # q_a_t_list.append(q_a_t_cpu)
        # del kfac_input_covs[i], input_covs_gpu, q_a, q_a_t
        del kfac_input_covs[i], input_covs_gpu, q_a
        torch.cuda.empty_cache()
        gc.collect()
    torch.distributed.barrier()
    del kfac_input_covs
    gc.collect()
    
    q_a_list.reverse()
    # q_a_t_list.reverse()
    # 存储
    file_path = os.path.join(output_dir, "q_a_list.pt")
    if local_rank == 0:
        torch.save(q_a_list, file_path)
    torch.distributed.barrier()
    # file_path = os.path.join(output_dir, "q_a_t_list.pt")
    # torch.save(q_a_t_list, file_path)
    # del q_a_list, q_a_t_list
    del q_a_list
    gc.collect()

    # kfac_grad_covs = torch.load(os.path.join(output_dir, "kfac_grad_covs.pt"))
    q_s_list = []
    # q_s_t_list = []
    for i in tqdm(range(len(mlp_blocks)-1, -1, -1)):
        # # print(f"Getting Layer {i}'s Q_list")
        # sys.stdout.flush()
        grad_covs_gpu = kfac_grad_covs[i].cuda()
        # q_s, _, q_s_t = torch.svd(grad_covs_gpu)
        _, q_s = torch.linalg.eigh(grad_covs_gpu)
        q_s_cpu = q_s.to(dtype).cpu()
        # q_s_t_cpu = q_s_t.cpu()
        q_s_list.append(q_s_cpu)
        # q_s_t_list.append(q_s_t_cpu)
        # del kfac_grad_covs[i], grad_covs_gpu, q_s, q_s_t
        del kfac_grad_covs[i], grad_covs_gpu, q_s
        torch.cuda.empty_cache()
        gc.collect()
    del kfac_grad_covs
    gc.collect()
    q_s_list.reverse()
    # q_s_t_list.reverse()
    file_path = os.path.join(output_dir, "q_s_list.pt")
    if local_rank == 0:
        torch.save(q_s_list, file_path)
    torch.distributed.barrier()
    # file_path = os.path.join(output_dir, "q_s_t_list.pt")
    # torch.save(q_s_t_list, file_path)
    # del q_s_list, q_s_t_list
    del q_s_list
    gc.collect()

    print("Successfully Getting Q Lists!")
    sys.stdout.flush()

# Compute lambda_ii for each layer
def get_lambda_ii_list(model, device, dataloader, mlp_blocks, q_a_list, q_s_list, rank, args):
    squared_projections_sum = [0.0] * len(mlp_blocks)
    # print("Getting lambda ii for every layer!")
    # sys.stdout.flush()

    lambda_ii_avg_list = [0.0] * len(mlp_blocks)

    layer_info = {}
    for name, module in model.named_modules():
        if filter_layers(name, module, args):
            layer_info[name] = {}

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch_grads = [[] for _ in range(len(mlp_blocks))]
        model.zero_grad()
        prepare_batch(batch, rank)

        loss = model(**batch).loss
        loss.backward()
        for name_p, p in model.named_parameters():
            if p.grad is not None:
                for name_m, module in model.named_modules():
                    if filter_layers(name_m, module, args) and module.weight is p:
                        layer_info[name_m]["weight_grad"] = p.grad.detach().clone()

        for i, [name, module] in enumerate(mlp_blocks):
            batch_grads[i].append(layer_info[name]["weight_grad"])

        squared_projections_sum = accumulate_squared_projections_sum(batch_grads, squared_projections_sum, q_a_list,
                                                                     q_s_list)

    lambda_ii_avg_list = [projections_sum_layer / len(dataloader) for projections_sum_layer in squared_projections_sum]
    # print("Successfully Get lambda_ii_avg_list!")
    # sys.stdout.flush()
    del squared_projections_sum, layer_info, batch_grads
    torch.cuda.empty_cache()
    gc.collect()

    return lambda_ii_avg_list

# Accumulate squared_projections_sum
def accumulate_squared_projections_sum(batch_grads, squared_projections_sum, q_a_list, q_s_list):
    for layer_num in range(len(batch_grads)):
        n_examples = len(batch_grads[0])
        for j in range(n_examples):
            dtheta = batch_grads[layer_num][j]
            dtheta = dtheta.cuda()
            q_a = q_a_list[layer_num].cuda()
            q_s = q_s_list[layer_num].cuda()
            result = (q_s @ dtheta @ q_a.T).view(-1)
            squared_projections_sum[layer_num] += (result ** 2).cpu()
        del q_a, q_s, dtheta, result
        torch.cuda.empty_cache()

    return squared_projections_sum

# main function to calculate K-FAC
def cal_ihvp(dataloader, model, output_dir, args, local_rank):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = next(model.parameters()).dtype

    # Get kfac_input_covs, kfac_grad_covs, mlp_blocks
    # print("Starting Loading ekfac factors!")
    # sys.stdout.flush()

    total_length = len(dataloader)
    kfac_input_covs = []
    kfac_grad_covs = []
    mlp_blocks = []
    
    layer_info = {}
    hook_handle = []
    for name, module in model.named_modules():
        if filter_layers(name, module, args):
            # Create ekfac_factors and pseudo_grads
            kfac_input_covs.append(torch.zeros(module.in_features, module.in_features))
            kfac_grad_covs.append(torch.zeros(module.out_features, module.out_features))
            mlp_blocks.append([name, module])
            # Register hooks
            layer_info[name] = {}
            hook_forward = module.register_forward_hook(forward_hook_fn(layer_info, name))
            hook_backward = module.register_full_backward_hook(backward_hook_fn(layer_info, name))
            hook_handle.append(hook_forward)
            hook_handle.append(hook_backward)
    
    for batch in tqdm(dataloader, total=total_length):
        model.zero_grad()
        prepare_batch(batch, local_rank)
        loss = model(**batch).loss
        # loss = model(input_ids=input, attention_mask=attention, labels=labels).loss

        # Compute input covariance matrix A_{l-1}
        for i, [name, module] in enumerate(mlp_blocks):
            input_batch = layer_info[name]["input"]
            # # print(f"No.{i} input batch: {input_batch.shape}")
            input_cov = torch.einsum("...i,...j->ij", input_batch, input_batch)
            input_cov_cpu = input_cov.cpu()
            kfac_input_covs[i] += input_cov_cpu / total_length
            del layer_info[name]["input"], input_cov, input_cov_cpu, input_batch
            torch.cuda.empty_cache()
            gc.collect()
        
        # # print(f"kfac_input_covs[0] shape:{kfac_input_covs[0].shape}")
        # sys.stdout.flush()
        loss.backward()

        # Compute output gradient covariance matrix S_l and weight gradient
        for i, [name, module] in enumerate(mlp_blocks):
            grad_output_batch = layer_info[name]["grad_output"]

            grad_cov = torch.einsum("...i,...j->ij", grad_output_batch, grad_output_batch)
            grad_cov_cpu = grad_cov.cpu()
            kfac_grad_covs[i] += grad_cov_cpu / total_length
            del layer_info[name]["grad_output"], grad_cov, grad_cov_cpu, grad_output_batch
            torch.cuda.empty_cache()
            gc.collect()
    '''
    file_path = os.path.join(output_dir, "kfac_input_covs.pt")
    torch.save(kfac_input_covs, file_path)
    file_path = os.path.join(output_dir, "kfac_grad_covs.pt")
    torch.save(kfac_grad_covs, file_path)
    
    del kfac_input_covs, kfac_grad_covs
    gc.collect()
    '''

    # print("Finish Get EFPG!")
    # sys.stdout.flush()
    # print(local_rank, "Finish Get EFPG!")
    # sys.stdout.flush()
    # torch.distributed.barrier()

    # delete IDLE resource
    del layer_info
    for hooks in hook_handle:
        hooks.remove()
    del hook_handle
    torch.cuda.empty_cache()
    gc.collect()

    # Get Q_list (length of MLP number)
    if local_rank == 0:
        print("Calculating Q List!")
        sys.stdout.flush()
    
    get_Q_list(kfac_input_covs, kfac_grad_covs, mlp_blocks, output_dir, dtype, local_rank)

    if local_rank == 0:
        print("Finish Getting Q list")
        sys.stdout.flush()
    
    q_a_list = torch.load(os.path.join(output_dir, "q_a_list.pt"))
    q_s_list = torch.load(os.path.join(output_dir, "q_s_list.pt"))
    
    # Get lambda_ii list (length of MLP number)
    if local_rank == 0:
        print("Start Calculating Lambda!")
        sys.stdout.flush()
    lambda_ii_avg_list = get_lambda_ii_list(model, device, dataloader, mlp_blocks, q_a_list, q_s_list, local_rank, args)
    # 存储
    file_path = os.path.join(output_dir, "lambda_list.pt")
    if local_rank == 0:
        torch.save(lambda_ii_avg_list, file_path)
    torch.distributed.barrier()

    ### ------------------------------------------------------------------------------------- ###
    del q_a_list, q_s_list, lambda_ii_avg_list
    torch.cuda.empty_cache()
    gc.collect()
    if local_rank == 0:
        print("Finish Getting Lambda list")
        sys.stdout.flush()


def cal_influence(hessian_path, train_grad_path, validation_grad_path, local_rank):
    q_a_list = torch.load(os.path.join(hessian_path, "q_a_list.pt"))
    q_s_list = torch.load(os.path.join(hessian_path, "q_s_list.pt"))
    # print("Finish Loading Q list")
    # sys.stdout.flush()

    lambda_ii_avg_list = torch.load(os.path.join(hessian_path, "lambda_list.pt"))
    if local_rank == 0:
        print("Finish Loading Lambda list")
        sys.stdout.flush()

    try:
        sub_train_grad_list = torch.load(train_grad_path)
        if local_rank == 0:
            print("Finish Loading Train Grad list")
            sys.stdout.flush()
    except:
        raise Exception("Train Grad list not found")

    try:
        validation_grad_list = torch.load(validation_grad_path)
        if local_rank == 0:
            print("Finish Loading Validation Grad list")
            sys.stdout.flush()
    except:
        raise Exception("Validation Grad list not found")

    if local_rank == 0:
        print('-'*20)
        print("Start Calculating Influence")
        sys.stdout.flush()

    damping = 0.000
    # time_in = time.time()
    influence_list = [0.0] * len(sub_train_grad_list)
    for i in range(len(sub_train_grad_list)):
        V = sub_train_grad_list[i].cuda()
        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, q_s = q_a_list[i], q_s_list[i]

        # Calculate the EK-FAC diagonal damping inverse matrix.
        lambda_ii = lambda_ii_avg_list[i]
        ekfacDiag_damped_inv = 1.0 / (lambda_ii + damping)
        ekfacDiag_damped_inv = ekfacDiag_damped_inv.reshape((V.shape[-2], V.shape[-1]))

        # calculate middle result
        q_a_t = q_a.t().cuda() # input * k
        intermediate_result = torch.einsum("ij,jk->ik", V, q_a_t) # 1 * output * input, input * k-> 1 * output * k
        del V, q_a_t, lambda_ii
        torch.cuda.empty_cache()

        q_s = q_s.cuda() # k * output
        intermediate_result = torch.einsum("ji,ik->jk", q_s, intermediate_result) # k * output, 1 * output * k -> 1 * k * k
        
        result = intermediate_result / ekfacDiag_damped_inv.cuda()
        del intermediate_result, ekfacDiag_damped_inv
        torch.cuda.empty_cache()

        # calculate the ihvp component
        q_a = q_a.cuda()
        ihvp_component = torch.einsum("ij,jk->ik", result, q_a) # 1 * k * k, k*input -> 1 * k * input
        ihvp_component = ihvp_component.cuda()
        del result, q_a
        torch.cuda.empty_cache()

        q_s_t = q_s.t().cuda()
        ihvp_component = torch.einsum("ji,ik->jk", q_s_t, ihvp_component)# output*k, 1*k*input -> 1*output*input
        # flattening the result except for the batch dimension
        ihvp_component = ihvp_component.view(-1) # (output*input)
        del q_s_t, q_s
        torch.cuda.empty_cache()

        M = validation_grad_list[i].cuda() # output*input
        M = M.view(-1)
        # # print(M.shape, ihvp_component.shape)
        final_result = torch.dot(M, ihvp_component)
        influence_list[i] = final_result.cpu()
        del final_result, M, ihvp_component
        torch.cuda.empty_cache()
    
    del q_a_list, q_s_list, lambda_ii_avg_list, sub_train_grad_list, validation_grad_list
    torch.cuda.empty_cache()
    gc.collect()

    # influence list represent the influence of dataset in each layer.
    # time_out = time.time()
    # print(f"total time: {time_out - time_in}")
    
    return influence_list

