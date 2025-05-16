'''
Developed by: Ming
'''
import torch
from torch import nn
from typing import Any
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM

def load_model(model_path: str,
               torch_dtype: Any = torch.bfloat16) -> Any:
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        # print("Using PreTrainedTokenizerFast.")
    except Exception as e:
        print(f"Failed to load PreTrainedTokenizerFast: {e}. Trying LlamaTokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)
    
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map="auto")

    return model, tokenizer

def filter_layers(name, module, args):
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