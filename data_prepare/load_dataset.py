import contextlib
import numpy as np
import torch
from tqdm import tqdm
import json
from typing import  Sequence
from collections.abc import Iterable
import os
from torch.utils.data import Dataset
import random

class LoadDataset(Dataset):
    def __init__(self, all_file_paths, tokenizer, max_seq_length=1024, sample_percentage=1.0, seed=0):
        self.file_paths_list = load_train_files(all_file_paths)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sample_percentage = sample_percentage
        self.data_indices = self._load_data_indices()
        self.data_indices = random.sample(self.data_indices, int(len(self.data_indices) * self.sample_percentage))
        self.prompt = ''
        self.answer_prefix = '\nAnswer: '
        self.pmt_len = 128
        self.ans_len = 128
    
    def _load_data_indices(self):
        data_indices = []
        for file_path in self.file_paths_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    data_indices.append((file_path, offset))
                    
        return data_indices

    def __len__(self):
        return len(self.data_indices)
    
    # def __getitem__(self, idx):
    #     file_path, line_idx = self.data_indices[idx]
    #     with open(file_path, 'r') as f:
    #         for current_idx, line in enumerate(f):
    #             if current_idx == line_idx:
    #                 return self.tokenizer(line)

    def __getitem__(self, idx):
        file_path, offset = self.data_indices[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            sample = json.loads(line.strip())
            question = sample['instruction'] + ' ' + sample['input']
            answer = '\n### ' + sample['output']
            
            input_text = self.prompt + question + self.answer_prefix
            target_text = answer + self.tokenizer.eos_token

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                # print("Pad token ID:", self.tokenizer.pad_token_id)

            encoding = self.tokenizer(
                input_text + target_text,
                add_special_tokens=False,
                padding='max_length',
                truncation=True,
                max_length=self.pmt_len + self.ans_len,
                return_tensors='pt'
            )

            labels = encoding['input_ids'].squeeze().clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[:self.pmt_len] = -100

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': labels
            }

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def load_train_files(all_file_paths):
    file_path_list = []
    has_subfolders = any(os.path.isdir(os.path.join(all_file_paths, item)) for item in os.listdir(all_file_paths))
    if has_subfolders:
        for dirpath, _, filenames in os.walk(all_file_paths):
            for filename in filenames:
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(dirpath, filename)
                    file_path_list.append(filepath)
    else:
        for filename in os.listdir(all_file_paths):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(all_file_paths, filename)
                file_path_list.append(filepath)
    return file_path_list