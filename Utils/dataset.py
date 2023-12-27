from curses import meta
import pandas as pd
import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import os
import numpy as np

class MarkdownDataset(Dataset):
    def __init__(self, meta_data: pd.DataFrame, tokenizer, fold: int = -1, mode='train', parameter=None, max_seq_length=2048):
        self.meta_data = meta_data.copy()
        self.meta_data.reset_index(drop=True, inplace=True)
        if mode == 'train':
            pass
        elif mode == 'valid':
            self.meta_data = self.meta_data[self.meta_data['fold_flag'] == fold].copy()
            self.meta_data = self.meta_data[self.meta_data['id'].isin(self.meta_data['id'].values[:1000])]
        elif mode == 'test':
            pass
        else:
            raise ValueError(mode)
        self.meta_data.reset_index(drop=True, inplace=True)
        if tokenizer.sep_token != '[SEP]':
            self.meta_data['source'] = self.meta_data['source'].apply(
                lambda x: [
                    y.replace(tokenizer.sep_token, '').replace(tokenizer.cls_token, '').replace(tokenizer.pad_token, '')
                    for y in x])
        self.parameter = parameter
        self.seq_length = max_seq_length
        self.source = self.meta_data['source'].values
        self.cell_type = self.meta_data['cell_type'].values
        self.rank = self.meta_data['rank'].values
        self.mode = mode
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        source = self.source[index]
        cell_type = self.cell_type[index]
        rank = self.rank[index]

        cell_inputs = self.tokenizer.batch_encode_plus(
            source,
            add_special_tokens=False,
            max_length=self.parameter.cell_max_length,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        seq, seq_mask, target_mask, target = self._create_target(cell_inputs["input_ids"], 
                                                                 cell_type, 
                                                                 rank)
        return seq, seq_mask, target_mask, target
    
    def __len__(self):
        return len(self.meta_data)
    
    def max_length_rule_base(self, input_ids, cell_type, rank):
        if len(input_ids) > self.parameter.cell_max_length:
            input_ids = input_ids[:self.parameter.cell_max_length]
            cell_type = cell_type[:self.parameter.cell_max_length]
            rank = rank[:self.parameter.cell_max_length]
        return input_ids, cell_type, rank
    