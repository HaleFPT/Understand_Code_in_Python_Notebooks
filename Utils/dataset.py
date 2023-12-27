from curses import meta
from mimetypes import init
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
    
    def max_length_rule_base(self, cell_inputs, cell_type, rank):
        init_length = [len(x) for x in cell_inputs]
        total_max_length = self.seq_length - len(init_length)
        min_length = total_max_length // len(init_length)
        cell_length = self.search_length(init_length, 
                                         min_length, 
                                         total_max_length, 
                                         len(init_length))
        seq = []
        for i in range(len(cell_inputs)):
            if cell_type[i] == 0:
                seq.append(self.tokenizer.cls_token_id)
            else:
                seq.append(self.tokenizer.sep_token_id)
            if cell_length[i] > 0:
                seq.extend(cell_inputs[i][:cell_length[i]])
            
        if len(seq) < self.seq_length:
            seq_mask = [1] * len(seq)
        else:
            seq_mask = [1] * self.seq_length
            seq = seq[:self.seq_length]
        seq, seq_mask = np.array(seq, dtype=np.int), np.array(seq_mask, dtype=np.int)
        target_mask = np.where((seq == self.tokenizer.cls_token_id) | (seq == self.tokenizer.sep_token_id), 1, 0)
        target = np.zeros(self.seq_length, dtype=np.float32)
        tmp = np.where(seq == self.tokenizer.cls_token_id) | (seq == self.tokenizer.sep_token_id)
        target[tmp] = rank
        return seq, seq_mask, target_mask, target
    
    @staticmethod
    def search_length(init_length, min_length, total_max_length, cell_count, step=4, max_search_count=50):
        if np.sum(init_length) <= total_max_length:
            return init_length
        
        res= [min(init_length[i], min_length) for i in range(cell_count)]
        for s_i in range(max_search_count):
            tmp = [min(init_length[i], res[i] + step) for i in range(cell_count)]
            if np.sum(res) <= total_max_length:
                res = tmp
            else:
                break
        for s_i in range(cell_count):
            tmp = [i for i in res]
            tmp[s_i] = min(init_length[s_i], tmp[s_i] + step)
            if np.sum(tmp) <= total_max_length:
                res = tmp
            else:
                break
        return res
    