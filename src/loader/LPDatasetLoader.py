# -*- coding: utf-8 -*-
'''
@Time    : 2021/8/24 16:01
@Author  : Wang Qiang
@FileName: LPDatasetLoader.py
'''
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl

class LPDataset(Dataset):

    def __init__(self, path, window_size):
        super(LPDataset, self).__init__()
        data_sp = pkl.load(open(path, 'rb'))
        data_numpy = np.array([x.toarray() for x in data_sp])
        self.data = torch.from_numpy(data_numpy)
        self.window_size = window_size - 1
        self.num = self.data.size(0) - window_size
        self.nb_nodes = self.data.size(-1)
        self.max_thres = np.max(data_numpy)
        self.data = self.data / self.max_thres

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size], self.data[item + self.window_size], self.data[item + 1: item + self.window_size + 1], self.data[item + self.window_size + 1]


class OfflineLPDataset(Dataset):

    def __init__(self, path, window_size):
        super(OfflineLPDataset, self).__init__()
        data_sp = pkl.load(open(path, 'rb'))
        data_numpy = np.array([x.toarray() for x in data_sp])
        self.data = torch.from_numpy(data_numpy)
        self.window_size = window_size - 1
        self.num = self.data.size(0) - window_size
        self.nb_nodes = self.data.size(-1)
        self.max_thres = np.max(data_numpy)
        self.data = self.data / self.max_thres

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data[item: item + self.window_size].unsqueeze(0), self.data[item + self.window_size].unsqueeze(0)

