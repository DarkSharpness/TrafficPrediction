import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        # if self.individual:
        #     self.Linear = nn.ModuleList()
        #     for i in range(self.channels):
        #         self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        # else:
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        # if self.individual:
        #     output = torch.zeros(
        #         [x.size(0), self.pred_len, x.size(2)],
        #         dtype=x.dtype, device=x.device)
        #     for i in range(self.channels):
        #         output[:, :, i] = self.Linear[i](x[:, :, i])
        #     x = output
        # else:
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


class Configs:
    seq_len = 24
    pred_len = 1
    individual = False
    enc_in = 1  # 单通道，即每个探头一个时间序列