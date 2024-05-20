import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import sys


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sensor_ids = data['iu_ac'].unique()
        self.samples = self.create_samples()
        
#数据的处理待定
    def create_samples(self):
        samples = []
        for sensor_id in self.sensor_ids:
            sensor_data = self.data[self.data['iu_ac'] == sensor_id]
            values = sensor_data['q'].values
            for i in range(0, len(values) - self.seq_len - self.pred_len + 1, self.seq_len + self.pred_len):
                seq_x = values[i:i+self.seq_len]
                seq_y = values[i+self.seq_len:i+self.seq_len+self.pred_len]
                samples.append((seq_x, seq_y))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_x, seq_y = self.samples[idx]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

file1 = pd.read_csv('iiid.csv')

file3 = pd.read_csv('1111.csv')

file2 = pd.read_csv('modified_file.csv')

output_file = open('DLinear.csv', 'w')
output_file.write('id,estimate_q\n')

seq_len = 24  # 使用过去24小时的数据
pred_len = 1  # 预测接下来1小时的数据

k = 1
result = pd.DataFrame()
class Configs:
    seq_len = 24
    pred_len = 1
    individual = False
    enc_in = 1  # 单通道，即每个探头一个时间序列

for index, row in file1.iterrows():
    current_id = row['id']

    train_data = file2[file2['iu_ac'] == current_id]

    dataset = TrafficDataset(train_data, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    configs = Configs()
    model = Model(configs)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    for epoch in range(500):  # 训练50个epoch
        epoch_loss = 0
        for seq_x, seq_y in dataloader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            optimizer.zero_grad()
            output = model(seq_x.unsqueeze(-1))  # 添加通道维度
            loss = criterion(output, seq_y.unsqueeze(-1))  # 添加通道维度
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
       # print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}')
    
    prediction = file3[file3['Camera ID'] == current_id]
    predict_data = pd.DataFrame()
    for index1, row in prediction.iterrows():
        current_id = row['Camera ID']
        current_time = row['Time']
        target_times = list(range(current_time - 24, current_time))
        filtered_data = train_data[(train_data['iu_ac'] == current_id) & (train_data['index'].isin(target_times))]
        final_data = pd.DataFrame()
        for t in target_times:
            if t in filtered_data['index'].values:
            # 如果时间点存在，直接添加
                final_data = final_data._append(filtered_data[filtered_data['index'] == t])
            else:
            # 如果时间点不存在，寻找相差24的倍数的数据
                candidates = train_data[(train_data['iu_ac'] == current_id) & ((train_data['index'] - t) % 24 == 0)]
            
                if len(candidates) > 0:
                # 取这些数据的平均数作为新的值
                    avg_row = candidates.mean(axis=0).round()
                    avg_row['iu_ac'] = current_id
                    avg_row['index'] = t
                    final_data = final_data._append(avg_row, ignore_index=True)
                else:
                # 如果一个都没有找到，取之前找到的数据的第一个作为填充值
                    if len(final_data) > 0:
                        fill_row = final_data.iloc[0].copy()
                    else:
                    # 如果之前也一个都没找到，取file2中这个ID下时间大于或小于它的第一个点
                        nearest_row = train_data[(train_data['iu_ac'] == current_id)]
                        fill_row = nearest_row.iloc[0].copy()
                    fill_row['index'] = t
                    final_data = final_data._append(fill_row, ignore_index=True)
        predict_data = pd.concat([predict_data, final_data])
    
    predict_dataset = TrafficDataset(predict_data, seq_len, 0)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
    
        for seq_x, seq_y in predict_loader:
            seq_x = seq_x.to(device)
            output = model(seq_x.unsqueeze(-1))
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    for  prediction in predictions:
        estimate_q = int(round(prediction.item()))
        output_file.write(f"{k},{estimate_q}\n")
        k += 1
    result = pd.concat([result, predict_data])
    del(model)
    print(index)


output_file.close()

result.reset_index(drop=True, inplace=True)

# 将结果保存到新的CSV文件中
result.to_csv('nnnn.csv', index=False)

