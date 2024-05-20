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
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]

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

file3 = pd.read_csv('nnnn.csv')

file2 = pd.read_csv('modified_file.csv')

output_file = open('pre.csv', 'w')
output_file.write('id,estimate_q\n')

seq_len = 24  # 使用过去24小时的数据
pred_len = 1  # 预测接下来1小时的数据

k = 1

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
        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}')
        

    predict_data = file3[file3['iu_ac'] == current_id]

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
    
    del(model)
    print(index)

output_file.close()

