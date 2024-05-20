import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channel) if configs.rev else None
        self.individual = configs.individual

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred, self.forward_loss(pred, y)

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

file3 = pd.read_csv('new_file.csv')

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
    drop = 0.2  # Dropout率
    rev = True  # 启用RevIN
    channel = 1  # 输入通道数量

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
            output, loss = model(seq_x.unsqueeze(-1), seq_y.unsqueeze(-1)) 
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
    
        for seq_x, _ in predict_loader:
            seq_x = seq_x.to(device)
            output, _ = model(seq_x.unsqueeze(-1), seq_x.unsqueeze(-1))
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    for  prediction in predictions:
        estimate_q = int(round(prediction.item()))
        output_file.write(f"{k},{estimate_q}\n")
        k += 1
    
    del(model)
    print(index)

output_file.close()

