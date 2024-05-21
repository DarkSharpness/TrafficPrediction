import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn

from TrafficDataset import TrafficDataset
from models.DLinear import Model, Configs

from tqdm import tqdm
import os

seq_len = 24  # 使用过去24小时的数据
pred_len = 1  # 预测接下来1小时的数据

configs = Configs()
model = Model(configs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import visdom
vis = visdom.Visdom()
losses = []
losses_idx = 0
def update_loss(loss):
    if len(losses) < 1000:
        losses.append(loss)
    else:
        losses[losses_idx] = loss
        losses_idx += 1
        losses_idx %= 1000
    vis.line(Y=np.array(losses), X=np.arange(len(losses)),
             win='loss', opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss'))

def train(train_data: pd.DataFrame):
    dataset = TrafficDataset(train_data, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    global model
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):  # 训练50个epoch
        epoch_loss = 0
        for seq_x, seq_y in dataloader:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            optimizer.zero_grad()
            output = model(seq_x.unsqueeze(-1))  # 添加通道维度
            loss = criterion(output, seq_y.unsqueeze(-1))  # 添加通道维度
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss = epoch_loss / len(dataloader)
    update_loss(loss)


def predict(predict_data: pd.DataFrame) -> list[float]:
    predict_dataset = TrafficDataset(predict_data, 24, 0)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
    model.eval()
    predictions = []
    datalen = len(predict_loader)
    with torch.no_grad():
        for seq_x, seq_y in predict_loader:
            seq_x = seq_x.to(device)
            output = model(seq_x.unsqueeze(-1))
            predictions.append(output.cpu().numpy().flatten())

    predictions = np.concatenate(predictions, axis=0)
    if (len(predictions) != datalen):
        print("error", f"{datalen =} {len(predictions) =}")
        exit(1)
    return predictions

SAVE_PATH = "DLinear.pth"

is_train = True

def main():
    print("start loading data")
    id4predict = pd.read_csv('data/id_for_predict.csv')
    train_data = pd.read_csv('data/train_data.csv')
    predict_data = pd.read_csv('data/predict_data.csv')

    global model
    if os.path.exists(SAVE_PATH):
        model = torch.load(SAVE_PATH)
    else:
        model = Model(configs)
    print("done loading data")

    if is_train:
        for index, row in tqdm(id4predict.iterrows(), total=1758):
            current_id = row['id']
            # print(f"{current_id =}")
            current_train_data = train_data[train_data['iu_ac'] == current_id]
            train(current_train_data)

        print("start save model")
        torch.save(model, SAVE_PATH)
        print("done save model")

    print("start predicting")
    result = []
    for index, row in tqdm(id4predict.iterrows(), total=1758):
        current_id = row['id']
        current_predict_data = predict_data[predict_data['iu_ac'] == current_id]
        predictions = predict(current_predict_data)
        result.extend(predictions)
    print("done predicting")
    print("start writing to file")
    with open("DLinear.csv", "w") as f:
        f.write("id,estimate_q\n")
        for i in range(len(result)):
            f.write(f"{i+1},{result[i]:.2f}\n")
    print("done writing to file")


if __name__ == '__main__':
    main()
