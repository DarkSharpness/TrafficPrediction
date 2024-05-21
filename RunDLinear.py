import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn

from TrafficDataset import TrafficDataset
from models.DLinear import Model, Configs


seq_len = 24  # 使用过去24小时的数据
pred_len = 1  # 预测接下来1小时的数据

configs = Configs()
model = Model(configs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hidden():
    # 所有要预测的摄像头的id
    file1 = pd.read_csv('iiid.csv')
    # 生成的预测用的数据
    file3 = pd.read_csv('1111.csv')
    # 训练用的数据
    file2 = pd.read_csv('modified_file.csv')

    output_file = open('DLinear.csv', 'w')
    output_file.write('id,estimate_q\n')

    k = 1
    result = pd.DataFrame()

    for index, row in file1.iterrows():
        current_id = row['id']

        train_data = file2[file2['iu_ac'] == current_id]

        # print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}')

        # prediction = file3[file3['Camera ID'] == current_id]
        # predict_data = pd.DataFrame()
        # for index1, row in prediction.iterrows():
        #     current_id = row['Camera ID']
        #     current_time = row['Time']
        #     target_times = list(range(current_time - 24, current_time))
        #     filtered_data = train_data[(train_data['iu_ac'] == current_id) & (
        #         train_data['index'].isin(target_times))]
        #     final_data = pd.DataFrame()
        #     for t in target_times:
        #         if t in filtered_data['index'].values:
        #             # 如果时间点存在，直接添加
        #             final_data = final_data._append(
        #                 filtered_data[filtered_data['index'] == t])
        #         else:
        #             # 如果时间点不存在，寻找相差24的倍数的数据
        #             candidates = train_data[(train_data['iu_ac'] == current_id) & (
        #                 (train_data['index'] - t) % 24 == 0)]

        #             if len(candidates) > 0:
        #                 # 取这些数据的平均数作为新的值
        #                 avg_row = candidates.mean(axis=0).round()
        #                 avg_row['iu_ac'] = current_id
        #                 avg_row['index'] = t
        #                 final_data = final_data._append(
        #                     avg_row, ignore_index=True)
        #             else:
        #                 # 如果一个都没有找到，取之前找到的数据的第一个作为填充值
        #                 if len(final_data) > 0:
        #                     fill_row = final_data.iloc[0].copy()
        #                 else:
        #                     # 如果之前也一个都没找到，取file2中这个ID下时间大于或小于它的第一个点
        #                     nearest_row = train_data[(
        #                         train_data['iu_ac'] == current_id)]
        #                     fill_row = nearest_row.iloc[0].copy()
        #                 fill_row['index'] = t
        #                 final_data = final_data._append(
        #                     fill_row, ignore_index=True)
        #     predict_data = pd.concat([predict_data, final_data])

    output_file.close()

    result.reset_index(drop=True, inplace=True)

    # 将结果保存到新的CSV文件中
    result.to_csv('nnnn.csv', index=False)


def train(train_data):
    global model
    dataset = TrafficDataset(train_data, seq_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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


def predict(predict_data: pd.DataFrame) -> list[float]:
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
    return predictions


def main():
    id4predict = pd.read_csv('id_for_prediction.csv')
    train_data = pd.read_csv('train_data.csv')
    predict_data = pd.read_csv('predict_data.csv')

    result = pd.DataFrame()
    for index, row in id4predict.iterrows():
        current_id = row['id']
        current_predict_data = predict_data[predict_data['iu_ac'] == current_id]
        current_train_data = train_data[train_data['iu_ac'] == current_id]

        train(current_train_data)
        predictions = predict(current_predict_data)
        for i in range(len(predictions)):
            result = result.append(
                {'id': current_id, 'estimate_q': predictions[i]}, ignore_index=True)

