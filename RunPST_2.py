import visdom
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from TrafficDataset import TrafficDatasetTrain, TrafficDataset
from models.PST import Model, Configs2

from tqdm import tqdm
import os


config = Configs2()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model = Model(config)
SAVE_PATH = 'PST2.pth'
SAVE_PATH_INTRAIN = 'PSTInTrain{}.pth'

vis = visdom.Visdom()
losses = []
losses_idx = 0


def update_loss(loss):
    losses.append(loss)
    if len(losses) > 1000:
        losses.pop(0)
    vis.line(Y=np.array(losses), X=np.arange(len(losses)),
             win='loss', opts=dict(title='Training Loss', xlabel='Epoch', ylabel='Loss'))

isTrain = True
isPredict = False
use_tqdm = True

# @torch.compile


def train_once(dataloader, model, criterion, optimizer) -> float:
    epoch_loss = 0
    # k = 0
    for seq_x, seq_y in (tqdm(dataloader) if use_tqdm else dataloader):
        # seq_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        output = model(seq_x)
        loss = criterion(output, seq_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # k += 1
        # if k % 500 == 0:
        #     update_loss(loss.item())
    return epoch_loss / len(dataloader)


def train(dataset: torch.utils.data.Dataset):
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(100):
        epoch_loss = train_once(dataloader, model, criterion, optimizer)
        print("epoch:", epoch, "  loss:", epoch_loss)


def predict(dataset: torch.utils.data.Dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for seq_x, seq_y in dataloader:
            seq_x = seq_x.to(device)
            output = model(seq_x)
            predictions.append(output.cpu().numpy().flatten())
    predictions = np.concatenate(predictions, axis=0)
    return predictions
    

def main():
    global model
    if os.path.exists(SAVE_PATH):
        model = torch.load(SAVE_PATH)
    else:
        model = Model(config)
    model = model.to(device)

    if isTrain:
        print("start loading data")
        train_data = pd.read_csv('data/train_data_flat_data.csv')
        usefull_idx = pd.read_csv('data/train_data_flat_idx.csv')
        print("done loading data")

        print("preparing dataset")
        idx = usefull_idx['idx'].to_list()
        value = train_data['q'].to_list()
        dataset = TrafficDatasetTrain(idx, value, config.seq_len, config.pred_len, device=device)
        print("done preparing dataset")

        try:
            train(dataset)
            print("Training finished")
        except KeyboardInterrupt:
            print("KeyboardInterrupted")
        except Exception as e:
            print(e)
        finally:
            torch.save(model, SAVE_PATH)

    if isPredict:
        print("loading data")
        id4predict = pd.read_csv('data/id_for_predict.csv')
        predict_data = pd.read_csv('data/predict_data.csv')
        train_data = pd.read_csv('data/train_data.csv')
        print("start predicting")
        result = []
        try:
            for index, row in tqdm(id4predict.iterrows(), total=1758):
                current_id = row['id']
                current_train_data = train_data[train_data['iu_ac'] == current_id]
                train_datset = TrafficDataset(current_train_data, 24, 1, device=device)
                train(train_datset) # retrain for specialize
                current_predict_data = predict_data[predict_data['iu_ac'] == current_id]
                current_predict_dataset = TrafficDataset(current_predict_data, 24, 0, device=device)
                predictions = predict(current_predict_dataset)
                result.extend(predictions)

        except KeyboardInterrupt:
            print("KeyboardInterrupted")
            exit(0)
        except Exception as e:
            print(e)
            exit(1)
        finally:
            torch.save(model, SAVE_PATH)

        print("done predicting")
        print("start writing to file")
        with open("PST.csv", "w") as f:
            f.write("id,estimate_q\n")
            for i in range(len(result)):
                f.write(f"{i+1},{result[i]:.2f}\n")
        print("done writing to file")


if __name__ == '__main__':
    main()
