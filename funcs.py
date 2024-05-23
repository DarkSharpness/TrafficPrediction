import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from glob import glob
import os
from TrafficDataset import *
import pandas as pd

def train_once(model, dataloader, criterion, optimizer, doLog: bool = False) -> float:
    k = 0
    epoch_loss = 0
    for seq_x, seq_y in (tqdm(dataloader) if doLog else dataloader):
        # seq_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        output = model(seq_x)
        loss = criterion(output, seq_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        k += 1
        if k % 500 == 0:
            break
    # return epoch_loss / len(dataloader)
    return epoch_loss / k


def train(model: nn.Module, dataset: Dataset, doLog: bool = True, learning_rate: float = 0.003, epochs: int = 100, doSave: bool = False, savepath: str = "unknown", saveEachEpoch: bool = False):
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss = None
    try:
        saved = False
        for epoch in range(epochs):
            epoch_loss = train_once(model, dataloader, criterion, optimizer, doLog=doLog)
            loss = epoch_loss
            if doLog: print("epoch:", epoch, "  loss:", epoch_loss)
            if saveEachEpoch:
                save_model(model, path=savepath + "_train", loss=loss)
                saved = True
    except KeyboardInterrupt:
        print("KeyboardInterrupted")
    finally:
        if doSave and not saved:
            save_model(model, path=savepath, loss=loss)

def predict(model: nn.Module, dataset: Dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    predictions = torch.tensor([])
    with torch.no_grad():
        for seq_x, _ in dataloader:
            output = model(seq_x)
            predictions = torch.cat((predictions, output), dim=0)
    predictions = predictions.flatten()
    return predictions


def save_prediction(predictions: torch.Tensor, filename_noext: str):
    k = 1
    while os.path.exists(filename_noext + f"_{k}.csv"):
        k += 1
    filename = filename_noext + f"_{k}.csv"
    with open(filename, "w") as f:
        f.write("id,estimate_q\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")

def load_model(path: str):
    if os.path.exists(path):
        model = torch.load(path)
        return model
    else:
        return None

def save_model(model: nn.Module, path: str, loss: float | None):
    def gen_filename(k):
        return f"{path}_{k}_{int(loss)}.pth" if loss is not None else f"{path}_{k}.pth"
    def gen_filename_star(k):
        return f"{path}_{k}*.pth"
    k = 1
    while len(glob(gen_filename_star(k))) > 0:
        k += 1
    filename = gen_filename(k)
    torch.save(model, filename)


def load_train_dataset(device: torch.device = torch.device('cpu')):
    print("start loading data")
    train_data = pd.read_csv('data/train_data_flat_data.csv')
    usefull_idx = pd.read_csv('data/train_data_flat_idx.csv')
    idx = usefull_idx['idx'].to_list()
    value = train_data['q'].to_list()
    dataset = TrafficDatasetTrain(idx, value, 24, 1, device=device)
    print("done preparing dataset")
    return dataset
