import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from glob import glob
import os
from TrafficDataset import *
import pandas as pd
from datapath import *

def load_model(path: str):
    if os.path.exists(path):
        model = torch.load(path, map_location=torch.device('cpu'))
        return model
    else:
        return None

def save_model(model: nn.Module, path: str, loss: float):
    def gen_filename(k):
        return f"{path}_{k}_{int(loss)}.pth" if loss is not None else f"{path}_{k}.pth"

    def gen_filename_star(k):
        return f"{path}_{k}*.pth"
    k = 1
    while len(glob(gen_filename_star(k))) > 0:
        k += 1
    filename = gen_filename(k)
    print("saving model to", filename)
    prefix = '/'.join(filename.split('/')[:-1])
    if len(prefix) > 0:
        os.makedirs(prefix, exist_ok=True)
    torch.save(model, filename)

def train_once(model, dataloader, criterion, optimizer, doLog: bool = False) -> float:
    epoch_loss = 0
    for seq_x, seq_y in (tqdm(dataloader) if doLog else dataloader):
        # seq_x, seq_y = seq_x.to(device), seq_y.to(device)
        optimizer.zero_grad()
        output = model(seq_x)
        loss = criterion(output, seq_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)
    # return epoch_loss / k

def train(model: nn.Module, dataset: Dataset, doLog: bool = True, learning_rate: float = 0.003, epochs: int = 100, doSave: bool = False, savepath: str = "unknown", saveEachEpoch: bool = False):
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss = None
    try:
        saved = False
        for epoch in range(epochs):
            epoch_loss = train_once(
                model, dataloader, criterion, optimizer, doLog=doLog)
            loss = epoch_loss
            if doLog:
                print("epoch:", epoch, "  loss:", epoch_loss)
            if saveEachEpoch:
                save_model(model, path=savepath + "_train", loss=loss)
                saved = True
    except KeyboardInterrupt:
        print("KeyboardInterrupted")
    finally:
        if doSave and not saved:
            save_model(model, path=savepath, loss=loss)

def predict(model: nn.Module, dataset: Dataset, device: torch.device = torch.device('cpu'), batch_size=32, doLog: bool = False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions = torch.tensor([], device=device)
    with torch.no_grad():
        for seq_x, _ in (tqdm(dataloader) if doLog else dataloader):
            output = model(seq_x)
            predictions = torch.cat((predictions, output), dim=0)
    predictions = predictions.flatten()
    return predictions

def save_prediction(predictions: torch.Tensor, filename_noext: str):
    k = 1
    while os.path.exists(filename_noext + f"_{k}.csv"):
        k += 1
    filename = filename_noext + f"_{k}.csv"
    print("saving to", filename)
    prefix = '/'.join(filename.split('/')[:-1])
    if len(prefix) > 0:
        os.makedirs(prefix, exist_ok=True)
    with open(filename, "w") as f:
        f.write("id,estimate_q\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i+1},{pred:.2f}\n")

def load_train_dataset(device: torch.device = torch.device('cpu'), seq_len=24, pred_len=1):
    print("start loading data")
    train_data = pd.read_csv('data/train_data_flat_data.csv')
    usefull_idx = pd.read_csv('data/train_data_flat_idx.csv')
    idx = usefull_idx['idx'].to_list()
    value = train_data['q'].to_list()
    dataset = TrafficDatasetTrain(idx, value, seq_len, pred_len, device=device)
    print("done preparing dataset")
    return dataset

def load_finetune_datasets(**kwargs):
    with open(FlatDataFile, "r") as f:
        lines = f.readlines()
    train_data = [float(line) for line in lines if line.strip() != ""]
    datasets: list[Dataset] = []
    with open(PredictDataIdxFile, "r") as f:
        lines = f.readlines()
    for line in lines:
        id, idx_str = line.split(':')
        idx = [int(i) for i in idx_str.split(',')]
        dataset = TrafficDatasetTrain(idx, train_data, **kwargs)
        datasets.append(dataset)
    return datasets

def load_predict_spec_datasets(**kwargs):
    total_dataset = TrafficDatasetPredict(PredictDataFile, **kwargs)
    with open(PredictDataIdxFile, "r") as f:
        lines = f.readlines()
    datasets: list[Dataset] = []
    for line in lines:
        id, L, R = line.split(',')
        dataset = TrafficDatasetPredictSpec(int(L), int(R), total_dataset)
        datasets.append(dataset)
    return datasets

def do_predict_xzydata(model: nn.Module, datafile: str, device: torch.device, savepath: str, seq_len: int = 24):
    dataset = TrafficDatasetPredict(datafile, seq_len, device)
    predictions = predict(model, dataset, device, doLog=True)
    save_prediction(predictions, savepath)

def do_predict_with_finetune(model: nn.Module, device: torch.device, savepath: str, **kwargs):
    print("loading train data")
    train_sets = load_finetune_datasets(device=device, **kwargs)
    predict_sets = load_predict_spec_datasets(device=device, **kwargs)
    if len(train_sets) != len(predict_sets):
        raise ValueError("train set and predict set must have the same length")
    init_state = model.state_dict()
    predictions = torch.tensor([], device=device)
    for train_set, predict_set in zip(train_sets, predict_sets):
        model.load_state_dict(init_state)
        train(model, train_set, doLog=False, epochs=10, doSave=False)
        prediction = predict(model, predict_set, device=device)
        predictions = torch.cat((predictions, prediction), dim=0)
    save_prediction(predictions, savepath)
