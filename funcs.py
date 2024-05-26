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


def save_model(model: nn.Module, path: str, loss: float, doLog: bool = True):
    def gen_filename(k):
        return f"{path}_{k}_{int(loss)}.pth" if loss is not None else f"{path}_{k}.pth"

    def gen_filename_star(k):
        return f"{path}_{k}*.pth"
    k = 1
    while len(glob(gen_filename_star(k))) > 0:
        k += 1
    filename = gen_filename(k)
    if doLog: print("saving model to", filename)
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
    interrupted = False
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
        interrupted = True
    finally:
        if doSave and not saved:
            save_model(model, path=savepath, loss=loss)
    if interrupted:
        exit(0)
    return loss


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


def save_prediction(predictions: torch.Tensor, filename_noext: str, doLog: bool = True):
    k = 1
    while os.path.exists(filename_noext + f"_{k}.csv"):
        k += 1
    filename = filename_noext + f"_{k}.csv"
    if doLog: print("saving to", filename)
    prefix = '/'.join(filename.split('/')[:-1])
    if len(prefix) > 0:
        os.makedirs(prefix, exist_ok=True)
    with open(filename, "w") as f:
        f.write("id,estimate_q\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i+1},{pred:.2f}\n")


def load_train_dataset(**kwargs):
    with open(FlatDataFile, "r") as f:
        lines = f.readlines()
    train_data = [float(line) for line in lines if line.strip() != ""]
    with open(FlatIdxFile, "r") as f:
        lines = f.readlines()
    idx = [int(line) for line in lines if line.strip() != ""]
    dataset = TrafficDatasetTrain(
        idx, train_data, seq_len=kwargs['seq_len'], pred_len=kwargs['pred_len'], device=kwargs['device'])
    return dataset


def load_finetune_datasets(**kwargs):
    with open(FlatDataFile, "r") as f:
        lines = f.readlines()
    train_data = [float(line) for line in lines if line.strip() != ""]
    datasets: list[Dataset] = []
    with open(FinetuneIdxFile, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == "":
            continue
        try:
            id, idx_str = line.split(':')
            idx = [int(i) for i in idx_str.split(',') if i.strip() != ""]
            dataset = TrafficDatasetTrain(
                idx, train_data, seq_len=kwargs['seq_len'], pred_len=kwargs['pred_len'], device=kwargs['device'])
            datasets.append(dataset)
        except Exception as e:
            print(f"error line:[{line}]")
            raise e
    return datasets


def load_predict_spec_datasets(datafile: str, **kwargs):
    total_dataset = TrafficDatasetPredict(
        datafile, seq_len=kwargs['seq_len'], device=kwargs['device'])
    with open(PredictDataIdxFile, "r") as f:
        lines = f.readlines()
    datasets: list[Dataset] = []
    for line in lines:
        if line.strip() == "":
            continue
        id, L, R = line.split(',')
        L = int(L)
        R = int(R)
        if R == 0: R = -1
        else: L, R = L - 1, R - 1
        dataset = TrafficDatasetPredictSpec(L, R, total_dataset)
        datasets.append(dataset)
    return datasets


def do_predict(model: nn.Module, datafile: str, device: torch.device, savepath: str, seq_len: int = 24):
    dataset = TrafficDatasetPredict(datafile, seq_len, device)
    predictions = predict(model, dataset, device, doLog=True)
    save_prediction(predictions, savepath)


def do_predict_with_finetune(model: nn.Module, datafile: str, device: torch.device,
                             savepath: str, finetune_epochs: int, finetune_savepath: str,
                             use_saved: bool,
                             **kwargs):
    if finetune_epochs <= 0:
        print("fallback to normal predict")
        return do_predict(model, PredictDataFile, device, savepath, seq_len=kwargs['seq_len'])

    print("loading train data")
    train_sets = load_finetune_datasets(device=device, **kwargs)
    predict_sets = load_predict_spec_datasets(datafile, device=device, **kwargs)
    if len(train_sets) != len(predict_sets):
        raise ValueError("train set and predict set must have the same length")
    
    total_train_size = 0
    for train_set in train_sets:
        total_train_size += len(train_set)
    total_predict_size = 439298

    init_state = model.state_dict()
    predictions = torch.tensor([], device=device)
    print("start predicting")
    cur_id = 0
    losses = []
    with tqdm(total=total_predict_size + total_train_size * finetune_epochs) as pbar:
        for train_set, predict_set in zip(train_sets, predict_sets):
            cur_id += 1
            if len(predict_set) == 0:
                pbar.update(len(train_set) * finetune_epochs)
                continue
            model.load_state_dict(init_state)
            if len(train_set) > 0 and use_saved:
                model = torch.load(f"{finetune_savepath}/{cur_id}.pth")
            if len(train_set) > 0:
                loss = train(model, train_set, doLog=False, epochs=finetune_epochs, doSave=False)
                pbar.update(len(train_set) * finetune_epochs)
                losses.append((cur_id, loss))
                torch.save(model, f"{finetune_savepath}/{cur_id}.pth")
            prediction = predict(model, predict_set, device=device)
            predictions = torch.cat((predictions, prediction), dim=0)
            pbar.update(len(predict_set))
    save_prediction(predictions, savepath)

    with open("finetune_losses.csv", "w") as f:
        f.write("id,loss\n")
        for id, loss in losses:
            f.write(f"{id},{loss}\n")
