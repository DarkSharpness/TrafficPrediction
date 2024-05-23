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
from models.RLinear import Model, Configs

from tqdm import tqdm
import os

from funcs import *

config = Configs()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

isTrain = True
isPredict = False

ModelName = "RLinear"

def main():
    model = load_model("save/RLinear_3_43534.pth")
    if model is None:
        model = Model(config)
    model = model.to(device)

    if isTrain:
        dataset = load_train_dataset(device=device)
        train(model, dataset, learning_rate=0.003, epochs=10, doLog=True,
              doSave=True,
              savepath="save/" + ModelName,
              saveEachEpoch=False)

    if isPredict:
        assert False
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
        with open("RLinear.csv", "w") as f:
            f.write("id,estimate_q\n")
            for i in range(len(result)):
                f.write(f"{i+1},{result[i]:.2f}\n")
        print("done writing to file")


if __name__ == '__main__':
    main()
