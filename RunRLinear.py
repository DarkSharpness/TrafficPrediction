import torch
import torch.utils.data
from models.RLinear import Model, Configs

from funcs import *

config = Configs()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

isTrain = False
isPredict = True

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
        do_predict(model, device=device, savepath="result/" + ModelName)

if __name__ == '__main__':
    main()
