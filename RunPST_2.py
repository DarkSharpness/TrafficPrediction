import visdom
import torch
from models.PST import Model, Configs2
from funcs import *

config = Configs2()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model = Model(config)
SAVE_PATH = 'PST2.pth'
SAVE_PATH_INTRAIN = 'PSTInTrain{}.pth'


isTrain = False
isPredict = True
ModelName = "PST2"

def main():
    model = load_model("save/PST2_15040.pth")
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
