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
    model = load_model("save/PST2_best/PST2_train_79_13295.pth")
    if model is None:
        print("warn")
        model = Model(config)
    model = model.to(device)

    if isTrain:
        dataset = load_train_dataset(device=device)
        train(model, dataset, learning_rate=0.01, epochs=100, doLog=True,
              doSave=True,
              savepath="save/" + ModelName,
              saveEachEpoch=True)

    if isPredict:
        # do_predict_xzydata(model, datafile="data/pred.fmt.csv", device=device, savepath="result/" + ModelName)
        do_predict_with_finetune(model, "__exe__/pred.fmt.csv", device, savepath="result/LastTry",
                                 seq_len=24, pred_len=1, finetune_epochs=0, finetune_savepath="save/PST2_best/finetune", use_saved=True)


if __name__ == '__main__':
    main()
