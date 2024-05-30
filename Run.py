import torch
from funcs import *
# change this line to switch model
from models.PST import Model, Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Configs()
model = Model(config)

isTrain = True
isPredict = False
ModelName = "set your model name here"

def main():
    model = load_model(f"save/{ModelName}/{ModelName}.pth")
    if model is None:
        print(f"\033[31m [ warn ] new model ? \033[0m")
        model = Model(config)
    model = model.to(device)

    if isTrain:
        dataset = load_train_dataset(device=device, seq_len=24, pred_len=1)
        train(model, dataset, learning_rate=0.001, epochs=100, doLog=True,
              doSave=True,
              savepath=f"save/{ModelName}/{ModelName}",
              saveEachEpoch=True)

    if isPredict:
        do_predict(model, datafile=PredictDataFile,
                   device=device, savepath="result/" + ModelName)
        # do_predict_with_finetune(model, device, savepath="result/PST2/PST2_finetune", seq_len=24, pred_len=1)

if __name__ == '__main__':
    main()
