import torch
from matplotlib import pyplot as plt
from funcs import *

device = torch.device("cpu")

dataset = load_train_dataset(seq_len=24, pred_len=1, device=device)
dataset.idx = dataset.idx[150:300]

std = [dataset[i][1] for i in range(len(dataset))]
std = torch.tensor(std).flatten()
# print(std)
# print(std.tolist())

# 1. PST graph
model = torch.load("save/PST2_best/PST2_train_79_13295.pth",
                   map_location=device)

pst = predict(model, dataset, device, doLog=True)

# 2. NLinear
model = torch.load("save/NLinear.pth", map_location=device)
nL = predict(model, dataset, device, doLog=True)

# 3. RLinear

model = torch.load("save/RLinear/RLinear.pth", map_location=device)
rL = predict(model, dataset, device, doLog=True)

# 4. DLinear
model = torch.load("save/DLinear.pth", map_location=device)
dL = predict(model, dataset, device, doLog=True)

# colors = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000",
#           "#4472c4", "#91d024", "#b235e6", "#02ae75"]

colors = [
    "#10439F",
    "#874CCC",
    "#C65BCF",
    "#F27BBD"
]

fontsize = 25

x = list(range(len(dataset)))


def print(name, data):
    plt.figure(figsize=(30, 7), dpi=192, facecolor='none')
    # 添加标题和标签并设置字体大小
    plt.title(f'Result of {name}', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('Flow', fontsize=fontsize)
    # 调整刻度字体大小
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.plot(x, std, label="fact", color="black")
    plt.plot(x, data, label=name, color="#7028e4", linewidth=3)
    # 添加图例并设置字体大小
    plt.legend(fontsize=fontsize)
    # plt.show()
    plt.savefig(f"fig-std-{name}.png", bbox_inches="tight")
    plt.savefig(f"fig-std-{name}.svg", format="svg", bbox_inches="tight")


print("PatchTST", pst)
print("NLinear", nL)
print("RLinear", rL)
print("DLinear", dL)
