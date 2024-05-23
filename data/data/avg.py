import os
from matplotlib import pyplot as plt

data_times  = []
data_value  = []

def prepare():
    x = int(input())
    path = f"index/{x}.csv"
    # if no {x}.csv, exit
    if os.path.exists(path) == False:
        print('File do not exists')
        exit()

    data_file = open(path, 'r')

    for line in data_file:
        # use , to split the line
        line = line.split(',')
        data_times.append(int(line[0]))
        data_value.append(float(line[1]))

    data_file.close()

prepare()

# Adjust the average of value to 0
average = sum(data_value) / len(data_value)
data_value = [value - average for value in data_value]

# Average sum of the value of certain time of certain day in the week
avg = []

for i in range(7 * 24):
    avg.append([])

for i in range(len(data_times)):
    index = data_times[i] % (7 * 24)
    avg[index].append(data_value[i])

for i in range(7 * 24):
    avg[i] = sum(avg[i]) / len(avg[i])

# Calculate the loss
data_predict = []
loss = 0
for i in range(len(data_times)):
    index = data_times[i] % (7 * 24)
    data_predict.append(avg[index])
    loss += (data_value[i] - avg[index]) ** 2
loss /= len(data_times)

print(loss)

def display(x,y,y_fit):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')  # 原始数据点
    plt.plot(x, y_fit, '-', label='Fitted Function')  # 拟合函数

    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title('Comparison of Original Data and Fitted Function')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # 显示图表
    plt.show()

display(data_times, data_value, data_predict)
