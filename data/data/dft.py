import os

data_times  = []
data_value  = []
data_which  = 0

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
    return x

data_which = prepare()

# Too short
assert len(data_times) > 1000

# Adjust the average of value to 0
average = sum(data_value) / len(data_value)
data_value = [value - average for value in data_value]

import sample as utils

# for i in range(1, 24):
#     periods.append(i);

# tx,ty,tz = input("Input the range:").split()
# tx,ty,tz = int(tx),int(ty),int(tz)

def func(tx, ty, tz):
    periods = []

    for i in range(1, tx):
        periods.append(24. / i)

    for i in range(1, ty):
        periods.append(24. * 7 / i)

    for i in range(1, tz):
        periods.append(24. * 365 / i)

    periods = list(set(sorted(periods)))

    # print("Setting: ", tx, ty, tz)
    params = utils.sample(data_times, data_value, periods)

    # print(params)
    loss = utils.calculate_loss(data_times, data_value, periods, params)
    print("Loss: ", loss)

    # Read a line
    line = input()
    data = line.split()

    # print(data)
    data = utils.predict(data, periods, params)
    # print(' '.join([str(x + average) for x in data]))

func(7, 30, 30)
