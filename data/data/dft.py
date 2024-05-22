import os

data_times  = []
data_value  = []

def prepare():
    x = int(input('Enter a number: '))
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

import dft_sample as utils

periods = [24, 24 * 7]

params = utils.sample(data_times, data_value, periods)

utils.plot_data_and_fit(data_times, data_value, periods, params)
