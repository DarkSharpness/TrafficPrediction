import pandas as pd
import numpy as np

input_file = 'result/PST2_2.csv'
name, ext = input_file.rsplit('.', 1)
output_file = f"{name}_noneg.{ext}"

# read Dlinear4.csv
df = pd.read_csv(input_file)
# read loop_sensor_test_baseline.csv
df2 = pd.read_csv('result/NLinear2_noneg.csv')

# get the index of the line with negative value
index = df[df['estimate_q'] < 0.05].index.tolist()

# substitute the negative value with the data in loop_sensor_test_baseline.csv
for i in index:
	df['estimate_q'][i] = df2['estimate_q'][i]

# save the new data to Dlinear4.csv
df.to_csv(output_file, index=False)
