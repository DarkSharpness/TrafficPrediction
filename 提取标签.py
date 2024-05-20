import pandas as pd

# 读取原始CSV文件
input_file = 'new_file.csv'
data = pd.read_csv(input_file)


# 提取第二列的数据（现在的第一列）
second_column = data.iloc[:, 1].unique()

# 将结果保存到新的CSV文件中，并在第一行添加标签
new_data = pd.DataFrame(second_column, columns=['id'])

new_data.to_csv('id.csv', index=False)

