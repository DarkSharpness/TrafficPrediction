import pandas as pd
import sys

# 读取第一个文件
file1 = pd.read_csv('1111.csv')

# 读取第二个文件
file2 = pd.read_csv('modified_file.csv')

# 创建一个空的DataFrame来保存结果
result = pd.DataFrame()

# 遍历第一个文件的每一行
for index, row in file1.iterrows():
    # 获取当前行的ID和时间标签
    current_id = row['Camera ID']
    current_time = row['Time']
    
    # 在第二个文件中找到相同ID的记录，并且时间标签在当前时间之前24个的24个数据
    selected_rows = file2[(file2['iu_ac'] == current_id) & (file2['index'] <= current_time)]

    if len(selected_rows) < 24:
        if len(selected_rows) == 0:
            num_missing = 24

            nmnmnm = file2[(file2['iu_ac'] == current_id) & (file2['index'] >= current_time)]

            nmnmnm = nmnmnm.head(24)

            padding_data = pd.DataFrame({col: [nmnmnm.iloc[0]['q']] * num_missing for col in file2.columns if col != 'index'})

            padding_data['index'] = [current_time - i  for i in range(num_missing, 0, -1)]

            padding_data['iu_ac'] = current_id  # 确保填充的数据ID正确
        
        else :

            num = 24 - len(selected_rows)

            padding_data = pd.concat([selected_rows.iloc[[0]]] * num, ignore_index=True)

            padding_data['index'] = [selected_rows.iloc[0]['index'] - i for i in range(num, 0, -1)]

        selected_rows = pd.concat([padding_data, selected_rows], ignore_index=True)
    else:
        # 只保留最后24个数据
        selected_rows = selected_rows.tail(24)
    # 将选定的行添加到结果中
    result = pd.concat([result, selected_rows])
    print(index)
    sys.stdout.flush()

# 重置结果的索引
result.reset_index(drop=True, inplace=True)

# 将结果保存到新的CSV文件中
result.to_csv('n.csv', index=False)




