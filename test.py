import pandas as pd

# 读取CSV文件
file1 = pd.read_csv('1111.csv')  # 假设file1.csv有两列：ID, Time
file2 = pd.read_csv('modified_file.csv')  # 假设file2.csv有多列：ID, Time, Data1, Data2, ...

# 创建一个空的DataFrame来保存结果
result = pd.DataFrame()
i = 0
# 遍历file1中的每一行
for index, row in file1.iterrows():
    current_id = row['Camera ID']
    current_time = row['Time']
    
    # 目标时间范围
    target_times = list(range(current_time - 24, current_time))
    
    # 找到file2中相同ID且时间在目标时间范围内的数据
    filtered_data = file2[(file2['iu_ac'] == current_id) & (file2['index'].isin(target_times))]

    # 初始化用于保存最终结果的DataFrame
    final_data = pd.DataFrame()
    
    # 检查缺少的时间点并进行填充
    for t in target_times:
        if t in filtered_data['index'].values:
            # 如果时间点存在，直接添加
            final_data = final_data._append(filtered_data[filtered_data['index'] == t])
        else:
            # 如果时间点不存在，寻找相差24的倍数的数据
            candidates = file2[(file2['iu_ac'] == current_id) & ((file2['index'] - t) % 24 == 0)]
            
            if len(candidates) > 0:
                # 取这些数据的平均数作为新的值
                avg_row = candidates.mean(axis=0)
                avg_row['iu_ac'] = current_id
                avg_row['index'] = t
                final_data = final_data._append(avg_row, ignore_index=True)
            else:
                # 如果一个都没有找到，取之前找到的数据的第一个作为填充值
                if len(final_data) > 0:
                    fill_row = final_data.iloc[0].copy()
                else:
                    # 如果之前也一个都没找到，取file2中这个ID下时间大于或小于它的第一个点
                    nearest_row = file2[(file2['iu_ac'] == current_id)]
                    fill_row = nearest_row.iloc[0].copy()
                fill_row['index'] = t
                final_data = final_data._append(fill_row, ignore_index=True)
    
    # 将结果添加到结果DataFrame中
    result = pd.concat([result, final_data])

    i+=1
    if i == 2:
        break

# 保存结果到新的CSV文件
result.to_csv('prediction_data.csv', index=False)
