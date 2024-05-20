import csv
import os

# 定义输入文件和输出目录
input_file = 'new_file.csv'
output_dir = 'pre'

# 创建输出目录（如果不存在）


# 创建一个字典来存储文件指针
file_pointers = {}

try:
    # 打开输入文件进行读取
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        
        # 遍历每一行
        for row in reader:
            id_value = row['iu_ac']  # 获取ID字段的值
            
            # 如果该ID的文件指针不存在，则创建一个新的文件指针
            if id_value not in file_pointers:
                output_file = os.path.join(output_dir, f'{id_value}.csv')
                file_pointers[id_value] = open(output_file, 'w', newline='')
                writer = csv.DictWriter(file_pointers[id_value], fieldnames=reader.fieldnames)
                writer.writeheader()  # 写入表头
            
            # 写入当前行到对应的文件
            writer = csv.DictWriter(file_pointers[id_value], fieldnames=reader.fieldnames)
            writer.writerow(row)
finally:
    # 关闭所有文件指针
    for fp in file_pointers.values():
        fp.close()
