# 关于项目架构

- data.h: 核心头文件
- fixP.cpp: 修复预测数据，补全前 24 小时的数据
- fixT.cpp: 清洗训练数据，去除训练数据中连续的 0 和前缀 0
- flatten.h: 展平数据的头文件
- flatten.cpp: 展平数据，生成  pre_train/finetune 的数据
- geo_*.cpp: 地理信息处理相关
- knn.cpp: 没啥用
- predict.cpp: 直接预处理预测数据
- result.cpp: 清洗预测结果
- spliter.cpp: 暂时没啥用，只是生成一个 list，可能用于地理信息处理
- train.cpp: 直接预处理训练数据
