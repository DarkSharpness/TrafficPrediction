
FlatDataFile = '__exe__/flat.raw.csv'
FlatIdxFile = '__exe__/flat.idx.csv'

# 每个摄像头可用的训练数据的下标
FinetuneIdxFile = '__exe__/finetune.idx.csv'

# 预测数据,每行为一个向量,24个数字
PredictDataFile = '__exe__/pred.fmt.csv'
# 每个摄像头的每条预测的输入数据的下标
PredictDataIdxFile = '__exe__/pred.map.csv'
# 需要预测的摄像头的编号,csv,一列数字
# PredictIdFile = '__exe__/id_for_predict.csv'

# 可以用地理位置来预测学习的摄像头编号
GeoIndexList = '__exe__/geo_list.csv'
# 地理位置相关的训练数据路径
GeoTrainFilePath = '__exe__/geo_training'
# 地理位置相关的预测数据路径
GeoPredictFilePath = '__exe__/geo_predict'
