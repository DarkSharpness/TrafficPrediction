import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.spatial import distance_matrix

# 读取数据文件
filename = '1.tmp'
data = pd.read_csv(filename, sep=';')

# 解析geo_point_2d为点
data['latitude'] = data['geo_point_2d'].apply(lambda x: float(x.split(',')[0].strip()))
data['longitude'] = data['geo_point_2d'].apply(lambda x: float(x.split(',')[1].strip()))
data['point'] = data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

# 解析geo_shape为LineString
data['geo_shape'] = data['geo_shape'].apply(lambda x: eval(x.replace('""', '"').replace('"', '\\"').replace('\\"', '"')))
data['geometry'] = data['geo_shape'].apply(lambda x: LineString(x['coordinates']) if x['type'] == 'LineString' else None)

# 创建GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry='geometry')

# 计算距离矩阵
coords = gdf['point'].apply(lambda geom: (geom.x, geom.y)).tolist()
dist_matrix = distance_matrix(coords, coords)

# 查找最近的10个节点
nearest_nodes = {}
for i, dist in enumerate(dist_matrix):
    sorted_indices = dist.argsort()[1:11]  # 排序并排除自身，取前10个
    nearest_nodes[gdf['iu_ac'].iloc[i]] = gdf['iu_ac'].iloc[sorted_indices].tolist()

# 按照 iu_ac 排序
nearest_nodes = dict(sorted(nearest_nodes.items(), key=lambda x: x[0]))

out = open('2.tmp', 'w')
for k, v in nearest_nodes.items():
    out.write(f"{k}: {v}\n")
