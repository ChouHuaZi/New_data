from osgeo import gdal
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import GridSearchCV
np.seterr(divide='ignore', invalid='ignore')

def read_tiff(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_GeoTransform = dataset.GetGeoTransform()
    img_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float64)
    del dataset
    return img_proj, im_GeoTransform, im_data

def write_tiff(filename, im_proj, im_GeoTransform, im_data):
    datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_GeoTransform)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def getds(tiff):
    img_proj, im_GeoTransform, im_data = read_tiff(tiff)
    dss = gdal.Open(tiff)
    band1 = dss.GetRasterBand(1)
    invalid_value = band1.GetNoDataValue()
    index = np.where(im_data[0, :, :] == invalid_value)
    width, height = im_data.shape[1], im_data.shape[2]
    b_blue,b_green, b_red, b_red2, b_nir = im_data[0], im_data[1], im_data[2], im_data[3], im_data[4]
    v1 = (b_red+b_nir)/b_green# (3+5)/2
    v1[index] = 0
    v2 = b_blue/(b_green+b_nir)# 1/(2+5)
    v2[index] = 0
    v3 = (b_green+b_red2)/b_nir#(2+4)/5
    v3[index] = 0
    v4 =(b_red2-b_nir)/b_blue#(5+4)/1
    v4[index] = 0
    v5= (b_red+b_green)*b_nir#(2+3)*5
    v5[index] = 0
    v6 = (b_red + b_nir) / b_blue  # (3+5)/1
    v6[index] = 0
    v7 = (b_red2 + b_red) / b_nir  # (3+4)/5
    v7[index] = 0
    v8 = (b_nir - b_red) / (b_nir + b_red)  # (5-3)/(5+3)
    v8[index] = 0
    v9 = (b_nir - b_green) / (b_nir + b_green)  # (5-2)/(5+2)
    v9[index] = 0
    V1, V2, V3,V4 ,V5,V6,V7,V8,V9= v1.ravel(), v2.ravel(),v3.ravel(),v4.ravel(),v5.ravel(),v6.ravel(),v7.ravel(),v8.ravel(),v9.ravel()
    gaochun = b_red
    index1 = np.where(gaochun != 0)
    gaochun[index1] = 1
    return img_proj, im_GeoTransform, V1, V2, V3,V4,V5,V6,V7,V8,V9,gaochun



# 读取叶绿素a数据集
datasets = pd.read_excel("PK05.xlsx", header=0, sheet_name='Chl-a')
ds = np.array(datasets, dtype=np.float64)
input, output = ds[:, :-1], ds[:, -1]
input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.3)
# 设置超参数搜索范围
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5,10, 20],
    'min_samples_leaf': [1, 2, 4],
}

# 初始化随机森林回归器
rf_chl = RandomForestRegressor(random_state=0, n_jobs=-1)

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(estimator=rf_chl, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(input_train, output_train)

# 输出最佳参数和最佳得分
print("最佳参数:", grid_search.best_params_)
print("最佳 R² 分数:", grid_search.best_score_)

# 使用最佳参数训练最终模型
best_rf = grid_search.best_estimator_
best_rf.fit(input_train, output_train)
# 训练随机森林模型

y_train_pred = best_rf.predict(input_train)
y_test_pred = best_rf.predict(input_test)

print('Chl-a MSE train: %.3f, test: %.3f' % (mean_squared_error(output_train, y_train_pred), mean_squared_error(output_test, y_test_pred)))
print('Chl-a R^2 train: %.3f, test: %.3f' % (r2_score(output_train, y_train_pred), r2_score(output_test, y_test_pred)))



# 读取溶解氧数据集
datasets_do = pd.read_excel("PK05.xlsx", header=0, sheet_name='DO')
ds_do = np.array(datasets_do, dtype=np.float64)
input_do, output_do = ds_do[:, :-1], ds_do[:, -1]
input_train_do, input_test_do, output_train_do, output_test_do = train_test_split(input_do, output_do, test_size=0.3)

# 初始化随机森林回归器
rf_do = RandomForestRegressor(random_state=0, n_jobs=-1)

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(estimator=rf_do, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(input_train, output_train)
# 训练随机森林模型

y_train_pred_do = rf_do.predict(input_train_do)
y_test_pred_do = rf_do.predict(input_test_do)

print('DO MSE train: %.3f, test: %.3f' % (mean_squared_error(output_train_do, y_train_pred_do), mean_squared_error(output_test_do, y_test_pred_do)))
print('DO R^2 train: %.3f, test: %.3f' % (r2_score(output_train_do, y_train_pred_do), r2_score(output_test_do, y_test_pred_do)))

# 循环处理文件夹内的所有TIFF文件
input_folder = '原始影像'
output_folder = '溶解氧出图'

for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        tiff_path = os.path.join(input_folder, filename)
        img_proj, im_GeoTransform, V1, V2, V3, V4, V5, V6,V7,V8,V9,gaochun = getds(tiff_path)

        # 使用叶绿素a模型预测叶绿素a含量
        Gaochun_chl_a = rf_chl.predict(np.c_[V2, V4, V6, V9])
        DO_chl_a_feature = Gaochun_chl_a.reshape(gaochun.shape) * gaochun
        DO_chl_a_feature[DO_chl_a_feature < 0] = 0.0
        DO_chl_a_feature[DO_chl_a_feature > 50] = 45

        # 生成溶解氧的输入特征
        tif_dataset_do = np.c_[V1, DO_chl_a_feature.reshape(-1, 1)]
        tif_dataset_do[np.isinf(tif_dataset_do)] = 0.0
        imputer = SimpleImputer(strategy='mean')
        tif_dataset_do = imputer.fit_transform(tif_dataset_do)

        # 溶解氧预测
        final_predictions_do = rf_do.predict(tif_dataset_do)
        DO_final = final_predictions_do.reshape(gaochun.shape) * gaochun
        DO_final[DO_final < 0] = 0.0
        DO_final[DO_final > 50] = 45

        # 写入预测结果TIFF文件
        output_path = os.path.join(output_folder, f'DO_final_{filename}')
        write_tiff(output_path, img_proj, im_GeoTransform, DO_final)
        print(f'Processed {filename} and saved to {output_path}')
