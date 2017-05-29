# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
kNN预测及kNN填充last_20缺失值
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time
import sklearn.neighbors
from sklearn import preprocessing

### 录入数据
# 平均时间
A2_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/A2_train_data.csv')
A3_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/A3_train_data.csv')
B1_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/B1_train_data.csv')
B3_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/B3_train_data.csv')
C1_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/C1_train_data.csv')
C3_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/C3_train_data.csv')

A2_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/A2_test_data.csv')
A3_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/A3_test_data.csv')
B1_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/B1_test_data.csv')
B3_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/B3_test_data.csv')
C1_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/C1_test_data.csv')
C3_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/C3_test_data.csv')

# 流量
V10_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V10_train_data.csv')
V11_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V11_train_data.csv')
V20_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V20_train_data.csv')
V30_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V30_train_data.csv')
V31_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V31_train_data.csv')

V10_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V10_test_data.csv')
V11_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V11_test_data.csv')
V20_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V20_test_data.csv')
V30_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V30_test_data.csv')
V31_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_独热_非量化天气/V31_test_data.csv')


### 分离特征与标签
A2_label = A2_train_data.avg_travel_time
A2_train_data = A2_train_data.drop(['avg_travel_time'], axis=1)

A3_label = A3_train_data.avg_travel_time
A3_train_data = A3_train_data.drop(['avg_travel_time'], axis=1)

B1_label = B1_train_data.avg_travel_time
B1_train_data = B1_train_data.drop(['avg_travel_time'], axis=1)

B3_label = B3_train_data.avg_travel_time
B3_train_data = B3_train_data.drop(['avg_travel_time'], axis=1)

C1_label = C1_train_data.avg_travel_time
C1_train_data = C1_train_data.drop(['avg_travel_time'], axis=1)

C3_label = C3_train_data.avg_travel_time
C3_train_data = C3_train_data.drop(['avg_travel_time'], axis=1)

V10_label = V10_train_data.volume
V10_train_data = V10_train_data.drop(['volume'], axis=1)

V11_label = V11_train_data.volume
V11_train_data = V11_train_data.drop(['volume'], axis=1)

V20_label = V20_train_data.volume
V20_train_data = V20_train_data.drop(['volume'], axis=1)

V30_label = V30_train_data.volume
V30_train_data = V30_train_data.drop(['volume'], axis=1)

V31_label = V31_train_data.volume
V31_train_data = V31_train_data.drop(['volume'], axis=1)

### 数据归一化
min_max_scaler = preprocessing.MinMaxScaler()
time_columns = A2_train_data.columns
volume_columns = V10_train_data.columns

# 归一化
temp = min_max_scaler.fit_transform(A2_train_data)
A2_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(A2_test_data)
A2_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(A3_train_data)
A3_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(A3_test_data)
A3_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(B1_train_data)
B1_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(B1_test_data)
B1_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(B3_train_data)
B3_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(B3_test_data)
B3_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(C1_train_data)
C1_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(C1_test_data)
C1_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(C3_train_data)
C3_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(C3_test_data)
C3_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(V10_train_data)
V10_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(V10_test_data)
V10_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(V11_train_data)
V11_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(V11_test_data)
V11_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(V20_train_data)
V20_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(V20_test_data)
V20_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(V30_train_data)
V30_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(V30_test_data)
V30_test_data = pd.DataFrame(temp, columns=time_columns)

temp = min_max_scaler.fit_transform(V31_train_data)
V31_train_data = pd.DataFrame(temp, columns=time_columns)
temp = min_max_scaler.transform(V31_test_data)
V31_test_data = pd.DataFrame(temp, columns=time_columns)

### kNN预测
#KNeighborsRegressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 5, weights='distance')
KNeighborsRegressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 4, weights='distance')

# 训练，预测
KNeighborsRegressor.fit(A2_train_data, A2_label)
A2_test_data['avg_travel_time'] = KNeighborsRegressor.predict(A2_test_data)

KNeighborsRegressor.fit(A3_train_data, A3_label)
A3_test_data['avg_travel_time'] = KNeighborsRegressor.predict(A3_test_data)

KNeighborsRegressor.fit(B1_train_data, B1_label)
B1_test_data['avg_travel_time'] = KNeighborsRegressor.predict(B1_test_data)

KNeighborsRegressor.fit(B3_train_data, B3_label)
B3_test_data['avg_travel_time'] = KNeighborsRegressor.predict(B3_test_data)

KNeighborsRegressor.fit(C1_train_data, C1_label)
C1_test_data['avg_travel_time'] = KNeighborsRegressor.predict(C1_test_data)

KNeighborsRegressor.fit(C3_train_data, C3_label)
C3_test_data['avg_travel_time'] = KNeighborsRegressor.predict(C3_test_data)

KNeighborsRegressor.fit(V10_train_data, V10_label)
V10_test_data['volume'] = KNeighborsRegressor.predict(V10_test_data)

KNeighborsRegressor.fit(V11_train_data, V11_label)
V11_test_data['volume'] = KNeighborsRegressor.predict(V11_test_data)

KNeighborsRegressor.fit(V20_train_data, V20_label)
V20_test_data['volume'] = KNeighborsRegressor.predict(V20_test_data)

KNeighborsRegressor.fit(V30_train_data, V30_label)
V30_test_data['volume'] = KNeighborsRegressor.predict(V30_test_data)

KNeighborsRegressor.fit(V31_train_data, V31_label)
V31_test_data['volume'] = KNeighborsRegressor.predict(V31_test_data)

# 整合
travel_time_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/原始数据/submission_sample_travelTime.csv')
volume_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/原始数据/submission_sample_volume.csv')

temp1 = pd.concat([A2_test_data, A3_test_data, B1_test_data, B3_test_data, C1_test_data, C3_test_data], axis=0)
temp2 = pd.concat([V10_test_data, V11_test_data, V20_test_data, V30_test_data, V31_test_data], axis=0)

travel_time_submission['avg_travel_time'] = np.array(temp1.avg_travel_time)
volume_submission['volume'] = np.array(temp2.volume)

# 输出
travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.0/kNN/travel_time_submission.csv', index=False)
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.0/kNN/volume_submission.csv', index=False)

"""
### kNN填充last_20min缺失值
travel_time_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/test_travel_time_data.csv')
volume_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/test_volume_data.csv')

# 排好序
travel_time_test_data['start_time'] = travel_time_test_data['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))
volume_test_data['start_time'] = volume_test_data['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))
travel_time_test_data = travel_time_test_data.sort_values(by = ['intersection_id', 'tollgate_id', 'start_time'])
travel_time_test_data.index = np.arange(len(travel_time_test_data))
del travel_time_test_data['start_time']
volume_test_data = volume_test_data.sort_values(by = ['tollgate_id', 'direction', 'start_time'])
volume_test_data.index = np.arange(len(volume_test_data))
del volume_test_data['start_time']

# 填充值
time_target = travel_time_submission['avg_travel_time'].values
travel_time_test_data = travel_time_test_data.set_index(np.arange(len(travel_time_test_data)))
for i in range(len(travel_time_test_data)):
    if i % 6 == 1:
        travel_time_test_data['last_20min'][i] = time_target[i-1]
    if i % 6 == 2:
        travel_time_test_data['last_20min'][i] = time_target[i-1]
        travel_time_test_data['last_40min'][i] = time_target[i-2]
    if i % 6 == 3:
        travel_time_test_data['last_20min'][i] = time_target[i-1]
        travel_time_test_data['last_40min'][i] = time_target[i-2]
        travel_time_test_data['last_60min'][i] = time_target[i-3]
    if i % 6 == 4:
        travel_time_test_data['last_20min'][i] = time_target[i-1]
        travel_time_test_data['last_40min'][i] = time_target[i-2]
        travel_time_test_data['last_60min'][i] = time_target[i-3]
        travel_time_test_data['last_80min'][i] = time_target[i-4]
    if i % 6 == 5:
        travel_time_test_data['last_20min'][i] = time_target[i-1]
        travel_time_test_data['last_40min'][i] = time_target[i-2]
        travel_time_test_data['last_60min'][i] = time_target[i-3]
        travel_time_test_data['last_80min'][i] = time_target[i-4]
        travel_time_test_data['last_100min'][i] = time_target[i-5]

volume_target = volume_submission['volume'].values
volume_test_data = volume_test_data.set_index(np.arange(len(volume_test_data)))
for i in range(len(volume_test_data)):
    if i % 6 == 1:
        volume_test_data['last_20min'][i] = volume_target[i-1]
    if i % 6 == 2:
        volume_test_data['last_20min'][i] = volume_target[i-1]
        volume_test_data['last_40min'][i] = volume_target[i-2]
    if i % 6 == 3:
        volume_test_data['last_20min'][i] = volume_target[i-1]
        volume_test_data['last_40min'][i] = volume_target[i-2]
        volume_test_data['last_60min'][i] = volume_target[i-3]
    if i % 6 == 4:
        volume_test_data['last_20min'][i] = volume_target[i-1]
        volume_test_data['last_40min'][i] = volume_target[i-2]
        volume_test_data['last_60min'][i] = volume_target[i-3]
        volume_test_data['last_80min'][i] = volume_target[i-4]
    if i % 6 == 5:
        volume_test_data['last_20min'][i] = volume_target[i-1]
        volume_test_data['last_40min'][i] = volume_target[i-2]
        volume_test_data['last_60min'][i] = volume_target[i-3]
        volume_test_data['last_80min'][i] = volume_target[i-4]
        volume_test_data['last_100min'][i] = volume_target[i-5]

# 写出数据
travel_time_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/travel_time_test_data.csv', index=False)
volume_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/volume_test_data.csv', index=False)
"""