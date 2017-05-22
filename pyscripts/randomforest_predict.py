# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
RandomForest for prediction
Created on 2017/05/17
"""


from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time


# 录入数据
travel_time_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/travel_time_train_data.csv')
volume_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/volume_train_data.csv')
travel_time_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/test_travel_time_data.csv')
volume_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/test_volume_data.csv')

'''
# 分割训练集和验证集
test_date = ['2016-10-11', '2016-10-12', '2016-10-13', '2016-10-14', '2016-10-15', '2016-10-16', '2016-10-17']
def isin(x):
    if x in test_date:
        return 1
    else:
        return 0
# 平均时间
travel_time_train_data['is_test'] = travel_time_train_data['date'].map(isin)
travel_time_train = travel_time_train_data[travel_time_train_data['is_test'] == 0]
travel_time_val = travel_time_train_data[travel_time_train_data['is_test'] == 1]
# 流量
volume_train_data['is_test'] = volume_train_data['date'].map(isin)
volume_train = volume_train_data[volume_train_data['is_test'] == 0]
volume_val = volume_train_data[volume_train_data['is_test'] == 1]
del travel_time_train_data['is_test']
del volume_train_data['is_test']
'''

# 分离标签和特征
Y_time = travel_time_train_data.avg_travel_time
X_time = travel_time_train_data.drop(['avg_travel_time'], axis=1)
#val_Y_time = travel_time_val.avg_travel_time
#val_X_time = travel_time_val.drop(['avg_travel_time'], axis=1)

Y_volume = volume_train_data.volume
X_volume = volume_train_data.drop(['volume'], axis=1)
#val_Y_volume = volume_val.volume
#val_X_volume = volume_val.drop(['volume'], axis=1)

# 平均时间
# 选择特征
X_time = X_time.drop(['intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time'], axis=1)
X_time = X_time.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
X_time = X_time.drop(['last_20min', 'last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
X_time = X_time.drop(['datemap'], axis=1)

#val_X_time = val_X_time.drop(['intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time'], axis=1)
#val_X_time = val_X_time.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
#val_X_time = val_X_time.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
#val_X_time = val_X_time.drop(['datemap', 'is_test'], axis=1)

# 测试集同步
travel_time_submission = travel_time_test_data[['intersection_id', 'tollgate_id', 'time_window']]
travel_time_test_data = travel_time_test_data.drop(['intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time'], axis=1)
travel_time_test_data = travel_time_test_data.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
travel_time_test_data = travel_time_test_data.drop(['last_20min', 'last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
travel_time_test_data = travel_time_test_data.drop(['datemap'], axis=1)

# 流量
# 选择特征
X_volume = X_volume.drop(['tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time'], axis=1)
X_volume = X_volume.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
X_volume = X_volume.drop(['last_20min', 'last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
X_volume = X_volume.drop(['datemap'], axis=1)

#val_X_volume = val_X_volume.drop(['tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time'], axis=1)
#val_X_volume = val_X_volume.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
#val_X_volume = val_X_volume.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
#val_X_volume = val_X_volume.drop(['datemap', 'is_test'], axis=1)

# 测试集同步
volume_submission = volume_test_data[['tollgate_id', 'time_window', 'direction']]
volume_test_data = volume_test_data.drop(['tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time'], axis=1)
volume_test_data = volume_test_data.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
volume_test_data = volume_test_data.drop(['last_20min', 'last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
volume_test_data = volume_test_data.drop(['datemap'], axis=1)

# 字符串特征转换数值
X_time['route'] = pd.factorize(X_time['route'])[0]
#val_X_time['route'] = pd.factorize(val_X_time['route'])[0]
travel_time_test_data['route'] = pd.factorize(travel_time_test_data['route'])[0]
X_volume['pair'] = pd.factorize(X_volume['pair'])[0]
#val_X_volume['pair'] = pd.factorize(val_X_volume['pair'])[0]
volume_test_data['pair'] = pd.factorize(volume_test_data['pair'])[0]

# 训练
# 自定义评价函数
def MAPE(preds, realval):
    return 'MAPE', float(sum(np.fabs((realval - preds) / realval))) / len(realval)

# 默认参数
rf_default = RandomForestRegressor(oob_score = True, random_state = 20)
rf_default.fit(X_time, Y_time)
print(rf_default.oob_score_)

# 预测
preds = rf_default.predict(travel_time_test_data)

# 输出
travel_time_submission['avg_travel_time'] = preds
travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/RF1.0/travel_time_submission.csv', index=False)

rf_default2 = RandomForestRegressor(oob_score = True, random_state = 20)
rf_default2.fit(X_volume, Y_volume)
print(rf_default2.oob_score_)

# 预测
preds2 = rf_default2.predict(volume_test_data)

# 输出
volume_submission['volume'] = preds2
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/RF1.0/volume_submission.csv', index=False)