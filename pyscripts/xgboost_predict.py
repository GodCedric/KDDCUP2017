# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
XGBoost for prediction
Created on 2017/05/17
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

# 记录时间
start_tiem = time()

# 录入数据
travel_time_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/travel_time_train_data.csv')
volume_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/volume_train_data.csv')
travel_time_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/test_travel_time_data.csv')
volume_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/2.0/以最近值填充的/test_volume_data.csv')

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


# 分离标签和特征
Y_time = travel_time_train.avg_travel_time
X_time = travel_time_train.drop(['avg_travel_time'], axis=1)
val_Y_time = travel_time_val.avg_travel_time
val_X_time = travel_time_val.drop(['avg_travel_time'], axis=1)


Y_volume = volume_train.volume
X_volume = volume_train.drop(['volume'], axis=1)
val_Y_volume = volume_val.volume
val_X_volume = volume_val.drop(['volume'], axis=1)

# 平均时间
# 选择特征
X_time = X_time.drop(['intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time'], axis=1)
X_time = X_time.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
X_time = X_time.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
X_time = X_time.drop(['datemap', 'is_test'], axis=1)

val_X_time = val_X_time.drop(['intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time'], axis=1)
val_X_time = val_X_time.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
val_X_time = val_X_time.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
val_X_time = val_X_time.drop(['datemap', 'is_test'], axis=1)

# 测试集同步
travel_time_submission = travel_time_test_data[['intersection_id', 'tollgate_id', 'time_window']]
travel_time_test_data = travel_time_test_data.drop(['intersection_id', 'tollgate_id', 'time_window', 'start_time', 'date', 'time'], axis=1)
travel_time_test_data = travel_time_test_data.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
travel_time_test_data = travel_time_test_data.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
travel_time_test_data = travel_time_test_data.drop(['datemap'], axis=1)

# 流量
# 选择特征
X_volume = X_volume.drop(['tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time'], axis=1)
X_volume = X_volume.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
X_volume = X_volume.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
X_volume = X_volume.drop(['datemap', 'is_test'], axis=1)

val_X_volume = val_X_volume.drop(['tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time'], axis=1)
val_X_volume = val_X_volume.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
val_X_volume = val_X_volume.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
val_X_volume = val_X_volume.drop(['datemap', 'is_test'], axis=1)

# 测试集同步
volume_submission = volume_test_data[['tollgate_id', 'time_window', 'direction']]
volume_test_data = volume_test_data.drop(['tollgate_id', 'direction', 'time_window', 'start_time', 'date', 'time'], axis=1)
volume_test_data = volume_test_data.drop(['wind_direction', 'wind_speed', 'precipitation', 'SSD'], axis=1)
volume_test_data = volume_test_data.drop(['last_40min', 'last_60min', 'last_80min', 'last_100min', 'last_120min'], axis=1)
volume_test_data = volume_test_data.drop(['datemap'], axis=1)

# 字符串特征转换数值
X_time['route'] = pd.factorize(X_time['route'])[0]
val_X_time['route'] = pd.factorize(val_X_time['route'])[0]
travel_time_test_data['route'] = pd.factorize(travel_time_test_data['route'])[0]
X_volume['pair'] = pd.factorize(X_volume['pair'])[0]
val_X_volume['pair'] = pd.factorize(val_X_volume['pair'])[0]
volume_test_data['pair'] = pd.factorize(volume_test_data['pair'])[0]

# 转化格式
time_train = xgb.DMatrix(X_time, label=Y_time)
time_val = xgb.DMatrix(val_X_time, label=val_Y_time)
time_test = xgb.DMatrix(travel_time_test_data)

volume_train = xgb.DMatrix(X_volume, label=Y_volume)
volume_val = xgb.DMatrix(val_X_volume, label=val_Y_volume)
volume_test = xgb.DMatrix(volume_test_data)

# 参数设置
params = {
    'booster':'gbtree',
    'objective':'reg:linear',
    'early_stopping_rounds':100,
    'eta':0.085,
    'gamma':0.44,
    'max_depth':9,
    'min_child_weight':1,
    'subsample':0.8,
    'colsample_bytree':0.67,
    'alpha':5,
    'lambda':112,
    'seed':0,
}

# 训练
# 自定义评价函数
def MAPE(preds, dtrain):
    labels = dtrain.get_label()
    return 'MAPE', float(sum(np.fabs((labels - preds) / labels))) / len(labels)
plst = list(params.items())
num_rounds = 150
watchlist = [(time_train, 'train'), (time_val, 'val')]

time_model = xgb.train(plst, time_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", time_model.best_ntree_limit)

# 预测
time_predict = time_model.predict(time_test, ntree_limit=time_model.best_ntree_limit)
# 输出
travel_time_submission['avg_travel_time'] = time_predict
travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/xgboost1.0/travel_time_submission.csv', index=False)


params = {
    'booster':'gbtree',
    'objective':'reg:linear',
    'early_stopping_rounds':100,
    'eta':0.05,
    'gamma':0.3,
    'max_depth':6,
    'min_child_weight':2,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'alpha':0,
    'lambda':120,
    'seed':2017,
}

plst = list(params.items())
num_rounds = 850
watchlist = [(volume_train, 'train'), (volume_val, 'val')]

volume_model = xgb.train(plst, volume_train, num_rounds, watchlist, feval=MAPE)
print("best best_ntree_limit", volume_model.best_ntree_limit )

volume_predict = volume_model.predict(volume_test, ntree_limit=volume_model.best_ntree_limit)

volume_submission['volume'] = volume_predict
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/xgboost1.0/volume_submission.csv', index=False)