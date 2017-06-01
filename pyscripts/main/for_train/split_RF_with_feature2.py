# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
import copy

# 录入数据
travel_time_train = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/travel_time_train_data.csv')
volume_train = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/volume_train_data.csv')

test_travel_time = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/test_travel_time_data.csv')
test_volume = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/7.75/test_volume_data.csv')

# 分割路径，对
A2_train = travel_time_train[travel_time_train['route'] == 'A-2']
A3_train = travel_time_train[travel_time_train['route'] == 'A-3']
B1_train = travel_time_train[travel_time_train['route'] == 'B-1']
B3_train = travel_time_train[travel_time_train['route'] == 'B-3']
C1_train = travel_time_train[travel_time_train['route'] == 'C-1']
C3_train = travel_time_train[travel_time_train['route'] == 'C-3']

A2_test = test_travel_time[test_travel_time['route'] == 'A-2']
A3_test = test_travel_time[test_travel_time['route'] == 'A-3']
B1_test = test_travel_time[test_travel_time['route'] == 'B-1']
B3_test = test_travel_time[test_travel_time['route'] == 'B-3']
C1_test = test_travel_time[test_travel_time['route'] == 'C-1']
C3_test = test_travel_time[test_travel_time['route'] == 'C-3']

V10_train =  volume_train[volume_train['pair'] == '1-0']
V11_train =  volume_train[volume_train['pair'] == '1-1']
V20_train =  volume_train[volume_train['pair'] == '2-0']
V30_train =  volume_train[volume_train['pair'] == '3-0']
V31_train =  volume_train[volume_train['pair'] == '3-1']

V10_test =  test_volume[test_volume['pair'] == '1-0']
V11_test =  test_volume[test_volume['pair'] == '1-1']
V20_test =  test_volume[test_volume['pair'] == '2-0']
V30_test =  test_volume[test_volume['pair'] == '3-0']
V31_test =  test_volume[test_volume['pair'] == '3-1']

# 选择特征
time_columns = ['avg_travel_time', 'is_true',
                'month', 'day', 'weekday','holiday','timemap',
                'pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature',
                'rel_humidity', 'precipitation',
                'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1',
                'last_20min_C3', 'last_20min_V10', 'last_20min_V11', 'last_20min_V20',
                'last_20min_V30', 'last_20min_V31'
               ]

time_columns2 = ['month', 'day', 'weekday','holiday','timemap',
                'pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature',
                'rel_humidity', 'precipitation',
                'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1',
                'last_20min_C3', 'last_20min_V10', 'last_20min_V11', 'last_20min_V20',
                'last_20min_V30', 'last_20min_V31'
                ]

volume_columns = ['volume', 'is_true',
                  'month', 'day', 'weekday', 'holiday', 'timemap',
                  'pressure', 'sea_pressure','wind_direction', 'wind_speed', 'temperature',
                  'rel_humidity', 'precipitation',
                  'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1', 'last_20min_C3',
                  'last_20min_V10', 'last_20min_V11', 'last_20min_V20', 'last_20min_V30',
                  'last_20min_V31'
                 ]

volume_columns2 = ['month', 'day', 'weekday', 'holiday', 'timemap',
                  'pressure', 'sea_pressure','wind_direction', 'wind_speed', 'temperature',
                  'rel_humidity', 'precipitation',
                  'last_20min_A2', 'last_20min_A3', 'last_20min_B1', 'last_20min_B3', 'last_20min_C1', 'last_20min_C3',
                  'last_20min_V10', 'last_20min_V11', 'last_20min_V20', 'last_20min_V30',
                  'last_20min_V31'
                  ]

A2_train = A2_train[time_columns]
A3_train = A3_train[time_columns]
B1_train = B1_train[time_columns]
B3_train = B3_train[time_columns]
C1_train = C1_train[time_columns]
C3_train = C3_train[time_columns]

A2_test = A2_test[time_columns2]
A3_test = A3_test[time_columns2]
B1_test = B1_test[time_columns2]
B3_test = B3_test[time_columns2]
C1_test = C1_test[time_columns2]
C3_test = C3_test[time_columns2]

V10_train =  V10_train[volume_columns]
V11_train =  V11_train[volume_columns]
V20_train =  V20_train[volume_columns]
V30_train =  V30_train[volume_columns]
V31_train =  V31_train[volume_columns]

V10_test =  V10_test[volume_columns2]
V11_test =  V11_test[volume_columns2]
V20_test =  V20_test[volume_columns2]
V30_test =  V30_test[volume_columns2]
V31_test =  V31_test[volume_columns2]

# 评价函数
# 自定义评分函数
def MAPE2(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
score = make_scorer(MAPE2, greater_is_better=False)

# 训练预测
# 准备数据
A2_train_true = A2_train[A2_train['is_true'] == True]
A2_train_false = A2_train[A2_train['is_true'] == False]

del A2_train['is_true']
del A2_train_true['is_true']
del A2_train_false['is_true']

# 对n_estimators进行网格搜索
param_test1= {'n_estimators':range(30,150,10),'max_depth':range(3,8,1)}
base_estimator = RandomForestRegressor(random_state=2017)
gsearch1= GridSearchCV(estimator = base_estimator, param_grid =param_test1, scoring=score, cv=5)
gsearch1.fit(A2_train_true.iloc[:,1:], A2_train_true.iloc[:,0],)

# 输出最优参数
print(gsearch1.best_params_)

rf_base = RandomForestRegressor(n_estimators = 100, max_depth = 5, oob_score = True, random_state = 21)
test_score = cross_val_score(rf_base, A2_train_true.iloc[:,1:], A2_train_true.iloc[:,0], cv=5, scoring=score)
print(test_score)

# A2
rf_base1 = RandomForestRegressor(n_estimators = 100, max_depth = 5, oob_score = True, random_state = 10)

rf_base1.fit(A2_train.iloc[:,1:], A2_train.iloc[:,0],)
A2time_predict = rf_base1.predict(A2_test)

# 准备数据
A3_train_true = A3_train[A3_train['is_true'] == True]
A3_train_false = A3_train[A3_train['is_true'] == False]

del A3_train['is_true']
del A3_train_true['is_true']
del A3_train_false['is_true']

# 对n_estimators进行网格搜索
param_test1= {'n_estimators':range(30,150,10),'max_depth':range(3,8,1)}
base_estimator = RandomForestRegressor(random_state=2017)
gsearch1= GridSearchCV(estimator = base_estimator, param_grid =param_test1, scoring=score, cv=5)
gsearch1.fit(A3_train_true.iloc[:,1:], A3_train_true.iloc[:,0],)

# 输出最优参数
print(gsearch1.best_params_)

rf_base = RandomForestRegressor(n_estimators = 100, max_depth = 4, oob_score = True, random_state = 21)
test_score = cross_val_score(rf_base, A3_train_true.iloc[:,1:], A3_train_true.iloc[:,0], cv=5, scoring=score)
print(test_score)

# A2
rf_base2 = RandomForestRegressor(n_estimators = 100, max_depth = 4, oob_score = True, random_state = 10)

rf_base2.fit(A3_train.iloc[:,1:], A3_train.iloc[:,0],)
A3time_predict = rf_base2.predict(A3_test)

# 准备数据
B1_train_true = B1_train[B1_train['is_true'] == True]
B1_train_false = B1_train[B1_train['is_true'] == False]

del B1_train['is_true']
del B1_train_true['is_true']
del B1_train_false['is_true']

# 对n_estimators进行网格搜索
param_test1= {'n_estimators':range(30,150,10),'max_depth':range(3,8,1)}
base_estimator = RandomForestRegressor(random_state=2017)
gsearch1= GridSearchCV(estimator = base_estimator, param_grid =param_test1, scoring=score, cv=5)
gsearch1.fit(B1_train_true.iloc[:,1:], B1_train_true.iloc[:,0],)

# 输出最优参数
print(gsearch1.best_params_)

rf_base = RandomForestRegressor(n_estimators = 40, max_depth = 4, oob_score = True, random_state = 21)
test_score = cross_val_score(rf_base, B1_train_true.iloc[:,1:], B1_train_true.iloc[:,0], cv=5, scoring=score)
print(test_score)

# A2
rf_base3 = RandomForestRegressor(n_estimators = 40, max_depth = 4, oob_score = True, random_state = 10)

rf_base3.fit(B1_train.iloc[:,1:], B1_train.iloc[:,0],)
B1time_predict = rf_base3.predict(B1_test)

# 准备数据
B3_train_true = B3_train[B3_train['is_true'] == True]
B3_train_false = B3_train[B3_train['is_true'] == False]

del B3_train['is_true']
del B3_train_true['is_true']
del B3_train_false['is_true']

# 对n_estimators进行网格搜索
param_test1= {'n_estimators':range(30,150,10),'max_depth':range(3,8,1)}
base_estimator = RandomForestRegressor(random_state=2017)
gsearch1= GridSearchCV(estimator = base_estimator, param_grid =param_test1, scoring=score, cv=5)
gsearch1.fit(B3_train_true.iloc[:,1:], B3_train_true.iloc[:,0],)

# 输出最优参数
print(gsearch1.best_params_)

rf_base = RandomForestRegressor(n_estimators = 70, max_depth = 3, oob_score = True, random_state = 21)
test_score = cross_val_score(rf_base, B3_train_true.iloc[:,1:], B3_train_true.iloc[:,0], cv=5, scoring=score)
print(test_score)

# A2
rf_base4 = RandomForestRegressor(n_estimators = 70, max_depth = 3, oob_score = True, random_state = 10)

rf_base4.fit(B3_train.iloc[:,1:], B3_train.iloc[:,0],)
B3time_predict = rf_base4.predict(B3_test)


# 准备数据
C1_train_true = C1_train[C1_train['is_true'] == True]
C1_train_false = C1_train[C1_train['is_true'] == False]

del C1_train['is_true']
del C1_train_true['is_true']
del C1_train_false['is_true']

# 对n_estimators进行网格搜索
param_test1= {'n_estimators':range(30,150,10),'max_depth':range(3,8,1)}
base_estimator = RandomForestRegressor(random_state=2017)
gsearch1= GridSearchCV(estimator = base_estimator, param_grid =param_test1, scoring=score, cv=5)
gsearch1.fit(C1_train_true.iloc[:,1:], C1_train_true.iloc[:,0],)

# 输出最优参数
print(gsearch1.best_params_)

rf_base = RandomForestRegressor(n_estimators = 110, max_depth = 5, oob_score = True, random_state = 21)
test_score = cross_val_score(rf_base, C1_train_true.iloc[:,1:], C1_train_true.iloc[:,0], cv=5, scoring=score)
print(test_score)

# A2
rf_base5 = RandomForestRegressor(n_estimators = 110, max_depth = 5, oob_score = True, random_state = 10)

rf_base5.fit(C1_train.iloc[:,1:], C1_train.iloc[:,0],)
C1time_predict = rf_base5.predict(C1_test)

# 准备数据
C3_train_true = C3_train[C3_train['is_true'] == True]
C3_train_false = C3_train[C3_train['is_true'] == False]

del C3_train['is_true']
del C3_train_true['is_true']
del C3_train_false['is_true']

# 对n_estimators进行网格搜索
param_test1= {'n_estimators':range(30,150,10),'max_depth':range(3,8,1)}
base_estimator = RandomForestRegressor(random_state=2017)
gsearch1= GridSearchCV(estimator = base_estimator, param_grid =param_test1, scoring=score, cv=5)
gsearch1.fit(C3_train_true.iloc[:,1:], C3_train_true.iloc[:,0],)

# 输出最优参数
print(gsearch1.best_params_)

rf_base = RandomForestRegressor(n_estimators = 60, max_depth = 3, oob_score = True, random_state = 21)
test_score = cross_val_score(rf_base, C3_train_true.iloc[:,1:], C3_train_true.iloc[:,0], cv=5, scoring=score)
print(test_score)

# A2
rf_base6 = RandomForestRegressor(n_estimators = 110, max_depth = 5, oob_score = True, random_state = 10)

rf_base6.fit(C3_train.iloc[:,1:], C3_train.iloc[:,0],)
C3time_predict = rf_base6.predict(C3_test)

submission_travel_time = test_travel_time[['intersection_id','tollgate_id','time_window']]
predict_result = np.concatenate([A2time_predict,A3time_predict,B1time_predict,B3time_predict,C1time_predict,C3time_predict], axis=0)
submission_travel_time['avg_travel_time'] = predict_result

submission_travel_time.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.5/RF/travel_time_submission.csv')

