# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
各路线及对分割的RF模型
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation

### 读入数据
# 平均时间
A2_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A2_train_data.csv')
A3_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A3_train_data.csv')
B1_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B1_train_data.csv')
B3_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B3_train_data.csv')
C1_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C1_train_data.csv')
C3_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C3_train_data.csv')

A2_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A2_test_data.csv')
A3_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A3_test_data.csv')
B1_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B1_test_data.csv')
B3_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B3_test_data.csv')
C1_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C1_test_data.csv')
C3_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C3_test_data.csv')

# 流量
V10_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V10_train_data.csv')
V11_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V11_train_data.csv')
V20_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V20_train_data.csv')
V30_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V30_train_data.csv')
V31_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V31_train_data.csv')

V10_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V10_test_data.csv')
V11_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V11_test_data.csv')
V20_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V20_test_data.csv')
V30_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V30_test_data.csv')
V31_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V31_test_data.csv')


### 分离特征，标签
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


### 参数设置
rf_base1 = RandomForestRegressor(n_estimators = 110, max_depth = 6, oob_score = True, random_state = 10)


### 训练，预测
rf_base1.fit(A2_train_data, A2_label)
A2_test_data['avg_travel_time'] = rf_base1.predict(A2_test_data)

rf_base1.fit(A3_train_data, A3_label)
A3_test_data['avg_travel_time'] = rf_base1.predict(A3_test_data)

rf_base1.fit(B1_train_data, B1_label)
B1_test_data['avg_travel_time'] = rf_base1.predict(B1_test_data)

rf_base1.fit(B3_train_data, B3_label)
B3_test_data['avg_travel_time'] = rf_base1.predict(B3_test_data)

rf_base1.fit(C1_train_data, C1_label)
C1_test_data['avg_travel_time'] = rf_base1.predict(C1_test_data)

rf_base1.fit(C3_train_data, C3_label)
C3_test_data['avg_travel_time'] = rf_base1.predict(C3_test_data)


rf_base1.fit(V10_train_data, V10_label)
V10_test_data['volume'] = rf_base1.predict(V10_test_data)

rf_base1.fit(V11_train_data, V11_label)
V11_test_data['volume'] = rf_base1.predict(V11_test_data)

rf_base1.fit(V20_train_data, V20_label)
V20_test_data['volume'] = rf_base1.predict(V20_test_data)

rf_base1.fit(V30_train_data, V30_label)
V30_test_data['volume'] = rf_base1.predict(V30_test_data)

rf_base1.fit(V31_train_data, V31_label)
V31_test_data['volume'] = rf_base1.predict(V31_test_data)


### 整合及输出
travel_time_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/travel_time_submission.csv')
volume_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/volume_submission.csv')

temp1 = pd.concat([A2_test_data, A3_test_data, B1_test_data, B3_test_data, C1_test_data, C3_test_data], axis=0)
temp2 = pd.concat([V10_test_data, V11_test_data, V20_test_data, V30_test_data, V31_test_data], axis=0)

travel_time_submission['avg_travel_time'] = np.array(temp1.avg_travel_time)
volume_submission['volume'] = np.array(temp2.volume)

### 输出
travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/splitRF1.0/travel_time_submission.csv', index=False)
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/splitRF1.0/volume_submission.csv', index=False)
