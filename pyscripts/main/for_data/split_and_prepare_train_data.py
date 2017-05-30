# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
分割及准备训练集数据
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

### 录入数据
travel_time_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/travel_time_train_data.csv')
volume_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/volume_train_data.csv')
travel_time_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/test_travel_time_data.csv')
volume_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/test_volume_data.csv')


"""
travel_time_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/5.0/travel_time/包含1个月的训练集和测试集/travel_time_train_data.csv')
volume_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/5.0/volume/volume_train_data.csv')
travel_time_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/5.0/travel_time/包含1个月的训练集和测试集/test_travel_time_data.csv')
volume_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/加工过的数据集/5.0/volume/test_volume_data.csv')
"""

### 只取9.19之后的数据
start_date = date(2016,9,19)
travel_time_train_data['date'] = pd.to_datetime(travel_time_train_data['date'], format='%Y-%m-%d')
travel_time_train_data = travel_time_train_data[travel_time_train_data['date'] >= start_date]


### 删除十一节假日的数据
# 删除流量的节假日数据
volume_train_data = volume_train_data[volume_train_data['is_workday'] != 3]
dropindex = volume_train_data[(volume_train_data['pair'] == '1-0') & (volume_train_data['date'] == date(2016,9,30))].index
volume_train_data = volume_train_data.drop(dropindex, axis=0)


### 根据路线，收费站进出对分割数据
A2_train_data = travel_time_train_data[travel_time_train_data['route'] == 'A-2']
A3_train_data = travel_time_train_data[travel_time_train_data['route'] == 'A-3']
B1_train_data = travel_time_train_data[travel_time_train_data['route'] == 'B-1']
B3_train_data = travel_time_train_data[travel_time_train_data['route'] == 'B-3']
C1_train_data = travel_time_train_data[travel_time_train_data['route'] == 'C-1']
C3_train_data = travel_time_train_data[travel_time_train_data['route'] == 'C-3']

A2_test_data = travel_time_test_data[travel_time_test_data['route'] == 'A-2']
A3_test_data = travel_time_test_data[travel_time_test_data['route'] == 'A-3']
B1_test_data = travel_time_test_data[travel_time_test_data['route'] == 'B-1']
B3_test_data = travel_time_test_data[travel_time_test_data['route'] == 'B-3']
C1_test_data = travel_time_test_data[travel_time_test_data['route'] == 'C-1']
C3_test_data = travel_time_test_data[travel_time_test_data['route'] == 'C-3']

V10_train_data = volume_train_data[volume_train_data['pair'] == '1-0']
V11_train_data = volume_train_data[volume_train_data['pair'] == '1-1']
V20_train_data = volume_train_data[volume_train_data['pair'] == '2-0']
V30_train_data = volume_train_data[volume_train_data['pair'] == '3-0']
V31_train_data = volume_train_data[volume_train_data['pair'] == '3-1']

V10_test_data = volume_test_data[volume_test_data['pair'] == '1-0']
V11_test_data = volume_test_data[volume_test_data['pair'] == '1-1']
V20_test_data = volume_test_data[volume_test_data['pair'] == '2-0']
V30_test_data = volume_test_data[volume_test_data['pair'] == '3-0']
V31_test_data = volume_test_data[volume_test_data['pair'] == '3-1']



### 特征选择
travel_time_features = ['avg_travel_time',\
                        'hour', 'minute', 'weekday', 'timemap',\
                        'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                        #'last_20min',\
                        'SSD',\
                        'is_workday'
                       ]
travel_time_features2 = ['hour', 'minute', 'weekday', 'timemap',\
                        'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                        #'last_20min',\
                        'SSD',\
                        'is_workday'
                       ]
volume_features = ['volume',\
                   'hour', 'minute', 'weekday', 'timemap',\
                   'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                   #'last_20min',\
                   'SSD',\
                   'is_workday'
                  ]
volume_features2 = ['hour', 'minute', 'weekday', 'timemap',\
                   'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                   #'last_20min',\
                   'SSD',\
                   'is_workday'
                  ]
"""

# 独热 非量化
travel_time_features = ['avg_travel_time',\
                        'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                        #'SSD',\
                        #'last_20min',\
                        'hour__0', 'hour__1', 'hour__2', 'hour__3', 'hour__4',\
                        'hour__5', 'hour__6', 'hour__7', 'hour__8', 'hour__9', 'hour__10',\
                        'hour__11', 'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16',\
                        'hour__17', 'hour__18', 'hour__19', 'hour__20', 'hour__21', 'hour__22',\
                        'hour__23', 'minute__0', 'minute__20', 'minute__40', 'weekday__0',\
                        'weekday__1', 'weekday__2', 'weekday__3', 'weekday__4', 'weekday__5',\
                        'weekday__6', 'workday__1', 'workday__2',
                       ]
travel_time_features2 = ['pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                        #'SSD',\
                        #'last_20min',\
                        'hour__0', 'hour__1', 'hour__2', 'hour__3', 'hour__4',\
                        'hour__5', 'hour__6', 'hour__7', 'hour__8', 'hour__9', 'hour__10',\
                        'hour__11', 'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16',\
                        'hour__17', 'hour__18', 'hour__19', 'hour__20', 'hour__21', 'hour__22',\
                        'hour__23', 'minute__0', 'minute__20', 'minute__40', 'weekday__0',\
                        'weekday__1', 'weekday__2', 'weekday__3', 'weekday__4', 'weekday__5',\
                        'weekday__6', 'workday__1', 'workday__2',
                       ]
volume_features = ['volume',\
                   'pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                   #'SSD',\
                   #'last_20min',\
                   'hour__0', 'hour__1', 'hour__2', 'hour__3', 'hour__4',\
                   'hour__5', 'hour__6', 'hour__7', 'hour__8', 'hour__9', 'hour__10',\
                   'hour__11', 'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16',\
                   'hour__17', 'hour__18', 'hour__19', 'hour__20', 'hour__21', 'hour__22',\
                   'hour__23', 'minute__0', 'minute__20', 'minute__40', 'weekday__0',\
                   'weekday__1', 'weekday__2', 'weekday__3', 'weekday__4', 'weekday__5',\
                   'weekday__6', 'workday__1', 'workday__2',
                  ]
volume_features2 = ['pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation',\
                   #'SSD',\
                   #'last_20min',\
                   'hour__0', 'hour__1', 'hour__2', 'hour__3', 'hour__4',\
                   'hour__5', 'hour__6', 'hour__7', 'hour__8', 'hour__9', 'hour__10',\
                   'hour__11', 'hour__12', 'hour__13', 'hour__14', 'hour__15', 'hour__16',\
                   'hour__17', 'hour__18', 'hour__19', 'hour__20', 'hour__21', 'hour__22',\
                   'hour__23', 'minute__0', 'minute__20', 'minute__40', 'weekday__0',\
                   'weekday__1', 'weekday__2', 'weekday__3', 'weekday__4', 'weekday__5',\
                   'weekday__6', 'workday__1', 'workday__2',
                  ]
"""

A2_train_data = A2_train_data[travel_time_features]
A3_train_data = A3_train_data[travel_time_features]
B1_train_data = B1_train_data[travel_time_features]
B3_train_data = B3_train_data[travel_time_features]
C1_train_data = C1_train_data[travel_time_features]
C3_train_data = C3_train_data[travel_time_features]

A2_test_data = A2_test_data[travel_time_features2]
A3_test_data = A3_test_data[travel_time_features2]
B1_test_data = B1_test_data[travel_time_features2]
B3_test_data = B3_test_data[travel_time_features2]
C1_test_data = C1_test_data[travel_time_features2]
C3_test_data = C3_test_data[travel_time_features2]

V10_train_data = V10_train_data[volume_features]
V11_train_data = V11_train_data[volume_features]
V20_train_data = V20_train_data[volume_features]
V30_train_data = V30_train_data[volume_features]
V31_train_data = V31_train_data[volume_features]

V10_test_data = V10_test_data[volume_features2]
V11_test_data = V11_test_data[volume_features2]
V20_test_data = V20_test_data[volume_features2]
V30_test_data = V30_test_data[volume_features2]
V31_test_data = V31_test_data[volume_features2]

### 写出数据
A2_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A2_train_data.csv', index=False)
A3_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A3_train_data.csv', index=False)
B1_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B1_train_data.csv', index=False)
B3_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B3_train_data.csv', index=False)
C1_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C1_train_data.csv', index=False)
C3_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C3_train_data.csv', index=False)

A2_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A2_test_data.csv', index=False)
A3_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/A3_test_data.csv', index=False)
B1_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B1_test_data.csv', index=False)
B3_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/B3_test_data.csv', index=False)
C1_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C1_test_data.csv', index=False)
C3_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/C3_test_data.csv', index=False)

V10_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V10_train_data.csv', index=False)
V11_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V11_train_data.csv', index=False)
V20_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V20_train_data.csv', index=False)
V30_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V30_train_data.csv', index=False)
V31_train_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V31_train_data.csv', index=False)

V10_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V10_test_data.csv', index=False)
V11_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V11_test_data.csv', index=False)
V20_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V20_test_data.csv', index=False)
V30_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V30_test_data.csv', index=False)
V31_test_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/训练数据/分割_非独热_非量化天气/V31_test_data.csv', index=False)