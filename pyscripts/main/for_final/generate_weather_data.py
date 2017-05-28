# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time
import copy

# 录入数据
weather_phase1 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/training/weather (table 7)_training_update.csv')
weather_test1 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSets/testing_phase1/weather (table 7)_test1.csv')
weather_phase2 = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/dataSets/dataSet_phase2/weather (table 7)_2.csv')

# 先把10月10号的天气补上
# 10.10号的用10.09和10.11的平均
date9_weather_data = weather_phase1[weather_phase1['date'] == '2016-10-09']
date11_weather_data = weather_phase1[weather_phase1['date'] == '2016-10-11']

date10_weather_data = date11_weather_data.copy()
date10_weather_data['date'] = '2016-10-10'
date10_weather_data['pressure'] = (date9_weather_data['pressure'].values + date11_weather_data['pressure'].values) / 2
date10_weather_data['sea_pressure'] = (date9_weather_data['sea_pressure'].values + date11_weather_data['sea_pressure'].values) / 2
date10_weather_data['wind_direction'] = (date9_weather_data['wind_direction'].values + date11_weather_data['wind_direction'].values) / 2
date10_weather_data['wind_speed'] = (date9_weather_data['wind_speed'].values + date11_weather_data['wind_speed'].values) / 2
date10_weather_data['temperature'] = (date9_weather_data['temperature'].values + date11_weather_data['temperature'].values) / 2
date10_weather_data['rel_humidity'] = (date9_weather_data['rel_humidity'].values + date11_weather_data['rel_humidity'].values) / 2
date10_weather_data['precipitation'] = (date9_weather_data['precipitation'].values + date11_weather_data['precipitation'].values) / 2

weather_phase1 = pd.concat([weather_phase1, date10_weather_data], axis=0)

weather_train = pd.concat([weather_phase1, weather_test1], axis=0)
weather_train.index = np.arange(len(weather_train))
weather_test = weather_phase2

# 然后填补其余缺失的天气值
full_time = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/pyscripts/analyze/full_time_forweather.csv')
weather_train = pd.merge(full_time, weather_train, on=['date','hour'], how='left')
weather_train = weather_train.fillna(method='ffill')

# 分割扩展
"""
weather_train['hour'] = weather_train['hour'].map(lambda x: ' ' + time(x, 0, 0).strftime('%H:%M:%S'))
weather_train['start_time'] = weather_train['date'] + weather_train['hour']
weather_train['start_time'] = weather_train['start_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
del weather_train['date']
del weather_train['hour']
num = len(weather_train)
for i in range(num):
    temp = weather_train.ix[i]
    temp1 = copy.deepcopy(temp)
    temp2 = copy.deepcopy(temp)
    temp3 = copy.deepcopy(temp)
    temp4 = copy.deepcopy(temp)
    temp5 = copy.deepcopy(temp)
    temp6 = copy.deepcopy(temp)
    temp7 = copy.deepcopy(temp)
    temp8 = copy.deepcopy(temp)
    stime = temp.start_time
    temp1.start_time = stime + timedelta(minutes=20)
    temp2.start_time = stime + timedelta(minutes=40)
    temp3.start_time = stime + timedelta(minutes=60)
    temp4.start_time = stime + timedelta(minutes=80)
    temp5.start_time = stime + timedelta(minutes=100)
    temp6.start_time = stime + timedelta(minutes=120)
    temp7.start_time = stime + timedelta(minutes=140)
    temp8.start_time = stime + timedelta(minutes=160)
    alltemp = [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8]
    alltemp = pd.DataFrame(alltemp)
    weather_train = pd.concat([weather_train, alltemp])
weather_train = weather_train.sort_values(by='start_time')
weather_train.index = np.arange(len(weather_train))

#
weather_test['hour'] = weather_test['hour'].map(lambda x: ' ' + time(x, 0, 0).strftime('%H:%M:%S'))
weather_test['start_time'] = weather_test['date'] + weather_test['hour']
weather_test['start_time'] = weather_test['start_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
del weather_test['date']
del weather_test['hour']
num = len(weather_test)
for i in range(num):
    temp = weather_test.ix[i]
    temp1 = copy.deepcopy(temp)
    temp2 = copy.deepcopy(temp)
    temp3 = copy.deepcopy(temp)
    temp4 = copy.deepcopy(temp)
    temp5 = copy.deepcopy(temp)
    temp6 = copy.deepcopy(temp)
    temp7 = copy.deepcopy(temp)
    temp8 = copy.deepcopy(temp)
    stime = temp.start_time
    temp1.start_time = stime + timedelta(minutes=20)
    temp2.start_time = stime + timedelta(minutes=40)
    temp3.start_time = stime + timedelta(minutes=60)
    temp4.start_time = stime + timedelta(minutes=80)
    temp5.start_time = stime + timedelta(minutes=100)
    temp6.start_time = stime + timedelta(minutes=120)
    temp7.start_time = stime + timedelta(minutes=140)
    temp8.start_time = stime + timedelta(minutes=160)
    alltemp = [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8]
    alltemp = pd.DataFrame(alltemp)
    weather_test = pd.concat([weather_test, alltemp])
weather_test = weather_test.sort_values(by='start_time')
weather_test.index = np.arange(len(weather_test))
"""

# 写出数据
weather_train.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/weather_train.csv', index=False)
weather_test.to_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/weather_test.csv', index=False)
