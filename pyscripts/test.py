# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta,date,time
import copy

in_file_path = '/home/godcedric/GitLocal/KDDCUP2017/result/training_20min_avg_travel_time.csv'
raw_data = pd.read_csv(in_file_path)
in_file_path = '/home/godcedric/GitLocal/KDDCUP2017/result/weather (table 7)_training_update.csv'
weather_data = pd.read_csv(in_file_path)

# 处理数据，合并平均时间和天气情况
raw_data['start_time'] = raw_data['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S'))
weather_data['date'] = weather_data['date'].astype(str)
weather_data['date'] = weather_data['date'].astype(str)
weather_data['hour'] = weather_data['hour'].map(lambda x: ' ' + time(x,0,0).strftime('%H:%M:%S'))
weather_data['start_time'] = weather_data['date'] + weather_data['hour']
weather_data['start_time'] = weather_data['start_time'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

num = len(weather_data)
for i in range(num):
    temp = weather_data.ix[i]
    temp1 = copy.deepcopy(temp);temp2 = copy.deepcopy(temp);temp3 = copy.deepcopy(temp);temp4 = copy.deepcopy(temp)
    temp5 = copy.deepcopy(temp);temp6 = copy.deepcopy(temp);temp7 = copy.deepcopy(temp);temp8 = copy.deepcopy(temp)
    stime = temp.start_time
    temp1.start_time = stime + timedelta(minutes=20)
    temp2.start_time = stime + timedelta(minutes=40)
    temp3.start_time = stime + timedelta(minutes=60)
    temp4.start_time = stime + timedelta(minutes=80)
    temp5.start_time = stime + timedelta(minutes=100)
    temp6.start_time = stime + timedelta(minutes=120)
    temp7.start_time = stime + timedelta(minutes=140)
    temp8.start_time = stime + timedelta(minutes=160)
    alltemp = [temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8]
    alltemp = pd.DataFrame(alltemp)
    weather_data = pd.concat([weather_data,alltemp])
weather_data.sort('start_time')

process_data = pd.merge(raw_data, weather_data, on='start_time', how='left')

# 分析