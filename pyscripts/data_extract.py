# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
提取，整合数据
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time
import copy

# 提取数据
def extract_data():

    #####**********读入原始数据**********#####

    # 读入平均时间，流量和天气数据（平均时间数据由aggregate_travel_time.py生成）
    in_file_path = '/home/godcedric/GitLocal/KDDCUP2017/result/training_20min_avg_travel_time.csv'
    raw_data = pd.read_csv(in_file_path)
    in_file_path = '/home/godcedric/GitLocal/KDDCUP2017/result/training_20min_avg_volume.csv'
    raw_data2 = pd.read_csv(in_file_path)
    in_file_path = '/home/godcedric/GitLocal/KDDCUP2017/result/weather (table 7)_training_update.csv'
    weather_data = pd.read_csv(in_file_path)

    #####**********提取平均时间数据**********#####

    # 整合平均时间与天气数据
    raw_data['start_time'] = raw_data['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))
    weather_data['date'] = weather_data['date'].astype(str)
    weather_data['hour'] = weather_data['hour'].map(lambda x: ' ' + time(x, 0, 0).strftime('%H:%M:%S'))
    weather_data['start_time'] = weather_data['date'] + weather_data['hour']
    weather_data['start_time'] = weather_data['start_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    num = len(weather_data)
    for i in range(num):
        temp = weather_data.ix[i]
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
        weather_data = pd.concat([weather_data, alltemp])
    weather_data.sort('start_time')

    process_data = pd.merge(raw_data, weather_data, on='start_time', how='left')
    del process_data['hour']
    process_data['time'] = process_data['start_time'].map(lambda x: x.time())

    # 增加路线和星期几两列
    process_data['weekday'] = process_data['start_time'].map(lambda x: x.weekday())
    process_data['route'] = process_data['intersection_id'].astype(str) + '-' + process_data['tollgate_id'].astype(str)

    # 增加前20分钟平均时间数据
    start_time = process_data['start_time']
    avg_travel_time = process_data['avg_travel_time']
    avg_travel_time.index = list(process_data['start_time'])
    last_20min = list(process_data['avg_travel_time'])
    for i in range(len(start_time)):
        cur_time = start_time[i]
        last_time = cur_time - timedelta(minutes=20)
        if i == 0:
            last_20min[i] = np.nan
            continue
        if last_time == start_time[i - 1]:
            last_20min[i] = avg_travel_time[last_time]
        else:
            last_20min[i] = np.nan
    process_data['last_20min'] = last_20min


    # 写出数据
    process_data.to_csv('travel_time_data.csv')


    #####**********提取流量数据**********#####

    # 整合流量与天气数据
    raw_data2['start_time'] = raw_data2['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S'))
    raw_data2['pair'] = raw_data2['tollgate_id'].astype(str) + '-' + raw_data2['direction'].astype(str)
    process_data2 = pd.merge(raw_data2, weather_data, on='start_time', how='left')

    # 增加星期几和时间窗口两列
    process_data2['weekday'] = process_data2['start_time'].map(lambda x: x.weekday())
    process_data2['time'] = process_data2['start_time'].map(lambda x: x.time())

    # 写出数据
    process_data2.to_csv('volume_data.csv')

def main():

    extract_data()


if __name__ == '__main__':
    main()
