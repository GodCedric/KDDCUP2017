# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
提取，整合天气数据
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time
import copy

# 提取数据
def extract_data(travel_time_infile, volume_infile, weather_infile, test_travel_time_infile, test_volume_infile, test_weather_infile):

  ###----------------训练集------------------###

    #####**********读入原始数据**********#####

    # 读入平均时间，流量和天气数据（平均时间数据由aggregate_travel_time.py生成）
    raw_data = pd.read_csv(travel_time_infile)
    raw_data2 = pd.read_csv(volume_infile)
    weather_data = pd.read_csv(weather_infile)

    #####**********提取平均时间数据**********#####

    """
    # 训练集缺失值补充，保证每一个时间段都有数据
    full_time = pd.read_csv('full_time.csv')
    full_time = full_time.sort(['intersection_id', 'tollgate_id'])
    raw_data = pd.merge(full_time, raw_data, on=['intersection_id', 'tollgate_id', 'time_window'], how='left')
    ffill = lambda g: g.fillna(method='ffill')
    bfill = lambda g: g.fillna(method='bfill')

    raw_data['avg_travel_time'] = raw_data.groupby(['intersection_id', 'tollgate_id'])['avg_travel_time'].apply(ffill)
    raw_data['avg_travel_time'] = raw_data.groupby(['intersection_id', 'tollgate_id'])['avg_travel_time'].apply(bfill)

    full_time2 = pd.read_csv('full_time2.csv')
    full_time2 = full_time2.sort(['tollgate_id', 'direction'])
    raw_data2 = pd.merge(full_time2, raw_data2, on=['tollgate_id', 'time_window', 'direction'], how='left')
    raw_data2['volume'] = raw_data2.groupby(['tollgate_id', 'direction'])['volume'].apply(ffill)
    raw_data2['volume'] = raw_data2.groupby(['tollgate_id', 'direction'])['volume'].apply(bfill)
    """

    # 填补10月10号的天气缺失数据
    # 10.10号的用10.09和10.11的平均
    date9_weather_data = weather_data[weather_data['date'] == '2016-10-09']
    date11_weather_data = weather_data[weather_data['date'] == '2016-10-11']

    date10_weather_data = date11_weather_data.copy()
    date10_weather_data['date'] = '2016-10-10'
    date10_weather_data['pressure'] = (date9_weather_data['pressure'].values + date11_weather_data['pressure'].values) / 2
    date10_weather_data['sea_pressure'] = (date9_weather_data['sea_pressure'].values + date11_weather_data[
        'sea_pressure'].values) / 2
    date10_weather_data['wind_direction'] = (date9_weather_data['wind_direction'].values + date11_weather_data[
        'wind_direction'].values) / 2
    date10_weather_data['wind_speed'] = (date9_weather_data['wind_speed'].values + date11_weather_data[
        'wind_speed'].values) / 2
    date10_weather_data['temperature'] = (date9_weather_data['temperature'].values + date11_weather_data[
        'temperature'].values) / 2
    date10_weather_data['rel_humidity'] = (date9_weather_data['rel_humidity'].values + date11_weather_data[
        'rel_humidity'].values) / 2
    date10_weather_data['precipitation'] = (date9_weather_data['precipitation'].values + date11_weather_data[
        'precipitation'].values) / 2

    weather_data = pd.concat([weather_data, date10_weather_data], axis=0)

    weather_data.index = np.arange(len(weather_data))


    # 整合平均时间与天气数据
    raw_data['start_time'] = raw_data['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))
    weather_data['date'] = weather_data['date'].astype(str)
    weather_data['hour'] = weather_data['hour'].map(lambda x: ' ' + time(x, 0, 0).strftime('%H:%M:%S'))
    weather_data['start_time'] = weather_data['date'] + weather_data['hour']
    weather_data['start_time'] = weather_data['start_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    del weather_data['date']
    del weather_data['hour']
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

    process_data = pd.merge(raw_data, weather_data, on='start_time', how='left')
    process_data['time'] = process_data['start_time'].map(lambda x: x.time())
    # 天气缺失补充
    process_data.fillna(method='ffill', inplace=True)
    process_data['date'] = process_data['start_time'].map(lambda x: x.date())

    # 增加路线和星期几，小时，分钟
    process_data['weekday'] = process_data['start_time'].map(lambda x: x.weekday())
    process_data['hour'] = process_data['start_time'].map(lambda x: x.hour)
    process_data['minute'] = process_data['start_time'].map(lambda x: x.minute)
    process_data['route'] = process_data['intersection_id'].astype(str) + '-' + process_data['tollgate_id'].astype(str)

    # 增加前2小时平均时间数据
    start_time = process_data['start_time']
    avg_travel_time = process_data['avg_travel_time']
    last_20min = process_data['avg_travel_time'].copy()
    last_40min = process_data['avg_travel_time'].copy()
    last_60min = process_data['avg_travel_time'].copy()
    last_80min = process_data['avg_travel_time'].copy()
    last_100min = process_data['avg_travel_time'].copy()
    last_120min = process_data['avg_travel_time'].copy()
    last_20min[:] = np.nan
    last_40min[:] = np.nan
    last_60min[:] = np.nan
    last_80min[:] = np.nan
    last_100min[:] = np.nan
    last_120min[:] = np.nan
    for i in range(len(start_time)):
        cur_time = start_time[i]  # 当前时间
        # 前20min
        if (i - 1) >= 0:
            last20 = cur_time - timedelta(minutes=20)
            if last20 == start_time[i - 1]:
                last_20min[i] = avg_travel_time[i - 1]
        # 前40min
        if (i - 2) >= 0:
            last40 = cur_time - timedelta(minutes=40)
            for j in range(2):
                if last40 == start_time[i - j - 1]:
                    last_40min[i] = avg_travel_time[i - j - 1]
                    break
        # 前60min
        if (i - 3) >= 0:
            last60 = cur_time - timedelta(minutes=60)
            for j in range(3):
                if last60 == start_time[i - j - 1]:
                    last_60min[i] = avg_travel_time[i - j - 1]
                    break
        # 前80min
        if (i - 4) >= 0:
            last80 = cur_time - timedelta(minutes=80)
            for j in range(4):
                if last80 == start_time[i - j - 1]:
                    last_80min[i] = avg_travel_time[i - j - 1]
                    break
        # 前100min
        if (i - 5) >= 0:
            last100 = cur_time - timedelta(minutes=100)
            for j in range(5):
                if last100 == start_time[i - j - 1]:
                    last_100min[i] = avg_travel_time[i - j - 1]
                    break
        # 前120min
        if (i - 6) >= 0:
            last120 = cur_time - timedelta(minutes=120)
            for j in range(6):
                if last120 == start_time[i - j - 1]:
                    last_120min[i] = avg_travel_time[i - j - 1]
                    break
    process_data['last_20min'] = last_20min
    process_data['last_40min'] = last_40min
    process_data['last_60min'] = last_60min
    process_data['last_80min'] = last_80min
    process_data['last_100min'] = last_100min
    process_data['last_120min'] = last_120min

    # 把time映射成1～72
    from collections import defaultdict
    time_start = datetime(2016, 10, 17, 0, 0, 0)
    timedic = defaultdict(int)
    for i in range(72):
        timedic[time_start.time()] = i + 1
        time_start = time_start + timedelta(minutes=20)
    process_data['timemap'] = process_data['time'].map(lambda x: timedic[x])

  # 增加法定节假日特征
    holiday = ['2016-09-15', '2016-09-16', '2016-09-17', '2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04',
               '2016-10-05', '2016-10-06', '2016-10-07']
    def ff(x):
        if x in holiday:
            return 1
        else:
            return 0
    process_data['holiday'] = process_data['date'].map(ff)

    # 写出数据
    process_data.to_csv('/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/travel_time_raw_data.csv', index=False)


    #####**********提取流量数据**********#####

    # 整合流量与天气数据
    raw_data2['start_time'] = raw_data2['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S'))
    raw_data2['pair'] = raw_data2['tollgate_id'].astype(str) + '-' + raw_data2['direction'].astype(str)
    process_data2 = pd.merge(raw_data2, weather_data, on='start_time', how='left')
    process_data2['date'] = process_data2['start_time'].map(lambda x: x.date())
    # 天气缺失补充
    process_data2.fillna(method='ffill', inplace=True)

    # 增加星期几和时间窗口两列
    process_data2['weekday'] = process_data2['start_time'].map(lambda x: x.weekday())
    process_data2['time'] = process_data2['start_time'].map(lambda x: x.time())
    process_data2['hour'] = process_data2['start_time'].map(lambda x: x.hour)
    process_data2['minute'] = process_data2['start_time'].map(lambda x: x.minute)

    # 增加前20分钟流量特征
    start_time = process_data2['start_time']
    volume = process_data2['volume']
    last_20min = process_data2['volume'].copy()
    last_40min = process_data2['volume'].copy()
    last_60min = process_data2['volume'].copy()
    last_80min = process_data2['volume'].copy()
    last_100min = process_data2['volume'].copy()
    last_120min = process_data2['volume'].copy()
    last_20min[:] = np.nan
    last_40min[:] = np.nan
    last_60min[:] = np.nan
    last_80min[:] = np.nan
    last_100min[:] = np.nan
    last_120min[:] = np.nan
    for i in range(len(start_time)):
        cur_time = start_time[i] #当前时间
        # 前20min
        if (i-1) >= 0:
            last20 = cur_time - timedelta(minutes=20)
            if last20 == start_time[i-1]:
                last_20min[i] = volume[i-1]
        # 前40min
        if (i-2) >= 0:
            last40 = cur_time - timedelta(minutes=40)
            for j in range(2):
                if last40 == start_time[i-j-1]:
                    last_40min[i] = volume[i-j-1]
                    break
        # 前60min
        if (i-3) >= 0:
            last60 = cur_time - timedelta(minutes=60)
            for j in range(3):
                if last60 == start_time[i-j-1]:
                    last_60min[i] = volume[i-j-1]
                    break
        # 前80min
        if (i-4) >= 0:
            last80 = cur_time - timedelta(minutes=80)
            for j in range(4):
                if last80 == start_time[i-j-1]:
                    last_80min[i] = volume[i-j-1]
                    break
        # 前100min
        if (i-5) >= 0:
            last100 = cur_time - timedelta(minutes=100)
            for j in range(5):
                if last100 == start_time[i-j-1]:
                    last_100min[i] = volume[i-j-1]
                    break
        # 前120min
        if (i-6) >= 0:
            last120 = cur_time - timedelta(minutes=120)
            for j in range(6):
                if last120 == start_time[i-j-1]:
                    last_120min[i] = volume[i-j-1]
                    break
    process_data2['last_20min'] = last_20min
    process_data2['last_40min'] = last_40min
    process_data2['last_60min'] = last_60min
    process_data2['last_80min'] = last_80min
    process_data2['last_100min'] = last_100min
    process_data2['last_120min'] = last_120min

    # 时间映射成1～72
    process_data2['timemap'] = process_data2['time'].map(lambda x: timedic[x])

    # holiday特征
    process_data2['holiday'] = process_data2['date'].map(ff)

    # 写出数据
    process_data2.to_csv('/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/volume_raw_data.csv', index=False)


  ###----------------测试集------------------###


    #####**********平均时间**********#####

    # 读入平均时间，流量和天气数据（平均时间和流量由aggregate_travel_time.py和aggregate_volume.py生成）
    test1 = pd.read_csv(test_travel_time_infile)
    test2 = pd.read_csv(test_volume_infile)
    weather_data = pd.read_csv(test_weather_infile)

    # 测试特征集
    test_travel_time = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/submission_sample_travelTime.csv')
    test_volume = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_sample/submission_sample_volume.csv')

    # start_time
    del test_travel_time['avg_travel_time']
    test_travel_time['start_time'] = test_travel_time['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S'))

    # 天气特征
    weather_data['hour'] = weather_data['hour'].map(lambda x: ' ' + time(x, 0, 0).strftime('%H:%M:%S'))
    weather_data['start_time'] = weather_data['date'] + weather_data['hour']
    weather_data['start_time'] = weather_data['start_time'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    del weather_data['date']
    del weather_data['hour']
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

    test_travel_time = pd.merge(test_travel_time, weather_data, on='start_time', how='left')
    test_travel_time['date'] = test_travel_time['start_time'].map(lambda x: x.date())
    test_travel_time['time'] = test_travel_time['start_time'].map(lambda x: x.time())
    test_travel_time['hour'] = test_travel_time['start_time'].map(lambda x: x.hour)
    test_travel_time['minute'] = test_travel_time['start_time'].map(lambda x: x.minute)

    # weekday特征
    test_travel_time['weekday'] = test_travel_time['start_time'].map(lambda x: x.weekday())

    # route特征
    test_travel_time['route'] = test_travel_time['intersection_id'].astype(str) + '-' + test_travel_time['tollgate_id'].astype(str)

    # last_20min特征
    test1['start_time'] = test1['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S') + timedelta(hours=2))
    del test1['time_window']
    test_for_last20min = pd.merge(test_travel_time, test1, on=['intersection_id','tollgate_id','start_time'], how='left')

    """
    ### 以均值填充缺失值
    fill_mean = lambda g:g.fillna(g.mean())
    test_for_last20min['avg_travel_time'] = test_for_last20min.groupby(['route','time'])['avg_travel_time'].apply(fill_mean)
    """
    ### 以最近值填充缺失值
    ffill = lambda g:g.fillna(method='ffill')
    bfill = lambda g:g.fillna(method='bfill')

    test_for_last20min['last_20min'] = test_for_last20min.groupby(['route','time'])['avg_travel_time'].apply(ffill)
    test_for_last20min['last_20min'] = test_for_last20min.groupby(['route','time'])['avg_travel_time'].apply(bfill)

    last_travel_time = test_for_last20min['avg_travel_time']
    last_20min = last_travel_time.copy()
    last_40min = last_travel_time.copy()
    last_60min = last_travel_time.copy()
    last_80min = last_travel_time.copy()
    last_100min = last_travel_time.copy()
    last_120min = last_travel_time.copy()
    last_20min[:] = np.nan
    last_40min[:] = np.nan
    last_60min[:] = np.nan
    last_80min[:] = np.nan
    last_100min[:] = np.nan
    last_120min[:] = np.nan
    for i in range(len(last_travel_time)):
        if i % 6 == 0:
            last_20min[i] = last_travel_time[i+5]
            last_40min[i] = last_travel_time[i+4]
            last_60min[i] = last_travel_time[i+3]
            last_80min[i] = last_travel_time[i+2]
            last_100min[i] = last_travel_time[i+1]
            last_120min[i] = last_travel_time[i]
        if i % 6 == 1:
            last_40min[i] = last_travel_time[i+4]
            last_60min[i] = last_travel_time[i+3]
            last_80min[i] = last_travel_time[i+2]
            last_100min[i] = last_travel_time[i+1]
            last_120min[i] = last_travel_time[i]
        if i % 6 == 2:
            last_60min[i] = last_travel_time[i+3]
            last_80min[i] = last_travel_time[i+2]
            last_100min[i] = last_travel_time[i+1]
            last_120min[i] = last_travel_time[i]
        if i % 6 == 3:
            last_80min[i] = last_travel_time[i+2]
            last_100min[i] = last_travel_time[i+1]
            last_120min[i] = last_travel_time[i]
        if i % 6 == 4:
            last_100min[i] = last_travel_time[i+1]
            last_120min[i] = last_travel_time[i]
        if i % 6 == 5:
            last_120min[i] = last_travel_time[i]
    test_travel_time['last_20min'] = last_20min
    test_travel_time['last_40min'] = last_40min
    test_travel_time['last_60min'] = last_60min
    test_travel_time['last_80min'] = last_80min
    test_travel_time['last_100min'] = last_100min
    test_travel_time['last_120min'] = last_120min

    # 把time映射成1～72
    from collections import defaultdict
    time_start = datetime(2016,10,17,0,0,0)
    timedic = defaultdict(int)
    for i in range(72):
        timedic[time_start.time()] = i+1
        time_start = time_start + timedelta(minutes=20)
    test_travel_time['timemap'] = test_travel_time['time'].map(lambda x:timedic[x])

    # 写出数据
    test_travel_time.to_csv('/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/test_travel_time_feature_ffill.csv', index=False)


    #####**********流量**********#####

    del test_volume['volume']
    # start_time特征
    test_volume['start_time'] = test_volume['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S'))

    test_volume = test_volume.sort(['tollgate_id', 'direction', 'start_time'])
    # pair特征
    test_volume['pair'] = test_volume['tollgate_id'].astype(str) + '-' + test_volume['direction'].astype(str)

    # 天气特征
    test_volume = pd.merge(test_volume, weather_data, on='start_time', how='left')

    # 星期几
    test_volume['weekday'] = test_volume['start_time'].map(lambda x: x.weekday())

    # 时间窗口
    test_volume['time'] = test_volume['start_time'].map(lambda x: x.time())
    test_volume['date'] = test_volume['start_time'].map(lambda x: x.date())

    del test2['etc']
    test_volume['hour'] = test_volume['start_time'].map(lambda x: x.hour)
    test_volume['minute'] = test_volume['start_time'].map(lambda x: x.minute)
    # last_20min
    test2['start_time'] = test2['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:], '%Y-%m-%d %H:%M:%S')+timedelta(hours=2))
    del test2['time_window']
    test2 = test2.sort(['tollgate_id','direction'])
    test_for_last20min2 = pd.merge(test_volume, test2, on=['tollgate_id','direction','start_time'], how='left')
    last_volume = test_for_last20min2['volume']
    last_20min = last_volume.copy()
    last_40min = last_volume.copy()
    last_60min = last_volume.copy()
    last_80min = last_volume.copy()
    last_100min = last_volume.copy()
    last_120min = last_volume.copy()
    last_20min[:] = np.nan
    last_40min[:] = np.nan
    last_60min[:] = np.nan
    last_80min[:] = np.nan
    last_100min[:] = np.nan
    last_120min[:] = np.nan
    for i in range(len(last_volume)):
        if i % 6 == 0:
            last_20min[i] = last_volume[i+5]
            last_40min[i] = last_volume[i+4]
            last_60min[i] = last_volume[i+3]
            last_80min[i] = last_volume[i+2]
            last_100min[i] = last_volume[i+1]
            last_120min[i] = last_volume[i]
        if i % 6 == 1:
            last_40min[i] = last_volume[i+4]
            last_60min[i] = last_volume[i+3]
            last_80min[i] = last_volume[i+2]
            last_100min[i] = last_volume[i+1]
            last_120min[i] = last_volume[i]
        if i % 6 == 2:
            last_60min[i] = last_volume[i+3]
            last_80min[i] = last_volume[i+2]
            last_100min[i] = last_volume[i+1]
            last_120min[i] = last_volume[i]
        if i % 6 == 3:
            last_80min[i] = last_volume[i+2]
            last_100min[i] = last_volume[i+1]
            last_120min[i] = last_volume[i]
        if i % 6 == 4:
            last_100min[i] = last_volume[i+1]
            last_120min[i] = last_volume[i]
        if i % 6 == 5:
            last_120min[i] = last_volume[i]
    test_volume['last_20min'] = last_20min
    test_volume['last_40min'] = last_40min
    test_volume['last_60min'] = last_60min
    test_volume['last_80min'] = last_80min
    test_volume['last_100min'] = last_100min
    test_volume['last_120min'] = last_120min

    # 无缺失值

    # 把time映射成1～72
    from collections import defaultdict
    time_start = datetime(2016,10,17,0,0,0)
    timedic = defaultdict(int)
    for i in range(72):
        timedic[time_start.time()] = i+1
        time_start = time_start + timedelta(minutes=20)
    test_volume['timemap'] = test_volume['time'].map(lambda x:timedic[x])

    # 写出数据
    test_volume.to_csv('/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/数据提取与合并/test_volume_feature_ffill.csv', index=False)


def main():

    travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/初始形成时间窗的数据集/training_20min_avg_travel_time.csv'
    volume_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/初始形成时间窗的数据集/training_20min_avg_volume.csv'
    weather_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/初始形成时间窗的数据集/weather (table 7)_training_update.csv'
    test_travel_time_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/初始形成时间窗的数据集/test1_20min_avg_travel_time.csv'
    test_volume_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/初始形成时间窗的数据集/test1_20min_avg_volume.csv'
    test_weather_infile = '/home/godcedric/GitLocal/KDDCUP2017/待加工数据集/初始形成时间窗的数据集/weather (table 7)_test1.csv'

    extract_data(travel_time_infile, volume_infile, weather_infile, test_travel_time_infile, test_volume_infile, test_weather_infile)


if __name__ == '__main__':
    main()
