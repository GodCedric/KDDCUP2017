# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Mean模型
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date,time

travel_time_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/travel_time_train_data.csv')
volume_train_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.0/volume_train_data.csv')
travel_time_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/test_travel_time_data.csv')
volume_test_data = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/加工好的数据/6.5/test_volume_data.csv')

travel_time_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/原始数据/submission_sample_travelTime.csv')
volume_submission = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/原始数据/submission_sample_volume.csv')


### 取平均预测

travel_time = travel_time_train_data[['avg_travel_time', 'route', 'start_time']]
travel_time['start_time'] = pd.to_datetime(travel_time['start_time'], format='%Y-%m-%d %H:%M:%S')
travel_time['hour'] = travel_time['start_time'].map(lambda x: x.hour)
# 留下8,9,17,18点的数据
travel_time = travel_time[(travel_time['hour']==8) | (travel_time['hour']==9) | (travel_time['hour']==17) | (travel_time['hour']==18)]

travel_time = travel_time.sort_values(by=['route', 'start_time'])
travel_time.index = np.arange(len(travel_time))

travel_time['time'] = travel_time['start_time'].map(lambda x: x.time())

del travel_time['start_time']
del travel_time['hour']

group_by_route = travel_time.groupby(['route', 'time'])
# 不分星期几
predict_value = group_by_route.mean().values
temp1 = predict_value[0:12]
temp2 = predict_value[12:24]
temp3 = predict_value[24:36]
temp4 = predict_value[36:48]
temp5 = predict_value[48:60]
temp6 = predict_value[60:72]
temp = np.concatenate((temp1, temp1, temp1, temp1, temp1, temp1, temp1, \
                       temp2, temp2, temp2, temp2, temp2, temp2, temp2, \
                       temp3, temp3, temp3, temp3, temp3, temp3, temp3, \
                       temp4, temp4, temp4, temp4, temp4, temp4, temp4, \
                       temp5, temp5, temp5, temp5, temp5, temp5, temp5,
                       temp6, temp6, temp6, temp6, temp6, temp6, temp6), axis=0)

travel_time_submission['avg_travel_time'] = temp

volume = volume_train_data[['volume', 'pair', 'start_time']]
volume['start_time'] = pd.to_datetime(volume['start_time'], format='%Y-%m-%d %H:%M:%S')
volume['hour'] = volume['start_time'].map(lambda x: x.hour)
# 留下8,9,17,18点的数据
volume = volume[(volume['hour']==8) | (volume['hour']==9) | (volume['hour']==17) | (volume['hour']==18)]

volume = volume.sort_values(by=['pair', 'start_time'])
volume.index = np.arange(len(volume))

volume['time'] = volume['start_time'].map(lambda x: x.time())
del volume['start_time']
del volume['hour']
group_by_pair = volume.groupby(['pair', 'time'])

# 不分星期几
predict_value = group_by_pair.mean().values
temp1 = predict_value[0:12]
temp2 = predict_value[12:24]
temp3 = predict_value[24:36]
temp4 = predict_value[36:48]
temp5 = predict_value[48:60]
temp = np.concatenate((temp1, temp1, temp1, temp1, temp1, temp1, temp1, \
                       temp2, temp2, temp2, temp2, temp2, temp2, temp2, \
                       temp3, temp3, temp3, temp3, temp3, temp3, temp3, \
                       temp4, temp4, temp4, temp4, temp4, temp4, temp4, \
                       temp5, temp5, temp5, temp5, temp5, temp5, temp5, ), axis=0)

volume_submission['volume'] = temp
#travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/mean1.5/travel_time_submission.csv', index=False)
#volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/mean1.5/volume_submission.csv', index=False)


### last_2hours修正

# 读取前两小时的数据
# 读取前两小时的数据
travel_time_before2hour = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/时间窗数据/test_20min_avg_travel_time.csv')
volume_before2hour = pd.read_csv('/home/godcedric/GitLocal/KDDCUP2017/final_data/时间窗数据/test_20min_avg_volume.csv')

travel_time_before2hour['start_time'] = travel_time_before2hour['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S'))
volume_before2hour['start_time'] = volume_before2hour['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S'))

del travel_time_before2hour['time_window']
del volume_before2hour['time_window']

travel_time_before2hour = travel_time_before2hour.sort_values(by=['intersection_id', 'tollgate_id', 'start_time'])
travel_time_before2hour.index = np.arange(len(travel_time_before2hour))
volume_before2hour = volume_before2hour.sort_values(by=['tollgate_id', 'direction', 'start_time'])
volume_before2hour.index = np.arange(len(volume_before2hour))

travel_time_before2hour_pre = travel_time_submission.copy()
travel_time_before2hour_pre['start_time'] = travel_time_before2hour_pre['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S') - timedelta(hours=2))
travel_time_before2hour_pre['time_window'] = travel_time_before2hour_pre['start_time'].map(lambda x: '[' + str(x) + ',' + str(x+timedelta(minutes=20)) + ']')
travel_time_before2hour_pre = travel_time_before2hour_pre.drop(['avg_travel_time'], axis=1)

travel_time_before2hour = pd.merge(travel_time_before2hour_pre, travel_time_before2hour, on=['intersection_id', 'tollgate_id', 'start_time'], how='left')
del travel_time_before2hour['start_time']

volume_before2hour_pre = volume_submission.copy()
volume_before2hour_pre['start_time'] = volume_before2hour_pre['time_window'].map(lambda x: datetime.strptime(x.split(',')[0][1:],'%Y-%m-%d %H:%M:%S') - timedelta(hours=2))
volume_before2hour_pre['time_window'] = volume_before2hour_pre['start_time'].map(lambda x: '[' + str(x) + ',' + str(x+timedelta(minutes=20)) + ']')
volume_before2hour_pre = volume_before2hour_pre.drop(['volume'], axis=1)

volume_before2hour = pd.merge(volume_before2hour_pre, volume_before2hour, on=['tollgate_id', 'direction', 'start_time'], how='left')
#del volume_before2hour['etc']
del volume_before2hour['start_time']

# 获取前2小时的均值预测值
# 获取前2小时的均值预测值
travel_time = travel_time_train_data[['avg_travel_time', 'route', 'start_time']]
travel_time['start_time'] = pd.to_datetime(travel_time['start_time'], format='%Y-%m-%d %H:%M:%S')
travel_time['hour'] = travel_time['start_time'].map(lambda x: x.hour)
travel_time = travel_time[(travel_time['hour']==6) | (travel_time['hour']==7) | (travel_time['hour']==15) | (travel_time['hour']==16)]

travel_time = travel_time.sort_values(by=['route', 'start_time'])
travel_time.index = np.arange(len(travel_time))

travel_time['time'] = travel_time['start_time'].map(lambda x: x.time())
del travel_time['start_time']
del travel_time['hour']

group_by_route = travel_time.groupby(['route', 'time'])
predict_value = group_by_route.mean().values
temp1 = predict_value[0:12]
temp2 = predict_value[12:24]
temp3 = predict_value[24:36]
temp4 = predict_value[36:48]
temp5 = predict_value[48:60]
temp6 = predict_value[60:72]
temp = np.concatenate((temp1, temp1, temp1, temp1, temp1, temp1, temp1,
                       temp2, temp2, temp2, temp2, temp2, temp2, temp2,
                       temp3, temp3, temp3, temp3, temp3, temp3, temp3,
                       temp4, temp4, temp4, temp4, temp4, temp4, temp4,
                       temp5, temp5, temp5, temp5, temp5, temp5, temp5,
                       temp6, temp6, temp6, temp6, temp6, temp6, temp6), axis=0)

travel_time_before2hour['predict'] = temp


volume = volume_train_data[['volume', 'pair', 'start_time']]
volume['start_time'] = pd.to_datetime(volume['start_time'], format='%Y-%m-%d %H:%M:%S')
volume['hour'] = volume['start_time'].map(lambda x: x.hour)
volume = volume[(volume['hour']==6) | (volume['hour']==7) | (volume['hour']==15) | (volume['hour']==16)]

volume = volume.sort_values(by=['pair', 'start_time'])
volume.index = np.arange(len(volume))

volume['time'] = volume['start_time'].map(lambda x: x.time())
del volume['start_time']
del volume['hour']

group_by_pair = volume.groupby(['pair', 'time'])
predict_value = group_by_pair.mean().values
temp1 = predict_value[0:12]
temp2 = predict_value[12:24]
temp3 = predict_value[24:36]
temp4 = predict_value[36:48]
temp5 = predict_value[48:60]
temp = np.concatenate((temp1, temp1, temp1, temp1, temp1, temp1, temp1, \
                       temp2, temp2, temp2, temp2, temp2, temp2, temp2, \
                       temp3, temp3, temp3, temp3, temp3, temp3, temp3, \
                       temp4, temp4, temp4, temp4, temp4, temp4, temp4, \
                       temp5, temp5, temp5, temp5, temp5, temp5, temp5, ), axis=0)

volume_before2hour['predict'] = temp

"""
###  均值修正

travel_time_before2hour = travel_time_before2hour.fillna(method = 'ffill')
travel_time_before2hour['delta'] = travel_time_before2hour['avg_travel_time'] - travel_time_before2hour['predict']

# 取均值修正
correct = travel_time_before2hour['delta'].copy().values
for i in range(6, len(correct)+1):
    if (i%6) == 0:
        co = (correct[i-6] + correct[i-5] + correct[i-4] + correct[i-3] + correct[i-2] + correct[i-1]) / 6
        correct[i-6:i] = round(co, 2)

travel_time_submission['avg_travel_time'] = travel_time_submission['avg_travel_time'] + correct


volume_before2hour['delta'] = volume_before2hour['volume'] - volume_before2hour['predict']

# 取均值修正
correct = volume_before2hour['delta'].copy().values
for i in range(6, len(correct)+1):
    if (i%6) == 0:
        co = (correct[i-6] + correct[i-5] + correct[i-4] + correct[i-3] + correct[i-2] + correct[i-1]) / 6
        correct[i-6:i] = round(co, 2)

volume_submission['volume'] = volume_submission['volume'] + correct

travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/mean1.5/2小时修正/travel_time_submission.csv', index=False)
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase1.5/mean1.5/2小时修正/volume_submission.csv', index=False)
"""

### 系数修正

travel_time_before2hour = travel_time_before2hour.fillna(method = 'ffill')
travel_time_before2hour['delta'] = travel_time_before2hour['avg_travel_time'] - travel_time_before2hour['predict']

correct = travel_time_before2hour['delta'].copy().values

# 不需要修正的路段
correct[0:84] = 0     #A2
correct[84:168] = 0   #A3
correct[168:252] = 0  #B1
correct[252:336] = 0  #B3
correct[336:420] = 0  #C1
correct[420:504] = 0  #C3

for i in range(6, len(correct)+1):
    if (i%6) == 0:
        co = 0.5*correct[i-6] + 0.25*correct[i-5] + 0.1*correct[i-4] + 0.05*correct[i-3] + 0.05*correct[i-2] + 0.05*correct[i-1]
        correct[i-6:i] = round(co, 2)

travel_time_submission['avg_travel_time'] = travel_time_submission['avg_travel_time'] + correct

volume_before2hour['delta'] = volume_before2hour['volume'] - volume_before2hour['predict']

correct = volume_before2hour['delta'].copy().values

# 不需要修正的pair
correct[0:84] = 0     #10
correct[84:168] = 0   #11
correct[168:252] = 0  #20
correct[252:336] = 0  #30
correct[336:420] = 0  #31
for i in range(6, len(correct)+1):
    if (i%6) == 0:
        co = 0.5*correct[i-6] + 0.25*correct[i-5] + 0.1*correct[i-4] + 0.05*correct[i-3] + 0.05*correct[i-2] + 0.05*correct[i-1]
        correct[i-6:i] = round(co, 2)

volume_submission['volume'] = volume_submission['volume'] + correct

travel_time_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.0/Mean/travel_time_submission.csv', index=False)
volume_submission.to_csv('/home/godcedric/GitLocal/KDDCUP2017/submission_result/phase2.0/Mean/volume_submission.csv', index=False)